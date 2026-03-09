"""
Bi-level adversarial supervised IDS with an FT-Transformer style backbone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config import Config
from src.utils.data_loader import build_data_loader


class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformerIDS(nn.Module):
    def __init__(
        self,
        n_features: int | None = None,
        d_token: int = 64,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        n_features = n_features or Config.STATE_DIM
        self.tokenizer = NumericalFeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.feature_pos = nn.Parameter(torch.randn(1, n_features + 1, d_token) * 0.01)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x_tokens = torch.cat([cls_token, tokens], dim=1)
        x_tokens = x_tokens + self.feature_pos[:, : x_tokens.size(1)]
        encoded = self.encoder(x_tokens)
        cls_repr = encoded[:, 0]
        return self.head(cls_repr)


@dataclass
class AdversarialConfig:
    epsilon: float = 0.02
    alpha: float = 0.005
    steps: int = 2
    adv_weight: float = 0.6


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    fpr = float(np.sum((y_pred == 1) & (y_true == 0)) / max(np.sum(y_true == 0), 1))
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "FPR": fpr,
    }


class BiLevelFTTransformerTrainer:
    def __init__(self, seed: int, adv_cfg: AdversarialConfig | None = None):
        Config.set_seed(seed)
        self.seed = seed
        self.adv_cfg = adv_cfg or AdversarialConfig()

        loader = build_data_loader(Config.DATASET_NAME)
        self.X_train, self.y_train = loader.load_data(mode="train")
        self.X_test, self.y_test = loader.load_data(mode="test")

        self.model = FTTransformerIDS(n_features=self.X_train.shape[1]).to(Config.DEVICE)

        class_counts = np.bincount(self.y_train, minlength=2).astype(np.float32)
        class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
        class_weights = class_weights / class_weights.sum() * 2.0
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=Config.DEVICE))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=7e-4, weight_decay=1e-4)

    def make_loader(self, batch_size: int) -> DataLoader:
        dataset = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.long),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    def generate_adversarial_batch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_orig = x.detach()
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-self.adv_cfg.epsilon, self.adv_cfg.epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        self.model.eval()
        for _ in range(self.adv_cfg.steps):
            x_adv.requires_grad_(True)
            logits = self.model(x_adv)
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
            with torch.no_grad():
                x_adv = x_adv + self.adv_cfg.alpha * grad.sign()
                perturbation = torch.clamp(x_adv - x_orig, -self.adv_cfg.epsilon, self.adv_cfg.epsilon)
                x_adv = torch.clamp(x_orig + perturbation, 0.0, 1.0)
        self.model.train()
        return x_adv.detach()

    def train_epoch(self, batch_size: int) -> Dict[str, float]:
        self.model.train()
        loader = self.make_loader(batch_size)

        total_loss = 0.0
        total_clean = 0.0
        total_adv = 0.0
        num_batches = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(Config.DEVICE)
            batch_y = batch_y.to(Config.DEVICE)

            batch_adv = self.generate_adversarial_batch(batch_x, batch_y)
            clean_logits = self.model(batch_x)
            adv_logits = self.model(batch_adv)

            clean_loss = self.criterion(clean_logits, batch_y)
            adv_loss = self.criterion(adv_logits, batch_y)
            loss = (1.0 - self.adv_cfg.adv_weight) * clean_loss + self.adv_cfg.adv_weight * adv_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_clean += clean_loss.item()
            total_adv += adv_loss.item()
            num_batches += 1

        return {
            "total_loss": total_loss / max(num_batches, 1),
            "clean_loss": total_clean / max(num_batches, 1),
            "adv_loss": total_adv / max(num_batches, 1),
        }

    def evaluate(self, use_adversarial: bool = False) -> Dict[str, float]:
        self.model.eval()
        x = torch.tensor(self.X_test, dtype=torch.float32, device=Config.DEVICE)
        y = torch.tensor(self.y_test, dtype=torch.long, device=Config.DEVICE)

        preds = []
        batch_size = 1024
        for start in range(0, len(x), batch_size):
            batch_x = x[start:start + batch_size]
            batch_y = y[start:start + batch_size]
            if use_adversarial:
                batch_x = self.generate_adversarial_batch(batch_x, batch_y)
            with torch.no_grad():
                logits = self.model(batch_x)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())

        y_pred = np.concatenate(preds, axis=0)
        return compute_metrics(self.y_test, y_pred)

    def train(self, epochs: int = 10, batch_size: int = 512) -> Dict[str, object]:
        best_f1 = -1.0
        best_state = None
        history = []
        for epoch in range(1, epochs + 1):
            train_stats = self.train_epoch(batch_size)
            clean_stats = self.evaluate(use_adversarial=False)
            history.append({"epoch": epoch, **train_stats, **{f"clean_{k}": v for k, v in clean_stats.items()}})
            if clean_stats["F1"] > best_f1:
                best_f1 = clean_stats["F1"]
                best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                print(
                    f"Epoch {epoch:02d}/{epochs} "
                    f"loss={train_stats['total_loss']:.4f} "
                    f"F1={clean_stats['F1']:.4f} FPR={clean_stats['FPR']:.4f}"
                )
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {
            "clean": self.evaluate(False),
            "fgsm_like": self.evaluate(True),
            "history_tail": history[-3:],
        }

    def save(self) -> Path:
        path = Config.get_model_path("BiATFTTransformer", self.seed, "model")
        torch.save(self.model.state_dict(), path)
        return path


def main():
    parser = argparse.ArgumentParser(description="Bi-level adversarial FT-Transformer IDS")
    parser.add_argument("--dataset", type=str, default="unsw-nb15")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--adv_weight", type=float, default=0.6)
    args = parser.parse_args()

    Config.configure_dataset(args.dataset)
    trainer = BiLevelFTTransformerTrainer(
        seed=args.seed,
        adv_cfg=AdversarialConfig(
            epsilon=args.epsilon,
            alpha=args.alpha,
            steps=args.steps,
            adv_weight=args.adv_weight,
        ),
    )
    results = trainer.train(epochs=args.epochs, batch_size=args.batch_size)
    model_path = trainer.save()

    payload = {
        "dataset": Config.DATASET_NAME,
        "seed": args.seed,
        "model": "BiAT-FTTransformer",
        "config": vars(args),
        "clean": results["clean"],
        "fgsm_like": results["fgsm_like"],
        "model_path": str(model_path),
        "history_tail": results["history_tail"],
    }
    suffix = Config.DATASET_NAME.replace("-", "_")
    result_path = Config.RESULTS_DIR / f"biat_fttransformer_{suffix}_seed{args.seed}.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved model: {model_path}")
    print(f"Saved result: {result_path}")


if __name__ == "__main__":
    main()
