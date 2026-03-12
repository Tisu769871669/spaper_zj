"""
Vanilla FT-Transformer IDS baseline (clean supervised training, no adversarial training).

This module provides a direct ablation counterpart to BiAT-FTTransformer:
  same FT-Transformer backbone, but trained only on clean data without any
  bilevel adversarial perturbations.

This completes the ablation chain:
  Vanilla Transformer (here) → BiAT-MLP → BiAT-FTTransformer

Reference backbone: Gorishniy et al., "Revisiting Deep Learning Models for
  Tabular Data", NeurIPS 2021. (FT-Transformer)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config import Config
from src.utils.data_loader import build_data_loader
from src.baselines.bilevel_fttransformer_ids import FTTransformerIDS


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    fpr = float(np.sum((y_pred == 1) & (y_true == 0)) / max(np.sum(y_true == 0), 1))
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "FPR": fpr,
    }


class VanillaFTTransformerTrainer:
    """
    Clean-only supervised training with FT-Transformer backbone.
    No adversarial perturbations; serves as the ablation baseline for
    'does bilevel adversarial training help beyond just using FT-Transformer?'
    """

    def __init__(
        self,
        seed: int,
        *,
        d_token: int = 64,
        dropout: float = 0.15,
        learning_rate: float = 7e-4,
        weight_decay: float = 1e-4,
    ):
        Config.set_seed(seed)
        self.seed = seed

        loader = build_data_loader(Config.DATASET_NAME)
        self.X_train, self.y_train = loader.load_data(mode="train")
        self.X_test, self.y_test = loader.load_data(mode="test")

        self.model = FTTransformerIDS(
            n_features=self.X_train.shape[1],
            d_token=d_token,
            dropout=dropout,
        ).to(Config.DEVICE)

        class_counts = np.bincount(self.y_train, minlength=2).astype(np.float32)
        class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
        class_weights = class_weights / class_weights.sum() * 2.0
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, device=Config.DEVICE)
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-5
        )

    def make_loader(self, batch_size: int) -> DataLoader:
        dataset = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.long),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    def train_epoch(self, batch_size: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for batch_x, batch_y in self.make_loader(batch_size):
            batch_x = batch_x.to(Config.DEVICE)
            batch_y = batch_y.to(Config.DEVICE)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        self.scheduler.step()
        return total_loss / max(num_batches, 1)

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        preds = []
        batch_size = 1024
        with torch.no_grad():
            for i in range(0, len(self.X_test), batch_size):
                bx = torch.tensor(
                    self.X_test[i:i + batch_size], dtype=torch.float32, device=Config.DEVICE
                )
                preds.append(torch.argmax(self.model(bx), dim=1).cpu().numpy())
        y_pred = np.concatenate(preds)
        return compute_metrics(self.y_test, y_pred)

    def train(self, epochs: int = 10, batch_size: int = 512) -> Dict[str, object]:
        import time
        best_f1 = -1.0
        best_state = None
        history = []
        t0 = time.time()
        for epoch in range(1, epochs + 1):
            t_ep = time.time()
            loss = self.train_epoch(batch_size)
            metrics = self.evaluate()
            elapsed_ep = time.time() - t_ep
            elapsed_total = time.time() - t0
            history.append({"epoch": epoch, "loss": loss, **{f"clean_{k}": v for k, v in metrics.items()}})
            if metrics["F1"] > best_f1:
                best_f1 = metrics["F1"]
                best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            print(
                f"Epoch {epoch:02d}/{epochs}  loss={loss:.4f}  "
                f"F1={metrics['F1']:.4f}  FPR={metrics['FPR']:.4f}  "
                f"ep={elapsed_ep:.0f}s  total={elapsed_total:.0f}s",
                flush=True,
            )
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {"clean": self.evaluate(), "history_tail": history[-3:]}

    def save(self) -> Path:
        path = Config.get_model_path("TransformerIDS", self.seed, "model")
        torch.save(self.model.state_dict(), path)
        return path


def main():
    parser = argparse.ArgumentParser(description="Vanilla FT-Transformer IDS (clean supervised)")
    parser.add_argument("--dataset", type=str, default="unsw-nb15")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--d_token", type=int, default=64)
    args = parser.parse_args()

    Config.configure_dataset(args.dataset)
    trainer = VanillaFTTransformerTrainer(
        seed=args.seed,
        d_token=args.d_token,
        dropout=args.dropout,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    results = trainer.train(epochs=args.epochs, batch_size=args.batch_size)
    model_path = trainer.save()

    payload = {
        "dataset": Config.DATASET_NAME,
        "seed": args.seed,
        "model": "Transformer-IDS",
        "config": vars(args),
        "clean": results["clean"],
        "model_path": str(model_path),
        "history_tail": results["history_tail"],
    }
    suffix = Config.DATASET_NAME.replace("-", "_")
    result_path = Config.RESULTS_DIR / f"transformer_ids_{suffix}_seed{args.seed}.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved model: {model_path}")
    print(f"Saved result: {result_path}")


if __name__ == "__main__":
    main()
