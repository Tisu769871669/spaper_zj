#!/usr/bin/env python
"""
Adversarial Robustness Evaluation: FGSM and PGD attacks on all model families.

Covers:
  - RL-based models: Bi-ARL, Vanilla PPO, MARL
  - Route-C models:  BiAT-MLP, BiAT-FTTransformer

Attack configurations evaluated:
  Clean | FGSM-eps0.1 | FGSM-eps0.3 | PGD-eps0.1 | PGD-eps0.3
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.data_loader import build_data_loader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


# ---------------------------------------------------------------------------
# Generic adversarial attack helpers (work on any nn.Module with logit output)
# ---------------------------------------------------------------------------

def fgsm_attack(model, x: torch.Tensor, y: torch.Tensor, epsilon: float) -> torch.Tensor:
    x = x.clone().detach().requires_grad_(True)
    y = y.to(x.device)
    loss = F.cross_entropy(model(x), y)
    model.zero_grad()
    loss.backward()
    return torch.clamp(x + epsilon * x.grad.sign(), 0.0, 1.0).detach()


def pgd_attack(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float = 0.01,
    steps: int = 40,
) -> torch.Tensor:
    x_orig = x.clone().detach()
    y = y.to(x.device)
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.clamp(x_orig + torch.clamp(x_adv - x_orig, -epsilon, epsilon), 0.0, 1.0).detach()
    return x_adv


def batch_attack(attack_fn, model, X: np.ndarray, y: np.ndarray, batch_size: int = 256, desc: str = "") -> np.ndarray:
    import time
    model.eval()
    out = []
    total_batches = (len(X) + batch_size - 1) // batch_size
    t0 = time.time()
    for batch_idx, i in enumerate(range(0, len(X), batch_size)):
        bx = torch.FloatTensor(X[i:i + batch_size]).to(Config.DEVICE)
        by = torch.LongTensor(y[i:i + batch_size]).to(Config.DEVICE)
        out.append(attack_fn(model, bx, by).cpu().numpy())
        # 每10个batch或最后一个batch打印进度
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.time() - t0
            pct = (batch_idx + 1) / total_batches
            eta = elapsed / pct * (1 - pct) if pct > 0 else 0
            print(f"    [{desc}] batch {batch_idx+1}/{total_batches} "
                  f"({pct*100:.0f}%)  elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                  flush=True)
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_rl_model(model_type: str, seed: int, model_file: str = "defender"):
    from src.agents.defender_agent import DefenderAgent

    path = Config.find_model_file(model_type, seed, model_file)
    if not path.exists():
        return None, path
    model = DefenderAgent().to(Config.DEVICE)
    model.load_state_dict(torch.load(path, map_location=Config.DEVICE, weights_only=True))
    model.eval()
    return model, path


def load_biat_mlp(seed: int, n_features: int):
    from src.baselines.bilevel_supervised_ids import MLPIDS

    path = Config.find_model_file("BiATMLP", seed, "model")
    if not path.exists():
        return None, path
    model = MLPIDS(input_dim=n_features).to(Config.DEVICE)
    model.load_state_dict(torch.load(path, map_location=Config.DEVICE, weights_only=True))
    model.eval()
    return model, path


def load_biat_fttransformer(seed: int, n_features: int, model_type_override: str = None):
    from src.baselines.bilevel_fttransformer_ids import FTTransformerIDS

    model_type = model_type_override or "BiATFTTransformer"
    path = Config.find_model_file(model_type, seed, "model")
    if not path.exists():
        return None, path
    model = FTTransformerIDS(n_features=n_features).to(Config.DEVICE)
    model.load_state_dict(torch.load(path, map_location=Config.DEVICE, weights_only=True))
    model.eval()
    return model, path


# ---------------------------------------------------------------------------
# RL-model prediction wrapper (maps discrete action index → binary label)
# ---------------------------------------------------------------------------

def rl_predict_batch(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        batch_size = 512
        for i in range(0, len(X), batch_size):
            bx = torch.FloatTensor(X[i:i + batch_size]).to(Config.DEVICE)
            actions = torch.argmax(model(bx), dim=1).cpu().numpy()
            preds.append((actions >= 5).astype(int))
    return np.concatenate(preds)


def supervised_predict_batch(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        batch_size = 1024
        for i in range(0, len(X), batch_size):
            bx = torch.FloatTensor(X[i:i + batch_size]).to(Config.DEVICE)
            preds.append(torch.argmax(model(bx), dim=1).cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    fpr = float(np.sum((y_pred == 1) & (y_true == 0)) / max(np.sum(y_true == 0), 1))
    return {
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "FPR": fpr,
    }


def evaluate_model_robustness(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    predict_fn,
    model_name: str,
    seed: int,
    is_rl: bool,
) -> list:
    """Run all attack conditions and return list of result dicts."""
    rows = []

    import argparse as _ap
    _ns = _ap.Namespace(fgsm_only=getattr(main, "_fgsm_only", False))
    if _ns.fgsm_only:
        attack_configs = [
            ("Clean",        None,        0.0),
            ("FGSM-eps0.1",  "fgsm",      0.1),
            ("FGSM-eps0.3",  "fgsm",      0.3),
        ]
    else:
        attack_configs = [
            ("Clean",        None,        0.0),
            ("FGSM-eps0.1",  "fgsm",      0.1),
            ("FGSM-eps0.3",  "fgsm",      0.3),
            ("PGD-eps0.1",   "pgd",       0.1),
            ("PGD-eps0.3",   "pgd",       0.3),
        ]

    import time
    for attack_name, attack_type, eps in attack_configs:
        t_start = time.time()
        print(f"  >> [{model_name} seed={seed}] 开始攻击: {attack_name} ...", flush=True)
        if attack_type is None:
            X_eval = X_test
        elif attack_type == "fgsm":
            attack_fn = lambda m, bx, by, e=eps: fgsm_attack(m, bx, by, e)
            X_eval = batch_attack(attack_fn, model, X_test, y_test,
                                  desc=f"{model_name}/{attack_name}")
        else:  # pgd
            # Use fewer steps for large supervised models (FTTransformer) to reduce eval time
            pgd_steps = 20 if "FTTransformer" in model_name else 40
            attack_fn = lambda m, bx, by, e=eps, s=pgd_steps: pgd_attack(m, bx, by, e, alpha=eps / 10, steps=s)
            X_eval = batch_attack(attack_fn, model, X_test, y_test,
                                  desc=f"{model_name}/{attack_name}")

        y_pred = predict_fn(model, X_eval)
        metrics = compute_metrics(y_test, y_pred)
        elapsed = time.time() - t_start
        rows.append({
            "Model": model_name,
            "Seed": seed,
            "Attack": attack_name,
            "Epsilon": eps,
            **metrics,
        })
        print(
            f"  {attack_name:<15}  "
            f"Recall={metrics['Recall']:.4f}  "
            f"F1={metrics['F1']:.4f}  "
            f"FPR={metrics['FPR']:.4f}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Adversarial robustness evaluation (FGSM + PGD)")
    parser.add_argument("--dataset", type=str, default="nsl-kdd",
                        help="Dataset: nsl-kdd / unsw-nb15")
    parser.add_argument("--fgsm_only", action="store_true",
                        help="Only run FGSM attacks (skip slow PGD for large models)")
    args = parser.parse_args()
    main._fgsm_only = args.fgsm_only

    Config.configure_dataset(args.dataset)

    print("\n" + "=" * 70)
    print(f"  Adversarial Robustness Evaluation  [{args.dataset}]")
    print("=" * 70 + "\n")

    loader = build_data_loader(Config.DATASET_NAME)
    loader.load_data(mode="train")
    X_test, y_test = loader.load_data(mode="test")
    n_features = X_test.shape[1]
    print(f"Test set: {len(X_test)} samples, {n_features} features\n")

    # (model_display_name, loader_fn, predict_fn, model_type, model_file)
    model_specs = [
        ("Bi-ARL",            "rl",      "BiARL",             "defender"),
        ("Vanilla PPO",        "rl",      "VanillaPPO",        "model"),
        ("MARL",               "rl",      "MARL",              "defender"),
        ("BiAT-MLP",           "biat_mlp",  None,              None),
        ("Transformer-IDS",    "biat_ft",   None,              "transformer_ids"),
        ("BiAT-FTTransformer", "biat_ft",   None,              None),
    ]

    all_rows = []

    for model_name, family, model_type, model_file in model_specs:
        print(f"\n{'=' * 70}")
        print(f"  Model: {model_name}")
        print("=" * 70)

        for seed in Config.SEEDS:
            print(f"\n  --- Seed {seed} ---")

            if family == "rl":
                model, path = load_rl_model(model_type, seed, model_file)
                predict_fn = rl_predict_batch
            elif family == "biat_mlp":
                model, path = load_biat_mlp(seed, n_features)
                predict_fn = supervised_predict_batch
            elif family == "biat_ft":
                # model_file holds optional model_type override (e.g. "transformer_ids" -> "TransformerIDS")
                override = None
                if model_file == "transformer_ids":
                    override = "TransformerIDS"
                model, path = load_biat_fttransformer(seed, n_features, model_type_override=override)
                predict_fn = supervised_predict_batch
            else:
                print(f"  [Skip] Unknown family: {family}")
                continue

            if model is None:
                print(f"  [Skip] Model not found: {path}")
                continue

            rows = evaluate_model_robustness(
                model, X_test, y_test, predict_fn, model_name, seed, is_rl=(family == "rl")
            )
            all_rows.extend(rows)

    if not all_rows:
        print("\n[Warning] No models were evaluated. Check that model files exist.")
        return

    df = pd.DataFrame(all_rows)
    suffix = Config.DATASET_NAME.replace("-", "_")

    detailed_path = Config.RESULTS_DIR / f"robustness_detailed_{suffix}.csv"
    df.to_csv(detailed_path, index=False)

    summary_df = (
        df.groupby(["Model", "Attack", "Epsilon"])[["Recall", "Precision", "F1", "FPR"]]
        .agg(["mean", "std"])
        .round(4)
        .reset_index()
    )
    summary_df.columns = [
        "_".join(c).strip("_") if c[1] else c[0]
        for c in summary_df.columns.values
    ]
    summary_path = Config.RESULTS_DIR / f"robustness_summary_{suffix}.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n\n" + "=" * 70)
    print("  Robustness Summary (Mean across seeds)")
    print("=" * 70)
    print(f"\n{'Model':<22} {'Attack':<15} {'Recall':>8} {'F1':>8} {'FPR':>8}")
    print("-" * 70)
    for _, row in summary_df.iterrows():
        print(
            f"{row['Model']:<22} {row['Attack']:<15} "
            f"{row['Recall_mean']:>8.4f} {row['F1_mean']:>8.4f} {row['FPR_mean']:>8.4f}"
        )

    print(f"\nDetailed results: {detailed_path}")
    print(f"Summary results:  {summary_path}")


if __name__ == "__main__":
    main()
