#!/usr/bin/env python
"""
聚合主结果表: Clean 条件下各模型在多个随机种子上的 Mean ± Std。
"""

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.data_loader import build_data_loader


def compute_metrics(y_true, y_pred):
    fpr = np.sum((y_pred == 1) & (y_true == 0)) / np.sum(y_true == 0)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "FPR": fpr,
    }


def evaluate_defender_model(model_path, X_test, y_test):
    import torch
    from src.agents.defender_agent import DefenderAgent

    model = DefenderAgent().to(Config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE, weights_only=True))
    model.eval()

    predictions = []
    batch_size = 1024
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_x = torch.FloatTensor(X_test[i:i + batch_size]).to(Config.DEVICE)
            actions = torch.argmax(model(batch_x), dim=1).cpu().numpy()
            predictions.extend((actions >= 5).astype(int).tolist())

    return compute_metrics(y_test, np.array(predictions))


def evaluate_lstm_model(model_path, X_test, y_test):
    import torch
    from src.baselines.lstm_ids import LSTMIDS

    model = LSTMIDS().to(Config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE, weights_only=True))
    model.eval()

    predictions = []
    with torch.no_grad():
        batch_size = 512
        for i in range(0, len(X_test), batch_size):
            batch_x = torch.FloatTensor(X_test[i:i + batch_size]).unsqueeze(1).to(Config.DEVICE)
            logits = model(batch_x)
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    return compute_metrics(y_test, np.array(predictions))


def evaluate_biat_mlp_model(model_path, X_test, y_test):
    import torch
    from src.baselines.bilevel_supervised_ids import MLPIDS

    model = MLPIDS(input_dim=X_test.shape[1]).to(Config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE, weights_only=True))
    model.eval()

    predictions = []
    with torch.no_grad():
        batch_size = 1024
        for i in range(0, len(X_test), batch_size):
            batch_x = torch.FloatTensor(X_test[i:i + batch_size]).to(Config.DEVICE)
            logits = model(batch_x)
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    return compute_metrics(y_test, np.array(predictions))


def evaluate_biat_fttransformer_model(model_path, X_test, y_test):
    import torch
    from src.baselines.bilevel_fttransformer_ids import FTTransformerIDS

    model = FTTransformerIDS(n_features=X_test.shape[1]).to(Config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE, weights_only=True))
    model.eval()

    predictions = []
    with torch.no_grad():
        batch_size = 1024
        for i in range(0, len(X_test), batch_size):
            batch_x = torch.FloatTensor(X_test[i:i + batch_size]).to(Config.DEVICE)
            logits = model(batch_x)
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    return compute_metrics(y_test, np.array(predictions))


def evaluate_hgbt_model(X_train, y_train, X_test, y_test, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=8,
        max_iter=300,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return compute_metrics(y_test, y_pred)


def evaluate_xgboost_model(X_train, y_train, X_test, y_test, seed):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("xgboost is not installed in the current environment") from exc

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=neg / max(pos, 1),
        random_state=seed,
        n_jobs=4,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return compute_metrics(y_test, y_pred)


def evaluate_lightgbm_model(X_train, y_train, X_test, y_test, seed):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError("lightgbm is not installed in the current environment") from exc

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    model = LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        scale_pos_weight=neg / max(pos, 1),
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return compute_metrics(y_test, y_pred)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nsl-kdd", help="Dataset: nsl-kdd / unsw-nb15")
    args = parser.parse_args()
    Config.configure_dataset(args.dataset)

    print("\n" + "=" * 70)
    print("  Main Results Evaluation")
    print("=" * 70 + "\n")

    loader = build_data_loader(Config.DATASET_NAME)
    if not loader.has_real_data():
        raise FileNotFoundError(
            f"Dataset files not found for {Config.DATASET_NAME}. "
            f"Expected: {loader.train_path} and {loader.test_path}"
        )
    X_train, y_train = loader.load_data(mode="train")
    X_test, y_test = loader.load_data(mode="test")

    model_specs = {
        "Bi-ARL": ("BiARL", "defender", "rl"),
        "Vanilla PPO": ("VanillaPPO", "model", "rl"),
        "MARL": ("MARL", "defender", "rl"),
        "LSTM-IDS": ("LSTM", "model", "lstm"),
        "Transformer-IDS": ("TransformerIDS", "model", "biat_fttransformer"),
        "BiAT-MLP": ("BiATMLP", "model", "biat_mlp"),
        "BiAT-FTTransformer": ("BiATFTTransformer", "model", "biat_fttransformer"),
        "HGBT-IDS": ("HGBT", "model", "hgbt"),
        "XGBoost-IDS": ("XGBoost", "model", "xgb"),
        "LightGBM-IDS": ("LightGBM", "model", "lgbm"),
    }

    import time
    detailed_rows = []
    total_jobs = sum(
        1 for name, (_, _, fam) in model_specs.items()
        for s in Config.SEEDS
        if fam not in ("hgbt", "xgb", "lgbm")
    ) + len([f for _, (_, _, f) in model_specs.items() if f in ("hgbt","xgb","lgbm")]) * len(Config.SEEDS)
    job_idx = 0
    t_global = time.time()

    for model_name, (model_type, model_file, family) in model_specs.items():
        print(f"\n[{model_name}]", flush=True)
        for seed in Config.SEEDS:
            job_idx += 1
            t_job = time.time()
            print(f"  ({job_idx}/{total_jobs}) seed={seed} ...", end=" ", flush=True)
            if family == "hgbt":
                metrics = evaluate_hgbt_model(X_train, y_train, X_test, y_test, seed)
            elif family == "xgb":
                try:
                    metrics = evaluate_xgboost_model(X_train, y_train, X_test, y_test, seed)
                except RuntimeError as exc:
                    print(f"[Skip] {exc}", flush=True)
                    continue
            elif family == "lgbm":
                try:
                    metrics = evaluate_lightgbm_model(X_train, y_train, X_test, y_test, seed)
                except RuntimeError as exc:
                    print(f"[Skip] {exc}", flush=True)
                    continue
            else:
                model_path = Config.find_model_file(model_type, seed, model_file)
                if not model_path.exists():
                    print(f"[Skip] not found", flush=True)
                    continue

                if family == "lstm":
                    metrics = evaluate_lstm_model(model_path, X_test, y_test)
                elif family == "biat_mlp":
                    metrics = evaluate_biat_mlp_model(model_path, X_test, y_test)
                elif family == "biat_fttransformer":
                    metrics = evaluate_biat_fttransformer_model(model_path, X_test, y_test)
                else:
                    metrics = evaluate_defender_model(model_path, X_test, y_test)

            elapsed_job = time.time() - t_job
            elapsed_total = time.time() - t_global
            row = {"Model": model_name, "Seed": seed}
            row.update(metrics)
            detailed_rows.append(row)
            print(
                f"F1={metrics['F1']:.4f}  FPR={metrics['FPR']:.4f}  "
                f"({elapsed_job:.1f}s, total {elapsed_total:.0f}s)",
                flush=True,
            )

    detailed_df = pd.DataFrame(detailed_rows)
    suffix = Config.DATASET_NAME.replace("-", "_")
    detailed_path = Config.RESULTS_DIR / f"main_results_detailed_{suffix}.csv"
    detailed_df.to_csv(detailed_path, index=False)

    summary_df = detailed_df.groupby("Model")[["Accuracy", "Recall", "Precision", "F1", "FPR"]].agg(["mean", "std"])
    summary_df.columns = [f"{metric}_{stat}" for metric, stat in summary_df.columns]
    summary_df = summary_df.reset_index()
    seed_counts = detailed_df.groupby("Model")["Seed"].nunique().reset_index(name="NumSeeds")
    summary_df = summary_df.merge(seed_counts, on="Model", how="left")
    std_cols = [col for col in summary_df.columns if col.endswith("_std")]
    summary_df[std_cols] = summary_df[std_cols].fillna(0.0)
    summary_path = Config.RESULTS_DIR / f"main_results_summary_{suffix}.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nClean Summary (Mean ± Std):")
    for _, row in summary_df.iterrows():
        print(
            f"{row['Model']:<15} "
            f"n={int(row['NumSeeds'])} "
            f"Recall={row['Recall_mean']:.4f}±{row['Recall_std']:.4f} "
            f"Precision={row['Precision_mean']:.4f}±{row['Precision_std']:.4f} "
            f"F1={row['F1_mean']:.4f}±{row['F1_std']:.4f} "
            f"FPR={row['FPR_mean']:.4f}±{row['FPR_std']:.4f}"
        )

    print(f"\nDetailed results saved: {detailed_path}")
    print(f"Summary results saved: {summary_path}")


if __name__ == "__main__":
    main()
