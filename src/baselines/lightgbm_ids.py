"""
LightGBM baseline for tabular IDS datasets.
"""

import argparse
import json
import sys
import os

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config import Config
from src.utils.data_loader import build_data_loader


def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "FPR": float(np.sum((y_pred == 1) & (y_true == 0)) / np.sum(y_true == 0)),
    }


def main():
    parser = argparse.ArgumentParser(description="LightGBM IDS baseline")
    parser.add_argument("--dataset", type=str, default="nsl-kdd")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Config.configure_dataset(args.dataset)
    Config.set_seed(args.seed)

    loader = build_data_loader(Config.DATASET_NAME)
    if not loader.has_real_data():
        raise FileNotFoundError(
            f"Dataset files not found for {Config.DATASET_NAME}. "
            f"Expected input under: {loader.train_path} and {loader.test_path}"
        )
    X_train, y_train = loader.load_data(mode="train")
    X_test, y_test = loader.load_data(mode="test")
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    model = LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=args.seed,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    output = {
        "dataset": Config.DATASET_NAME,
        "seed": args.seed,
        "model": "LightGBM",
        **metrics,
    }
    print(json.dumps(output, indent=2))

    suffix = Config.DATASET_NAME.replace("-", "_")
    output_path = Config.RESULTS_DIR / f"lightgbm_baseline_{suffix}_seed{args.seed}.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved baseline result: {output_path}")


if __name__ == "__main__":
    main()
