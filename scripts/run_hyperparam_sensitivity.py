#!/usr/bin/env python
"""
Hyperparameter Sensitivity Analysis.

Experiment A — NSL-KDD: K_inner sweep for Bi-ARL
  K_inner in {1, 3, 5, 10}  x  seeds {42, 123}

Experiment B — UNSW-NB15: lambda (adv_weight) sweep for BiAT-FTTransformer
  lambda in {0.3, 0.5, 0.7, 0.9}  x  seeds {42, 123}

Outputs:
  outputs/results/sensitivity_K_inner_nsl_kdd.csv
  outputs/results/sensitivity_lambda_unsw_nb15.csv
  outputs/results/sensitivity_K_inner_nsl_kdd.tex
  outputs/results/sensitivity_lambda_unsw_nb15.tex
  figures/generated/sensitivity/sensitivity_K_inner.pdf
  figures/generated/sensitivity/sensitivity_lambda.pdf
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config

SENSITIVITY_SEEDS = [42, 123]


# ---------------------------------------------------------------------------
# Experiment A: K_inner sweep on NSL-KDD (Bi-ARL)
# ---------------------------------------------------------------------------

def run_k_inner_sweep():
    from src.utils.data_loader import build_data_loader
    from src.algorithms.bilevel_trainer import BiLevelTrainer
    from src.envs.network_security_game import NetworkSecurityGame
    from src.agents.attacker_agent import AttackerAgent
    from src.agents.defender_agent import DefenderAgent
    from src.utils.ppo import PPO
    from sklearn.metrics import f1_score, recall_score, precision_score
    import torch

    Config.configure_dataset("nsl-kdd")
    loader = build_data_loader(Config.DATASET_NAME)
    loader.load_data(mode="train")
    X_test, y_test = loader.load_data(mode="test")

    K_INNER_VALUES = [1, 3, 5, 10]
    rows = []

    for k_inner in K_INNER_VALUES:
        for seed in SENSITIVITY_SEEDS:
            print(f"\n[K_inner={k_inner}, seed={seed}]", flush=True)
            Config.set_seed(seed)
            Config.INNER_LOOP_STEPS = k_inner

            env = NetworkSecurityGame()
            attacker = AttackerAgent().to(Config.DEVICE)
            defender = DefenderAgent().to(Config.DEVICE)
            opt_att = torch.optim.Adam(attacker.parameters(), lr=Config.LR)
            opt_def = torch.optim.Adam(defender.parameters(), lr=Config.LR)
            ppo_att = PPO(attacker, opt_att, Config.LR, Config.GAMMA, Config.EPS_CLIP, Config.K_EPOCHS)
            ppo_def = PPO(defender, opt_def, Config.LR, Config.GAMMA, Config.EPS_CLIP, Config.K_EPOCHS)

            trainer = BiLevelTrainer(
                env=env, attacker=attacker, defender=defender,
                ppo_attacker=ppo_att, ppo_defender=ppo_def, config=Config
            )
            for episode in range(1, 101):
                trainer.train_one_episode(episode)
                if episode % 25 == 0:
                    print(f"  Episode {episode}/100", flush=True)

            # Evaluate clean (batch inference)
            defender.eval()
            preds = []
            with torch.no_grad():
                batch_size = 1024
                for i in range(0, len(X_test), batch_size):
                    bx = torch.FloatTensor(X_test[i:i + batch_size]).to(Config.DEVICE)
                    actions = torch.argmax(defender(bx), dim=1).cpu().numpy()
                    preds.extend((actions >= 5).astype(int).tolist())
            preds = np.array(preds)
            fpr = float(np.sum((preds == 1) & (y_test == 0)) / max(np.sum(y_test == 0), 1))

            rows.append({
                "K_inner": k_inner,
                "Seed": seed,
                "Recall": recall_score(y_test, preds, zero_division=0),
                "Precision": precision_score(y_test, preds, zero_division=0),
                "F1": f1_score(y_test, preds, zero_division=0),
                "FPR": fpr,
            })
            print(
                f"  F1={rows[-1]['F1']:.4f}  FPR={rows[-1]['FPR']:.4f}"
            )

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("K_inner")[["F1", "FPR", "Recall"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()

    out_csv = Config.RESULTS_DIR / "sensitivity_K_inner_nsl_kdd.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nK_inner sensitivity results: {out_csv}")

    # LaTeX table
    tex = _make_sensitivity_table(
        summary,
        param_col="K_inner",
        param_label="$K_{\\text{inner}}$",
        caption="NSL-KDD: Bi-ARL sensitivity to inner-loop steps $K_{\\text{inner}}$ (mean $\\pm$ std, 2 seeds).",
        label="tab:sensitivity_kinner",
    )
    tex_path = Config.RESULTS_DIR / "sensitivity_K_inner_nsl_kdd.tex"
    tex_path.write_text(tex, encoding="utf-8")

    _plot_sensitivity(
        summary,
        param_col="K_inner",
        xlabel="Inner-loop steps $K_{\\mathrm{inner}}$",
        title="Bi-ARL Sensitivity to $K_{\\mathrm{inner}}$ (NSL-KDD)",
        out_path=Config.FIGURES_DIR / "sensitivity" / "sensitivity_K_inner.pdf",
    )
    return summary


# ---------------------------------------------------------------------------
# Experiment B: lambda sweep on UNSW-NB15 (BiAT-FTTransformer)
# ---------------------------------------------------------------------------

def run_lambda_sweep():
    from src.utils.data_loader import build_data_loader
    from src.baselines.bilevel_fttransformer_ids import BiLevelFTTransformerTrainer, AdversarialConfig

    Config.configure_dataset("unsw-nb15")

    LAMBDA_VALUES = [0.3, 0.5, 0.7, 0.9]
    rows = []

    for lam in LAMBDA_VALUES:
        for seed in SENSITIVITY_SEEDS:
            print(f"\n[lambda={lam}, seed={seed}]")
            trainer = BiLevelFTTransformerTrainer(
                seed=seed,
                adv_cfg=AdversarialConfig(
                    epsilon=0.02,
                    alpha=0.005,
                    steps=2,
                    adv_weight=lam,
                ),
            )
            results = trainer.train(epochs=5, batch_size=512)
            m = results["clean"]
            rows.append({
                "lambda": lam,
                "Seed": seed,
                "Recall": m["Recall"],
                "Precision": m["Precision"],
                "F1": m["F1"],
                "FPR": m["FPR"],
            })
            print(f"  F1={m['F1']:.4f}  FPR={m['FPR']:.4f}")

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("lambda")[["F1", "FPR", "Recall"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()

    out_csv = Config.RESULTS_DIR / "sensitivity_lambda_unsw_nb15.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nLambda sensitivity results: {out_csv}")

    tex = _make_sensitivity_table(
        summary,
        param_col="lambda",
        param_label="$\\lambda$",
        caption="UNSW-NB15: BiAT-FTTransformer sensitivity to adversarial loss weight $\\lambda$ (mean $\\pm$ std, 2 seeds).",
        label="tab:sensitivity_lambda",
    )
    tex_path = Config.RESULTS_DIR / "sensitivity_lambda_unsw_nb15.tex"
    tex_path.write_text(tex, encoding="utf-8")

    _plot_sensitivity(
        summary,
        param_col="lambda",
        xlabel="Adversarial loss weight $\\lambda$",
        title="BiAT-FTTransformer Sensitivity to $\\lambda$ (UNSW-NB15)",
        out_path=Config.FIGURES_DIR / "sensitivity" / "sensitivity_lambda.pdf",
    )
    return summary


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_sensitivity_table(
    df: pd.DataFrame,
    param_col: str,
    param_label: str,
    caption: str,
    label: str,
) -> str:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        f"{param_label} & F1 (\\%) & FPR (\\%) & Recall (\\%) \\\\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        f1m = row.get("F1_mean", float("nan")) * 100
        f1s = row.get("F1_std", 0.0) * 100
        fprm = row.get("FPR_mean", float("nan")) * 100
        fprs = row.get("FPR_std", 0.0) * 100
        recm = row.get("Recall_mean", float("nan")) * 100
        recs = row.get("Recall_std", 0.0) * 100
        lines.append(
            f"{row[param_col]} & "
            f"${f1m:.2f}\\pm{f1s:.2f}$ & "
            f"${fprm:.2f}\\pm{fprs:.2f}$ & "
            f"${recm:.2f}\\pm{recs:.2f}$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _plot_sensitivity(
    df: pd.DataFrame,
    param_col: str,
    xlabel: str,
    title: str,
    out_path: Path,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for ax, (metric, ylabel) in zip(
            axes,
            [("F1", "F1 Score (%)"), ("FPR", "FPR (%)")],
        ):
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            xs = df[param_col].astype(str).tolist()
            ys = df[mean_col].values * 100
            errs = df.get(std_col, pd.Series(np.zeros(len(df)))).values * 100
            ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=4, linewidth=2)
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"{metric} vs. {param_col}", fontsize=12)
            ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved: {out_path}")
    except Exception as exc:
        print(f"[Warning] Plotting failed: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sensitivity analysis")
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "k_inner", "lambda"],
        help="Which experiment to run",
    )
    args = parser.parse_args()

    Config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.experiment in ("all", "k_inner"):
        print("\n" + "=" * 60)
        print("  Experiment A: K_inner sweep (NSL-KDD, Bi-ARL)")
        print("=" * 60)
        run_k_inner_sweep()

    if args.experiment in ("all", "lambda"):
        print("\n" + "=" * 60)
        print("  Experiment B: Lambda sweep (UNSW-NB15, BiAT-FTTransformer)")
        print("=" * 60)
        run_lambda_sweep()


if __name__ == "__main__":
    main()
