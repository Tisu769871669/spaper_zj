#!/usr/bin/env python
"""
Statistical Significance Tests for key model comparisons.

Reads per-seed results from main_results_detailed_*.csv and runs:
  - Wilcoxon signed-rank test (non-parametric, appropriate for n=4 seeds)
  - Cohen's d effect size
  - 95% confidence interval

Key comparisons:
  NSL-KDD:   Bi-ARL      vs. Vanilla PPO  (F1, FPR)
  UNSW-NB15: BiAT-FTT    vs. BiAT-MLP     (F1, FPR)
  UNSW-NB15: BiAT-FTT    vs. LightGBM-IDS (F1, FPR)

Outputs:
  - Console report
  - outputs/results/significance_report.txt
  - outputs/results/significance_table.tex  (LaTeX table for paper)
"""

import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.statistical_tests import compare_models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_per_seed(dataset_slug: str) -> pd.DataFrame:
    path = Config.RESULTS_DIR / f"main_results_detailed_{dataset_slug}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Per-seed results not found: {path}\n"
            f"Run: python scripts/evaluate_main_results.py --dataset {dataset_slug.replace('_', '-')}"
        )
    return pd.read_csv(path)


def get_metric_list(df: pd.DataFrame, model: str, metric: str) -> list:
    """Return per-seed metric values aligned to Config.SEEDS order."""
    sub = df[df["Model"] == model].sort_values("Seed")
    if sub.empty:
        return []
    return sub[metric].tolist()


def sig_stars(p: float) -> str:
    if p is None or np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Core comparison runner
# ---------------------------------------------------------------------------

def run_comparison(df, model_a: str, model_b: str, metric: str, dataset_label: str) -> dict:
    vals_a = get_metric_list(df, model_a, metric)
    vals_b = get_metric_list(df, model_b, metric)

    if len(vals_a) < 2 or len(vals_b) < 2:
        return {
            "dataset": dataset_label,
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric,
            "mean_a": np.mean(vals_a) if vals_a else float("nan"),
            "mean_b": np.mean(vals_b) if vals_b else float("nan"),
            "diff": float("nan"),
            "p_wilcoxon": float("nan"),
            "cohens_d": float("nan"),
            "stars": "n/a",
            "note": "insufficient data",
        }

    # Align length to minimum (handle if seed counts differ)
    n = min(len(vals_a), len(vals_b))
    vals_a, vals_b = vals_a[:n], vals_b[:n]

    result = compare_models(vals_a, vals_b, model_a, model_b, metric)

    return {
        "dataset": dataset_label,
        "model_a": model_a,
        "model_b": model_b,
        "metric": metric,
        "mean_a": result.mean_a,
        "std_a": result.std_a,
        "mean_b": result.mean_b,
        "std_b": result.std_b,
        "diff": result.mean_diff,
        "p_ttest": result.p_value_ttest,
        "p_wilcoxon": result.p_value_wilcoxon,
        "cohens_d": result.cohens_d,
        "stars": sig_stars(result.p_value_wilcoxon if result.p_value_wilcoxon is not None else result.p_value_ttest),
        "effect_size": result.effect_size_interpretation(),
        "note": "",
    }


# ---------------------------------------------------------------------------
# LaTeX table generator
# ---------------------------------------------------------------------------

def make_latex_table(rows: list) -> str:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Statistical Significance of Key Model Comparisons (Wilcoxon signed-rank test, $n{=}4$ seeds)}",
        r"\label{tab:significance}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Dataset & Comparison & Metric & $\bar{A}$ & $\bar{B}$ & $\Delta$ & $p$ \\",
        r"\midrule",
    ]
    prev_ds = None
    for r in rows:
        ds = r["dataset"] if r["dataset"] != prev_ds else ""
        prev_ds = r["dataset"]
        cmp = f"{r['model_a']} vs.\\ {r['model_b']}"
        mean_a = f"{r['mean_a'] * 100:.2f}\\%"
        mean_b = f"{r['mean_b'] * 100:.2f}\\%"
        diff_sign = "+" if r["diff"] > 0 else ""
        delta = f"{diff_sign}{r['diff'] * 100:.2f}\\%"
        p_val = r.get("p_wilcoxon")
        if p_val is not None and not (isinstance(p_val, float) and np.isnan(p_val)):
            p_str = f"{p_val:.3f}{r['stars']}"
        else:
            p_str = r["stars"]
        lines.append(
            f"{ds} & {cmp} & {r['metric']} & {mean_a} & {mean_b} & {delta} & {p_str} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\multicolumn{7}{l}{\scriptsize{$^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$, ns: not significant}}",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    comparisons_spec = [
        # (dataset_slug, dataset_label, model_a, model_b, metrics)
        (
            "nsl_kdd",
            "NSL-KDD",
            "Bi-ARL",
            "Vanilla PPO",
            ["F1", "FPR"],
        ),
        (
            "nsl_kdd",
            "NSL-KDD",
            "Bi-ARL",
            "MARL",
            ["F1"],
        ),
        (
            "unsw_nb15",
            "UNSW-NB15",
            "BiAT-FTTransformer",
            "BiAT-MLP",
            ["F1", "FPR"],
        ),
        (
            "unsw_nb15",
            "UNSW-NB15",
            "BiAT-FTTransformer",
            "LightGBM-IDS",
            ["F1", "FPR"],
        ),
        (
            "cic_ids2017_random",
            "CIC-IDS2017 (random)",
            "BiAT-FTTransformer",
            "BiAT-MLP",
            ["F1", "FPR"],
        ),
    ]

    all_results = []
    report_lines = []

    loaded_dfs = {}

    for dataset_slug, dataset_label, model_a, model_b, metrics in comparisons_spec:
        if dataset_slug not in loaded_dfs:
            try:
                loaded_dfs[dataset_slug] = load_per_seed(dataset_slug)
            except FileNotFoundError as e:
                print(f"[Warning] {e}")
                loaded_dfs[dataset_slug] = None

        df = loaded_dfs[dataset_slug]
        if df is None:
            continue

        for metric in metrics:
            r = run_comparison(df, model_a, model_b, metric, dataset_label)
            all_results.append(r)

            report_lines.append(
                f"[{dataset_label}] {model_a} vs {model_b} | {metric}: "
                f"{r.get('mean_a', float('nan')) * 100:.2f}% vs "
                f"{r.get('mean_b', float('nan')) * 100:.2f}%  "
                f"Δ={r.get('diff', float('nan')) * 100:+.2f}%  "
                f"p={r.get('p_wilcoxon') if r.get('p_wilcoxon') is not None else 'n/a'}  "
                f"{r['stars']}  d={r.get('cohens_d', float('nan')):.2f} ({r.get('effect_size', 'n/a')})"
            )
            print(report_lines[-1])

    if not all_results:
        print("\n[Warning] No comparisons completed. Check that per-seed CSV files exist.")
        return

    # Save report text
    report_path = Config.RESULTS_DIR / "significance_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport saved: {report_path}")

    # Save CSV
    csv_path = Config.RESULTS_DIR / "significance_results.csv"
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    # Save LaTeX table
    latex_path = Config.RESULTS_DIR / "significance_table.tex"
    latex_path.write_text(make_latex_table(all_results), encoding="utf-8")
    print(f"LaTeX table: {latex_path}")


if __name__ == "__main__":
    main()
