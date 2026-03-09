from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting_utils import (
    DATASET_LABELS,
    MODEL_COLORS,
    RESULTS_DIR,
    apply_percentage_axis,
    ensure_figure_dirs,
    ordered_models,
    save_figure,
    setup_plot_style,
)


SUMMARY_FILES = [
    "main_results_summary.csv",
    "main_results_summary_unsw_nb15.csv",
    "main_results_summary_cic_ids2017_random.csv",
    "main_results_summary_cic_ids2017.csv",
]

METRICS = [
    ("F1_mean", "F1", "F1_std"),
    ("Recall_mean", "Recall", "Recall_std"),
    ("Precision_mean", "Precision", "Precision_std"),
    ("FPR_mean", "FPR", "FPR_std"),
]


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Dataset"] = DATASET_LABELS.get(path.name, path.stem)
    return df


def plot_dataset_panels(df: pd.DataFrame, dataset_label: str, out_dir: Path, latex_dir: Path) -> None:
    models = ordered_models(df["Model"].tolist())
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.8))
    axes = axes.flatten()
    x = np.arange(len(models))

    for ax, (metric_col, title, std_col) in zip(axes, METRICS):
        values = [df.loc[df["Model"] == model, metric_col].iloc[0] for model in models]
        colors = [MODEL_COLORS.get(model, "#4c4c4c") for model in models]
        ax.bar(x, values, color=colors, linewidth=0, width=0.72)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=25, ha="right")
        apply_percentage_axis(ax, 1.0)
        ax.set_ylabel("Percentage")

    fig.suptitle(f"Main Results on {dataset_label}", y=0.99, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97], pad=0.8)
    save_figure(fig, f"main_results_{slugify(dataset_label)}", out_dir, latex_dir)
    plt.close(fig)


def plot_cross_dataset(df: pd.DataFrame, out_dir: Path, latex_dir: Path) -> None:
    focus_metrics = [("F1_mean", "F1"), ("FPR_mean", "FPR")]
    datasets = list(dict.fromkeys(df["Dataset"].tolist()))
    models = ordered_models(df["Model"].unique().tolist())
    x = np.arange(len(datasets))
    width = 0.12

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), sharex=True)
    for ax, (metric_col, title) in zip(axes, focus_metrics):
        for idx, model in enumerate(models):
            subset = df[df["Model"] == model].set_index("Dataset")
            vals = [subset.loc[d, metric_col] if d in subset.index else np.nan for d in datasets]
            offset = (idx - (len(models) - 1) / 2) * width
            ax.bar(
                x + offset,
                vals,
                width=width,
                label=model,
                color=MODEL_COLORS.get(model, "#4c4c4c"),
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=18, ha="right")
        apply_percentage_axis(ax, 1.0)

    axes[0].set_ylabel("Percentage")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.suptitle("Cross-Dataset Comparison", y=1.08, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.83], pad=0.8)
    save_figure(fig, "cross_dataset_main_results", out_dir, latex_dir)
    plt.close(fig)


def slugify(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def main() -> None:
    setup_plot_style()
    out_dir, latex_dir = ensure_figure_dirs("main_results")
    all_frames = []
    for filename in SUMMARY_FILES:
        path = RESULTS_DIR / filename
        if not path.exists():
            continue
        df = load_summary(path)
        all_frames.append(df)
        plot_dataset_panels(df, df["Dataset"].iloc[0], out_dir, latex_dir)

    if not all_frames:
        raise FileNotFoundError("No main result summary files were found in outputs/results.")

    merged = pd.concat(all_frames, ignore_index=True)
    plot_cross_dataset(merged, out_dir, latex_dir)
    print(f"Saved main result figures to {out_dir}")


if __name__ == "__main__":
    main()
