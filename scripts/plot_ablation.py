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
    save_figure,
    setup_plot_style,
)


ABLATION_FILES = [
    "ablation_results.csv",
    "ablation_results_unsw_nb15.csv",
]

VARIANT_ORDER = [
    "Full Bi-ARL",
    "w/o Inner Loop",
    "w/o Bi-level (MARL)",
    "w/o Attacker (Vanilla PPO)",
]


def slugify(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def plot_ablation_file(path: Path, out_dir: Path, latex_dir: Path) -> None:
    df = pd.read_csv(path)
    dataset_label = DATASET_LABELS.get(path.name, path.stem)
    conditions = ["Clean", "Stress"]
    metrics = [("F1", "F1"), ("FPR", "FPR")]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    x = np.arange(len(VARIANT_ORDER))
    width = 0.34

    for ax, (metric_col, title) in zip(axes, metrics):
        for idx, condition in enumerate(conditions):
            subset = df[df["Condition"] == condition].set_index("Variant")
            vals = [subset.loc[v, metric_col] if v in subset.index else np.nan for v in VARIANT_ORDER]
            offset = (idx - 0.5) * width
            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                label=condition,
                color=["#1f4e79", "#e76f51"][idx],
                alpha=0.9,
            )
            if idx == 0:
                for bar, variant in zip(bars, VARIANT_ORDER):
                    bar.set_edgecolor(MODEL_COLORS.get(variant, "#4c4c4c"))
                    bar.set_linewidth(0.8)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(VARIANT_ORDER, rotation=20, ha="right")
        apply_percentage_axis(ax, 1.0)

    axes[0].set_ylabel("Percentage")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.6,
    )
    fig.suptitle(f"Ablation Study on {dataset_label}", y=1.08, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.84], pad=0.8)
    save_figure(fig, f"ablation_{slugify(dataset_label)}", out_dir, latex_dir)
    plt.close(fig)


def main() -> None:
    setup_plot_style()
    out_dir, latex_dir = ensure_figure_dirs("ablation")
    found = False
    for filename in ABLATION_FILES:
        path = RESULTS_DIR / filename
        if not path.exists():
            continue
        found = True
        plot_ablation_file(path, out_dir, latex_dir)

    if not found:
        raise FileNotFoundError("No ablation result files were found in outputs/results.")

    print(f"Saved ablation figures to {out_dir}")


if __name__ == "__main__":
    main()
