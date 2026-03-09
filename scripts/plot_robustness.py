from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from plotting_utils import (
    MODEL_COLORS,
    RESULTS_DIR,
    apply_percentage_axis,
    ensure_figure_dirs,
    ordered_models,
    save_figure,
    setup_plot_style,
)


def plot_robustness(df: pd.DataFrame, out_dir, latex_dir) -> None:
    models = ordered_models(df["Model"].unique().tolist())
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharex=True)
    metrics = [("F1", "F1"), ("FPR", "FPR")]

    for ax, (metric_col, title) in zip(axes, metrics):
        for model in models:
            subset = df[df["Model"] == model].sort_values("Epsilon")
            ax.plot(
                subset["Epsilon"],
                subset[metric_col],
                marker="o",
                linewidth=2,
                markersize=5,
                label=model,
                color=MODEL_COLORS.get(model, "#4c4c4c"),
            )
        ax.set_title(title)
        ax.set_xlabel("Attack Strength (epsilon)")
        apply_percentage_axis(ax, 1.0)

    axes[0].set_ylabel("Percentage")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.suptitle("Adversarial Robustness under FGSM", y=1.08, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.84], pad=0.8)
    save_figure(fig, "adversarial_robustness", out_dir, latex_dir)
    plt.close(fig)


def main() -> None:
    setup_plot_style()
    out_dir, latex_dir = ensure_figure_dirs("robustness")
    path = RESULTS_DIR / "adversarial_robustness.csv"
    if not path.exists():
        raise FileNotFoundError("adversarial_robustness.csv was not found in outputs/results.")

    df = pd.read_csv(path)
    plot_robustness(df, out_dir, latex_dir)
    print(f"Saved robustness figures to {out_dir}")


if __name__ == "__main__":
    main()
