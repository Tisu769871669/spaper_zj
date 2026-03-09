from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "outputs" / "results"
FIGURES_DIR = ROOT / "outputs" / "figures"
LATEX_FIGURES_DIR = ROOT / "latex_source" / "figures" / "generated"

DATASET_LABELS = {
    "main_results_summary.csv": "NSL-KDD",
    "main_results_summary_nsl_kdd.csv": "NSL-KDD",
    "main_results_summary_unsw_nb15.csv": "UNSW-NB15",
    "main_results_summary_cic_ids2017.csv": "CIC-IDS2017 (Day Split)",
    "main_results_summary_cic_ids2017_random.csv": "CIC-IDS2017 (Random Split)",
    "ablation_results.csv": "NSL-KDD",
    "ablation_results_unsw_nb15.csv": "UNSW-NB15",
}

MODEL_ORDER = [
    "Bi-ARL",
    "MARL",
    "Vanilla PPO",
    "LSTM-IDS",
    "HGBT-IDS",
    "XGBoost-IDS",
    "LightGBM-IDS",
]

MODEL_COLORS = {
    "Bi-ARL": "#1f4e79",
    "MARL": "#2a9d8f",
    "Vanilla PPO": "#e76f51",
    "LSTM-IDS": "#6d597a",
    "HGBT-IDS": "#bc6c25",
    "XGBoost-IDS": "#264653",
    "LightGBM-IDS": "#8ab17d",
    "Full Bi-ARL": "#1f4e79",
    "w/o Inner Loop": "#577590",
    "w/o Attacker (Vanilla PPO)": "#e76f51",
    "w/o Bi-level (MARL)": "#2a9d8f",
}


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "font.size": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.25,
            "savefig.bbox": "tight",
        }
    )


def ensure_figure_dirs(*parts: str) -> tuple[Path, Path]:
    out_dir = FIGURES_DIR.joinpath(*parts)
    latex_dir = LATEX_FIGURES_DIR.joinpath(*parts)
    out_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, latex_dir


def save_figure(fig: plt.Figure, stem: str, out_dir: Path, latex_dir: Path) -> None:
    for directory in (out_dir, latex_dir):
        fig.savefig(directory / f"{stem}.pdf")
        fig.savefig(directory / f"{stem}.png")


def ordered_models(models: list[str]) -> list[str]:
    known = [m for m in MODEL_ORDER if m in models]
    unknown = [m for m in models if m not in MODEL_ORDER]
    return known + sorted(unknown)


def apply_percentage_axis(ax: plt.Axes, ymax: float = 1.0) -> None:
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", linestyle="--", alpha=0.25)
