from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "plot_main_results.py",
    "plot_ablation.py",
    "plot_robustness.py",
]


def main() -> None:
    for script in SCRIPTS:
        try:
            subprocess.run([sys.executable, str(ROOT / script)], check=True)
        except subprocess.CalledProcessError as exc:
            if script in {"plot_ablation.py", "plot_robustness.py"}:
                print(f"[Skip] {script}: {exc}")
                continue
            raise
    print("All figures generated successfully.")


if __name__ == "__main__":
    main()
