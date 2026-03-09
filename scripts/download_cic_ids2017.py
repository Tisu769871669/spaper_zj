#!/usr/bin/env python
"""
Download CIC-IDS2017 machine-learning parquet files from a public Hugging Face mirror.
"""

import argparse
from pathlib import Path
from urllib.request import urlretrieve


FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv.parquet",
    "Tuesday-WorkingHours.pcap_ISCX.csv.parquet",
    "Wednesday-workingHours.pcap_ISCX.csv.parquet",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv.parquet",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv.parquet",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet",
]

BASE_URL = "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning"


def main():
    parser = argparse.ArgumentParser(description="Download CIC-IDS2017 parquet files")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data") / "CIC_IDS2017_machine_learning",
        help="Output directory for parquet files",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        target = args.output_dir / filename
        if target.exists():
            print(f"[Skip] {target}")
            continue
        url = f"{BASE_URL}/{filename}"
        print(f"[Download] {url}")
        urlretrieve(url, target)
        print(f"[Saved] {target}")


if __name__ == "__main__":
    main()
