#!/usr/bin/env python
"""
Download a tiny public CICIoT2023-compatible sample for pipeline smoke tests.

This is not the full official benchmark and must not be used for paper results.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "data" / "CICIoT2023"
MAX_LINES = 5000

SAMPLE_FILES = {
    "BenignTraffic3.pcap.csv": "https://huggingface.co/datasets/baalajimaestro/DDoS-CICIoT2023/resolve/main/BenignTraffic3.pcap.csv",
    "DDoS-ICMP_Flood.pcap.csv": "https://huggingface.co/datasets/baalajimaestro/DDoS-CICIoT2023/resolve/main/DDoS-ICMP_Flood.pcap.csv",
}


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in SAMPLE_FILES.items():
        target = TARGET_DIR / filename
        if target.exists():
            print(f"Skip existing: {target}")
            continue
        print(f"Streaming sample {filename} ...")
        with urlopen(url, timeout=60) as response, target.open("w", encoding="utf-8", newline="") as fout:
            for idx, raw_line in enumerate(response):
                fout.write(raw_line.decode("utf-8"))
                if idx + 1 >= MAX_LINES:
                    break
        print(f"Saved sample: {target}")

    print("\nSample files downloaded.")
    print("These files are only for smoke tests and local pipeline validation.")


if __name__ == "__main__":
    main()
