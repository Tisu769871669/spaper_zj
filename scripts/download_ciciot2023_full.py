#!/usr/bin/env python
"""
Download the full public CICIoT2023 mirror from Hugging Face.

The script is resumable: existing files are skipped when size matches.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "data" / "CICIoT2023"
API_URL = "https://huggingface.co/api/datasets/baalajimaestro/DDoS-CICIoT2023/tree/main?recursive=1"
DEFAULT_BASE_URL = "https://huggingface.co/datasets/baalajimaestro/DDoS-CICIoT2023/resolve/main/"
BASE_URL = os.getenv("HF_FILE_BASE_URL", DEFAULT_BASE_URL)
MANIFEST_PATH = TARGET_DIR / "_manifest.json"
CHUNK_SIZE = 1024 * 1024
TIMEOUT = (30, 300)
MAX_RETRIES = 5


def fetch_manifest() -> list[dict]:
    response = requests.get(API_URL, timeout=60)
    response.raise_for_status()
    data = response.json()
    files = [item for item in data if item.get("path", "").endswith(".csv")]
    files.sort(key=lambda x: x["path"])
    return files


def load_existing_manifest() -> list[dict] | None:
    if not MANIFEST_PATH.exists():
        return None
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(files: list[dict]) -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(files, indent=2), encoding="utf-8")


def download_file(path: str, expected_size: int) -> None:
    target = TARGET_DIR / path
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".part")

    if target.exists() and target.stat().st_size == expected_size:
        print(f"Skip existing: {path}")
        return

    if tmp_path.exists() and expected_size and tmp_path.stat().st_size > expected_size:
        print(f"Removing oversized partial file: {tmp_path.name}")
        tmp_path.unlink()

    url = BASE_URL + path
    for attempt in range(1, MAX_RETRIES + 1):
        existing = tmp_path.stat().st_size if tmp_path.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}
        mode = "ab" if existing > 0 else "wb"
        print(f"Downloading: {path} (attempt {attempt}, resume={existing})")
        try:
            with requests.get(url, stream=True, timeout=TIMEOUT, headers=headers) as response:
                if response.status_code == 416:
                    if tmp_path.exists():
                        print(f"Resetting invalid partial file for {path}")
                        tmp_path.unlink()
                    continue
                if response.status_code not in (200, 206):
                    response.raise_for_status()
                if existing > 0 and response.status_code == 200:
                    mode = "wb"
                    existing = 0
                with tmp_path.open(mode) as fout:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            fout.write(chunk)
            actual_tmp = tmp_path.stat().st_size
            if expected_size and actual_tmp != expected_size:
                if actual_tmp < expected_size:
                    print(f"Incomplete file for {path}: {actual_tmp}/{expected_size}, retrying...")
                    continue
                raise RuntimeError(f"Size mismatch for {path}: expected {expected_size}, got {actual_tmp}")
            tmp_path.replace(target)
            break
        except requests.RequestException as exc:
            print(f"Retryable download error for {path}: {exc}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(min(10 * attempt, 30))
    actual = target.stat().st_size
    if expected_size and actual != expected_size:
        raise RuntimeError(f"Size mismatch for {path}: expected {expected_size}, got {actual}")


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    files = fetch_manifest()
    save_manifest(files)
    total_bytes = sum(item.get("size", 0) for item in files)
    print(f"Found {len(files)} CSV files, total {total_bytes / 1e9:.3f} GB")
    print(f"File download base: {BASE_URL}")

    start = time.time()
    downloaded = 0
    for idx, item in enumerate(files, start=1):
        path = item["path"]
        size = int(item.get("size", 0))
        download_file(path, size)
        downloaded += size
        elapsed_min = (time.time() - start) / 60.0
        print(f"[{idx}/{len(files)}] {(downloaded / 1e9):.3f} GB logical processed, elapsed {elapsed_min:.1f} min")

    print("CICIoT2023 mirror download complete.")


if __name__ == "__main__":
    main()
