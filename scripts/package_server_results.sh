#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PACKAGE_DIR="${PACKAGE_DIR:-$ROOT_DIR/outputs/packages}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
HOST_TAG="${HOST_TAG:-$(hostname)}"
ARCHIVE_NAME="${ARCHIVE_NAME:-spaper_results_${HOST_TAG}_${STAMP}.tar.gz}"
ARCHIVE_PATH="$PACKAGE_DIR/$ARCHIVE_NAME"

mkdir -p "$PACKAGE_DIR"

MANIFEST="$PACKAGE_DIR/package_manifest_${STAMP}.txt"
{
  echo "timestamp=$STAMP"
  echo "hostname=$HOST_TAG"
  echo "root=$ROOT_DIR"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "python=$(python --version 2>&1 || true)"
  echo "contents="
  echo "  outputs/results"
  echo "  outputs/models"
  echo "  outputs/figures"
  echo "  outputs/logs/server_runs"
  echo "  docs/optimization_decision_log.md"
} > "$MANIFEST"

tar -czf "$ARCHIVE_PATH" \
  outputs/results \
  outputs/models \
  outputs/figures \
  outputs/logs/server_runs \
  docs/optimization_decision_log.md \
  docs/server_conda_setup.md \
  "$MANIFEST"

echo "Packaged results: $ARCHIVE_PATH"
