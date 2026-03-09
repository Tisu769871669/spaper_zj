#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${SERVER_USER:?Please set SERVER_USER}"
: "${SERVER_HOST:?Please set SERVER_HOST}"

REMOTE_DIR="${REMOTE_DIR:-~/spaper_zj}"
REMOTE_ARCHIVE_GLOB="${REMOTE_ARCHIVE_GLOB:-outputs/packages/spaper_results_*.tar.gz}"
LOCAL_DIR="${LOCAL_DIR:-$ROOT_DIR/outputs/server_fetch}"
RSYNC_BIN="${RSYNC_BIN:-rsync}"

mkdir -p "$LOCAL_DIR"

"$RSYNC_BIN" -avz \
  "${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/${REMOTE_ARCHIVE_GLOB}" \
  "$LOCAL_DIR/"

echo "Fetched archives to $LOCAL_DIR"
