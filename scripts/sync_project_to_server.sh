#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${SERVER_USER:?Please set SERVER_USER}"
: "${SERVER_HOST:?Please set SERVER_HOST}"

REMOTE_DIR="${REMOTE_DIR:-~/spaper_zj}"
SYNC_DATA="${SYNC_DATA:-1}"
RSYNC_BIN="${RSYNC_BIN:-rsync}"

REMOTE="${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}"

"$RSYNC_BIN" -avz --delete \
  --exclude ".git/" \
  --exclude ".agent/" \
  --exclude ".vscode/" \
  --exclude "outputs/" \
  --exclude "runs/" \
  --exclude "latex_source/out/" \
  --exclude "*.pyc" \
  --exclude "__pycache__/" \
  ./ "$REMOTE/"

if [[ "$SYNC_DATA" == "1" ]]; then
  "$RSYNC_BIN" -avz \
    data/ "$REMOTE/data/"
fi

echo "Synced project to $REMOTE"
