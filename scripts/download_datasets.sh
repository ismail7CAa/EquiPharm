#!/bin/bash
set -euo pipefail

MODE="${1:-all}"
DUD_E_URL="${DUD_E_URL:-http://dude.docking.org/db/subsets/all/all.tar.gz}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-.downloads}"
DUD_E_ARCHIVE="$DOWNLOAD_DIR/dude_all.tar.gz"
DUD_E_EXTRACT_DIR="$DOWNLOAD_DIR/dude_all"

download_file() {
  local url="$1"
  local output="$2"

  mkdir -p "$(dirname "$output")"
  if command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$output"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$output" "$url"
  else
    echo "Error: install curl or wget to download datasets." >&2
    exit 1
  fi
}

download_dude() {
  echo "Downloading DUD-E from $DUD_E_URL"
  download_file "$DUD_E_URL" "$DUD_E_ARCHIVE"

  rm -rf "$DUD_E_EXTRACT_DIR"
  mkdir -p "$DUD_E_EXTRACT_DIR"
  tar -xzf "$DUD_E_ARCHIVE" -C "$DUD_E_EXTRACT_DIR"

  python scripts/prepare_dude.py \
    --source-dir "$DUD_E_EXTRACT_DIR" \
    --output-dir data/DUD-E
}

download_qm9() {
  echo "Downloading QM9 with PyTorch Geometric"
  python scripts/download_qm9.py --output-dir data/QM9
}

case "$MODE" in
  all)
    download_dude
    download_qm9
    ;;
  dude)
    download_dude
    ;;
  qm9)
    download_qm9
    ;;
  *)
    echo "Usage: bash scripts/download_datasets.sh [all|dude|qm9]" >&2
    exit 1
    ;;
esac
