#!/bin/bash
set -euo pipefail

MODE="${1:-all}"
DUD_E_URL="${DUD_E_URL:-http://dude.docking.org/db/subsets/all/all.tar.gz}"
LIT_PCBA_URL="${LIT_PCBA_URL:-}"
DEKOIS2_URL="${DEKOIS2_URL:-}"
BAYESBIND_URL="${BAYESBIND_URL:-}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-.downloads}"
DUD_E_ARCHIVE="$DOWNLOAD_DIR/dude_all.tar.gz"
DUD_E_EXTRACT_DIR="$DOWNLOAD_DIR/dude_all"

download_file() {
  local url="$1"
  local output="$2"

  if [[ "$url" =~ ^\[(.*)\]\((.*)\)$ ]]; then
    url="${BASH_REMATCH[2]}"
  fi

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

extract_archive() {
  local archive="$1"
  local destination="$2"

  rm -rf "$destination"
  mkdir -p "$destination"
  case "$archive" in
    *.tar.gz|*.tgz)
      tar -xzf "$archive" -C "$destination"
      ;;
    *.tar)
      tar -xf "$archive" -C "$destination"
      ;;
    *.zip)
      unzip -q "$archive" -d "$destination"
      ;;
    *)
      echo "Error: unsupported archive format: $archive" >&2
      exit 1
      ;;
  esac
}

download_dude() {
  echo "Downloading DUD-E from $DUD_E_URL"
  download_file "$DUD_E_URL" "$DUD_E_ARCHIVE"

  extract_archive "$DUD_E_ARCHIVE" "$DUD_E_EXTRACT_DIR"

  python scripts/prepare_dude.py \
    --source-dir "$DUD_E_EXTRACT_DIR" \
    --output-dir data/DUD-E
}

download_screening_dataset() {
  local dataset_name="$1"
  local dataset_url="$2"
  local extract_dir="$3"
  local output_dir="$4"

  if [[ -z "$dataset_url" ]]; then
    echo "Error: set ${dataset_name}_URL to a dataset archive URL, or download the dataset manually and run scripts/prepare_screening_dataset.py." >&2
    exit 1
  fi

  local archive_name
  archive_name="$(basename "${dataset_url%%\?*}")"
  local archive_path="$DOWNLOAD_DIR/$archive_name"

  echo "Downloading $dataset_name from $dataset_url"
  download_file "$dataset_url" "$archive_path"
  extract_archive "$archive_path" "$extract_dir"

  if [[ "$dataset_name" == "DEKOIS2" ]]; then
    python scripts/prepare_screening_dataset.py \
      --source-dir "$extract_dir" \
      --output-dir "$output_dir" \
      --query-from-first-active
  else
    python scripts/prepare_screening_dataset.py \
      --source-dir "$extract_dir" \
      --output-dir "$output_dir"
  fi
}

download_lit_pcba() {
  download_screening_dataset \
    "LIT_PCBA" \
    "$LIT_PCBA_URL" \
    "$DOWNLOAD_DIR/lit_pcba" \
    "data/LIT-PCBA"
}

download_dekois2() {
  download_screening_dataset \
    "DEKOIS2" \
    "$DEKOIS2_URL" \
    "$DOWNLOAD_DIR/dekois2" \
    "data/DEKOIS2"
}

download_bayesbind() {
  download_screening_dataset \
    "BAYESBIND" \
    "$BAYESBIND_URL" \
    "$DOWNLOAD_DIR/bayesbind" \
    "data/BayesBind"
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
  lit-pcba)
    download_lit_pcba
    ;;
  dekois2)
    download_dekois2
    ;;
  bayesbind)
    download_bayesbind
    ;;
  *)
    echo "Usage: bash scripts/download_datasets.sh [all|dude|qm9|lit-pcba|dekois2|bayesbind]" >&2
    exit 1
    ;;
esac
