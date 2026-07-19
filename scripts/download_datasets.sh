#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
MODE="${1:-all}"
DUD_E_URL="${DUD_E_URL:-http://dude.docking.org/db/subsets/all/all.tar.gz}"
LIT_PCBA_URL="${LIT_PCBA_URL:-}"
DEKOIS2_URL="${DEKOIS2_URL:-}"
BAYESBIND_URL="${BAYESBIND_URL:-}"
PHARMACOMATCH_URL="${PHARMACOMATCH_URL:-https://ndownloader.figshare.com/files/52234646}"
ANI2X_URL="${ANI2X_URL:-https://zenodo.org/records/10108942/files/ANI-2x-wB97X-631Gd.tar.gz?download=1}"
SPICE_URL="${SPICE_URL:-https://zenodo.org/records/10975225/files/SPICE-2.0.1.hdf5?download=1}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$ROOT_DIR/.downloads}"
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
    curl --fail --location --continue-at - "$url" --output "$output"
  elif command -v wget >/dev/null 2>&1; then
    wget --continue --output-document="$output" "$url"
  else
    echo "Error: install curl or wget to download datasets." >&2
    exit 1
  fi
}

verify_md5() {
  local expected="$1"
  local file="$2"
  if command -v md5sum >/dev/null 2>&1; then
    echo "$expected  $file" | md5sum -c -
  elif command -v md5 >/dev/null 2>&1; then
    [[ "$(md5 -q "$file")" == "$expected" ]] || {
      echo "Error: checksum mismatch for $file." >&2
      exit 1
    }
  else
    echo "Error: md5sum or md5 is required to verify $file." >&2
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

download_pharmacomatch() {
  local archive="$DOWNLOAD_DIR/pharmacomatch_data.zip"
  echo "Downloading official PharmacoMatch data from $PHARMACOMATCH_URL"
  download_file "$PHARMACOMATCH_URL" "$archive"
  if command -v md5sum >/dev/null 2>&1; then
    echo "387168f6dfb6ae821b08f7a3b92ab056  $archive" | md5sum -c -
  elif command -v md5 >/dev/null 2>&1; then
    [[ "$(md5 -q "$archive")" == "387168f6dfb6ae821b08f7a3b92ab056" ]] || {
      echo "Error: PharmacoMatch archive checksum mismatch." >&2
      exit 1
    }
  else
    echo "Error: md5sum or md5 is required to verify the dataset archive." >&2
    exit 1
  fi
  extract_archive "$archive" ".downloads/pharmacomatch_extract"
  if [[ ! -d ".downloads/pharmacomatch_extract/data" ]]; then
    echo "Error: PharmacoMatch archive does not contain a top-level data directory." >&2
    exit 1
  fi
  cp -R ".downloads/pharmacomatch_extract/data/." data/
  echo "PharmacoMatch data ready under data/"
}

download_ani2x() {
  local archive="$DOWNLOAD_DIR/ANI-2x-wB97X-631Gd.tar.gz"
  echo "Downloading ANI-2x (wB97X/6-31G(d)) from $ANI2X_URL"
  download_file "$ANI2X_URL" "$archive"
  verify_md5 "cb1d9effb3d07fc1cc6ced7cd0b1e1f2" "$archive"
  extract_archive "$archive" "$DATA_DIR/ANI-2x"
  echo "ANI-2x ready under $DATA_DIR/ANI-2x/"
}

download_spice() {
  local dataset_dir="$DATA_DIR/SPICE"
  local dataset_file="$dataset_dir/SPICE-2.0.1.hdf5"
  mkdir -p "$dataset_dir"
  echo "Downloading SPICE 2.0.1 from $SPICE_URL"
  download_file "$SPICE_URL" "$dataset_file"
  verify_md5 "bfba2224b6540e1390a579569b475510" "$dataset_file"
  echo "SPICE ready at $dataset_file"
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
  pharmacomatch)
    download_pharmacomatch
    ;;
  ani2x)
    download_ani2x
    ;;
  spice)
    download_spice
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
    echo "Usage: bash scripts/download_datasets.sh [all|dude|qm9|pharmacomatch|ani2x|spice|lit-pcba|dekois2|bayesbind]" >&2
    exit 1
    ;;
esac
