#!/usr/bin/env bash
set -euo pipefail

TARGET="$1"
BATCH_SIZE="${2:-34}"

ROOT="/data/db6/Izzy/EquiPharm"
cd "$ROOT"

PHARM_ROOT="external/PharmacoMatch"
CDPKIT_BIN="external/CDPKit/Bin"
OUT_DIR="pharmacophore/results/PharmacoMatch/$TARGET"
LOG_DIR="pharmacophore/results/PharmacoMatch_logs"

SRC_DIR="data/DUD-E/$TARGET"
EXT_DIR="$PHARM_ROOT/data/DUD-E/$TARGET"
PREP="$EXT_DIR/preprocessing"
RAW="$EXT_DIR/raw"

mkdir -p "$PREP" "$RAW" "$OUT_DIR" "$LOG_DIR"

echo "=== TARGET: $TARGET ==="

if [ ! -f "$SRC_DIR/actives_final.sdf" ]; then
  echo "[FAIL] Missing $SRC_DIR/actives_final.sdf"
  exit 1
fi

if [ ! -f "$SRC_DIR/decoys_final.sdf" ]; then
  echo "[FAIL] Missing $SRC_DIR/decoys_final.sdf"
  exit 1
fi

echo "[1] Copy valid original DUD-E SDFs"
cp "$SRC_DIR/actives_final.sdf" "$PREP/actives.sdf"
cp "$SRC_DIR/decoys_final.sdf" "$PREP/inactives.sdf"

echo "[2] Create PSD files"
"$CDPKIT_BIN/psdcreate" \
  -i "$PREP/actives.sdf" \
  -o "$RAW/actives.psd" \
  -d \
  -v ERROR \
  -l "$LOG_DIR/${TARGET}_actives_psdcreate.log"

"$CDPKIT_BIN/psdcreate" \
  -i "$PREP/inactives.sdf" \
  -o "$RAW/inactives.psd" \
  -d \
  -v ERROR \
  -l "$LOG_DIR/${TARGET}_inactives_psdcreate.log"

echo "[3] Run PharmacoMatch"
python -m pharmacophore.PharmacoMatch.cli \
  --prepare-target-dir "$SRC_DIR" \
  --output-dir "$OUT_DIR" \
  --pharmacomatch-root "$PHARM_ROOT" \
  --accelerator cuda \
  --devices 1 \
  --batch-size "$BATCH_SIZE"

echo "[OK] Finished $TARGET"
