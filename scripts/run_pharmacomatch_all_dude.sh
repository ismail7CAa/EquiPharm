#!/usr/bin/env bash
set -u

ROOT="/data/db6/Izzy/EquiPharm"
cd "$ROOT" || exit 1

PHARM_ROOT="external/PharmacoMatch"
CDPKIT_BIN="external/CDPKit/Bin"
OUT_ROOT="pharmacophore/results/PharmacoMatch"
LOG_ROOT="pharmacophore/results/PharmacoMatch_logs"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

echo "Running PharmacoMatch on all DUD-E targets"
echo "ROOT=$ROOT"
echo "PHARM_ROOT=$PHARM_ROOT"
echo "OUT_ROOT=$OUT_ROOT"
echo

for TARGET_DIR in data/DUD-E/*; do
    [ -d "$TARGET_DIR" ] || continue

    TARGET="$(basename "$TARGET_DIR")"
    echo "============================================================"
    echo "TARGET: $TARGET"
    echo "============================================================"

    ACTIVES="$TARGET_DIR/actives_final.sdf"
    INACTIVES="$TARGET_DIR/decoys_final.sdf"
    LIGAND="$TARGET_DIR/crystal_ligand.mol2"

    if [ ! -f "$ACTIVES" ]; then
        echo "[SKIP] Missing $ACTIVES"
        continue
    fi

    if [ ! -f "$INACTIVES" ]; then
        echo "[SKIP] Missing $INACTIVES"
        continue
    fi

    if [ ! -f "$LIGAND" ]; then
        echo "[WARN] Missing $LIGAND. The wrapper may fail if query ligand is required."
    fi

    EXT_TARGET="$PHARM_ROOT/data/DUD-E/$TARGET"
    PREP="$EXT_TARGET/preprocessing"
    RAW="$EXT_TARGET/raw"

    mkdir -p "$PREP" "$RAW" "$OUT_ROOT/$TARGET"

    echo "[1/4] Initializing PharmacoMatch target structure if needed..."

    if [ ! -d "$EXT_TARGET" ] || [ ! -f "$PREP/actives.sdf" ]; then
        python -m pharmacophore.PharmacoMatch.cli \
          --prepare-target-dir "$TARGET_DIR" \
          --output-dir "$OUT_ROOT/$TARGET" \
          --pharmacomatch-root "$PHARM_ROOT" \
          --accelerator cuda \
          --devices 1 \
          > "$LOG_ROOT/${TARGET}_initial_prepare.log" 2>&1 || true
    fi

    echo "[2/4] Replacing preprocessing SDFs with valid original DUD-E files..."

    cp "$ACTIVES" "$PREP/actives.sdf"
    cp "$INACTIVES" "$PREP/inactives.sdf"

    echo "[3/4] Creating CDPKit PSD databases..."

    "$CDPKIT_BIN/psdcreate" \
      -i "$PREP/actives.sdf" \
      -o "$RAW/actives.psd" \
      -d \
      -v ERROR \
      -l "$LOG_ROOT/${TARGET}_actives_psdcreate.log"

    if [ $? -ne 0 ]; then
        echo "[WARN] actives psdcreate with -d failed. Retrying without -d..."
        "$CDPKIT_BIN/psdcreate" \
          -i "$PREP/actives.sdf" \
          -o "$RAW/actives.psd" \
          -v ERROR \
          -l "$LOG_ROOT/${TARGET}_actives_psdcreate_no_d.log" || {
            echo "[FAIL] Could not create actives.psd for $TARGET"
            continue
          }
    fi

    "$CDPKIT_BIN/psdcreate" \
      -i "$PREP/inactives.sdf" \
      -o "$RAW/inactives.psd" \
      -d \
      -v ERROR \
      -l "$LOG_ROOT/${TARGET}_inactives_psdcreate.log"

    if [ $? -ne 0 ]; then
        echo "[WARN] inactives psdcreate with -d failed. Retrying without -d..."
        "$CDPKIT_BIN/psdcreate" \
          -i "$PREP/inactives.sdf" \
          -o "$RAW/inactives.psd" \
          -v ERROR \
          -l "$LOG_ROOT/${TARGET}_inactives_psdcreate_no_d.log" || {
            echo "[FAIL] Could not create inactives.psd for $TARGET"
            continue
          }
    fi

    echo "[4/4] Running PharmacoMatch screening..."

    python -m pharmacophore.PharmacoMatch.cli \
      --prepare-target-dir "$TARGET_DIR" \
      --output-dir "$OUT_ROOT/$TARGET" \
      --pharmacomatch-root "$PHARM_ROOT" \
      --accelerator cuda \
      --devices 1 \
      > "$LOG_ROOT/${TARGET}_run.log" 2>&1

    if [ $? -eq 0 ]; then
        echo "[OK] Finished $TARGET"
    else
        echo "[FAIL] PharmacoMatch failed for $TARGET"
        echo "Check log: $LOG_ROOT/${TARGET}_run.log"
        tail -40 "$LOG_ROOT/${TARGET}_run.log"
    fi

    echo
done

echo "Done. Results:"
find "$OUT_ROOT" -maxdepth 2 -type f | sort
