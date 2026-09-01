#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/data/db6/Izzy/EquiPharm}"
cd "$ROOT" || exit 1

DATA_ROOT="${DATA_ROOT:-data/BayesBind_prepared}"
PHARM_ROOT="${PHARM_ROOT:-external/PharmacoMatch}"
CDPKIT_BIN="${CDPKIT_BIN:-external/CDPKit/Bin}"

N_ACTIVES="${N_ACTIVES:-50}"
N_DECOYS="${N_DECOYS:-500}"
SEEDS="${SEEDS:-1 2 3}"
BATCH_SIZE="${BATCH_SIZE:-34}"

RUN_LABEL="${N_ACTIVES}A_${N_DECOYS}D"

OUT_ROOT="${OUT_ROOT:-pharmacophore/results/PharmacoMatch_BayesBind_subset_${RUN_LABEL}}"
LOG_ROOT="${LOG_ROOT:-pharmacophore/results/PharmacoMatch_BayesBind_subset_${RUN_LABEL}_logs}"
WORK_ROOT="${WORK_ROOT:-pharmacophore/work/PharmacoMatch_BayesBind_subset_${RUN_LABEL}}"

HELPER="scripts/create_shuffled_sdf_subset.py"
AGGREGATOR="scripts/aggregate_seed_metrics.py"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 TARGET [TARGET ...]"
    echo "Example: $0 ABL1 EGFR SRC"
    echo 'Optional: SEEDS="1 2 3" N_ACTIVES=50 N_DECOYS=500 BATCH_SIZE=34 bash scripts/run_pharmacomatch_bayesbind_subset_targets.sh ABL1'
    exit 1
fi

for REQUIRED in "$HELPER" "$AGGREGATOR"; do
    if [ ! -f "$REQUIRED" ]; then
        echo "[FAIL] Missing helper: $REQUIRED"
        exit 1
    fi
done

if [ ! -x "$CDPKIT_BIN/psdcreate" ]; then
    echo "[FAIL] CDPKit psdcreate not found or not executable: $CDPKIT_BIN/psdcreate"
    exit 1
fi

resolve_target() {
    local requested="$1" candidate

    for candidate in \
        "$DATA_ROOT/$requested" \
        "$DATA_ROOT/${requested^^}" \
        "$DATA_ROOT/${requested,,}"
    do
        if [ -d "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$WORK_ROOT"

for REQUESTED_TARGET in "$@"; do

    SOURCE_TARGET="$(resolve_target "$REQUESTED_TARGET")" || {
        echo "[SKIP] BayesBind target not found: $REQUESTED_TARGET"
        continue
    }

    TARGET="$(basename "$SOURCE_TARGET")"

    for REQUIRED in actives_sdf decoys_sdf; do
        if [ ! -d "$SOURCE_TARGET/$REQUIRED" ]; then
            echo "[SKIP] Missing directory: $SOURCE_TARGET/$REQUIRED"
            continue 2
        fi
    done

    if [ ! -f "$SOURCE_TARGET/crystal_ligand.mol2" ]; then
        echo "[SKIP] Missing query: $SOURCE_TARGET/crystal_ligand.mol2"
        continue
    fi

    for SEED in $SEEDS; do

        RUN_NAME="bayesbind_${TARGET}_seed${SEED}"

        SUBSET_TARGET="$WORK_ROOT/seed_$SEED/$RUN_NAME"
        PREPARED_TARGET="$PHARM_ROOT/data/BayesBind/$RUN_NAME"

        TARGET_OUT="$OUT_ROOT/seed_$SEED/$TARGET"
        TARGET_LOG="$LOG_ROOT/seed_$SEED/$TARGET"

        rm -rf \
            "$SUBSET_TARGET" \
            "$PREPARED_TARGET" \
            "$TARGET_OUT" \
            "$TARGET_LOG"

        mkdir -p \
            "$SUBSET_TARGET/actives_sdf" \
            "$SUBSET_TARGET/decoys_sdf" \
            "$TARGET_OUT" \
            "$TARGET_LOG"

        cp \
            "$SOURCE_TARGET/crystal_ligand.mol2" \
            "$SUBSET_TARGET/crystal_ligand.mol2"

        echo "[$TARGET seed $SEED] Selecting $N_ACTIVES actives and $N_DECOYS decoys"

        # --------------------------------------
        # ACTIVES
        # --------------------------------------
        python "$HELPER" \
            --input "$SOURCE_TARGET/actives_sdf" \
            --output "$SUBSET_TARGET/actives_sdf/actives.sdf" \
            --count "$N_ACTIVES" \
            --seed "$SEED" \
            --target "$TARGET" \
            --class-label active \
            --manifest "$TARGET_OUT/selected_actives.json" \
            > "$TARGET_LOG/select_actives.log" 2>&1 || {

                echo "[FAIL] Active selection failed: $TARGET seed $SEED"
                tail -40 "$TARGET_LOG/select_actives.log"
                continue
            }

        # --------------------------------------
        # DECOYS
        # --------------------------------------
        python "$HELPER" \
            --input "$SOURCE_TARGET/decoys_sdf" \
            --output "$SUBSET_TARGET/decoys_sdf/decoys.sdf" \
            --count "$N_DECOYS" \
            --seed "$SEED" \
            --target "$TARGET" \
            --class-label inactive \
            --manifest "$TARGET_OUT/selected_decoys.json" \
            > "$TARGET_LOG/select_decoys.log" 2>&1 || {

                echo "[FAIL] Decoy selection failed: $TARGET seed $SEED"
                tail -40 "$TARGET_LOG/select_decoys.log"
                continue
            }

        # --------------------------------------
        # PHARMACOMATCH
        # --------------------------------------
        python -m pharmacophore.PharmacoMatch.cli \
            --prepare-target-dir "$SUBSET_TARGET" \
            --prepared-vs-dir "$PREPARED_TARGET" \
            --output-dir "$TARGET_OUT" \
            --pharmacomatch-root "$PHARM_ROOT" \
            --cdpkit-bin "$CDPKIT_BIN" \
            --force-prepare \
            --accelerator cuda \
            --devices 1 \
            --batch-size "$BATCH_SIZE" \
            > "$TARGET_LOG/run.log" 2>&1

        if [ $? -eq 0 ]; then
            echo "[OK] Finished $TARGET seed $SEED: $TARGET_OUT"
        else
            echo "[FAIL] PharmacoMatch failed: $TARGET seed $SEED"
            tail -60 "$TARGET_LOG/run.log"
        fi

    done

    # ------------------------------------------
    # AGGREGATE 3 SEEDS
    # ------------------------------------------
    METRIC_FILES=()

    for SEED in $SEEDS; do
        METRIC_FILES+=(
            "$OUT_ROOT/seed_$SEED/$TARGET/metrics.json"
        )
    done

    python "$AGGREGATOR" \
        --target "$TARGET" \
        --output "$OUT_ROOT/seed_mean/$TARGET/metrics.json" \
        "${METRIC_FILES[@]}" \
        > "$LOG_ROOT/${TARGET}_seed_mean.log" 2>&1 || {

            echo "[WARN] Could not calculate seed mean for $TARGET"
            tail -40 "$LOG_ROOT/${TARGET}_seed_mean.log"
        }

done

echo "Done. Results: $OUT_ROOT"