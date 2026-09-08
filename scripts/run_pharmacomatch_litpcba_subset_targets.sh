#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/data/db6/Izzy/EquiPharm}"
cd "$ROOT" || exit 1

DATA_ROOT="${DATA_ROOT:-data/LIT-PCBA}"
PHARM_ROOT="${PHARM_ROOT:-external/PharmacoMatch}"
CDPKIT_BIN="${CDPKIT_BIN:-external/CDPKit/Bin}"

N_ACTIVES="${N_ACTIVES:-50}"
N_DECOYS="${N_DECOYS:-500}"
SEEDS="${SEEDS:-1 2 3}"
BATCH_SIZE="${BATCH_SIZE:-34}"

RUN_LABEL="${N_ACTIVES}A_${N_DECOYS}D"

OUT_ROOT="${OUT_ROOT:-pharmacophore/results/PharmacoMatch_LIT-PCBA_subset_${RUN_LABEL}}"
LOG_ROOT="${LOG_ROOT:-pharmacophore/results/PharmacoMatch_LIT-PCBA_subset_${RUN_LABEL}_logs}"
WORK_ROOT="${WORK_ROOT:-pharmacophore/work/PharmacoMatch_LIT-PCBA_subset_${RUN_LABEL}}"

HELPER="scripts/create_shuffled_sdf_subset.py"
AGGREGATOR="scripts/aggregate_seed_metrics.py"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 TARGET [TARGET ...]"
    echo
    echo "Example:"
    echo "  $0 ALDH1 ESR_ANT"
    echo
    echo "Optional:"
    echo '  SEEDS="1 2 3" N_ACTIVES=40 N_DECOYS=500 BATCH_SIZE=34 \'
    echo '  bash scripts/run_pharmacomatch_litpcba_subset_targets.sh ALDH1'
    exit 1
fi


# ------------------------------------------------------------
# Check required files
# ------------------------------------------------------------

for REQUIRED in "$HELPER" "$AGGREGATOR"; do
    if [ ! -f "$REQUIRED" ]; then
        echo "[FAIL] Missing helper: $REQUIRED"
        exit 1
    fi
done

if [ ! -x "$CDPKIT_BIN/psdcreate" ]; then
    echo "[FAIL] CDPKit psdcreate not found or not executable:"
    echo "       $CDPKIT_BIN/psdcreate"
    exit 1
fi


# ------------------------------------------------------------
# Resolve target directory case-insensitively
# ------------------------------------------------------------

resolve_target() {
    local requested="$1"
    local candidate

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

    # Fallback: proper case-insensitive search
    candidate="$(
        find "$DATA_ROOT" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -iname "$requested" \
            -print \
            -quit
    )"

    if [ -n "$candidate" ]; then
        printf '%s\n' "$candidate"
        return 0
    fi

    return 1
}


# ------------------------------------------------------------
# Resolve active directory/file
# ------------------------------------------------------------

resolve_actives() {
    local target_dir="$1"
    local candidate

    for candidate in \
        "$target_dir/actives_sdf" \
        "$target_dir/actives.sdf" \
        "$target_dir/active.sdf"
    do
        if [ -e "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}


# ------------------------------------------------------------
# Resolve inactive / decoy directory/file
# LIT-PCBA normally refers to inactive compounds rather than
# artificial decoys, so support both naming conventions.
# ------------------------------------------------------------

resolve_inactives() {
    local target_dir="$1"
    local candidate

    for candidate in \
        "$target_dir/inactives_sdf" \
        "$target_dir/decoys_sdf" \
        "$target_dir/inactives.sdf" \
        "$target_dir/decoys.sdf" \
        "$target_dir/inactive.sdf"
    do
        if [ -e "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}


# ------------------------------------------------------------
# Resolve query ligand
# PharmacoMatch currently expects crystal_ligand.mol2 in the
# temporary target directory, so find a suitable MOL2 source.
# ------------------------------------------------------------

resolve_query() {
    local target_dir="$1"
    local candidate

    for candidate in \
        "$target_dir/crystal_ligand.mol2" \
        "$target_dir/query.mol2" \
        "$target_dir/ligand.mol2"
    do
        if [ -f "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}


mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$WORK_ROOT"


# ============================================================
# Run targets
# ============================================================

for REQUESTED_TARGET in "$@"; do

    if ! SOURCE_TARGET="$(resolve_target "$REQUESTED_TARGET")"; then
        echo "[SKIP] LIT-PCBA target not found: $REQUESTED_TARGET"
        continue
    fi

    TARGET="$(basename "$SOURCE_TARGET")"

    echo
    echo "============================================================"
    echo "LIT-PCBA target: $TARGET"
    echo "Source:          $SOURCE_TARGET"
    echo "============================================================"


    # --------------------------------------------------------
    # Locate active compounds
    # --------------------------------------------------------

    if ! ACTIVES_INPUT="$(resolve_actives "$SOURCE_TARGET")"; then
        echo "[SKIP] No active SDF source found for $TARGET"
        echo "       Checked:"
        echo "       actives_sdf/"
        echo "       actives.sdf"
        echo "       active.sdf"
        continue
    fi


    # --------------------------------------------------------
    # Locate inactive compounds
    # --------------------------------------------------------

    if ! INACTIVES_INPUT="$(resolve_inactives "$SOURCE_TARGET")"; then
        echo "[SKIP] No inactive/decoy SDF source found for $TARGET"
        echo "       Checked:"
        echo "       inactives_sdf/"
        echo "       decoys_sdf/"
        echo "       inactives.sdf"
        echo "       decoys.sdf"
        continue
    fi


    # --------------------------------------------------------
    # Locate pharmacophore query
    # --------------------------------------------------------

    if ! QUERY_FILE="$(resolve_query "$SOURCE_TARGET")"; then
        echo "[SKIP] No query MOL2 found for $TARGET"
        echo "       Checked:"
        echo "       crystal_ligand.mol2"
        echo "       query.mol2"
        echo "       ligand.mol2"
        continue
    fi


    echo "Actives:   $ACTIVES_INPUT"
    echo "Inactives: $INACTIVES_INPUT"
    echo "Query:     $QUERY_FILE"


    # ========================================================
    # Seeds
    # ========================================================

    for SEED in $SEEDS; do

        RUN_NAME="litpcba_${TARGET}_seed${SEED}"

        SUBSET_TARGET="$WORK_ROOT/seed_$SEED/$RUN_NAME"

        PREPARED_TARGET="$PHARM_ROOT/data/LIT-PCBA/$RUN_NAME"

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


        # PharmacoMatch preparation expects this exact name
        cp "$QUERY_FILE" "$SUBSET_TARGET/crystal_ligand.mol2"


        echo
        echo "[$TARGET seed $SEED] Selecting:"
        echo "  Actives:   $N_ACTIVES"
        echo "  Inactives: $N_DECOYS"


        # ----------------------------------------------------
        # Select actives
        # ----------------------------------------------------

        python "$HELPER" \
            --input "$ACTIVES_INPUT" \
            --output "$SUBSET_TARGET/actives_sdf/actives.sdf" \
            --count "$N_ACTIVES" \
            --seed "$SEED" \
            --target "$TARGET" \
            --class-label active \
            --manifest "$TARGET_OUT/selected_actives.json" \
            > "$TARGET_LOG/select_actives.log" 2>&1 || {

                echo "[FAIL] Active selection failed:"
                echo "       $TARGET seed $SEED"

                tail -40 "$TARGET_LOG/select_actives.log"

                continue
            }


        # ----------------------------------------------------
        # Select LIT-PCBA inactives
        #
        # PharmacoMatch calls these "decoys" internally, so
        # output is still written as decoys.sdf.
        # ----------------------------------------------------

        python "$HELPER" \
            --input "$INACTIVES_INPUT" \
            --output "$SUBSET_TARGET/decoys_sdf/decoys.sdf" \
            --count "$N_DECOYS" \
            --seed "$SEED" \
            --target "$TARGET" \
            --class-label inactive \
            --manifest "$TARGET_OUT/selected_inactives.json" \
            > "$TARGET_LOG/select_inactives.log" 2>&1 || {

                echo "[FAIL] Inactive selection failed:"
                echo "       $TARGET seed $SEED"

                tail -40 "$TARGET_LOG/select_inactives.log"

                continue
            }


        # ----------------------------------------------------
        # Run PharmacoMatch
        # ----------------------------------------------------

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
            echo "[OK] Finished $TARGET seed $SEED"
            echo "     $TARGET_OUT"
        else
            echo "[FAIL] PharmacoMatch failed:"
            echo "       $TARGET seed $SEED"

            tail -60 "$TARGET_LOG/run.log"
        fi

    done


    # ========================================================
    # Aggregate metrics over seeds
    # ========================================================

    METRIC_FILES=()

    for SEED in $SEEDS; do
        METRIC_FILES+=(
            "$OUT_ROOT/seed_$SEED/$TARGET/metrics.json"
        )
    done


    mkdir -p "$OUT_ROOT/seed_mean/$TARGET"


    python "$AGGREGATOR" \
        --target "$TARGET" \
        --output "$OUT_ROOT/seed_mean/$TARGET/metrics.json" \
        "${METRIC_FILES[@]}" \
        > "$LOG_ROOT/${TARGET}_seed_mean.log" 2>&1 || {

            echo "[WARN] Could not calculate seed mean for $TARGET"

            tail -40 "$LOG_ROOT/${TARGET}_seed_mean.log"
        }

done


echo
echo "============================================================"
echo "Done."
echo "Results: $OUT_ROOT"
echo "============================================================"