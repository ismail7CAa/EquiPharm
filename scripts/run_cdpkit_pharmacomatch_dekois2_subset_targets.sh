#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/data/db6/Izzy/EquiPharm}"
cd "$ROOT" || exit 1

DATA_ROOT="${DATA_ROOT:-data/DEKOIS2.0}"
PHARM_ROOT="${PHARM_ROOT:-external/PharmacoMatch}"
CDPKIT_BIN="${CDPKIT_BIN:-external/CDPKit/Bin}"
N_ACTIVES="${N_ACTIVES:-40}"
N_DECOYS="${N_DECOYS:-500}"
SEEDS="${SEEDS:-1 2 3}"

RUN_LABEL="${N_ACTIVES}A_${N_DECOYS}D"
OUT_ROOT="${OUT_ROOT:-pharmacophore/results/CDPKit_PharmacoMatchAlignment_DEKOIS2.0_subset_${RUN_LABEL}}"
LOG_ROOT="${LOG_ROOT:-pharmacophore/results/CDPKit_PharmacoMatchAlignment_DEKOIS2.0_subset_${RUN_LABEL}_logs}"
WORK_ROOT="${WORK_ROOT:-pharmacophore/work/CDPKit_DEKOIS2.0_subset_${RUN_LABEL}}"
HELPER="scripts/create_shuffled_sdf_subset.py"
AGGREGATOR="scripts/aggregate_seed_metrics.py"
ALIGNER="scripts/run_pharmacomatch_cdpkit_alignment.py"
EVALUATOR="scripts/eval_pharmacomatch_cdpkit_alignment.py"
QUERY_GENERATOR="$PHARM_ROOT/data_processing/python_scripts/cdpl/pharm_gen_mol_ph4s.py"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 TARGET [TARGET ...]"
    echo "Example: $0 ace ache akt1"
    echo 'Optional: SEEDS="1 2 3" N_ACTIVES=40 N_DECOYS=500 bash scripts/run_cdpkit_pharmacomatch_dekois2_subset_targets.sh ace'
    exit 1
fi

for REQUIRED in "$HELPER" "$AGGREGATOR" "$ALIGNER" "$EVALUATOR" "$QUERY_GENERATOR"; do
    if [ ! -f "$REQUIRED" ]; then echo "[FAIL] Missing file: $REQUIRED"; exit 1; fi
done
if [ ! -x "$CDPKIT_BIN/psdcreate" ]; then
    echo "[FAIL] Missing executable: $CDPKIT_BIN/psdcreate"
    exit 1
fi

resolve_target() {
    local requested="$1" candidate
    for candidate in "$DATA_ROOT/$requested" "$DATA_ROOT/${requested^^}" "$DATA_ROOT/${requested,,}"; do
        if [ -d "$candidate" ]; then printf '%s\n' "$candidate"; return 0; fi
    done
    return 1
}

mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$WORK_ROOT"

for REQUESTED_TARGET in "$@"; do
    if ! SOURCE_TARGET="$(resolve_target "$REQUESTED_TARGET")"; then
        echo "[SKIP] DEKOIS 2.0 target not found: $REQUESTED_TARGET"
        continue
    fi
    TARGET="$(basename "$SOURCE_TARGET")"
    if [ ! -d "$SOURCE_TARGET/actives_sdf" ] || [ ! -d "$SOURCE_TARGET/decoys_sdf" ] || [ ! -f "$SOURCE_TARGET/crystal_ligand.mol2" ]; then
        echo "[SKIP] $TARGET must contain actives_sdf/, decoys_sdf/, and crystal_ligand.mol2"
        continue
    fi

    for SEED in $SEEDS; do
        RUN_NAME="dekois2_${TARGET}_seed${SEED}"
        TARGET_WORK="$WORK_ROOT/seed_$SEED/$RUN_NAME"
        RAW="$TARGET_WORK/raw"
        PREP="$TARGET_WORK/preprocessing"
        VS="$TARGET_WORK/vs"
        TARGET_OUT="$OUT_ROOT/seed_$SEED/$TARGET"
        TARGET_LOG="$LOG_ROOT/seed_$SEED/$TARGET"

        rm -rf "$TARGET_WORK" "$TARGET_OUT" "$TARGET_LOG"
        mkdir -p "$RAW" "$PREP" "$VS" "$TARGET_OUT" "$TARGET_LOG"

        python "$HELPER" --input "$SOURCE_TARGET/actives_sdf" --output "$PREP/actives.sdf" \
            --count "$N_ACTIVES" --seed "$SEED" --target "$TARGET" --class-label active \
            --manifest "$TARGET_OUT/selected_actives.json" > "$TARGET_LOG/select_actives.log" 2>&1 || { echo "[FAIL] Active selection: $TARGET seed $SEED"; continue; }
        python "$HELPER" --input "$SOURCE_TARGET/decoys_sdf" --output "$PREP/inactives.sdf" \
            --count "$N_DECOYS" --seed "$SEED" --target "$TARGET" --class-label inactive \
            --manifest "$TARGET_OUT/selected_decoys.json" > "$TARGET_LOG/select_decoys.log" 2>&1 || { echo "[FAIL] Decoy selection: $TARGET seed $SEED"; continue; }

        python "$QUERY_GENERATOR" -i "$SOURCE_TARGET/crystal_ligand.mol2" -o "$RAW/query.pml" \
            > "$TARGET_LOG/query.log" 2>&1 || { echo "[FAIL] Query generation: $TARGET seed $SEED"; continue; }
        "$CDPKIT_BIN/psdcreate" -i "$PREP/actives.sdf" -o "$RAW/actives.psd" -d -v ERROR -l "$TARGET_LOG/actives_psdcreate.log" || \
            "$CDPKIT_BIN/psdcreate" -i "$PREP/actives.sdf" -o "$RAW/actives.psd" -v ERROR -l "$TARGET_LOG/actives_psdcreate_no_d.log" || \
            { echo "[FAIL] Active PSD: $TARGET seed $SEED"; continue; }
        "$CDPKIT_BIN/psdcreate" -i "$PREP/inactives.sdf" -o "$RAW/inactives.psd" -d -v ERROR -l "$TARGET_LOG/inactives_psdcreate.log" || \
            "$CDPKIT_BIN/psdcreate" -i "$PREP/inactives.sdf" -o "$RAW/inactives.psd" -v ERROR -l "$TARGET_LOG/inactives_psdcreate_no_d.log" || \
            { echo "[FAIL] Decoy PSD: $TARGET seed $SEED"; continue; }

        python "$ALIGNER" "$TARGET_WORK" > "$TARGET_LOG/alignment.log" 2>&1 || { echo "[FAIL] Alignment: $TARGET seed $SEED"; tail -40 "$TARGET_LOG/alignment.log"; continue; }
        python "$EVALUATOR" "$TARGET_WORK" "$TARGET_OUT/metrics.json" > "$TARGET_LOG/evaluation.log" 2>&1

        if [ $? -eq 0 ]; then
            echo "[OK] Finished $TARGET seed $SEED: $TARGET_OUT"
        else
            echo "[FAIL] Evaluation: $TARGET seed $SEED"
            tail -40 "$TARGET_LOG/evaluation.log"
        fi
    done

    METRIC_FILES=()
    for SEED in $SEEDS; do
        METRIC_FILES+=("$OUT_ROOT/seed_$SEED/$TARGET/metrics.json")
    done
    python "$AGGREGATOR" --target "$TARGET" \
        --output "$OUT_ROOT/seed_mean/$TARGET/metrics.json" \
        "${METRIC_FILES[@]}" \
        > "$LOG_ROOT/${TARGET}_seed_mean.log" 2>&1 || {
            echo "[WARN] Could not calculate seed mean for $TARGET"
            tail -40 "$LOG_ROOT/${TARGET}_seed_mean.log"
        }
done

echo "Done. Results: $OUT_ROOT"
