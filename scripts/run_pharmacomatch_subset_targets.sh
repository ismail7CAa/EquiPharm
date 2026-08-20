#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/data/db6/Izzy/EquiPharm}"
cd "$ROOT" || exit 1

PHARM_ROOT="${PHARM_ROOT:-external/PharmacoMatch}"
CDPKIT_BIN="${CDPKIT_BIN:-external/CDPKit/Bin}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-data/DUD-E}"

N_ACTIVES="${N_ACTIVES:-50}"
N_INACTIVES="${N_INACTIVES:-500}"
SEEDS="${SEEDS:-1 2 3}"

OUT_ROOT="${OUT_ROOT:-pharmacophore/results/PharmacoMatch_subset_50A_500D}"
LOG_ROOT="${LOG_ROOT:-pharmacophore/results/PharmacoMatch_subset_50A_500D_logs}"
WORK_ROOT="${WORK_ROOT:-pharmacophore/work/PharmacoMatch_subset_50A_500D}"

HELPER="scripts/create_shuffled_sdf_subset.py"

mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$WORK_ROOT" scripts

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <target1> [target2 ...]"
    echo "Example: $0 ada andr egfr"
    echo
    echo "Optional environment variables:"
    echo '  SEEDS="1 2 3" N_ACTIVES=50 N_INACTIVES=500 bash run_pharmacomatch_subset_targets.sh ada andr'
    exit 1
fi

if [ ! -x "$CDPKIT_BIN/psdcreate" ]; then
    echo "[FAIL] CDPKit psdcreate not found or not executable: $CDPKIT_BIN/psdcreate"
    exit 1
fi

cat > "$HELPER" <<'PY'
import argparse
import json
import random
from pathlib import Path

from rdkit import Chem


def molecule_name(mol, fallback):
    if mol.HasProp("_Name"):
        name = mol.GetProp("_Name").strip()
        if name:
            return name

    for key in ("ID", "Id", "id", "MOL_ID", "mol_id", "NAME", "Name"):
        if mol.HasProp(key):
            value = mol.GetProp(key).strip()
            if value:
                return value

    return fallback


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle valid SDF records reproducibly and write a fixed-size subset."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--target", required=True)
    parser.add_argument("--class-label", required=True, choices=("active", "inactive"))
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    supplier = Chem.SDMolSupplier(str(input_path), sanitize=False, removeHs=False)

    valid_records = []
    invalid_indices = []

    for original_index, mol in enumerate(supplier):
        if mol is None:
            invalid_indices.append(original_index)
            continue

        valid_records.append(
            {
                "original_index_zero_based": original_index,
                "original_index_one_based": original_index + 1,
                "name": molecule_name(mol, f"record_{original_index + 1}"),
                "mol": mol,
            }
        )

    if len(valid_records) < args.count:
        raise RuntimeError(
            f"Requested {args.count} {args.class_label}s, but only "
            f"{len(valid_records)} valid molecules were found in {input_path}."
        )

    # DUD-E records may have a meaningful/sorted source order. Shuffle the full
    # valid record order immediately before taking the requested subset.
    shuffled = valid_records.copy()
    rng = random.Random(args.seed)
    rng.shuffle(shuffled)
    selected = shuffled[: args.count]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output_path))
    if writer is None:
        raise RuntimeError(f"Could not open output SDF: {output_path}")

    try:
        for item in selected:
            writer.write(item["mol"])
    finally:
        writer.close()

    manifest = {
        "target": args.target,
        "class": args.class_label,
        "seed": args.seed,
        "selection": "shuffle_all_valid_records_then_take_first_n",
        "input_sdf": str(input_path),
        "output_sdf": str(output_path),
        "requested_count": args.count,
        "available_valid_count": len(valid_records),
        "invalid_record_count": len(invalid_indices),
        "invalid_original_indices_zero_based": invalid_indices,
        "selected": [
            {
                "selection_rank_one_based": rank,
                "original_index_zero_based": item["original_index_zero_based"],
                "original_index_one_based": item["original_index_one_based"],
                "name": item["name"],
            }
            for rank, item in enumerate(selected, start=1)
        ],
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        json.dumps(
            {
                "target": args.target,
                "class": args.class_label,
                "seed": args.seed,
                "selected": len(selected),
                "available_valid": len(valid_records),
                "invalid": len(invalid_indices),
                "output": str(output_path),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
PY

echo "Running PharmacoMatch shuffled-subset benchmark"
echo "ROOT=$ROOT"
echo "SOURCE_DATA_ROOT=$SOURCE_DATA_ROOT"
echo "TARGETS=$*"
echo "SEEDS=$SEEDS"
echo "N_ACTIVES=$N_ACTIVES"
echo "N_INACTIVES=$N_INACTIVES"
echo "OUT_ROOT=$OUT_ROOT"
echo

for TARGET_ARG in "$@"; do
    TARGET="$TARGET_ARG"
    SOURCE_TARGET="$SOURCE_DATA_ROOT/$TARGET"

    # Convenience fallback for target names entered in uppercase.
    if [ ! -d "$SOURCE_TARGET" ]; then
        TARGET_LOWER="$(printf '%s' "$TARGET_ARG" | tr '[:upper:]' '[:lower:]')"
        if [ -d "$SOURCE_DATA_ROOT/$TARGET_LOWER" ]; then
            TARGET="$TARGET_LOWER"
            SOURCE_TARGET="$SOURCE_DATA_ROOT/$TARGET"
        fi
    fi

    if [ ! -d "$SOURCE_TARGET" ]; then
        echo "[SKIP] Target directory not found: $SOURCE_TARGET"
        continue
    fi

    SOURCE_ACTIVES="$SOURCE_TARGET/actives_final.sdf"
    SOURCE_INACTIVES="$SOURCE_TARGET/decoys_final.sdf"
    SOURCE_LIGAND="$SOURCE_TARGET/crystal_ligand.mol2"

    if [ ! -f "$SOURCE_ACTIVES" ]; then
        echo "[SKIP] Missing $SOURCE_ACTIVES"
        continue
    fi

    if [ ! -f "$SOURCE_INACTIVES" ]; then
        echo "[SKIP] Missing $SOURCE_INACTIVES"
        continue
    fi

    for SEED in $SEEDS; do
        echo "============================================================"
        echo "TARGET: $TARGET | SEED: $SEED"
        echo "============================================================"

        RUN_NAME="${TARGET}_seed${SEED}"
        SUBSET_TARGET_DIR="$WORK_ROOT/seed_$SEED/$RUN_NAME"
        TARGET_OUT="$OUT_ROOT/seed_$SEED/$TARGET"
        TARGET_LOG="$LOG_ROOT/seed_$SEED/$TARGET"

        # PharmacoMatch creates its own target working directory from the
        # basename of --prepare-target-dir. A seed-specific basename prevents
        # overwriting the original full-target PharmacoMatch files.
        EXT_TARGET="$PHARM_ROOT/data/DUD-E/$RUN_NAME"
        PREP="$EXT_TARGET/preprocessing"
        RAW="$EXT_TARGET/raw"

        rm -rf "$SUBSET_TARGET_DIR" "$EXT_TARGET" "$TARGET_OUT" "$TARGET_LOG"
        mkdir -p "$SUBSET_TARGET_DIR" "$PREP" "$RAW" "$TARGET_OUT" "$TARGET_LOG"

        echo "[1/5] Shuffling and selecting $N_ACTIVES actives..."
        python "$HELPER" \
          --input "$SOURCE_ACTIVES" \
          --output "$SUBSET_TARGET_DIR/actives_final.sdf" \
          --count "$N_ACTIVES" \
          --seed "$SEED" \
          --target "$TARGET" \
          --class-label active \
          --manifest "$TARGET_OUT/selected_actives.json" \
          > "$TARGET_LOG/select_actives.log" 2>&1

        if [ $? -ne 0 ]; then
            echo "[FAIL] Active selection failed for $TARGET seed $SEED"
            tail -40 "$TARGET_LOG/select_actives.log"
            echo
            continue
        fi

        echo "[2/5] Shuffling and selecting $N_INACTIVES decoys..."
        python "$HELPER" \
          --input "$SOURCE_INACTIVES" \
          --output "$SUBSET_TARGET_DIR/decoys_final.sdf" \
          --count "$N_INACTIVES" \
          --seed "$SEED" \
          --target "$TARGET" \
          --class-label inactive \
          --manifest "$TARGET_OUT/selected_inactives.json" \
          > "$TARGET_LOG/select_inactives.log" 2>&1

        if [ $? -ne 0 ]; then
            echo "[FAIL] Decoy selection failed for $TARGET seed $SEED"
            tail -40 "$TARGET_LOG/select_inactives.log"
            echo
            continue
        fi

        if [ -f "$SOURCE_LIGAND" ]; then
            cp "$SOURCE_LIGAND" "$SUBSET_TARGET_DIR/crystal_ligand.mol2"
        else
            echo "[WARN] Missing $SOURCE_LIGAND. PharmacoMatch may require the query ligand."
        fi

        echo "[3/5] Initializing seed-specific PharmacoMatch target..."
        python -m pharmacophore.PharmacoMatch.cli \
          --prepare-target-dir "$SUBSET_TARGET_DIR" \
          --output-dir "$TARGET_OUT" \
          --pharmacomatch-root "$PHARM_ROOT" \
          --accelerator cuda \
          --devices 1 \
          > "$TARGET_LOG/initial_prepare.log" 2>&1 || true

        # Always replace any generated preprocessing inputs with the exact
        # shuffled SDF subsets selected above.
        mkdir -p "$PREP" "$RAW"
        cp "$SUBSET_TARGET_DIR/actives_final.sdf" "$PREP/actives.sdf"
        cp "$SUBSET_TARGET_DIR/decoys_final.sdf" "$PREP/inactives.sdf"

        echo "[4/5] Creating CDPKit PSD databases..."

        rm -f "$RAW/actives.psd" "$RAW/inactives.psd"

        "$CDPKIT_BIN/psdcreate" \
          -i "$PREP/actives.sdf" \
          -o "$RAW/actives.psd" \
          -d \
          -v ERROR \
          -l "$TARGET_LOG/actives_psdcreate.log"

        if [ $? -ne 0 ]; then
            echo "[WARN] Active psdcreate with -d failed; retrying without -d..."
            "$CDPKIT_BIN/psdcreate" \
              -i "$PREP/actives.sdf" \
              -o "$RAW/actives.psd" \
              -v ERROR \
              -l "$TARGET_LOG/actives_psdcreate_no_d.log" || {
                echo "[FAIL] Could not create actives.psd for $TARGET seed $SEED"
                echo
                continue
              }
        fi

        "$CDPKIT_BIN/psdcreate" \
          -i "$PREP/inactives.sdf" \
          -o "$RAW/inactives.psd" \
          -d \
          -v ERROR \
          -l "$TARGET_LOG/inactives_psdcreate.log"

        if [ $? -ne 0 ]; then
            echo "[WARN] Inactive psdcreate with -d failed; retrying without -d..."
            "$CDPKIT_BIN/psdcreate" \
              -i "$PREP/inactives.sdf" \
              -o "$RAW/inactives.psd" \
              -v ERROR \
              -l "$TARGET_LOG/inactives_psdcreate_no_d.log" || {
                echo "[FAIL] Could not create inactives.psd for $TARGET seed $SEED"
                echo
                continue
              }
        fi

        echo "[5/5] Running PharmacoMatch screening..."
        python -m pharmacophore.PharmacoMatch.cli \
          --prepare-target-dir "$SUBSET_TARGET_DIR" \
          --output-dir "$TARGET_OUT" \
          --pharmacomatch-root "$PHARM_ROOT" \
          --accelerator cuda \
          --devices 1 \
          > "$TARGET_LOG/run.log" 2>&1

        if [ $? -eq 0 ]; then
            echo "[OK] Finished $TARGET seed $SEED"
            echo "Output: $TARGET_OUT"
            echo "Selection manifests:"
            echo "  $TARGET_OUT/selected_actives.json"
            echo "  $TARGET_OUT/selected_inactives.json"
        else
            echo "[FAIL] PharmacoMatch failed for $TARGET seed $SEED"
            echo "Check log: $TARGET_LOG/run.log"
            tail -40 "$TARGET_LOG/run.log"
        fi

        echo
    done
done

echo "Done. Result files:"
find "$OUT_ROOT" -maxdepth 4 -type f | sort