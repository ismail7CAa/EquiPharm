#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 METHOD TARGET [extra EquiPharm CLI arguments]" >&2
  exit 2
fi

METHOD="$1"
TARGET="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/DUD-E}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/pharmacophore/results}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$REPO_ROOT/models_checkpt/checkpoint_02-05-26/best_model.pt}"
DEVICE="${DEVICE:-cuda}"
TARGET_DIR="$DATA_ROOT/$TARGET"
RUN_ROOT="$OUTPUT_ROOT/$METHOD/${TARGET}_50a_500d_3seeds"
SUBSET_ROOT="$REPO_ROOT/.screening_subsets/$METHOD/$TARGET"
SEEDS=(1 2 3)

if [[ ! -f "$TARGET_DIR/crystal_ligand.mol2" ]]; then
  echo "Missing query ligand: $TARGET_DIR/crystal_ligand.mol2" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Missing checkpoint: $CHECKPOINT_PATH" >&2
  exit 1
fi

cd "$REPO_ROOT"
for SEED in "${SEEDS[@]}"; do
  SUBSET_DIR="$SUBSET_ROOT/seed_$SEED"
  SEED_OUTPUT="$RUN_ROOT/seed_$SEED"

  python "$SCRIPT_DIR/equipharm_seed_tools.py" prepare \
    --source "$TARGET_DIR" \
    --destination "$SUBSET_DIR" \
    --seed "$SEED"

  python -m "pharmacophore.${METHOD}.cli" \
    --target-name "$TARGET" \
    --query-ligand "$TARGET_DIR/crystal_ligand.mol2" \
    --actives-dir "$SUBSET_DIR/actives_sdf" \
    --decoys-dir "$SUBSET_DIR/decoys_sdf" \
    --checkpoint "$CHECKPOINT_PATH" \
    --output-dir "$SEED_OUTPUT" \
    --device "$DEVICE" \
    "$@"
done

python "$SCRIPT_DIR/equipharm_seed_tools.py" aggregate \
  --output-root "$RUN_ROOT" \
  --seeds "${SEEDS[@]}"

echo "Three-seed summary: $RUN_ROOT/three_seed_summary.csv"
