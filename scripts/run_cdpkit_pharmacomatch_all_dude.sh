#!/usr/bin/env bash
set -u

ROOT="/data/db6/Izzy/EquiPharm"
cd "$ROOT" || exit 1

PHARM_ROOT="external/PharmacoMatch"
DATA_ROOT="$PHARM_ROOT/data/DUD-E"
OUT_ROOT="pharmacophore/results/CDPKit_PharmacoMatchAlignment"
LOG_ROOT="pharmacophore/results/CDPKit_PharmacoMatchAlignment_logs"

mkdir -p "$OUT_ROOT" "$LOG_ROOT" scripts

echo "Running CDPKit PharmacoMatch-alignment baseline on all DUD-E targets"
echo "ROOT=$ROOT"
echo "DATA_ROOT=$DATA_ROOT"
echo "OUT_ROOT=$OUT_ROOT"
echo

cat > scripts/run_pharmacomatch_cdpkit_alignment.py <<'PY'
import sys
from pathlib import Path

repo = Path("external/PharmacoMatch").resolve()
sys.path.insert(0, str(repo))

from pharmacomatch.virtual_screening.alignment import PharmacophoreAlignment

target_root = sys.argv[1]
aligner = PharmacophoreAlignment(target_root)
aligner.align_preprocessed_ligands_to_query()
print("Alignment time:", getattr(aligner, "alignment_time", None))
PY

cat > scripts/eval_pharmacomatch_cdpkit_alignment.py <<'PY'
import sys
import json
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def max_pool_by_mol(scores, mol_ids):
    pooled = {}
    for score, mol_id in zip(scores, mol_ids):
        mol_id = int(mol_id)
        score = float(score)
        if mol_id not in pooled or score > pooled[mol_id]:
            pooled[mol_id] = score
    return np.array([pooled[k] for k in sorted(pooled)], dtype=float)


def enrichment_factor(y_true, y_score, alpha=0.01):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    n = len(y_true)
    n_top = max(1, int(np.ceil(alpha * n)))

    order = np.argsort(-y_score)
    top = y_true[order[:n_top]]

    n_actives = y_true.sum()
    if n_actives == 0:
        return 0.0

    expected = n_top * (n_actives / n)
    if expected == 0:
        return 0.0

    return float(top.sum() / expected)


def bedroc_score(y_true, y_score, alpha=20.0):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    n = len(y_true)
    n_actives = int(y_true.sum())

    if n == 0 or n_actives == 0 or n_actives == n:
        return float("nan")

    order = np.argsort(-y_score)
    active_ranks = np.where(y_true[order] == 1)[0] + 1

    rie = (n / n_actives) * np.sum(np.exp(-alpha * active_ranks / n))

    ra = n_actives / n
    numerator = rie * ra * np.sinh(alpha / 2.0)
    denominator = np.cosh(alpha / 2.0) - np.cosh(alpha / 2.0 - alpha * ra)

    bedroc = numerator / denominator + 1.0 / (1.0 - np.exp(alpha * (1.0 - ra)))
    return float(bedroc)


target_root = Path(sys.argv[1])
out_file = Path(sys.argv[2])
target_name = target_root.name

act_path = target_root / "vs" / "all_actives_aligned.pt"
ina_path = target_root / "vs" / "all_inactives_aligned.pt"

if not act_path.exists():
    raise FileNotFoundError(f"Missing {act_path}")

if not ina_path.exists():
    raise FileNotFoundError(f"Missing {ina_path}")

act = torch.load(act_path)
ina = torch.load(ina_path)

act_ph4_scores = (act[:, 0] + act[:, 1]).cpu().numpy()
ina_ph4_scores = (ina[:, 0] + ina[:, 1]).cpu().numpy()

act_mol_ids = act[:, 3].cpu().numpy()
ina_mol_ids = ina[:, 3].cpu().numpy()

act_scores = max_pool_by_mol(act_ph4_scores, act_mol_ids)
ina_scores = max_pool_by_mol(ina_ph4_scores, ina_mol_ids)

y_true = np.concatenate([
    np.ones(len(act_scores), dtype=int),
    np.zeros(len(ina_scores), dtype=int),
])

y_pred = np.concatenate([act_scores, ina_scores])

metrics = {
    "target": target_name,
    "pipeline": "CDPKit_PharmacoMatchAlignment",
    "n_actives": int(len(act_scores)),
    "n_inactives": int(len(ina_scores)),
    "n_total": int(len(y_true)),
    "roc_auc": float(roc_auc_score(y_true, y_pred)),
    "pr_auc": float(average_precision_score(y_true, y_pred)),
    "bedroc_alpha20": float(bedroc_score(y_true, y_pred, alpha=20.0)),
    "ef1_percent": float(enrichment_factor(y_true, y_pred, alpha=0.01)),
}

out_file.parent.mkdir(parents=True, exist_ok=True)
out_file.write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
PY

for TARGET_ROOT in "$DATA_ROOT"/*; do
    [ -d "$TARGET_ROOT" ] || continue

    TARGET="$(basename "$TARGET_ROOT")"

    echo "============================================================"
    echo "TARGET: $TARGET"
    echo "============================================================"

    RAW="$TARGET_ROOT/raw"
    VS="$TARGET_ROOT/vs"
    TARGET_OUT="$OUT_ROOT/$TARGET"

    QUERY="$RAW/query.pml"
    ACTIVES_PSD="$RAW/actives.psd"
    INACTIVES_PSD="$RAW/inactives.psd"

    if [ ! -f "$QUERY" ]; then
        echo "[SKIP] Missing query: $QUERY"
        continue
    fi

    if [ ! -f "$ACTIVES_PSD" ]; then
        echo "[SKIP] Missing actives PSD: $ACTIVES_PSD"
        continue
    fi

    if [ ! -f "$INACTIVES_PSD" ]; then
        echo "[SKIP] Missing inactives PSD: $INACTIVES_PSD"
        continue
    fi

    mkdir -p "$VS" "$TARGET_OUT"

    echo "[1/2] Running CDPL pharmacophore alignment..."

    rm -f "$VS/all_actives_aligned.pt"
    rm -f "$VS/all_inactives_aligned.pt"

    python scripts/run_pharmacomatch_cdpkit_alignment.py "$TARGET_ROOT" \
      > "$LOG_ROOT/${TARGET}_alignment.log" 2>&1

    if [ $? -ne 0 ]; then
        echo "[FAIL] Alignment failed for $TARGET"
        echo "Check log: $LOG_ROOT/${TARGET}_alignment.log"
        tail -40 "$LOG_ROOT/${TARGET}_alignment.log"
        echo
        continue
    fi

    if [ ! -f "$VS/all_actives_aligned.pt" ] || [ ! -f "$VS/all_inactives_aligned.pt" ]; then
        echo "[FAIL] Alignment finished but .pt files are missing for $TARGET"
        ls -lh "$VS"
        echo
        continue
    fi

    echo "[2/2] Evaluating metrics..."

    python scripts/eval_pharmacomatch_cdpkit_alignment.py \
      "$TARGET_ROOT" \
      "$TARGET_OUT/metrics.json" \
      > "$LOG_ROOT/${TARGET}_eval.log" 2>&1

    if [ $? -eq 0 ]; then
        echo "[OK] Finished $TARGET"
        cat "$TARGET_OUT/metrics.json"
    else
        echo "[FAIL] Evaluation failed for $TARGET"
        echo "Check log: $LOG_ROOT/${TARGET}_eval.log"
        tail -40 "$LOG_ROOT/${TARGET}_eval.log"
    fi

    echo
done

echo "Done. Results:"
find "$OUT_ROOT" -maxdepth 2 -type f | sort
