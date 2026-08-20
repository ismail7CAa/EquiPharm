#!/usr/bin/env bash
set -uo pipefail

ROOT="/data/db6/Izzy/EquiPharm"
cd "$ROOT" || exit 1

PHARM_ROOT="external/PharmacoMatch"
DATA_ROOT="$PHARM_ROOT/data/DUD-E"

N_ACTIVES="${N_ACTIVES:-50}"
N_INACTIVES="${N_INACTIVES:-500}"
SEED="${SEED:-42}"
SELECTION_MODE="${SELECTION_MODE:-random}"   # random or first

RUN_NAME="subset_${N_ACTIVES}A_${N_INACTIVES}D_${SELECTION_MODE}_seed${SEED}"
OUT_ROOT="pharmacophore/results/CDPKit_PharmacoMatchAlignment/${RUN_NAME}"
LOG_ROOT="pharmacophore/results/CDPKit_PharmacoMatchAlignment_logs/${RUN_NAME}"
HELPER_ROOT="scripts/cdpkit_pharmacomatch_subset"

mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$HELPER_ROOT"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 TARGET [TARGET ...]"
    echo "Example: $0 ADA ANDR EGFR"
    echo "Optional: SEED=7 N_ACTIVES=50 N_INACTIVES=500 SELECTION_MODE=random $0 ADA ANDR"
    exit 1
fi

if [ "$SELECTION_MODE" != "random" ] && [ "$SELECTION_MODE" != "first" ]; then
    echo "ERROR: SELECTION_MODE must be 'random' or 'first', got: $SELECTION_MODE"
    exit 1
fi

cat > "$HELPER_ROOT/align_subset.py" <<'PY'
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch

repo = Path("external/PharmacoMatch").resolve()
sys.path.insert(0, str(repo))

import CDPL.Pharm as Pharm


def stable_seed(base_seed: int, target: str, dataset_name: str) -> int:
    """Return a deterministic seed for each target/class pair."""
    key = f"{target}:{dataset_name}".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
    return (base_seed + offset) % (2**32)


class SubsetPharmacophoreAlignment:
    def __init__(
        self,
        target_root: Path,
        output_dir: Path,
        n_actives: int,
        n_inactives: int,
        seed: int,
        selection_mode: str,
    ) -> None:
        self.target_root = target_root
        self.output_dir = output_dir
        self.n_actives = n_actives
        self.n_inactives = n_inactives
        self.seed = seed
        self.selection_mode = selection_mode
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.selection_manifest = {
            "target": self.target_root.name,
            "selection_mode": selection_mode,
            "base_seed": seed,
            "datasets": {},
        }

    def run(self) -> None:
        tic = time.perf_counter()
        self._alignment("actives", self.n_actives)
        self._alignment("inactives", self.n_inactives)
        self.alignment_time = time.perf_counter() - tic

        manifest_path = self.output_dir / "selected_molecule_ids.json"
        manifest_path.write_text(
            json.dumps(self.selection_manifest, indent=2), encoding="utf-8"
        )
        print(f"Selection manifest: {manifest_path}")
        print(f"Total alignment time: {self.alignment_time:.3f} s")

    def _alignment(self, dataset_name: str, requested_molecules: int) -> None:
        ref_ph4_file = self.target_root / "raw" / "query.pml"
        in_file = self.target_root / "raw" / f"{dataset_name}.psd"
        out_file = self.output_dir / f"all_{dataset_name}_aligned.pt"

        if not ref_ph4_file.exists():
            raise FileNotFoundError(ref_ph4_file)
        if not in_file.exists():
            raise FileNotFoundError(in_file)

        ref_ph4 = self._read_ref_pharmacophore(ref_ph4_file)
        db_accessor = Pharm.PSDScreeningDBAccessor(str(in_file))
        num_ph4s = int(db_accessor.getNumPharmacophores())

        # Select compounds, not individual conformers/pharmacophores.
        all_molecule_ids = sorted(
            {int(db_accessor.getMoleculeIndex(i)) for i in range(num_ph4s)}
        )

        if requested_molecules > len(all_molecule_ids):
            raise ValueError(
                f"Requested {requested_molecules} {dataset_name}, but only "
                f"{len(all_molecule_ids)} unique molecule IDs exist in {in_file}"
            )

        class_seed = stable_seed(
            self.seed, self.target_root.name, dataset_name
        )

        if self.selection_mode == "first":
            selected_ids = all_molecule_ids[:requested_molecules]
        else:
            rng = random.Random(class_seed)
            selected_ids = sorted(
                rng.sample(all_molecule_ids, requested_molecules)
            )

        selected_set = set(selected_ids)

        mol_ph4 = Pharm.BasicPharmacophore()
        alignment = Pharm.PharmacophoreAlignment(True)
        self._clear_feature_orientations(ref_ph4)
        alignment.addFeatures(ref_ph4, True)
        alignment.performExhaustiveSearch(False)

        fit_score = Pharm.PharmacophoreFitScore(
            match_cnt_weight=1.0,
            pos_match_weight=0.9,
            geom_match_weight=0.0,
        )

        alignment_scores = []
        scored_molecule_ids = set()
        selected_pharmacophore_count = 0

        for i in range(num_ph4s):
            mol_idx = int(db_accessor.getMoleculeIndex(i))
            if mol_idx not in selected_set:
                continue

            db_accessor.getPharmacophore(i, mol_ph4)
            conf_idx = int(db_accessor.getConformationIndex(i))
            selected_pharmacophore_count += 1

            if mol_ph4.getNumFeatures() == 0:
                continue

            self._clear_feature_orientations(mol_ph4)
            alignment.clearEntities(False)
            alignment.addFeatures(mol_ph4, False)

            solutions = []
            while alignment.nextAlignment():
                score = float(
                    fit_score(ref_ph4, mol_ph4, alignment.getTransform())
                )
                solutions.append(score)

            if solutions:
                solution = max(solutions)
                row = [
                    int(solution),
                    solution % 1,
                    mol_ph4.getNumFeatures(),
                    mol_idx,
                    conf_idx,
                ]
            else:
                row = [
                    0,
                    0.0,
                    mol_ph4.getNumFeatures(),
                    mol_idx,
                    conf_idx,
                ]

            alignment_scores.append(row)
            scored_molecule_ids.add(mol_idx)

        if not alignment_scores:
            raise RuntimeError(
                f"No alignment rows were produced for {dataset_name}"
            )

        scores_tensor = torch.tensor(alignment_scores, dtype=torch.float32)
        torch.save(scores_tensor, out_file)

        missing_ids = sorted(selected_set - scored_molecule_ids)
        self.selection_manifest["datasets"][dataset_name] = {
            "input_psd": str(in_file),
            "available_unique_molecules": len(all_molecule_ids),
            "requested_molecules": requested_molecules,
            "class_seed": class_seed,
            "selected_molecule_ids": selected_ids,
            "selected_pharmacophores": selected_pharmacophore_count,
            "scored_unique_molecules": len(scored_molecule_ids),
            "missing_after_alignment": missing_ids,
            "output_tensor": str(out_file),
        }

        print(
            f"{dataset_name}: selected {len(selected_ids)} molecules, "
            f"processed {selected_pharmacophore_count} pharmacophores, "
            f"scored {len(scored_molecule_ids)} unique molecules, "
            f"saved {scores_tensor.shape[0]} rows to {out_file}"
        )
        if missing_ids:
            print(
                f"WARNING: {len(missing_ids)} selected {dataset_name} molecules "
                "had no scored pharmacophore rows. See the manifest."
            )

    @staticmethod
    def _read_ref_pharmacophore(filename: Path):
        reader = Pharm.PharmacophoreReader(str(filename))
        ph4 = Pharm.BasicPharmacophore()
        if not reader.read(ph4):
            raise RuntimeError(
                f"Reading reference pharmacophore failed: {filename}"
            )
        return ph4

    @staticmethod
    def _clear_feature_orientations(ph4) -> None:
        for feature in ph4:
            Pharm.clearOrientation(feature)
            Pharm.setGeometry(feature, Pharm.FeatureGeometry.SPHERE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-actives", type=int, default=50)
    parser.add_argument("--n-inactives", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection-mode",
        choices=("random", "first"),
        default="random",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = SubsetPharmacophoreAlignment(
        target_root=args.target_root.resolve(),
        output_dir=args.output_dir.resolve(),
        n_actives=args.n_actives,
        n_inactives=args.n_inactives,
        seed=args.seed,
        selection_mode=args.selection_mode,
    )
    runner.run()
PY

cat > "$HELPER_ROOT/evaluate_subset.py" <<'PY'
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def load_tensor(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def max_pool_by_molecule(scores, molecule_ids):
    pooled = {}
    for score, molecule_id in zip(scores, molecule_ids):
        molecule_id = int(molecule_id)
        score = float(score)
        if molecule_id not in pooled or score > pooled[molecule_id]:
            pooled[molecule_id] = score
    return np.asarray([pooled[key] for key in sorted(pooled)], dtype=float)


def enrichment_factor(y_true, y_score, alpha=0.01):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    n_top = max(1, int(np.ceil(alpha * n)))
    order = np.argsort(-y_score)
    n_actives = y_true.sum()
    if n_actives == 0:
        return 0.0
    expected = n_top * (n_actives / n)
    return 0.0 if expected == 0 else float(y_true[order[:n_top]].sum() / expected)


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
    active_ratio = n_actives / n
    numerator = rie * active_ratio * np.sinh(alpha / 2.0)
    denominator = np.cosh(alpha / 2.0) - np.cosh(
        alpha / 2.0 - alpha * active_ratio
    )
    return float(
        numerator / denominator
        + 1.0 / (1.0 - np.exp(alpha * (1.0 - active_ratio)))
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--alignment-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    act = load_tensor(args.alignment_dir / "all_actives_aligned.pt")
    ina = load_tensor(args.alignment_dir / "all_inactives_aligned.pt")

    act_scores = max_pool_by_molecule(
        (act[:, 0] + act[:, 1]).numpy(), act[:, 3].numpy()
    )
    ina_scores = max_pool_by_molecule(
        (ina[:, 0] + ina[:, 1]).numpy(), ina[:, 3].numpy()
    )

    y_true = np.concatenate(
        [
            np.ones(len(act_scores), dtype=int),
            np.zeros(len(ina_scores), dtype=int),
        ]
    )
    y_pred = np.concatenate([act_scores, ina_scores])

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    requested_actives = manifest["datasets"]["actives"]["requested_molecules"]
    requested_inactives = manifest["datasets"]["inactives"]["requested_molecules"]

    metrics = {
        "target": args.target,
        "pipeline": "CDPKit_PharmacoMatchAlignment_subset",
        "selection_mode": manifest["selection_mode"],
        "seed": manifest["base_seed"],
        "requested_actives": int(requested_actives),
        "requested_inactives": int(requested_inactives),
        "n_actives": int(len(act_scores)),
        "n_inactives": int(len(ina_scores)),
        "n_total": int(len(y_true)),
        "roc_auc": float(roc_auc_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, y_pred)),
        "bedroc_alpha20": float(bedroc_score(y_true, y_pred, alpha=20.0)),
        "ef1_percent": float(enrichment_factor(y_true, y_pred, alpha=0.01)),
        "ef5_percent": float(enrichment_factor(y_true, y_pred, alpha=0.05)),
        "ef10_percent": float(enrichment_factor(y_true, y_pred, alpha=0.10)),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    if len(act_scores) != requested_actives or len(ina_scores) != requested_inactives:
        print(
            "WARNING: evaluated molecule counts differ from requested counts. "
            "Inspect selected_molecule_ids.json for missing molecules."
        )
PY

resolve_target_root() {
    local requested="$1"
    local candidate

    for candidate in \
        "$DATA_ROOT/$requested" \
        "$DATA_ROOT/${requested^^}" \
        "$DATA_ROOT/${requested,,}"; do
        if [ -d "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

echo "Running CDPKit/PharmacoMatch subset alignment"
echo "Targets: $*"
echo "Actives per target: $N_ACTIVES"
echo "Inactives/decoys per target: $N_INACTIVES"
echo "Selection: $SELECTION_MODE"
echo "Seed: $SEED"
echo "Output: $OUT_ROOT"
echo

for REQUESTED_TARGET in "$@"; do
    if ! TARGET_ROOT="$(resolve_target_root "$REQUESTED_TARGET")"; then
        echo "[SKIP] Target directory not found for: $REQUESTED_TARGET"
        continue
    fi

    TARGET="$(basename "$TARGET_ROOT")"
    TARGET_OUT="$OUT_ROOT/$TARGET"
    ALIGNMENT_OUT="$TARGET_OUT/alignment"
    TARGET_LOG="$LOG_ROOT/$TARGET"

    mkdir -p "$ALIGNMENT_OUT" "$TARGET_LOG"
    rm -f \
        "$ALIGNMENT_OUT/all_actives_aligned.pt" \
        "$ALIGNMENT_OUT/all_inactives_aligned.pt" \
        "$ALIGNMENT_OUT/selected_molecule_ids.json" \
        "$TARGET_OUT/metrics.json"

    echo "============================================================"
    echo "TARGET: $TARGET"
    echo "============================================================"
    echo "[1/2] Selecting molecules and running CDPL alignment..."

    python "$HELPER_ROOT/align_subset.py" \
        --target-root "$TARGET_ROOT" \
        --output-dir "$ALIGNMENT_OUT" \
        --n-actives "$N_ACTIVES" \
        --n-inactives "$N_INACTIVES" \
        --seed "$SEED" \
        --selection-mode "$SELECTION_MODE" \
        > "$TARGET_LOG/alignment.log" 2>&1

    if [ $? -ne 0 ]; then
        echo "[FAIL] Alignment failed for $TARGET"
        echo "Log: $TARGET_LOG/alignment.log"
        tail -60 "$TARGET_LOG/alignment.log"
        echo
        continue
    fi

    echo "[2/2] Evaluating molecule-level metrics..."

    python "$HELPER_ROOT/evaluate_subset.py" \
        --target "$TARGET" \
        --alignment-dir "$ALIGNMENT_OUT" \
        --manifest "$ALIGNMENT_OUT/selected_molecule_ids.json" \
        --output "$TARGET_OUT/metrics.json" \
        > "$TARGET_LOG/evaluation.log" 2>&1

    if [ $? -eq 0 ]; then
        echo "[OK] Finished $TARGET"
        cat "$TARGET_OUT/metrics.json"
    else
        echo "[FAIL] Evaluation failed for $TARGET"
        echo "Log: $TARGET_LOG/evaluation.log"
        tail -60 "$TARGET_LOG/evaluation.log"
    fi

    echo
 done

echo "Done. Result files:"
find "$OUT_ROOT" -maxdepth 4 -type f | sort