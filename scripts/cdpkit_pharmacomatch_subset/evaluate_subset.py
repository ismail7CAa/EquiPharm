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
