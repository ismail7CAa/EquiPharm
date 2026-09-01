"""Shared SPICE model configuration and CLI helpers for screening variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pharmacophore.core.matching_screening import screen_actives_decoys_matching
from pharmacophore.core.screening import screen_actives_decoys


SPICE_MODEL = {
    "model_module": "pharm_training.equiformer_encoder_pharmaco_feat",
    "model_class": "SPICEPharmacophoreEncoder",
}


def run_pooled(**kwargs):
    # These isolated pipelines must never silently fall back to a QM9 model,
    # even when an old JSON configuration contains model overrides.
    kwargs.update(SPICE_MODEL)
    kwargs.setdefault("pipeline_name", "EquiPharm_SPICE")
    kwargs.setdefault("use_pharmacophore_features", True)
    kwargs.setdefault("rotatable_only", False)
    kwargs.setdefault("heavy_only", True)
    kwargs.setdefault("exclude_rings", True)
    kwargs.setdefault("one_per_bond", False)
    kwargs.setdefault("write_named_roc_curve", True)
    return screen_actives_decoys(**kwargs)


def run_matching(pipeline_name: str, matching_method: str, matching_score_mode: str, **kwargs):
    kwargs.update(SPICE_MODEL)
    kwargs.setdefault("pipeline_name", f"{pipeline_name}_SPICE")
    kwargs.setdefault("matching_method", matching_method)
    kwargs.setdefault("matching_score_mode", matching_score_mode)
    kwargs.setdefault("rotatable_only", False)
    kwargs.setdefault("heavy_only", True)
    kwargs.setdefault("exclude_rings", True)
    kwargs.setdefault("one_per_bond", False)
    return screen_actives_decoys_matching(**kwargs)


def run_cli(runner, description: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument("--target-name")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--query-ligand", type=Path)
    parser.add_argument("--actives-dir", type=Path)
    parser.add_argument("--decoys-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--maxiter", type=int)
    parser.add_argument("--popsize", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = {}
    if args.config is not None:
        config = json.loads(args.config.read_text())
    if args.target_dir is not None:
        config.update(
            query_ligand=str(args.target_dir / "crystal_ligand.mol2"),
            actives_dir=str(args.target_dir / "actives_sdf"),
            decoys_dir=str(args.target_dir / "decoys_sdf"),
        )
    overrides = {
        "checkpoint_path": args.checkpoint,
        "query_ligand": args.query_ligand,
        "actives_dir": args.actives_dir,
        "decoys_dir": args.decoys_dir,
        "output_dir": args.output_dir,
        "target_name": args.target_name,
        "device": args.device,
        "maxiter": args.maxiter,
        "popsize": args.popsize,
        "limit": args.limit,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    if args.no_optimize:
        config["optimize"] = False
    required = ("checkpoint_path", "query_ligand", "actives_dir", "decoys_dir", "output_dir")
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit(f"Missing required settings: {', '.join(missing)}")
    print(json.dumps(runner(**config), indent=2, sort_keys=True))
