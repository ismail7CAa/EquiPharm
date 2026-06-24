#!/usr/bin/env python
"""CLI for EquiPharm Hungarian v4 tiered scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .screening import run_equipharm_hungarian_v4_screening
except ImportError:
    from screening import run_equipharm_hungarian_v4_screening


def parse_args():
    parser = argparse.ArgumentParser(description="Run EquiPharm Hungarian v4 for any target.")
    parser.add_argument("--config", type=Path, help="JSON config file.")
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument("--target-name")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--query-ligand", type=Path)
    parser.add_argument("--actives-dir", type=Path)
    parser.add_argument("--decoys-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--distance-sigma", type=float)
    parser.add_argument("--geometry-penalty-weight", type=float)
    parser.add_argument("--allow-family-mismatch", action="store_true")
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--maxiter", type=int)
    parser.add_argument("--popsize", type=int)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {}
    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

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
        "distance_sigma": args.distance_sigma,
        "geometry_penalty_weight": args.geometry_penalty_weight,
        "maxiter": args.maxiter,
        "popsize": args.popsize,
        "limit": args.limit,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    if args.allow_family_mismatch:
        config["enforce_feature_family"] = False
    if args.no_optimize:
        config["optimize"] = False

    required = ["checkpoint_path", "query_ligand", "actives_dir", "decoys_dir", "output_dir"]
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit(f"Missing required settings: {', '.join(missing)}")

    metrics = run_equipharm_hungarian_v4_screening(**config)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
