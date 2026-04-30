#!/usr/bin/env python
"""CLI for the EquiPharm pharmacophore-feature screening pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .screening import run_equipharm_screening
except ImportError:
    from screening import run_equipharm_screening


def parse_args():
    parser = argparse.ArgumentParser(description="Run EquiPharm screening for any DUD-E target.")
    parser.add_argument("--config", type=Path, help="JSON config file.")
    parser.add_argument("--target-dir", type=Path, help="Directory with crystal_ligand.mol2, actives_sdf, decoys_sdf.")
    parser.add_argument("--target-name", help="Protein target name used in metrics and plot titles.")
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
    return parser.parse_args()


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.target_dir is not None:
        config["query_ligand"] = str(args.target_dir / "crystal_ligand.mol2")
        config["actives_dir"] = str(args.target_dir / "actives_sdf")
        config["decoys_dir"] = str(args.target_dir / "decoys_sdf")

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

    required = ["checkpoint_path", "query_ligand", "actives_dir", "decoys_dir", "output_dir"]
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit(f"Missing required settings: {', '.join(missing)}")

    metrics = run_equipharm_screening(**config)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
