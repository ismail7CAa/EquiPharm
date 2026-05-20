#!/usr/bin/env python
"""CLI for the optional PharmacoMatch baseline adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .screening import run_pharmacomatch_dataset_screening, run_pharmacomatch_screening
except ImportError:
    from screening import run_pharmacomatch_dataset_screening, run_pharmacomatch_screening


def parse_args():
    parser = argparse.ArgumentParser(description="Run PharmacoMatch through a command-template adapter.")
    parser.add_argument("--config", type=Path, help="JSON config file.")
    parser.add_argument("--dataset-dir", type=Path, help="DUD-E-style dataset root containing one directory per target.")
    parser.add_argument("--target-dir", type=Path, help="Directory with crystal_ligand.mol2, actives_sdf, decoys_sdf.")
    parser.add_argument("--query-ligand", type=Path)
    parser.add_argument("--actives-dir", type=Path)
    parser.add_argument("--decoys-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-name")
    parser.add_argument("--command-template")
    parser.add_argument("--score-regex")
    parser.add_argument("--score-json-key")
    parser.add_argument("--work-dir", type=Path)
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
        "dataset_dir": args.dataset_dir,
        "query_ligand": args.query_ligand,
        "actives_dir": args.actives_dir,
        "decoys_dir": args.decoys_dir,
        "output_dir": args.output_dir,
        "target_name": args.target_name,
        "command_template": args.command_template,
        "score_regex": args.score_regex,
        "score_json_key": args.score_json_key,
        "work_dir": args.work_dir,
        "limit": args.limit,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value

    if "dataset_dir" in config:
        required = ["command_template", "dataset_dir", "output_dir"]
        missing = [key for key in required if key not in config]
        if missing:
            raise SystemExit(f"Missing required settings: {', '.join(missing)}")
        if "score_regex" not in config and "score_json_key" not in config:
            raise SystemExit("Provide score_regex or score_json_key so PharmacoMatch scores can be parsed.")
        metrics = run_pharmacomatch_dataset_screening(**config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

    required = ["command_template", "query_ligand", "actives_dir", "decoys_dir", "output_dir"]
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit(f"Missing required settings: {', '.join(missing)}")
    if "score_regex" not in config and "score_json_key" not in config:
        raise SystemExit("Provide score_regex or score_json_key so PharmacoMatch scores can be parsed.")

    metrics = run_pharmacomatch_screening(**config)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
