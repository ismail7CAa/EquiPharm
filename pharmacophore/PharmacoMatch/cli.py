#!/usr/bin/env python
"""CLI for the optional PharmacoMatch baseline adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .screening import (
        prepare_official_pharmacomatch_target,
        run_official_pharmacomatch_dataset_screening,
        run_official_pharmacomatch_screening,
        run_pharmacomatch_dataset_screening,
        run_pharmacomatch_screening,
    )
except ImportError:
    from screening import (
        prepare_official_pharmacomatch_target,
        run_official_pharmacomatch_dataset_screening,
        run_official_pharmacomatch_screening,
        run_pharmacomatch_dataset_screening,
        run_pharmacomatch_screening,
    )


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
    parser.add_argument("--official-vs-dir", type=Path, help="Official PharmacoMatch preprocessed target directory.")
    parser.add_argument("--official-dataset-dir", type=Path, help="Official PharmacoMatch dataset root.")
    parser.add_argument("--prepare-target-dir", type=Path, help="DUD-E-style target to prepare and run with official PharmacoMatch.")
    parser.add_argument("--prepared-vs-dir", type=Path, help="Where to write/read prepared PharmacoMatch raw files.")
    parser.add_argument("--cdpkit-bin", type=Path, help="Directory containing CDPKit psdcreate.")
    parser.add_argument("--query-pharmacophore", type=Path, help="Existing query.pml to use instead of generating one from crystal_ligand.mol2.")
    parser.add_argument("--force-prepare", action="store_true", help="Regenerate prepared PharmacoMatch inputs.")
    parser.add_argument("--pharmacomatch-root", type=Path, default=Path("external/PharmacoMatch"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="1")
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
        "official_vs_dir": args.official_vs_dir,
        "official_dataset_dir": args.official_dataset_dir,
        "prepare_target_dir": args.prepare_target_dir,
        "prepared_vs_dir": args.prepared_vs_dir,
        "cdpkit_bin": args.cdpkit_bin,
        "query_pharmacophore": args.query_pharmacophore,
        "force_prepare": args.force_prepare,
        "pharmacomatch_root": args.pharmacomatch_root,
        "model_path": args.model_path,
        "batch_size": args.batch_size,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "limit": args.limit,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value

    if "prepare_target_dir" in config:
        required = ["prepare_target_dir", "output_dir"]
        missing = [key for key in required if key not in config]
        if missing:
            raise SystemExit(f"Missing required settings: {', '.join(missing)}")
        prepared_vs_dir = prepare_official_pharmacomatch_target(
            target_dir=config["prepare_target_dir"],
            prepared_vs_dir=config.get("prepared_vs_dir"),
            pharmacomatch_root=config.get("pharmacomatch_root", "external/PharmacoMatch"),
            cdpkit_bin=config.get("cdpkit_bin"),
            query_pharmacophore=config.get("query_pharmacophore"),
            force=bool(config.get("force_prepare", False)),
        )
        metrics = run_official_pharmacomatch_screening(
            official_vs_dir=prepared_vs_dir,
            output_dir=config["output_dir"],
            pharmacomatch_root=config.get("pharmacomatch_root", "external/PharmacoMatch"),
            model_path=config.get("model_path"),
            target_name=config.get("target_name"),
            batch_size=config.get("batch_size"),
            accelerator=config.get("accelerator", "auto"),
            devices=_parse_devices(config.get("devices", "1")),
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

    if "official_dataset_dir" in config:
        required = ["official_dataset_dir", "output_dir"]
        missing = [key for key in required if key not in config]
        if missing:
            raise SystemExit(f"Missing required settings: {', '.join(missing)}")
        metrics = run_official_pharmacomatch_dataset_screening(
            official_dataset_dir=config["official_dataset_dir"],
            output_dir=config["output_dir"],
            pharmacomatch_root=config.get("pharmacomatch_root", "external/PharmacoMatch"),
            model_path=config.get("model_path"),
            batch_size=config.get("batch_size"),
            accelerator=config.get("accelerator", "auto"),
            devices=_parse_devices(config.get("devices", "1")),
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

    if "official_vs_dir" in config:
        required = ["official_vs_dir", "output_dir"]
        missing = [key for key in required if key not in config]
        if missing:
            raise SystemExit(f"Missing required settings: {', '.join(missing)}")
        metrics = run_official_pharmacomatch_screening(
            official_vs_dir=config["official_vs_dir"],
            output_dir=config["output_dir"],
            pharmacomatch_root=config.get("pharmacomatch_root", "external/PharmacoMatch"),
            model_path=config.get("model_path"),
            target_name=config.get("target_name"),
            batch_size=config.get("batch_size"),
            accelerator=config.get("accelerator", "auto"),
            devices=_parse_devices(config.get("devices", "1")),
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

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


def _parse_devices(value):
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


if __name__ == "__main__":
    main()
