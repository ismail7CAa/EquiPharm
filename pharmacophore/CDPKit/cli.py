#!/usr/bin/env python
"""CLI for the optional CDPKit pharmacophore-screening baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .screening import ensure_cdpkit_query, run_cdpkit_dataset_screening, run_cdpkit_screening
except ImportError:
    from screening import ensure_cdpkit_query, run_cdpkit_dataset_screening, run_cdpkit_screening


def parse_args():
    parser = argparse.ArgumentParser(description="Run the CDPKit psdscreen baseline.")
    parser.add_argument("--config", type=Path, help="JSON config file.")
    parser.add_argument("--dataset-dir", type=Path, help="DUD-E-style dataset root containing one directory per target.")
    parser.add_argument("--target-dir", type=Path, help="Directory with actives_sdf, decoys_sdf, and optionally query.cdf/query.pml.")
    parser.add_argument("--query-dir", type=Path, help="Optional root with one query folder per target for dataset runs.")
    parser.add_argument("--query-pharmacophore", type=Path, help="CDPKit query pharmacophore, e.g. .cdf or .pml.")
    parser.add_argument("--actives-dir", type=Path)
    parser.add_argument("--decoys-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-name")
    parser.add_argument("--psdcreate-bin", default=None)
    parser.add_argument("--psdscreen-bin", default=None)
    parser.add_argument("--query-format")
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--max-omitted", type=int, default=None)
    parser.add_argument("--skip-missing-queries", action="store_true")
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
        config["actives_dir"] = str(args.target_dir / "actives_sdf")
        config["decoys_dir"] = str(args.target_dir / "decoys_sdf")
        if "query_pharmacophore" not in config:
            config["query_pharmacophore"] = str(ensure_cdpkit_query(args.target_dir))

    overrides = {
        "dataset_dir": args.dataset_dir,
        "query_dir": args.query_dir,
        "query_pharmacophore": args.query_pharmacophore,
        "actives_dir": args.actives_dir,
        "decoys_dir": args.decoys_dir,
        "output_dir": args.output_dir,
        "target_name": args.target_name,
        "psdcreate_bin": args.psdcreate_bin,
        "psdscreen_bin": args.psdscreen_bin,
        "query_format": args.query_format,
        "num_threads": args.num_threads,
        "max_omitted": args.max_omitted,
        "skip_missing_queries": args.skip_missing_queries or None,
        "limit": args.limit,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value

    if "dataset_dir" in config:
        required = ["dataset_dir", "output_dir"]
        missing = [key for key in required if key not in config]
        if missing:
            raise SystemExit(f"Missing required settings: {', '.join(missing)}")
        metrics = run_cdpkit_dataset_screening(**config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return

    required = ["query_pharmacophore", "actives_dir", "decoys_dir", "output_dir"]
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit(f"Missing required settings: {', '.join(missing)}")
    if Path(config["query_pharmacophore"]).suffix.lower() not in {".cdf", ".pml", ".psd"}:
        raise SystemExit("CDPKit psdscreen requires query_pharmacophore to be a .cdf, .pml, or .psd file.")

    metrics = run_cdpkit_screening(**config)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
