#!/usr/bin/env python3
"""Prepare deterministic screening subsets and aggregate EquiPharm seed metrics."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import statistics
from pathlib import Path


METRICS = ("ef1_percent", "ef5_percent", "ef10_percent", "bedroc_alpha20", "roc_auc")


def prepare_subset(source: Path, destination: Path, seed: int) -> None:
    classes = (("actives_sdf", 50), ("decoys_sdf", 500))
    selections: dict[str, list[Path]] = {}
    rng = random.Random(seed)

    for directory, count in classes:
        files = sorted((source / directory).glob("*.sdf"))
        if len(files) < count:
            raise SystemExit(
                f"{source / directory} contains {len(files)} SDF files; "
                f"{count} are required."
            )
        selections[directory] = rng.sample(files, count)

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    for directory, files in selections.items():
        output = destination / directory
        output.mkdir()
        for source_file in files:
            (output / source_file.name).symlink_to(source_file.resolve())

    manifest = {
        "seed": seed,
        "source": str(source.resolve()),
        "n_actives": len(selections["actives_sdf"]),
        "n_decoys": len(selections["decoys_sdf"]),
        "actives": [file.name for file in selections["actives_sdf"]],
        "decoys": [file.name for file in selections["decoys_sdf"]],
    }
    (destination / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def aggregate(output_root: Path, seeds: list[int]) -> None:
    seed_rows = []
    for seed in seeds:
        metrics_path = output_root / f"seed_{seed}" / "metrics.json"
        if not metrics_path.is_file():
            raise SystemExit(f"Missing seed metrics: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        missing = [name for name in METRICS if name not in metrics]
        if missing:
            raise SystemExit(f"{metrics_path} is missing metrics: {', '.join(missing)}")
        seed_rows.append({"seed": seed, **{name: float(metrics[name]) for name in METRICS}})

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "three_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("seed", *METRICS))
        writer.writeheader()
        writer.writerows(seed_rows)

    summary = {}
    for name in METRICS:
        values = [row[name] for row in seed_rows]
        summary[name] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    summary.update(seeds=seeds, number_of_seeds=len(seeds))

    (output_root / "three_seed_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_root / "three_seed_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "mean", "std"))
        writer.writeheader()
        for name in METRICS:
            writer.writerow({"metric": name, **summary[name]})


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--destination", type=Path, required=True)
    prepare.add_argument("--seed", type=int, required=True)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--output-root", type=Path, required=True)
    aggregate_parser.add_argument("--seeds", type=int, nargs="+", required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare_subset(args.source, args.destination, args.seed)
    else:
        aggregate(args.output_root, args.seeds)


if __name__ == "__main__":
    main()
