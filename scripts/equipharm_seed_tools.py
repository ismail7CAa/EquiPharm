#!/usr/bin/env python3
"""Prepare deterministic screening subsets and aggregate EquiPharm seed metrics."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import shutil
import statistics
from pathlib import Path


METRICS = ("ef1_percent", "ef5_percent", "ef10_percent", "bedroc_alpha20", "roc_auc")


def open_sdf(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def sample_combined_sdf(path: Path, count: int, rng: random.Random) -> tuple[list[str], int]:
    """Reservoir-sample molecule records without loading a full DUD-E file."""
    selected: list[str] = []
    record: list[str] = []
    seen = 0
    with open_sdf(path) as handle:
        for line in handle:
            record.append(line)
            if line.strip() != "$$$$":
                continue
            seen += 1
            molecule = "".join(record)
            if len(selected) < count:
                selected.append(molecule)
            else:
                replacement = rng.randrange(seen)
                if replacement < count:
                    selected[replacement] = molecule
            record = []
    if record and any(line.strip() for line in record):
        seen += 1
        molecule = "".join(record)
        if not molecule.rstrip().endswith("$$$$"):
            molecule += "\n$$$$\n"
        if len(selected) < count:
            selected.append(molecule)
        else:
            replacement = rng.randrange(seen)
            if replacement < count:
                selected[replacement] = molecule
    return selected, seen


def find_combined_sdf(source: Path, stems: tuple[str, ...]) -> Path | None:
    for stem in stems:
        for suffix in (".sdf", ".sdf.gz"):
            candidate = source / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def prepare_subset(source: Path, destination: Path, seed: int) -> None:
    classes = (
        ("actives_sdf", "active", 50, ("actives_final", "actives", "active")),
        ("decoys_sdf", "decoy", 500, ("decoys_final", "decoys", "inactive", "inactives")),
    )
    selections: dict[str, tuple[str, list[Path] | list[str], str]] = {}
    rng = random.Random(seed)

    for directory, prefix, count, combined_stems in classes:
        files = sorted((source / directory).glob("*.sdf"))
        if len(files) >= count:
            selections[directory] = ("links", rng.sample(files, count), str(source / directory))
            continue

        combined = find_combined_sdf(source, combined_stems)
        if combined is None:
            raise SystemExit(
                f"{source / directory} contains {len(files)} SDF files; "
                f"{count} are required, and no combined {'/'.join(combined_stems)}.sdf[.gz] file was found."
            )
        records, available = sample_combined_sdf(combined, count, rng)
        if available < count:
            raise SystemExit(f"{combined} contains {available} molecules; {count} are required.")
        selections[directory] = ("records", records, str(combined))

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    manifest_entries: dict[str, list[str]] = {}
    for directory, prefix, _, _ in classes:
        mode, molecules, selection_source = selections[directory]
        output = destination / directory
        output.mkdir()
        names = []
        if mode == "links":
            for source_file in molecules:
                (output / source_file.name).symlink_to(source_file.resolve())
                names.append(source_file.name)
        else:
            for index, record in enumerate(molecules, start=1):
                name = f"{prefix}_{index:05d}.sdf"
                (output / name).write_text(record, encoding="utf-8")
                names.append(name)
        manifest_entries[directory] = names
        manifest_entries[f"{directory}_source"] = selection_source

    manifest = {
        "seed": seed,
        "source": str(source.resolve()),
        "n_actives": len(manifest_entries["actives_sdf"]),
        "n_decoys": len(manifest_entries["decoys_sdf"]),
        "actives": manifest_entries["actives_sdf"],
        "decoys": manifest_entries["decoys_sdf"],
        "actives_source": manifest_entries["actives_sdf_source"],
        "decoys_source": manifest_entries["decoys_sdf_source"],
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
