#!/usr/bin/env python
"""Prepare downloaded DUD-E targets for the EquiPharm screening CLI."""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def split_sdf(source: Path, destination: Path, prefix: str) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    for old_file in destination.glob("*.sdf"):
        old_file.unlink()

    count = 0
    buffer: list[str] = []
    with open_text(source) as handle:
        for line in handle:
            buffer.append(line)
            if line.strip() == "$$$$":
                count += 1
                output = destination / f"{prefix}_{count:05d}.sdf"
                output.write_text("".join(buffer), encoding="utf-8")
                buffer = []

    if buffer:
        count += 1
        output = destination / f"{prefix}_{count:05d}.sdf"
        output.write_text("".join(buffer), encoding="utf-8")

    return count


def find_targets(source: Path) -> list[Path]:
    targets = [path.parent for path in source.rglob("crystal_ligand.mol2")]
    return sorted(set(targets), key=lambda path: path.name)


def find_sdf(target_dir: Path, stem: str) -> Path | None:
    candidates = [
        target_dir / f"{stem}.sdf.gz",
        target_dir / f"{stem}.sdf",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def prepare_target(target_dir: Path, output_root: Path) -> tuple[str, int, int]:
    target_name = target_dir.name
    output_dir = output_root / target_name
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(target_dir / "crystal_ligand.mol2", output_dir / "crystal_ligand.mol2")

    actives = find_sdf(target_dir, "actives_final")
    decoys = find_sdf(target_dir, "decoys_final")
    if actives is None or decoys is None:
        raise FileNotFoundError(f"Missing actives_final/decoys_final SDF for {target_name}")

    active_count = split_sdf(actives, output_dir / "actives_sdf", "active")
    decoy_count = split_sdf(decoys, output_dir / "decoys_sdf", "decoy")
    return target_name, active_count, decoy_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DUD-E target folders for EquiPharm.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/DUD-E"))
    args = parser.parse_args()

    targets = find_targets(args.source_dir)
    if not targets:
        raise RuntimeError(f"No DUD-E targets found under {args.source_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = 0
    for target in targets:
        name, active_count, decoy_count = prepare_target(target, args.output_dir)
        prepared += 1
        print(f"{name}: {active_count} actives, {decoy_count} decoys")

    print(f"DUD-E ready at {args.output_dir} ({prepared} targets).")


if __name__ == "__main__":
    main()
