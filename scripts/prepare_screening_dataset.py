#!/usr/bin/env python
"""Normalize virtual-screening datasets into the EquiPharm target layout."""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path


QUERY_PATTERNS = (
    "crystal_ligand.mol2",
    "crystal_ligand.sdf",
    "query_ligand.mol2",
    "query_ligand.sdf",
    "ligand.mol2",
    "ligand.sdf",
    "*_ligand.mol2",
    "*_ligand.sdf",
)

ACTIVE_PATTERNS = (
    "active.smi",
    "actives.smi",
    "active_T.smi",
    "active_V.smi",
    "active.sdf",
    "actives.sdf",
    "actives_final.sdf",
    "active.sdf.gz",
    "actives.sdf.gz",
    "actives_final.sdf.gz",
)

DECOY_PATTERNS = (
    "inactive.smi",
    "inactives.smi",
    "inactive_T.smi",
    "inactive_V.smi",
    "decoy.smi",
    "decoys.smi",
    "*decoy*.smi",
    "inactive.sdf",
    "inactives.sdf",
    "decoys.sdf",
    "*decoy*.sdf",
    "decoys_final.sdf",
    "inactive.sdf.gz",
    "inactives.sdf.gz",
    "decoys.sdf.gz",
    "*decoy*.sdf.gz",
    "decoys_final.sdf.gz",
)


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def find_first(root: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def find_all(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    paths = []
    for pattern in patterns:
        paths.extend(sorted(root.glob(pattern)))
    seen = set()
    unique = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def discover_targets(source_dir: Path, *, query_from_first_active: bool = False) -> list[Path]:
    targets = []
    for path in sorted(p for p in source_dir.rglob("*") if p.is_dir()):
        has_query = find_first(path, QUERY_PATTERNS) is not None
        has_actives = bool(find_all(path, ACTIVE_PATTERNS))
        has_decoys = bool(find_all(path, DECOY_PATTERNS))
        if (has_query or query_from_first_active) and has_actives and has_decoys:
            targets.append(path)
    if source_dir not in targets:
        has_query = find_first(source_dir, QUERY_PATTERNS) is not None
        has_actives = bool(find_all(source_dir, ACTIVE_PATTERNS))
        has_decoys = bool(find_all(source_dir, DECOY_PATTERNS))
        if (has_query or query_from_first_active) and has_actives and has_decoys:
            targets.append(source_dir)
    return sorted(set(targets), key=lambda p: str(p.relative_to(source_dir)))


def split_sdf(source: Path, destination: Path, prefix: str, *, start_index: int = 0, limit: int | None = None) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    count = start_index
    buffer: list[str] = []
    with open_text(source) as handle:
        for line in handle:
            buffer.append(line)
            if line.strip() == "$$$$":
                count += 1
                (destination / f"{prefix}_{count:05d}.sdf").write_text("".join(buffer), encoding="utf-8")
                buffer = []
                if limit is not None and count - start_index >= limit:
                    return count

    if buffer and (limit is None or count - start_index < limit):
        count += 1
        (destination / f"{prefix}_{count:05d}.sdf").write_text("".join(buffer), encoding="utf-8")
    return count


def smiles_records(path: Path):
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            if not parts:
                continue
            if line_number == 1 and parts[0].lower() in {"smiles", "smile", "smi"}:
                continue
            name = parts[1] if len(parts) > 1 else f"mol_{line_number:05d}"
            yield parts[0], name


def write_smiles_as_sdf(
    source: Path,
    destination: Path,
    prefix: str,
    *,
    start_index: int = 0,
    limit: int | None = None,
) -> int:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RuntimeError("Preparing SMILES datasets requires RDKit.") from exc

    destination.mkdir(parents=True, exist_ok=True)
    count = start_index
    for smiles, name in smiles_records(source):
        if limit is not None and count - start_index >= limit:
            break
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = count + 1
        if AllChem.EmbedMolecule(mol, params) != 0:
            continue
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
        mol.SetProp("_Name", name)
        count += 1
        writer = Chem.SDWriter(str(destination / f"{prefix}_{count:05d}.sdf"))
        try:
            writer.write(mol)
        finally:
            writer.close()
    return count


def prepare_molecule_files(paths: list[Path], destination: Path, prefix: str, *, limit: int | None = None) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    for old_file in destination.glob("*.sdf"):
        old_file.unlink()

    count = 0
    for source in paths:
        remaining = None if limit is None else max(0, limit - count)
        if remaining == 0:
            break
        suffixes = "".join(source.suffixes).lower()
        if suffixes.endswith(".smi"):
            count = write_smiles_as_sdf(source, destination, prefix, start_index=count, limit=remaining)
        elif suffixes.endswith(".sdf") or suffixes.endswith(".sdf.gz"):
            count = split_sdf(source, destination, prefix, start_index=count, limit=remaining)
        else:
            raise ValueError(f"Unsupported molecule file format: {source}")
    return count


def copy_query_ligand(target_dir: Path, output_dir: Path) -> Path:
    query = find_first(target_dir, QUERY_PATTERNS)
    if query is None:
        raise FileNotFoundError(f"No query ligand found in {target_dir}")
    suffix = query.suffix.lower()
    if suffix not in {".mol2", ".sdf"}:
        raise ValueError(f"Unsupported query ligand format: {query}")
    destination = output_dir / f"crystal_ligand{suffix}"
    shutil.copy2(query, destination)
    return destination


def use_first_active_as_query(output_dir: Path) -> None:
    active_files = sorted((output_dir / "actives_sdf").glob("*.sdf"))
    if not active_files:
        raise FileNotFoundError(f"Cannot create query ligand because no active SDF files were written in {output_dir}")
    first_active = active_files[0]
    shutil.copy2(first_active, output_dir / "crystal_ligand.sdf")
    first_active.unlink()


def prepare_target(
    source_root: Path,
    target_dir: Path,
    output_root: Path,
    *,
    active_limit: int | None = None,
    decoy_limit: int | None = None,
    query_from_first_active: bool = False,
) -> tuple[str, int, int]:
    target_name = target_dir.name
    if target_name.lower() in {"actives", "decoys", "inactive", "inactives"}:
        target_name = target_dir.parent.name
    output_dir = output_root / target_name
    output_dir.mkdir(parents=True, exist_ok=True)

    active_files = find_all(target_dir, ACTIVE_PATTERNS)
    decoy_files = find_all(target_dir, DECOY_PATTERNS)
    if not active_files or not decoy_files:
        relative = target_dir.relative_to(source_root)
        raise FileNotFoundError(f"Missing active/inactive files for {relative}")

    active_count = prepare_molecule_files(active_files, output_dir / "actives_sdf", "active", limit=active_limit)
    decoy_count = prepare_molecule_files(decoy_files, output_dir / "decoys_sdf", "decoy", limit=decoy_limit)
    if find_first(target_dir, QUERY_PATTERNS) is not None:
        copy_query_ligand(target_dir, output_dir)
    elif query_from_first_active:
        use_first_active_as_query(output_dir)
        active_count -= 1
    else:
        raise FileNotFoundError(f"No query ligand found in {target_dir}")
    return target_name, active_count, decoy_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize a screening dataset into EquiPharm target folders.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--active-limit", type=int)
    parser.add_argument("--decoy-limit", type=int)
    parser.add_argument(
        "--query-from-first-active",
        action="store_true",
        help="Use the first generated active SDF as crystal_ligand.sdf when no query ligand is provided.",
    )
    args = parser.parse_args()

    targets = discover_targets(args.source_dir, query_from_first_active=args.query_from_first_active)
    if not targets:
        raise RuntimeError(
            f"No targets found under {args.source_dir}. Expected active/inactive SMI or SDF files"
            f"{' plus a query ligand' if not args.query_from_first_active else ''}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for target in targets:
        name, active_count, decoy_count = prepare_target(
            args.source_dir,
            target,
            args.output_dir,
            active_limit=args.active_limit,
            decoy_limit=args.decoy_limit,
            query_from_first_active=args.query_from_first_active,
        )
        print(f"{name}: {active_count} actives, {decoy_count} decoys")

    print(f"Screening dataset ready at {args.output_dir} ({len(targets)} targets).")


if __name__ == "__main__":
    main()
