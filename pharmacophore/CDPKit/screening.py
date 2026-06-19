"""CDPKit psdcreate/psdscreen baseline wrapper."""

from __future__ import annotations

from pathlib import Path

try:
    from ..core.external_baselines import (
        build_rows_from_scores,
        collect_labeled_sdf_files,
        discover_dude_targets,
        first_sdf_mol,
        find_cdpkit_query,
        find_query_ligand,
        infer_target_name,
        run_command,
        write_baseline_outputs,
        write_dataset_summary,
    )
except ImportError:
    from pharmacophore.core.external_baselines import (
        build_rows_from_scores,
        collect_labeled_sdf_files,
        discover_dude_targets,
        first_sdf_mol,
        find_cdpkit_query,
        find_query_ligand,
        infer_target_name,
        run_command,
        write_baseline_outputs,
        write_dataset_summary,
    )


DEFAULT_SCORE_PROPERTIES = (
    "score",
    "Score",
    "PHARM_SCORE",
    "PHARMACOPHORE_SCORE",
    "CDPL_SCREENING_SCORE",
)


def run_cdpkit_screening(
    *,
    query_pharmacophore: str | Path,
    actives_dir: str | Path,
    decoys_dir: str | Path,
    output_dir: str | Path,
    target_name: str | None = None,
    pipeline_name: str = "CDPKit",
    psdcreate_bin: str = "psdcreate",
    psdscreen_bin: str = "psdscreen",
    query_format: str | None = None,
    num_threads: int | None = None,
    max_omitted: int | None = None,
    hit_score: float = 1.0,
    miss_score: float = 0.0,
    score_properties: list[str] | tuple[str, ...] = DEFAULT_SCORE_PROPERTIES,
    limit: int | None = None,
) -> dict:
    """Run CDPKit screening against DUD-E-style active/decoy directories.

    The external CDPKit executables are invoked only when this function is called.
    """
    query_path = Path(query_pharmacophore)
    if query_path.suffix.lower() not in {".cdf", ".pml", ".psd"}:
        raise ValueError("CDPKit psdscreen requires query_pharmacophore to be a .cdf, .pml, or .psd file.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target_name = target_name or infer_target_name(query_pharmacophore, actives_dir, decoys_dir, output_dir)

    candidates = collect_labeled_sdf_files(actives_dir, decoys_dir, limit=limit)
    combined_sdf = output_path / "cdpkit_input.sdf"
    database_path = output_path / "cdpkit_screening_database.psd"
    hits_sdf = output_path / "cdpkit_hits.sdf"
    report_path = output_path / "cdpkit_report.txt"

    write_combined_sdf(candidates, combined_sdf)

    create_cmd = [
        psdcreate_bin,
        "-i",
        str(combined_sdf),
        "-o",
        str(database_path),
    ]
    if num_threads is not None:
        create_cmd.extend(["-t", str(num_threads)])
    run_command(create_cmd)

    screen_cmd = [
        psdscreen_bin,
        "-d",
        str(database_path),
        "-q",
        str(query_path),
        "-o",
        str(hits_sdf),
        "-r",
        str(report_path),
        "-O",
        "sdf",
        "-S",
        "true",
        "-I",
        "true",
    ]
    if query_format is not None:
        screen_cmd.extend(["-Q", query_format])
    if max_omitted is not None:
        screen_cmd.extend(["-M", str(max_omitted)])
    if num_threads is not None:
        screen_cmd.extend(["-t", str(num_threads)])
    run_command(screen_cmd)

    scores = parse_hit_scores(hits_sdf, hit_score=hit_score, score_properties=score_properties)
    
    # Build score dictionary for every molecule.
    # CDPKit hits keep their CDPKit score.
    # Non-hits get miss_score.
    # A tiny deterministic jitter breaks ties without using active/decoy labels
    # and without changing the meaningful CDPKit ranking.
    scored = {}
    for sdf_path, _label in candidates:
        name = sdf_path.stem
        base_score = scores.get(name, miss_score)
        scored[name] = float(base_score) + deterministic_jitter(name)
    
    rows = build_rows_from_scores(
        candidates,
        scored,
        pipeline_name=pipeline_name,
        target_name=target_name,
        default_score=miss_score,
    )
    return write_baseline_outputs(output_dir, rows, pipeline_name=pipeline_name, target_name=target_name)


def run_cdpkit_dataset_screening(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    query_dir: str | Path | None = None,
    pipeline_name: str = "CDPKit",
    psdcreate_bin: str = "psdcreate",
    psdscreen_bin: str = "psdscreen",
    query_format: str | None = None,
    num_threads: int | None = None,
    max_omitted: int | None = None,
    hit_score: float = 1.0,
    miss_score: float = 0.0,
    score_properties: list[str] | tuple[str, ...] = DEFAULT_SCORE_PROPERTIES,
    limit: int | None = None,
    skip_missing_queries: bool = False,
) -> dict:
    """Run CDPKit on every DUD-E-style target and write one summary CSV."""
    output_path = Path(output_dir)
    metrics_rows = []

    for target_dir in discover_dude_targets(dataset_dir):
        target_name = target_dir.name
        query_root = Path(query_dir) / target_name if query_dir else target_dir
        query_path = find_cdpkit_query(query_root)
        if query_path is None and query_dir is None:
            try:
                query_path = ensure_cdpkit_query(target_dir)
            except Exception:
                query_path = None
        if query_path is None:
            if skip_missing_queries:
                metrics_rows.append(
                    {
                        "pipeline": pipeline_name,
                        "target": target_name,
                        "status": "skipped",
                        "error": "missing CDPKit query pharmacophore",
                    }
                )
                continue
            raise FileNotFoundError(
                f"No CDPKit query pharmacophore found for {target_name}. "
                "Expected query.cdf, query.pml, query.psd, crystal_ligand.cdf, or crystal_ligand.pml."
            )

        try:
            metrics = run_cdpkit_screening(
                query_pharmacophore=query_path,
                actives_dir=target_dir / "actives_sdf",
                decoys_dir=target_dir / "decoys_sdf",
                output_dir=output_path / target_name,
                target_name=target_name,
                pipeline_name=pipeline_name,
                psdcreate_bin=psdcreate_bin,
                psdscreen_bin=psdscreen_bin,
                query_format=query_format,
                num_threads=num_threads,
                max_omitted=max_omitted,
                hit_score=hit_score,
                miss_score=miss_score,
                score_properties=score_properties,
                limit=limit,
            )
            metrics["status"] = "ok"
        except Exception as exc:
            metrics = {
                "pipeline": pipeline_name,
                "target": target_name,
                "status": "failed",
                "error": str(exc),
            }
        metrics_rows.append(metrics)

    summary_path = write_dataset_summary(output_path, metrics_rows)
    return {
        "pipeline": pipeline_name,
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_path),
        "summary_csv": str(summary_path),
        "n_targets": len(metrics_rows),
        "n_ok": sum(row.get("status") == "ok" for row in metrics_rows),
        "n_failed": sum(row.get("status") == "failed" for row in metrics_rows),
        "n_skipped": sum(row.get("status") == "skipped" for row in metrics_rows),
    }


def ensure_cdpkit_query(target_dir: str | Path, *, query_pharmacophore: str | Path | None = None) -> Path:
    """Return an existing CDPKit query or generate query.pml from the target ligand."""
    if query_pharmacophore is not None:
        return Path(query_pharmacophore)

    target_path = Path(target_dir)
    query_path = find_cdpkit_query(target_path)
    if query_path is not None:
        return query_path

    ligand_path = find_query_ligand(target_path)
    print("[CDPKIT QUERY LIGAND]", ligand_path)
    if ligand_path is None:
        raise FileNotFoundError(
            f"No CDPKit query pharmacophore or query ligand found in {target_path}."
        )

    generated_query = target_path / "query.pml"
    print("[CDPKIT GENERATED QUERY]", generated_query)
    if not generated_query.exists():
        create_ligand_query_pharmacophore(ligand_path, generated_query)
    return generated_query


def create_ligand_query_pharmacophore(ligand_path: str | Path, output_path: str | Path) -> Path:
    """Generate a ligand pharmacophore query with Python CDPL."""
    try:
        import CDPL.Chem as Chem
        import CDPL.Pharm as Pharm
    except ImportError as exc:
        raise RuntimeError("Generating CDPKit query.pml requires Python CDPL. Install it with `pip install CDPKit`.") from exc

    ligand_path = Path(ligand_path)
    output_path = Path(output_path)
    reader = cdpl_molecule_reader(str(ligand_path), Chem)
    Chem.setMultiConfImportParameter(reader, False)
    writer = cdpl_pharmacophore_writer(str(output_path), Pharm)
    molecule = Chem.BasicMolecule()
    pharmacophore = Pharm.BasicPharmacophore()
    generator = Pharm.DefaultPharmacophoreGenerator()

    if not reader.read(molecule):
        raise ValueError(f"No readable molecule found in {ligand_path}")
    Pharm.prepareForPharmacophoreGeneration(molecule)
    generator.generate(molecule, pharmacophore)
    name = Chem.getName(molecule).strip() or ligand_path.stem
    Pharm.setName(pharmacophore, name)
    if pharmacophore.getNumFeatures() == 0:
        raise ValueError(f"Generated empty pharmacophore from {ligand_path}")
    if not writer.write(pharmacophore):
        raise RuntimeError(f"Could not write generated query pharmacophore to {output_path}")
    return output_path


def cdpl_molecule_reader(filename: str, Chem):
    suffix = Path(filename).suffix.lower().lstrip(".")
    handler = Chem.MoleculeIOManager.getInputHandlerByFileExtension(suffix)
    if not handler:
        raise ValueError(f"Unsupported molecule input format: {filename}")
    return handler.createReader(filename)


def cdpl_pharmacophore_writer(filename: str, Pharm):
    suffix = Path(filename).suffix.lower().lstrip(".")
    handler = Pharm.FeatureContainerIOManager.getOutputHandlerByFileExtension(suffix)
    if not handler:
        raise ValueError(f"Unsupported pharmacophore output format: {filename}")
    return handler.createWriter(filename)


def write_combined_sdf(candidates: list[tuple[Path, int]], output_path: Path) -> None:
    from rdkit import Chem

    writer = Chem.SDWriter(str(output_path))
    try:
        for sdf_path, label in candidates:
            mol = first_sdf_mol(sdf_path)
            mol.SetProp("_Name", sdf_path.stem)
            mol.SetProp("label", str(label))
            mol.SetProp("source_path", str(sdf_path))
            writer.write(mol)
    finally:
        writer.close()


def parse_hit_scores(
    hits_sdf: Path,
    *,
    hit_score: float,
    score_properties: list[str] | tuple[str, ...],
) -> dict[str, float]:
    from pathlib import Path
    from rdkit import Chem

    hits_sdf = Path(hits_sdf)

    # CDPKit can produce an empty hits file when no molecules match.
    # RDKit SDMolSupplier can crash or behave badly on a 0-byte SDF.
    if not hits_sdf.exists() or hits_sdf.stat().st_size == 0:
        print(f"[WARN] CDPKit produced no hits or empty hits file: {hits_sdf}")
        return {}

    scores = {}
    supplier = Chem.SDMolSupplier(str(hits_sdf), sanitize=False, removeHs=False)

    for mol in supplier:
        if mol is None:
            continue

        name = mol.GetProp("_Name") if mol.HasProp("_Name") else None
        if not name:
            continue

        scores[name] = get_score_property(mol, score_properties, default=hit_score)

    return scores


def get_score_property(
    mol,
    score_properties: list[str] | tuple[str, ...],
    *,
    default: float,
) -> float:
    for prop_name in score_properties:
        if mol.HasProp(prop_name):
            try:
                return float(mol.GetProp(prop_name))
            except ValueError:
                pass
    return float(default)

    
def deterministic_jitter(name: str, scale: float = 1e-12) -> float:
    import hashlib

    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12)
    return value * scale