"""CDPKit/CDPL pharmacophore-alignment baseline wrapper."""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path

try:
    from ..core.external_baselines import (
        build_rows_from_scores,
        collect_labeled_sdf_files,
        discover_dude_targets,
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
        find_cdpkit_query,
        find_query_ligand,
        infer_target_name,
        run_command,
        write_baseline_outputs,
        write_dataset_summary,
    )


def run_cdpkit_screening(
    *,
    query_pharmacophore: str | Path,
    actives_dir: str | Path,
    decoys_dir: str | Path,
    output_dir: str | Path,
    target_name: str | None = None,
    pipeline_name: str = "CDPKit",
    psdcreate_bin: str | None = "psdcreate",
    psdscreen_bin: str | None = None,
    query_format: str | None = None,
    num_threads: int | None = None,
    max_omitted: int | None = None,
    hit_score: float = 1.0,
    miss_score: float = 0.0,
    score_properties=None,
    limit: int | None = None,
) -> dict:
    """Run PharmacoMatch-style CDPL pharmacophore alignment.

    The public function name is kept for existing runners, but this no longer
    calls CDPKit `psdscreen`. It creates active/decoy PSD files, aligns every
    pharmacophore to the query with CDPL, max-pools alignment scores per
    molecule, and writes the standard screening outputs.
    """
    del psdscreen_bin, query_format, max_omitted, hit_score, score_properties

    query_path = Path(query_pharmacophore)
    if query_path.suffix.lower() != ".pml":
        raise ValueError("CDPL alignment requires query_pharmacophore to be a .pml file.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target_name = target_name or infer_target_name(query_pharmacophore, actives_dir, decoys_dir, output_dir)

    active_candidates = collect_labeled_sdf_files(actives_dir, decoys_dir, limit=limit)
    active_files = [path for path, label in active_candidates if label == 1]
    decoy_files = [path for path, label in active_candidates if label == 0]

    raw_path = output_path / "raw"
    preprocessing_path = output_path / "preprocessing"
    vs_path = output_path / "vs"
    raw_path.mkdir(parents=True, exist_ok=True)
    preprocessing_path.mkdir(parents=True, exist_ok=True)
    vs_path.mkdir(parents=True, exist_ok=True)

    local_query = raw_path / "query.pml"
    if query_path.resolve() != local_query.resolve():
        shutil.copyfile(query_path, local_query)

    actives_sdf = preprocessing_path / "actives.sdf"
    inactives_sdf = preprocessing_path / "inactives.sdf"
    actives_psd = raw_path / "actives.psd"
    inactives_psd = raw_path / "inactives.psd"

    write_sdf_bundle(active_files, actives_sdf)
    write_sdf_bundle(decoy_files, inactives_sdf)
    resolved_psdcreate = resolve_psdcreate(psdcreate_bin)
    create_psd_database(resolved_psdcreate, actives_sdf, actives_psd, num_threads=num_threads)
    create_psd_database(resolved_psdcreate, inactives_sdf, inactives_psd, num_threads=num_threads)

    active_scores = align_psd_to_query(local_query, actives_psd, vs_path / "all_actives_aligned.pt")
    decoy_scores = align_psd_to_query(local_query, inactives_psd, vs_path / "all_inactives_aligned.pt")

    scored = {}
    scored.update(scores_by_file_stem(active_files, active_scores, default_score=miss_score))
    scored.update(scores_by_file_stem(decoy_files, decoy_scores, default_score=miss_score))

    rows = build_rows_from_scores(
        active_candidates,
        scored,
        pipeline_name=pipeline_name,
        target_name=target_name,
        default_score=miss_score,
    )
    return write_baseline_outputs(output_dir, rows, pipeline_name=pipeline_name, target_name=target_name)


def score_cdpkit_alignment(
    *,
    query_pharmacophore: str | Path,
    candidate_sdf: str | Path,
    work_dir: str | Path,
    psdcreate_bin: str | Path | None = "psdcreate",
    num_threads: int | None = None,
    default_score: float = 0.0,
) -> float:
    """Return the PharmacoMatch-style CDPL alignment score for one SDF file."""
    scores = score_cdpkit_alignment_batch(
        query_pharmacophore=query_pharmacophore,
        candidate_sdfs=[candidate_sdf],
        work_dir=work_dir,
        psdcreate_bin=psdcreate_bin,
        num_threads=num_threads,
        default_score=default_score,
    )
    return float(scores[Path(candidate_sdf).stem])


def score_cdpkit_alignment_batch(
    *,
    query_pharmacophore: str | Path,
    candidate_sdfs: Iterable[str | Path],
    work_dir: str | Path,
    psdcreate_bin: str | Path | None = "psdcreate",
    num_threads: int | None = None,
    default_score: float = 0.0,
) -> dict[str, float]:
    """Return max-pooled CDPL pharmacophore alignment scores keyed by file stem.

    This is the small functional API intended for reuse outside the screening
    runner. It mirrors the CDPKit comparison used around PharmacoMatch:
    generate ligand pharmacophores, align them to `query.pml`, score every
    generated pharmacophore with CDPL's fit score, then keep the best score per
    input molecule.
    """
    query_path = Path(query_pharmacophore)
    if query_path.suffix.lower() != ".pml":
        raise ValueError("CDPL alignment requires query_pharmacophore to be a .pml file.")

    candidates = [Path(path) for path in candidate_sdfs]
    if not candidates:
        raise ValueError("candidate_sdfs must contain at least one SDF file.")

    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    bundled_sdf = work_path / "candidates.sdf"
    database_psd = work_path / "candidates.psd"
    aligned_pt = work_path / "candidates_aligned.pt"

    write_sdf_bundle(candidates, bundled_sdf)
    resolved_psdcreate = resolve_psdcreate(psdcreate_bin)
    create_psd_database(resolved_psdcreate, bundled_sdf, database_psd, num_threads=num_threads)
    alignment_rows = align_psd_to_query(query_path, database_psd, aligned_pt)
    return scores_by_file_stem(candidates, alignment_rows, default_score=default_score)


def run_cdpkit_dataset_screening(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    query_dir: str | Path | None = None,
    pipeline_name: str = "CDPKit",
    psdcreate_bin: str | None = "psdcreate",
    psdscreen_bin: str | None = None,
    query_format: str | None = None,
    num_threads: int | None = None,
    max_omitted: int | None = None,
    hit_score: float = 1.0,
    miss_score: float = 0.0,
    score_properties=None,
    limit: int | None = None,
    skip_missing_queries: bool = False,
) -> dict:
    """Run CDPL alignment on every DUD-E-style target and write one summary CSV."""
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
                        "error": "missing CDPL query pharmacophore",
                    }
                )
                continue
            raise FileNotFoundError(
                f"No CDPL query pharmacophore found for {target_name}. "
                "Expected query.pml or crystal_ligand.pml, or a query ligand to generate query.pml."
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
    """Return an existing CDPL query or generate query.pml from the target ligand."""
    if query_pharmacophore is not None:
        return Path(query_pharmacophore)

    target_path = Path(target_dir)
    query_path = find_cdpkit_query(target_path)
    if query_path is not None:
        if query_path.suffix.lower() != ".pml":
            raise ValueError(f"CDPL alignment requires a .pml query, got: {query_path}")
        return query_path

    ligand_path = find_query_ligand(target_path)
    if ligand_path is None:
        raise FileNotFoundError(f"No query pharmacophore or query ligand found in {target_path}.")

    generated_query = target_path / "query.pml"
    if not generated_query.exists():
        create_ligand_query_pharmacophore(ligand_path, generated_query)
    return generated_query


def create_ligand_query_pharmacophore(ligand_path: str | Path, output_path: str | Path) -> Path:
    """Generate a ligand pharmacophore query with Python CDPL."""
    try:
        import CDPL.Chem as Chem
        import CDPL.Pharm as Pharm
    except ImportError as exc:
        raise RuntimeError("Generating query.pml requires Python CDPL. Install CDPKit.") from exc

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


def resolve_psdcreate(psdcreate_bin: str | Path | None) -> Path | None:
    """Return psdcreate if available, otherwise use the Python CDPL fallback."""
    if psdcreate_bin is None:
        return None
    path = Path(psdcreate_bin)
    if path.name == "psdcreate" and path.exists():
        return path
    if path.is_dir() and (path / "psdcreate").exists():
        return path / "psdcreate"
    executable = shutil.which(str(psdcreate_bin))
    if executable:
        return Path(executable)
    try:
        import CDPL.Chem  # noqa: F401
        import CDPL.Pharm  # noqa: F401
    except ImportError as exc:
        raise FileNotFoundError(
            "CDPKit psdcreate was not found and Python CDPL is not importable. "
            "Install CDPKit, add psdcreate to PATH, or pass --psdcreate-bin."
        ) from exc
    return None


def create_psd_database(
    psdcreate: Path | None,
    input_sdf: Path,
    output_psd: Path,
    *,
    num_threads: int | None = None,
) -> None:
    """Create a CDPL pharmacophore screening database."""
    if psdcreate is not None:
        command = [str(psdcreate), "-i", str(input_sdf), "-o", str(output_psd), "-d"]
        if num_threads is not None:
            command.extend(["-t", str(num_threads)])
        run_command(command)
        return
    create_psd_with_cdpl_python(input_sdf, output_psd)


def create_psd_with_cdpl_python(input_sdf: Path, output_psd: Path) -> None:
    """Create a PSD file with Python CDPL when psdcreate is unavailable."""
    try:
        import CDPL.Chem as Chem
        import CDPL.Pharm as Pharm
    except ImportError as exc:
        raise RuntimeError("Creating PSD files without psdcreate requires Python CDPL.") from exc

    reader = cdpl_molecule_reader(str(input_sdf), Chem)
    Chem.setMultiConfImportParameter(reader, False)
    writer = cdpl_pharmacophore_writer(str(output_psd), Pharm)
    generator = Pharm.DefaultPharmacophoreGenerator()
    molecule = Chem.BasicMolecule()
    pharmacophore = Pharm.BasicPharmacophore()

    while reader.read(molecule):
        pharmacophore.clear()
        Pharm.prepareForPharmacophoreGeneration(molecule)
        generator.generate(molecule, pharmacophore)
        if pharmacophore.getNumFeatures() == 0:
            continue
        Pharm.setName(pharmacophore, Chem.getName(molecule).strip())
        if not writer.write(pharmacophore):
            raise RuntimeError(f"Could not write pharmacophore to {output_psd}")


def align_psd_to_query(query_pml: Path, input_psd: Path, output_pt: Path) -> list[dict]:
    """Align every PSD pharmacophore to query.pml and return raw scores."""
    try:
        import CDPL.Pharm as Pharm
        import torch
    except ImportError as exc:
        raise RuntimeError("CDPL alignment requires Python CDPKit and torch.") from exc

    ref_ph4 = read_reference_pharmacophore(query_pml, Pharm)
    clear_feature_orientations(ref_ph4, Pharm)
    db_accessor = Pharm.PSDScreeningDBAccessor(str(input_psd))
    mol_ph4 = Pharm.BasicPharmacophore()
    alignment = Pharm.PharmacophoreAlignment(True)
    alignment.addFeatures(ref_ph4, True)
    alignment.performExhaustiveSearch(False)
    fit_score = Pharm.PharmacophoreFitScore(
        match_cnt_weight=1.0,
        pos_match_weight=0.9,
        geom_match_weight=0.0,
    )

    rows = []
    for index in range(db_accessor.getNumPharmacophores()):
        db_accessor.getPharmacophore(index, mol_ph4)
        mol_idx = int(db_accessor.getMoleculeIndex(index))
        conf_idx = int(db_accessor.getConformationIndex(index))
        if mol_ph4.getNumFeatures() == 0:
            rows.append([0.0, 0.0, 0, mol_idx, conf_idx])
            continue

        clear_feature_orientations(mol_ph4, Pharm)
        alignment.clearEntities(False)
        alignment.addFeatures(mol_ph4, False)
        solutions = []
        while alignment.nextAlignment():
            solutions.append(float(fit_score(ref_ph4, mol_ph4, alignment.getTransform())))
        best_score = max(solutions) if solutions else 0.0
        rows.append(
            [
                float(int(best_score)),
                float(best_score % 1.0),
                float(mol_ph4.getNumFeatures()),
                float(mol_idx),
                float(conf_idx),
            ]
        )

    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(rows, dtype=torch.float32), output_pt)
    return [
        {
            "score": row[0] + row[1],
            "num_features": int(row[2]),
            "mol_idx": int(row[3]),
            "conf_idx": int(row[4]),
        }
        for row in rows
    ]


def scores_by_file_stem(files: list[Path], rows: list[dict], *, default_score: float = 0.0) -> dict[str, float]:
    """Max-pool pharmacophore alignment scores by original SDF file stem."""
    scores = {path.stem: float(default_score) for path in files}
    for row in rows:
        mol_idx = row["mol_idx"]
        if 0 <= mol_idx < len(files):
            name = files[mol_idx].stem
            scores[name] = max(scores[name], float(row["score"]))
    return scores


def read_reference_pharmacophore(filename: Path, Pharm):
    reader = Pharm.PharmacophoreReader(str(filename))
    pharmacophore = Pharm.BasicPharmacophore()
    if not reader.read(pharmacophore):
        raise ValueError(f"Could not read reference pharmacophore: {filename}")
    return pharmacophore


def clear_feature_orientations(pharmacophore, Pharm) -> None:
    for feature in pharmacophore:
        Pharm.clearOrientation(feature)
        Pharm.setGeometry(feature, Pharm.FeatureGeometry.SPHERE)


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


def write_sdf_bundle(files: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        for sdf_file in files:
            with sdf_file.open("rb") as handle:
                shutil.copyfileobj(handle, output)
            output.write(b"\n")
