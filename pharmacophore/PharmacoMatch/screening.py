"""Command-template adapter for a PharmacoMatch baseline."""

from __future__ import annotations

import sys
import shutil
from pathlib import Path

try:
    from ..core.external_baselines import (
        collect_labeled_sdf_files,
        discover_dude_targets,
        format_command,
        infer_target_name,
        parse_score_from_process,
        run_command,
        write_baseline_outputs,
        write_dataset_summary,
    )
except ImportError:
    from pharmacophore.core.external_baselines import (
        collect_labeled_sdf_files,
        discover_dude_targets,
        format_command,
        infer_target_name,
        parse_score_from_process,
        run_command,
        write_baseline_outputs,
        write_dataset_summary,
    )


def run_pharmacomatch_screening(
    *,
    command_template: str,
    query_ligand: str | Path,
    actives_dir: str | Path,
    decoys_dir: str | Path,
    output_dir: str | Path,
    target_name: str | None = None,
    pipeline_name: str = "PharmacoMatch",
    score_regex: str | None = None,
    score_json_key: str | None = None,
    work_dir: str | Path | None = None,
    limit: int | None = None,
) -> dict:
    """Run a PharmacoMatch command once per active/decoy candidate.

    `command_template` is formatted with:
    `{query_ligand}`, `{candidate}`, `{candidate_name}`, `{output_dir}`, `{target_name}`,
    `{label}`, and `{idx}`.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target_name = target_name or infer_target_name(query_ligand, actives_dir, decoys_dir, output_dir)
    candidates = collect_labeled_sdf_files(actives_dir, decoys_dir, limit=limit)

    rows = []
    errors = []
    for idx, (candidate, label) in enumerate(candidates):
        command = format_command(
            command_template,
            query_ligand=query_ligand,
            candidate=candidate,
            candidate_name=candidate.stem,
            output_dir=output_path,
            target_name=target_name,
            label=label,
            idx=idx,
        )
        try:
            result = run_command(command, cwd=work_dir)
            score = parse_score_from_process(
                result,
                score_regex=score_regex,
                score_json_key=score_json_key,
            )
            error = None
        except Exception as exc:
            score = float("nan")
            error = str(exc)
            errors.append(f"{candidate.name}: {error}")

        row = {
            "pipeline": pipeline_name,
            "target": target_name,
            "name": candidate.stem,
            "path": str(candidate),
            "label": label,
            "score": score,
            "torsion_count": 0,
        }
        if error is not None:
            row["error"] = error
        rows.append(row)

    rows = [row for row in rows if row.get("score") == row.get("score")]
    if not rows:
        detail = "; ".join(errors[:3])
        if len(errors) > 3:
            detail += f"; ... {len(errors) - 3} more"
        raise RuntimeError(f"PharmacoMatch produced no valid scores for {target_name}. First errors: {detail}")
    return write_baseline_outputs(output_dir, rows, pipeline_name=pipeline_name, target_name=target_name)


def run_official_pharmacomatch_screening(
    *,
    official_vs_dir: str | Path,
    output_dir: str | Path,
    pharmacomatch_root: str | Path = "external/PharmacoMatch",
    model_path: str | Path | None = None,
    target_name: str | None = None,
    pipeline_name: str = "PharmacoMatch",
    batch_size: int | None = None,
    accelerator: str = "auto",
    devices: int | str = 1,
) -> dict:
    """Run the official PharmacoMatch model on a preprocessed VS target directory.

    The official implementation expects:
    `raw/query.pml`, `raw/actives.psd`, `raw/inactives.psd`, and processed `.pt` files.
    PharmacoMatch returns a penalty where lower is better; this exports `score=-penalty`
    so the shared metrics code can rank scores descending.
    """
    root_path = Path(pharmacomatch_root).resolve()
    vs_path = Path(official_vs_dir)
    output_path = Path(output_dir)
    target_name = target_name or vs_path.name
    model_path = Path(model_path) if model_path is not None else root_path / "trained_model" / "trained_model.ckpt"

    _validate_official_pharmacomatch_inputs(vs_path, model_path)
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))

    try:
        import torch
        from lightning import Trainer
        from pharmacomatch.dataset import VirtualScreeningDataModule
        from pharmacomatch.model import PharmacoMatch
        from pharmacomatch.virtual_screening import VirtualScreeningEmbedder, VirtualScreener
    except ImportError as exc:
        raise RuntimeError(
            "Official PharmacoMatch dependencies are not installed. Activate the "
            "pharmaco_match environment and run `pip install -e external/PharmacoMatch`."
        ) from exc

    model = PharmacoMatch.load_from_checkpoint(str(model_path), map_location=torch.device("cpu"))
    trainer = Trainer(
        num_nodes=1,
        devices=devices,
        accelerator=accelerator,
        logger=False,
        enable_checkpointing=False,
    )
    datamodule = VirtualScreeningDataModule(str(vs_path), batch_size=batch_size)
    datamodule.setup(stage="screening")
    screener = VirtualScreener(VirtualScreeningEmbedder(model, datamodule, trainer))

    rows = _rows_from_official_screener(
        screener=screener,
        metadata=datamodule.metadata,
        target_name=target_name,
        pipeline_name=pipeline_name,
        vs_path=vs_path,
    )
    return write_baseline_outputs(output_path, rows, pipeline_name=pipeline_name, target_name=target_name)


def prepare_official_pharmacomatch_target(
    *,
    target_dir: str | Path,
    prepared_vs_dir: str | Path | None = None,
    pharmacomatch_root: str | Path = "external/PharmacoMatch",
    cdpkit_bin: str | Path | None = None,
    query_pharmacophore: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Prepare one DUD-E-style target for the official PharmacoMatch runner."""
    target_path = Path(target_dir)
    if not target_path.is_dir():
        raise FileNotFoundError(f"Target directory not found: {target_path}")

    target_name = target_path.name
    prepared_path = Path(prepared_vs_dir) if prepared_vs_dir is not None else Path(pharmacomatch_root) / "data" / "DUD-E" / target_name
    raw_path = prepared_path / "raw"
    preprocessing_path = prepared_path / "preprocessing"
    processed_path = prepared_path / "processed"
    raw_path.mkdir(parents=True, exist_ok=True)
    preprocessing_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    cdpkit_path = _resolve_cdpkit_bin(cdpkit_bin)
    psdcreate = cdpkit_path / "psdcreate"
    query_output = raw_path / "query.pml"
    actives_sdf = preprocessing_path / "actives.sdf"
    inactives_sdf = preprocessing_path / "inactives.sdf"
    actives_psd = raw_path / "actives.psd"
    inactives_psd = raw_path / "inactives.psd"

    _prepare_query_pharmacophore(
        target_path=target_path,
        query_output=query_output,
        pharmacomatch_root=Path(pharmacomatch_root),
        query_pharmacophore=Path(query_pharmacophore) if query_pharmacophore is not None else None,
        force=force,
    )
    _combine_sdf_directory(target_path / "actives_sdf", actives_sdf, force=force)
    _combine_sdf_directory(target_path / "decoys_sdf", inactives_sdf, force=force)
    _run_psdcreate(psdcreate, actives_sdf, actives_psd, force=force)
    _run_psdcreate(psdcreate, inactives_sdf, inactives_psd, force=force)
    return prepared_path


def run_official_pharmacomatch_dataset_screening(
    *,
    official_dataset_dir: str | Path,
    output_dir: str | Path,
    pharmacomatch_root: str | Path = "external/PharmacoMatch",
    model_path: str | Path | None = None,
    pipeline_name: str = "PharmacoMatch",
    batch_size: int | None = None,
    accelerator: str = "auto",
    devices: int | str = 1,
) -> dict:
    """Run official PharmacoMatch on every preprocessed target in a dataset root."""
    output_path = Path(output_dir)
    metrics_rows = []
    targets = _discover_official_pharmacomatch_targets(official_dataset_dir)
    for target_dir in targets:
        try:
            metrics = run_official_pharmacomatch_screening(
                official_vs_dir=target_dir,
                output_dir=output_path / target_dir.name,
                pharmacomatch_root=pharmacomatch_root,
                model_path=model_path,
                target_name=target_dir.name,
                pipeline_name=pipeline_name,
                batch_size=batch_size,
                accelerator=accelerator,
                devices=devices,
            )
            metrics["status"] = "ok"
        except Exception as exc:
            metrics = {
                "pipeline": pipeline_name,
                "target": target_dir.name,
                "status": "failed",
                "error": str(exc),
            }
        metrics_rows.append(metrics)

    summary_path = write_dataset_summary(output_path, metrics_rows)
    return {
        "pipeline": pipeline_name,
        "dataset_dir": str(official_dataset_dir),
        "output_dir": str(output_path),
        "summary_csv": str(summary_path),
        "n_targets": len(metrics_rows),
        "n_ok": sum(row.get("status") == "ok" for row in metrics_rows),
        "n_failed": sum(row.get("status") == "failed" for row in metrics_rows),
        "n_skipped": 0,
    }


def _validate_official_pharmacomatch_inputs(vs_path: Path, model_path: Path) -> None:
    missing = []
    for relative_path in (
        "raw/query.pml",
        "raw/actives.psd",
        "raw/inactives.psd",
    ):
        if not (vs_path / relative_path).exists():
            missing.append(str(vs_path / relative_path))
    if not model_path.exists():
        missing.append(str(model_path))
    if missing:
        raise FileNotFoundError(
            "Official PharmacoMatch inputs are missing. Download/preprocess the "
            "PharmacoMatch data first. Missing: " + ", ".join(missing)
        )


def _resolve_cdpkit_bin(cdpkit_bin: str | Path | None) -> Path:
    if cdpkit_bin is not None:
        path = Path(cdpkit_bin)
        if (path / "psdcreate").exists():
            return path
        raise FileNotFoundError(f"CDPKit psdcreate not found in: {path}")

    vendored_path = Path("external/CDPKit/Bin")
    if (vendored_path / "psdcreate").exists():
        return vendored_path

    executable = shutil.which("psdcreate")
    if executable:
        return Path(executable).parent
    raise FileNotFoundError(
        "CDPKit psdcreate was not found. Install CDPKit into `external/CDPKit` "
        "with `python scripts/install_cdpkit.py`, add psdcreate to PATH, or pass "
        "`--cdpkit-bin /path/to/CDPKit/Bin`."
    )


def _prepare_query_pharmacophore(
    *,
    target_path: Path,
    query_output: Path,
    pharmacomatch_root: Path,
    query_pharmacophore: Path | None,
    force: bool,
) -> None:
    if query_output.exists() and not force:
        return

    if query_pharmacophore is None:
        for candidate_name in ("query.pml", "crystal_ligand.pml"):
            candidate = target_path / candidate_name
            if candidate.exists():
                query_pharmacophore = candidate
                break

    if query_pharmacophore is not None:
        if query_pharmacophore.suffix.lower() != ".pml":
            raise ValueError(f"Official PharmacoMatch expects query.pml, got: {query_pharmacophore}")
        shutil.copyfile(query_pharmacophore, query_output)
        return

    query_ligand = target_path / "crystal_ligand.mol2"
    if not query_ligand.exists():
        raise FileNotFoundError(
            f"No query pharmacophore or crystal_ligand.mol2 found in {target_path}."
        )

    script = pharmacomatch_root / "data_processing" / "python_scripts" / "cdpl" / "pharm_gen_mol_ph4s.py"
    if not script.exists():
        raise FileNotFoundError(f"PharmacoMatch query generation script not found: {script}")
    run_command([sys.executable, str(script), "-i", str(query_ligand), "-o", str(query_output)])


def _combine_sdf_directory(sdf_dir: Path, output_sdf: Path, *, force: bool) -> None:
    if output_sdf.exists() and not force:
        return
    if not sdf_dir.is_dir():
        raise FileNotFoundError(f"SDF directory not found: {sdf_dir}")
    sdf_files = sorted(sdf_dir.glob("*.sdf"))
    if not sdf_files:
        raise ValueError(f"No SDF files found in {sdf_dir}.")

    with output_sdf.open("wb") as output:
        for sdf_file in sdf_files:
            with sdf_file.open("rb") as handle:
                shutil.copyfileobj(handle, output)
            output.write(b"\n")


def _run_psdcreate(psdcreate: Path, input_sdf: Path, output_psd: Path, *, force: bool) -> None:
    if output_psd.exists() and not force:
        return
    run_command([str(psdcreate), "-i", str(input_sdf), "-o", str(output_psd), "-d"])


def _discover_official_pharmacomatch_targets(dataset_dir: str | Path) -> list[Path]:
    dataset_path = Path(dataset_dir)
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Official PharmacoMatch dataset directory not found: {dataset_path}")

    targets = []
    for target_dir in sorted(path for path in dataset_path.iterdir() if path.is_dir()):
        required = (
            target_dir / "raw" / "query.pml",
            target_dir / "raw" / "actives.psd",
            target_dir / "raw" / "inactives.psd",
        )
        if all(path.exists() for path in required):
            targets.append(target_dir)
    if not targets:
        raise ValueError(f"No official PharmacoMatch targets found in {dataset_path}.")
    return targets


def _rows_from_official_screener(
    *,
    screener,
    metadata,
    target_name: str,
    pipeline_name: str,
    vs_path: Path,
) -> list[dict]:
    active_names = _ligand_names_from_metadata(metadata.active)
    inactive_names = _ligand_names_from_metadata(metadata.inactive)
    active_scores = (-screener.active_ligand_score).detach().cpu().tolist()
    inactive_scores = (-screener.inactive_ligand_score).detach().cpu().tolist()

    if len(active_names) != len(active_scores):
        raise ValueError(f"Active metadata/score length mismatch: {len(active_names)} names, {len(active_scores)} scores.")
    if len(inactive_names) != len(inactive_scores):
        raise ValueError(f"Inactive metadata/score length mismatch: {len(inactive_names)} names, {len(inactive_scores)} scores.")

    rows = []
    for name, score in zip(active_names, active_scores):
        rows.append(_official_score_row(pipeline_name, target_name, name, vs_path / "raw" / "actives.psd", 1, score))
    for name, score in zip(inactive_names, inactive_scores):
        rows.append(_official_score_row(pipeline_name, target_name, name, vs_path / "raw" / "inactives.psd", 0, score))
    return rows


def _ligand_names_from_metadata(metadata) -> list[str]:
    return [str(name) for name in metadata.drop_duplicates("name", keep="first")["name"].tolist()]


def _official_score_row(pipeline_name: str, target_name: str, name: str, path: Path, label: int, score: float) -> dict:
    return {
        "pipeline": pipeline_name,
        "target": target_name,
        "name": name,
        "path": str(path),
        "label": label,
        "score": float(score),
        "torsion_count": 0,
    }


def run_pharmacomatch_dataset_screening(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    command_template: str,
    pipeline_name: str = "PharmacoMatch",
    score_regex: str | None = None,
    score_json_key: str | None = None,
    work_dir: str | Path | None = None,
    limit: int | None = None,
) -> dict:
    """Run PharmacoMatch on every DUD-E-style target and write one summary CSV."""
    output_path = Path(output_dir)
    metrics_rows = []

    for target_dir in discover_dude_targets(dataset_dir):
        target_name = target_dir.name
        try:
            metrics = run_pharmacomatch_screening(
                command_template=command_template,
                query_ligand=target_dir / "crystal_ligand.mol2",
                actives_dir=target_dir / "actives_sdf",
                decoys_dir=target_dir / "decoys_sdf",
                output_dir=output_path / target_name,
                target_name=target_name,
                pipeline_name=pipeline_name,
                score_regex=score_regex,
                score_json_key=score_json_key,
                work_dir=work_dir,
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
        "n_skipped": 0,
    }
