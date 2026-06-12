"""Reusable command-template adapters for external pharmacophore screeners."""

from __future__ import annotations

from pathlib import Path

try:
    from .external_baselines import (
        collect_labeled_sdf_files,
        discover_dude_targets,
        find_query_ligand,
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
        find_query_ligand,
        format_command,
        infer_target_name,
        parse_score_from_process,
        run_command,
        write_baseline_outputs,
        write_dataset_summary,
    )


def run_command_baseline_screening(
    *,
    command_template: str,
    query_ligand: str | Path,
    actives_dir: str | Path,
    decoys_dir: str | Path,
    output_dir: str | Path,
    pipeline_name: str,
    target_name: str | None = None,
    score_regex: str | None = None,
    score_json_key: str | None = None,
    work_dir: str | Path | None = None,
    limit: int | None = None,
) -> dict:
    """Run one external command per candidate and parse one numeric score."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target_name = target_name or infer_target_name(query_ligand, actives_dir, decoys_dir, output_dir)
    candidates = collect_labeled_sdf_files(actives_dir, decoys_dir, limit=limit)

    rows = []
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
    return write_baseline_outputs(output_dir, rows, pipeline_name=pipeline_name, target_name=target_name)


def run_command_baseline_dataset_screening(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    command_template: str,
    pipeline_name: str,
    score_regex: str | None = None,
    score_json_key: str | None = None,
    work_dir: str | Path | None = None,
    limit: int | None = None,
) -> dict:
    """Run a command-template baseline on every normalized screening target."""
    output_path = Path(output_dir)
    metrics_rows = []

    for target_dir in discover_dude_targets(dataset_dir):
        target_name = target_dir.name
        try:
            query_ligand = find_query_ligand(target_dir)
            if query_ligand is None:
                raise FileNotFoundError(f"No query ligand found for {target_name}.")
            metrics = run_command_baseline_screening(
                command_template=command_template,
                query_ligand=query_ligand,
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
