"""Incremental screening artifacts for downstream analysis."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch


ARTIFACT_DIRNAME = "analysis_artifacts"


def initialize_artifacts(output_dir: str | Path, run_config: dict) -> Path:
    """Create the per-target artifact directory and write the run config."""
    artifact_dir = Path(output_dir) / ARTIFACT_DIRNAME
    (artifact_dir / "molecules").mkdir(parents=True, exist_ok=True)
    (artifact_dir / "failures").mkdir(parents=True, exist_ok=True)
    config = dict(run_config)
    config["artifact_created_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(artifact_dir / "run_config.json", make_jsonable(config))
    append_jsonl(
        artifact_dir / "run_events.jsonl",
        {
            "event": "run_initialized",
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline": config.get("pipeline_name"),
            "target": config.get("target_name"),
        },
    )
    return artifact_dir


def save_query_artifact(output_dir: str | Path, *, query_ligand: str | Path, encoding) -> None:
    """Save query embeddings once per target."""
    artifact_dir = Path(output_dir) / ARTIFACT_DIRNAME
    payload = {"query_ligand": str(query_ligand)}
    payload.update(encoding_payload(encoding))
    torch.save(payload, artifact_dir / "query.pt")


def save_molecule_artifact(
    output_dir: str | Path,
    *,
    row: dict,
    encoding,
    opt_meta: dict | None = None,
    match_details: list[dict] | None = None,
    score_components: dict | None = None,
) -> None:
    """Save one completed molecule artifact and append its index row."""
    artifact_dir = Path(output_dir) / ARTIFACT_DIRNAME
    artifact_path = artifact_dir / "molecules" / f"{safe_artifact_name(row.get('name') or Path(row['path']).stem)}.pt"
    payload = {
        "row": make_jsonable(row),
        "optimization": make_jsonable(opt_meta or {}),
        "matching_details": make_jsonable(match_details or []),
        "score_components": make_jsonable(score_components or {}),
    }
    payload.update(encoding_payload(encoding))
    torch.save(payload, artifact_path)
    append_jsonl(
        artifact_dir / "molecule_index.jsonl",
        {
            "name": row.get("name"),
            "path": row.get("path"),
            "label": row.get("label"),
            "score": row.get("score"),
            "artifact_path": str(artifact_path),
            "torsion_count": row.get("torsion_count"),
            "time_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    append_jsonl(
        artifact_dir / "run_events.jsonl",
        {
            "event": "molecule_completed",
            "name": row.get("name"),
            "path": row.get("path"),
            "score": row.get("score"),
            "time_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def save_failure_artifact(output_dir: str | Path, *, row: dict, error: str) -> None:
    """Append a failed molecule entry for later debugging."""
    artifact_dir = Path(output_dir) / ARTIFACT_DIRNAME
    append_jsonl(
        artifact_dir / "failures.jsonl",
        {
            "name": row.get("name"),
            "path": row.get("path"),
            "label": row.get("label"),
            "error": error,
            "time_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    append_jsonl(
        artifact_dir / "run_events.jsonl",
        {
            "event": "molecule_failed",
            "name": row.get("name"),
            "path": row.get("path"),
            "error": error,
            "time_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def encoding_payload(encoding) -> dict:
    """Normalize model outputs into a CPU-serializable artifact payload."""
    if isinstance(encoding, dict):
        return {
            key: tensor_to_cpu(value)
            for key, value in encoding.items()
            if key in {"global_embedding", "feature_embeddings", "feature_metadata"}
        }
    return {"global_embedding": tensor_to_cpu(encoding)}


def tensor_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: tensor_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tensor_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tensor_to_cpu(item) for item in value)
    return value


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(make_jsonable(row), sort_keys=True) + "\n")


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def make_jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [make_jsonable(item) for item in value]
    return value


def safe_artifact_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._")
    return safe or "molecule"
