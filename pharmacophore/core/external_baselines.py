"""Helpers for optional external screening baselines."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path


TARGET_QUERY_NAMES = (
    "query.pml",
    "crystal_ligand.pml",
)

LIGAND_QUERY_NAMES = (
    "crystal_ligand.mol2",
    "crystal_ligand.sdf",
    "query_ligand.mol2",
    "query_ligand.sdf",
    "ligand.mol2",
    "ligand.sdf",
)


def collect_labeled_sdf_files(
    actives_dir: str | Path,
    decoys_dir: str | Path,
    *,
    limit: int | None = None,
) -> list[tuple[Path, int]]:
    active_path = Path(actives_dir)
    decoy_path = Path(decoys_dir)
    if not active_path.is_dir():
        raise FileNotFoundError(f"Actives directory not found: {active_path}")
    if not decoy_path.is_dir():
        raise FileNotFoundError(f"Decoys directory not found: {decoy_path}")

    active_files = [(path, 1) for path in sorted(active_path.glob("*.sdf"))]
    decoy_files = [(path, 0) for path in sorted(decoy_path.glob("*.sdf"))]
    candidates = active_files + decoy_files
    if limit is not None:
        candidates = candidates[:limit]
    if not candidates:
        raise ValueError(f"No SDF candidates found in {active_path} or {decoy_path}.")
    return candidates


def discover_dude_targets(dataset_dir: str | Path) -> list[Path]:
    dataset_path = Path(dataset_dir)
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    targets = []
    for target_dir in sorted(path for path in dataset_path.iterdir() if path.is_dir()):
        if (target_dir / "actives_sdf").is_dir() and (target_dir / "decoys_sdf").is_dir():
            targets.append(target_dir)
    if not targets:
        raise ValueError(f"No DUD-E-style targets found in {dataset_path}.")
    return targets


def find_cdpkit_query(target_dir: str | Path) -> Path | None:
    target_path = Path(target_dir)
    for query_name in TARGET_QUERY_NAMES:
        query_path = target_path / query_name
        if query_path.exists():
            return query_path
    return None


def find_query_ligand(target_dir: str | Path) -> Path | None:
    target_path = Path(target_dir)
    for query_name in LIGAND_QUERY_NAMES:
        query_path = target_path / query_name
        if query_path.exists():
            return query_path
    mol2_candidates = sorted(target_path.glob("*_ligand.mol2"))
    if mol2_candidates:
        return mol2_candidates[0]
    sdf_candidates = sorted(target_path.glob("*_ligand.sdf"))
    if sdf_candidates:
        return sdf_candidates[0]
    return None


def infer_target_name(*paths) -> str:
    for raw_path in paths:
        path = Path(raw_path)
        parts = path.parts
        if "DUD-E" in parts:
            idx = parts.index("DUD-E")
            if idx + 1 < len(parts):
                return parts[idx + 1]
    output_name = Path(paths[-1]).name
    return output_name or "unknown_target"


def build_rows_from_scores(
    candidates: list[tuple[Path, int]],
    scores: dict[str, float],
    *,
    pipeline_name: str,
    target_name: str,
    default_score: float = 0.0,
) -> list[dict]:
    rows = []
    for idx, (sdf_path, label) in enumerate(candidates):
        name = sdf_path.stem
        rows.append(
            {
                "pipeline": pipeline_name,
                "target": target_name,
                "name": name,
                "path": str(sdf_path),
                "label": label,
                "score": float(scores.get(name, default_score)),
                "torsion_count": 0,
            }
        )
    return rows


def run_command(command: list[str], *, cwd: str | Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


def parse_score_from_process(
    result: subprocess.CompletedProcess,
    *,
    score_regex: str | None = None,
    score_json_key: str | None = None,
) -> float:
    text = "\n".join(part for part in [result.stdout, result.stderr] if part)
    if score_json_key:
        payload = json.loads(result.stdout)
        value = payload
        for key in score_json_key.split("."):
            value = value[key]
        return float(value)
    if score_regex:
        match = re.search(score_regex, text)
        if not match:
            raise ValueError(f"Could not parse score with regex: {score_regex}")
        group = match.group(1) if match.groups() else match.group(0)
        return float(group)
    raise ValueError("Provide score_regex or score_json_key for command-based scoring.")


def format_command(template: str, **values) -> list[str]:
    formatted = template.format(**{key: shlex.quote(str(value)) for key, value in values.items()})
    return shlex.split(formatted)


def write_baseline_outputs(
    output_dir: str | Path,
    rows: list[dict],
    *,
    pipeline_name: str,
    target_name: str,
) -> dict:
    try:
        from .metrics import write_outputs
    except ImportError:
        from pharmacophore.core.metrics import write_outputs

    return write_outputs(
        output_dir,
        rows,
        pipeline_name=pipeline_name,
        target_name=target_name,
        write_roc_curve_image=False,
    )


def write_dataset_summary(
    output_dir: str | Path,
    metrics_rows: list[dict],
    *,
    filename: str = "dataset_metrics.csv",
) -> Path:
    import pandas as pd

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / filename
    pd.DataFrame(metrics_rows).sort_values("target").to_csv(summary_path, index=False)
    return summary_path


def first_sdf_mol(path: Path):
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    for mol in supplier:
        if mol is not None:
            return mol
    raise ValueError(f"No readable molecule found in {path}")
