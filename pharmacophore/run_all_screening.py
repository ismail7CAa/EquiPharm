#!/usr/bin/env python
"""Run all pharmacophore screening methods and aggregate their result tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .CDPKit.screening import run_cdpkit_screening
    from .DiscoveryStudio.screening import run_discovery_studio_screening
    from .EquiPharm.screening import run_equipharm_screening
    from .EquiPharm_Hungarian.screening import run_equipharm_hungarian_screening
    from .EquiPharm_Hungarian_3D.screening import run_equipharm_hungarian_3d_screening
    from .EquiPharm_Hungarian_Cosine.screening import run_equipharm_hungarian_cosine_screening
    from .EquiPharm_Hungarian_Cosine_v2.screening import run_equipharm_hungarian_cosine_v2_screening
    from .EquiPharm_Hungarian_v2.screening import run_equipharm_hungarian_v2_screening
    from .EquiPharm_Hungarian_v3.screening import run_equipharm_hungarian_v3_screening
    from .EquiPharm_Hungarian_v4.screening import run_equipharm_hungarian_v4_screening
    from .OpenPharmaco.screening import run_openpharmaco_screening
    from .PharmacoMatch.screening import run_pharmacomatch_screening
    from .Pharmit.screening import run_pharmit_screening
    from .SchrodingerPhase.screening import run_schrodinger_phase_screening
    from .core.external_baselines import discover_dude_targets, find_cdpkit_query, find_query_ligand, write_dataset_summary
except ImportError:
    from pharmacophore.CDPKit.screening import run_cdpkit_screening
    from pharmacophore.DiscoveryStudio.screening import run_discovery_studio_screening
    from pharmacophore.EquiPharm.screening import run_equipharm_screening
    from pharmacophore.EquiPharm_Hungarian.screening import run_equipharm_hungarian_screening
    from pharmacophore.EquiPharm_Hungarian_3D.screening import run_equipharm_hungarian_3d_screening
    from pharmacophore.EquiPharm_Hungarian_Cosine.screening import run_equipharm_hungarian_cosine_screening
    from pharmacophore.EquiPharm_Hungarian_Cosine_v2.screening import run_equipharm_hungarian_cosine_v2_screening
    from pharmacophore.EquiPharm_Hungarian_v2.screening import run_equipharm_hungarian_v2_screening
    from pharmacophore.EquiPharm_Hungarian_v3.screening import run_equipharm_hungarian_v3_screening
    from pharmacophore.EquiPharm_Hungarian_v4.screening import run_equipharm_hungarian_v4_screening
    from pharmacophore.OpenPharmaco.screening import run_openpharmaco_screening
    from pharmacophore.PharmacoMatch.screening import run_pharmacomatch_screening
    from pharmacophore.Pharmit.screening import run_pharmit_screening
    from pharmacophore.SchrodingerPhase.screening import run_schrodinger_phase_screening
    from pharmacophore.core.external_baselines import discover_dude_targets, find_cdpkit_query, find_query_ligand, write_dataset_summary


MODEL_PIPELINES = (
    "CDPKit",
    "PharmacoMatch",
    "SchrodingerPhase",
    "OpenPharmaco",
    "Pharmit",
    "DiscoveryStudio",
    "EquiPharm",
    "EquiPharm_Hungarian",
    "EquiPharm_Hungarian_v2",
    "EquiPharm_Hungarian_v3",
    "EquiPharm_Hungarian_v4",
    "EquiPharm_Hungarian_3D",
    "EquiPharm_Hungarian_Cosine",
    "EquiPharm_Hungarian_Cosine_v2",
)


COMMAND_BASELINES = {
    "PharmacoMatch": {
        "runner": run_pharmacomatch_screening,
        "arg_prefix": "pharmacomatch",
        "label": "PharmacoMatch",
    },
    "SchrodingerPhase": {
        "runner": run_schrodinger_phase_screening,
        "arg_prefix": "phase",
        "label": "SchrodingerPhase",
    },
    "OpenPharmaco": {
        "runner": run_openpharmaco_screening,
        "arg_prefix": "openpharmaco",
        "label": "OpenPharmaco",
    },
    "Pharmit": {
        "runner": run_pharmit_screening,
        "arg_prefix": "pharmit",
        "label": "Pharmit",
    },
    "DiscoveryStudio": {
        "runner": run_discovery_studio_screening,
        "arg_prefix": "discoverystudio",
        "label": "DiscoveryStudio",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run all configured screening methods on active-decoy targets.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", help="Name used for dataset-specific result folders. Defaults to dataset-dir name.")
    parser.add_argument("--output-dir", type=Path, default=Path("pharmacophore/results"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--target", action="append", help="Run only this DUD-E target. Can be repeated.")
    parser.add_argument(
        "--exclude-pipeline",
        action="append",
        choices=MODEL_PIPELINES,
        default=[],
        help="Skip this pipeline. Can be repeated.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--maxiter", type=int, default=3)
    parser.add_argument("--popsize", type=int, default=2)
    parser.add_argument("--v4-distance-sigma", type=float, default=1.0)
    parser.add_argument("--v4-geometry-penalty-weight", type=float, default=1.0)
    parser.add_argument("--cdpkit-query-dir", type=Path)
    parser.add_argument("--psdcreate-bin", default="psdcreate")
    parser.add_argument("--psdscreen-bin", default="psdscreen")
    parser.add_argument("--pharmacomatch-command-template")
    parser.add_argument("--pharmacomatch-score-regex")
    parser.add_argument("--pharmacomatch-score-json-key")
    parser.add_argument("--pharmacomatch-work-dir", type=Path)
    add_command_baseline_args(parser, "phase")
    add_command_baseline_args(parser, "openpharmaco")
    add_command_baseline_args(parser, "pharmit")
    add_command_baseline_args(parser, "discoverystudio")
    return parser.parse_args()


def add_command_baseline_args(parser, prefix: str) -> None:
    parser.add_argument(f"--{prefix}-command-template")
    parser.add_argument(f"--{prefix}-score-regex")
    parser.add_argument(f"--{prefix}-score-json-key")
    parser.add_argument(f"--{prefix}-work-dir", type=Path)


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset_name or infer_dataset_name(args.dataset_dir)
    output_root = resolve_output_root(args.output_dir, dataset_name)
    output_root.mkdir(parents=True, exist_ok=True)

    requested_targets = set(args.target or [])
    target_dirs = [
        target_dir
        for target_dir in discover_dude_targets(args.dataset_dir)
        if not requested_targets or target_dir.name in requested_targets
    ]
    if requested_targets:
        found = {target_dir.name for target_dir in target_dirs}
        missing = sorted(requested_targets - found)
        if missing:
            raise SystemExit(f"Requested target(s) not found under {args.dataset_dir}: {', '.join(missing)}")

    metrics_rows = []
    for target_dir in target_dirs:
        target_name = target_dir.name
        for pipeline in selected_pipelines(args):
            try:
                metrics = run_one_pipeline(args, pipeline, target_dir, output_root)
                metrics["dataset"] = dataset_name
                metrics["status"] = "ok"
            except Exception as exc:
                metrics = {
                    "dataset": dataset_name,
                    "pipeline": pipeline,
                    "target": target_name,
                    "status": "failed",
                    "error": str(exc),
                }
            metrics_rows.append(metrics)

    metrics_path = write_dataset_summary(
        output_root,
        metrics_rows,
        filename="all_screening_metrics.csv",
    )
    scores_path = write_combined_scores(output_root, metrics_rows)

    result = {
        "output_dir": str(output_root),
        "dataset": dataset_name,
        "metrics_csv": str(metrics_path),
        "scores_csv": str(scores_path),
        "n_targets": len(target_dirs),
        "n_runs": len(metrics_rows),
        "n_ok": sum(row.get("status") == "ok" for row in metrics_rows),
        "n_failed": sum(row.get("status") == "failed" for row in metrics_rows),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def infer_dataset_name(dataset_dir: str | Path) -> str:
    dataset_path = Path(dataset_dir)
    return dataset_path.name or dataset_path.resolve().name


def resolve_output_root(output_dir: str | Path, dataset_name: str) -> Path:
    output_path = Path(output_dir)
    if output_path.name == dataset_name:
        return output_path
    return output_path / dataset_name


def selected_pipelines(args) -> tuple[str, ...]:
    excluded = set(args.exclude_pipeline or [])
    return tuple(pipeline for pipeline in MODEL_PIPELINES if pipeline not in excluded)


def run_one_pipeline(args, pipeline: str, target_dir: Path, output_root: Path) -> dict:
    target_name = target_dir.name
    common_dirs = {
        "actives_dir": target_dir / "actives_sdf",
        "decoys_dir": target_dir / "decoys_sdf",
        "target_name": target_name,
        "limit": args.limit,
    }

    if pipeline == "CDPKit":
        query_root = Path(args.cdpkit_query_dir) / target_name if args.cdpkit_query_dir else target_dir
        query_pharmacophore = find_cdpkit_query(query_root)
        if query_pharmacophore is None:
            raise FileNotFoundError(f"No CDPKit query pharmacophore found for {target_name}.")
        return run_cdpkit_screening(
            query_pharmacophore=query_pharmacophore,
            output_dir=output_root / pipeline / target_name,
            psdcreate_bin=args.psdcreate_bin,
            psdscreen_bin=args.psdscreen_bin,
            **common_dirs,
        )

    if pipeline in COMMAND_BASELINES:
        baseline = COMMAND_BASELINES[pipeline]
        prefix = baseline["arg_prefix"]
        command_template = getattr(args, f"{prefix}_command_template")
        if not command_template:
            raise ValueError(f"Provide --{prefix}-command-template to run {baseline['label']}.")
        return baseline["runner"](
            command_template=command_template,
            query_ligand=require_query_ligand(target_dir),
            output_dir=output_root / pipeline / target_name,
            score_regex=getattr(args, f"{prefix}_score_regex"),
            score_json_key=getattr(args, f"{prefix}_score_json_key"),
            work_dir=getattr(args, f"{prefix}_work_dir"),
            **common_dirs,
        )

    model_kwargs = {
        "checkpoint_path": args.checkpoint,
        "query_ligand": require_query_ligand(target_dir),
        "output_dir": output_root / pipeline / target_name,
        "device": args.device,
        "optimize": not args.no_optimize,
        "maxiter": args.maxiter,
        "popsize": args.popsize,
        **common_dirs,
    }
    if pipeline == "EquiPharm":
        return run_equipharm_screening(**model_kwargs)
    if pipeline == "EquiPharm_Hungarian":
        return run_equipharm_hungarian_screening(**model_kwargs)
    if pipeline == "EquiPharm_Hungarian_v2":
        return run_equipharm_hungarian_v2_screening(**model_kwargs)
    if pipeline == "EquiPharm_Hungarian_v3":
        return run_equipharm_hungarian_v3_screening(**model_kwargs)
    if pipeline == "EquiPharm_Hungarian_v4":
        return run_equipharm_hungarian_v4_screening(
            distance_sigma=args.v4_distance_sigma,
            geometry_penalty_weight=args.v4_geometry_penalty_weight,
            **model_kwargs,
        )
    if pipeline == "EquiPharm_Hungarian_3D":
        return run_equipharm_hungarian_3d_screening(**model_kwargs)
    if pipeline == "EquiPharm_Hungarian_Cosine":
        return run_equipharm_hungarian_cosine_screening(**model_kwargs)
    if pipeline == "EquiPharm_Hungarian_Cosine_v2":
        return run_equipharm_hungarian_cosine_v2_screening(**model_kwargs)
    raise ValueError(f"Unknown pipeline: {pipeline}")


def require_query_ligand(target_dir: Path) -> Path:
    query_ligand = find_query_ligand(target_dir)
    if query_ligand is None:
        raise FileNotFoundError(f"No query ligand found for {target_dir.name}.")
    return query_ligand


def write_combined_scores(output_root: Path, metrics_rows: list[dict]) -> Path:
    score_tables = []
    for row in metrics_rows:
        if row.get("status") != "ok":
            continue
        score_path = output_root / row["pipeline"] / row["target"] / "scores.csv"
        if score_path.exists():
            df = pd.read_csv(score_path)
            if "dataset" not in df.columns:
                df.insert(0, "dataset", row.get("dataset", output_root.name))
            score_tables.append(df)

    combined_path = output_root / "all_screening_scores.csv"
    if score_tables:
        pd.concat(score_tables, ignore_index=True).to_csv(combined_path, index=False)
    else:
        pd.DataFrame().to_csv(combined_path, index=False)
    return combined_path


if __name__ == "__main__":
    main()
