"""Shared SPICE model configuration and CLI helpers for screening variants."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from pathlib import Path

from pharmacophore.core.matching_screening import screen_actives_decoys_matching
from pharmacophore.core.seed_aggregation import run_seeded
from pharmacophore.core.screening import screen_actives_decoys


SPICE_MODEL = {
    "model_module": "pharm_training.equiformer_encoder_pharmaco_feat",
    "model_class": "SPICEPharmacophoreEncoder",
    "checkpoint_path": "runs/pharm_training/spice_search/trial_0d798538b1/checkpoints/best.pt",
}

SPICE_SAMPLE = {
    "num_actives": 50,
    "num_decoys": 500,
    "seed": 1,
}

SPICE_SEEDS = (1, 2, 3)


class Tee:
    """Write CLI output to both the terminal and a persistent log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def default_output_dir(runner, config: dict) -> Path:
    """Return the standard SPICE result directory for a CLI invocation."""
    variant = runner.__module__.split(".")[-2]
    target_name = config.get("target_name")
    if not target_name:
        for key in ("query_ligand", "actives_dir", "decoys_dir"):
            value = config.get(key)
            if value:
                path = Path(value)
                target_name = path.parent.name if key == "query_ligand" else path.parent.name
                break
    target_name = target_name or "unknown_target"
    return Path("pharmacophore/results/pharmacophore_spice") / variant / target_name


def run_pooled(**kwargs):
    # These isolated pipelines must never silently fall back to a QM9 model,
    # even when an old JSON configuration contains model overrides.
    for key, value in SPICE_MODEL.items():
        kwargs.setdefault(key, value)
    for key, value in SPICE_SAMPLE.items():
        kwargs.setdefault(key, value)
    kwargs.setdefault("pipeline_name", "EquiPharm_SPICE")
    kwargs.setdefault("use_pharmacophore_features", True)
    kwargs.setdefault("rotatable_only", False)
    kwargs.setdefault("heavy_only", True)
    kwargs.setdefault("exclude_rings", True)
    kwargs.setdefault("one_per_bond", False)
    kwargs.setdefault("write_named_roc_curve", True)
    return screen_actives_decoys(**kwargs)


def run_matching(pipeline_name: str, matching_method: str, matching_score_mode: str, **kwargs):
    for key, value in SPICE_MODEL.items():
        kwargs.setdefault(key, value)
    for key, value in SPICE_SAMPLE.items():
        kwargs.setdefault(key, value)
    kwargs.setdefault("pipeline_name", f"{pipeline_name}_SPICE")
    kwargs.setdefault("matching_method", matching_method)
    kwargs.setdefault("matching_score_mode", matching_score_mode)
    kwargs.setdefault("rotatable_only", False)
    kwargs.setdefault("heavy_only", True)
    kwargs.setdefault("exclude_rings", True)
    kwargs.setdefault("one_per_bond", False)
    return screen_actives_decoys_matching(**kwargs)


def run_cli(runner, description: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument("--target-name")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--query-ligand", type=Path)
    parser.add_argument("--actives-dir", type=Path)
    parser.add_argument("--decoys-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--maxiter", type=int)
    parser.add_argument("--popsize", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-actives", type=int)
    parser.add_argument("--num-decoys", type=int)
    parser.add_argument("--seed", type=int, help="Run one seed instead of the default seeds 1, 2, and 3.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Seeds to run (default: 1 2 3).")
    args = parser.parse_args()

    config = {}
    if args.config is not None:
        config = json.loads(args.config.read_text())
    if args.target_dir is not None:
        config.update(
            query_ligand=str(args.target_dir / "crystal_ligand.mol2"),
            actives_dir=str(args.target_dir / "actives_sdf"),
            decoys_dir=str(args.target_dir / "decoys_sdf"),
        )
    overrides = {
        "checkpoint_path": args.checkpoint,
        "query_ligand": args.query_ligand,
        "actives_dir": args.actives_dir,
        "decoys_dir": args.decoys_dir,
        "output_dir": args.output_dir,
        "target_name": args.target_name,
        "device": args.device,
        "maxiter": args.maxiter,
        "popsize": args.popsize,
        "limit": args.limit,
        "num_actives": args.num_actives,
        "num_decoys": args.num_decoys,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    if args.no_optimize:
        config["optimize"] = False
    if "output_dir" not in config:
        config["output_dir"] = str(default_output_dir(runner, config))
    required = ("checkpoint_path", "query_ligand", "actives_dir", "decoys_dir", "output_dir")
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit(f"Missing required settings: {', '.join(missing)}")
    seeds = args.seeds if args.seeds is not None else ([args.seed] if args.seed is not None else SPICE_SEEDS)
    output_root = Path(config["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "run.log"
    with log_path.open("a", encoding="utf-8") as log_handle:
        stdout = Tee(sys.stdout, log_handle)
        stderr = Tee(sys.stderr, log_handle)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print(f"Writing SPICE screening results to {output_root}")
            print(f"Logging terminal output to {log_path}")
            try:
                result = run_seeded(runner, config, seeds)
            except Exception:
                traceback.print_exc()
                raise
            print(json.dumps(result, indent=2, sort_keys=True))
