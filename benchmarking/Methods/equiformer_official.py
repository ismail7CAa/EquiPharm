#!/usr/bin/env python
"""Launch the authors' official Equiformer QM9 implementation reproducibly.

This launcher deliberately executes the official repository instead of
reimplementing the architecture.  The published QM9 setup is single-target, so
presets expand to independent target/seed runs rather than a multi-output model.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


OFFICIAL_REPOSITORY = "https://github.com/atomicarchitects/equiformer"

QM9_TARGETS = {
    "dipole_moment": 0,
    "polarizability": 1,
    "homo": 2,
    "lumo": 3,
    "homo_lumo_gap": 4,
    "electronic_spatial_extent": 5,
    "zpve": 6,
    "u0": 7,
    "u": 8,
    "h": 9,
    "g": 10,
    "heat_capacity": 11,
}

TARGET_PRESETS = {
    "electronic": [
        "dipole_moment",
        "polarizability",
        "homo",
        "lumo",
        "homo_lumo_gap",
        "electronic_spatial_extent",
    ],
    "official_all": list(QM9_TARGETS),
}


@dataclass(frozen=True)
class AuthorConfig:
    model_name: str
    batch_size: int
    epochs: int
    num_basis: int
    weight_decay: float
    lr: float
    alpha_description: str
    distributed_gpus: int
    standardize: bool


def author_config(target_index: int) -> AuthorConfig:
    """Return the target-specific settings from the official training scripts."""
    if target_index <= 4 or target_index == 11:
        return AuthorConfig(
            model_name="graph_attention_transformer_nonlinear_l2",
            batch_size=128,
            epochs=300,
            num_basis=128,
            weight_decay=5e-3,
            lr=5e-4,
            alpha_description="Gaussian basis, attention dropout 0.2",
            distributed_gpus=1,
            standardize=True,
        )
    if target_index == 5:
        return AuthorConfig(
            model_name="graph_attention_transformer_nonlinear_bessel_l2_drop01",
            batch_size=128,
            epochs=300,
            num_basis=8,
            weight_decay=5e-3,
            lr=5e-4,
            alpha_description="Bessel basis, attention dropout 0.1",
            distributed_gpus=1,
            standardize=True,
        )
    if target_index == 6:
        return AuthorConfig(
            model_name="graph_attention_transformer_nonlinear_bessel_l2",
            batch_size=128,
            epochs=300,
            num_basis=8,
            weight_decay=5e-3,
            lr=5e-4,
            alpha_description="Bessel basis, attention dropout 0.2",
            distributed_gpus=1,
            standardize=True,
        )
    return AuthorConfig(
        model_name="graph_attention_transformer_nonlinear_bessel_l2_drop00",
        batch_size=32,
        epochs=600,
        num_basis=8,
        weight_decay=0.0,
        lr=1.5e-4,
        alpha_description="Bessel basis, attention dropout 0.0",
        distributed_gpus=2,
        standardize=False,
    )


def resolve_targets(args: argparse.Namespace) -> list[str]:
    if args.target:
        return [args.target]
    return TARGET_PRESETS[args.target_preset]


def build_command(
    *,
    repo: Path,
    data_path: Path,
    output_dir: Path,
    target_name: str,
    seed: int,
) -> list[str]:
    target_index = QM9_TARGETS[target_name]
    config = author_config(target_index)
    prefix = [sys.executable]
    if config.distributed_gpus == 2:
        prefix += ["-m", "torch.distributed.launch", "--nproc_per_node=2", "--use_env"]

    command = prefix + [
        str(repo / "main_qm9.py"),
        "--output-dir", str(output_dir / target_name / f"seed_{seed}"),
        "--model-name", config.model_name,
        "--input-irreps", "5x0e",
        "--target", str(target_index),
        "--data-path", str(data_path),
        "--feature-type", "one_hot",
        "--batch-size", str(config.batch_size),
        "--radius", "5.0",
        "--num-basis", str(config.num_basis),
        "--drop-path", "0.0",
        "--weight-decay", str(config.weight_decay),
        "--lr", str(config.lr),
        "--epochs", str(config.epochs),
        "--min-lr", "1e-6",
        "--seed", str(seed),
        "--no-model-ema",
        "--no-amp",
    ]
    if not config.standardize:
        command.append("--no-standardize")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the authors' official single-target Equiformer QM9 experiments."
    )
    parser.add_argument(
        "--official-repo",
        type=Path,
        default=Path("external/equiformer_official"),
        help="Clone of the official atomicarchitects/equiformer repository.",
    )
    parser.add_argument("--data-path", type=Path, default=Path("data/QM9_official"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/EquiformerOfficial"))
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--target", choices=QM9_TARGETS)
    targets.add_argument(
        "--target-preset",
        choices=TARGET_PRESETS,
        default="official_all",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without importing or running the official implementation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.official_repo.resolve()
    data_path = args.data_path.resolve()
    output_dir = args.output_dir.resolve()

    if not args.dry_run and not (repo / "main_qm9.py").is_file():
        raise FileNotFoundError(
            f"Official Equiformer checkout not found at {repo}. Clone {OFFICIAL_REPOSITORY} "
            f"to that path or pass --official-repo."
        )

    for target_name in resolve_targets(args):
        config = author_config(QM9_TARGETS[target_name])
        print(f"\n### {target_name}: {config.alpha_description}", flush=True)
        for seed in args.seeds:
            command = build_command(
                repo=repo,
                data_path=data_path,
                output_dir=output_dir,
                target_name=target_name,
                seed=seed,
            )
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=repo, check=True)


if __name__ == "__main__":
    main()
