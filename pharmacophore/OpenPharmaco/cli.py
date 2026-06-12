#!/usr/bin/env python
"""CLI for the optional OpenPharmaco baseline adapter."""

from __future__ import annotations

try:
    from .screening import run_openpharmaco_dataset_screening, run_openpharmaco_screening
    from ..core.command_baseline_cli import run_command_baseline_cli
except ImportError:
    from screening import run_openpharmaco_dataset_screening, run_openpharmaco_screening
    from pharmacophore.core.command_baseline_cli import run_command_baseline_cli


def main() -> None:
    run_command_baseline_cli(
        description="Run OpenPharmaco through a command-template adapter.",
        run_screening=run_openpharmaco_screening,
        run_dataset_screening=run_openpharmaco_dataset_screening,
    )


if __name__ == "__main__":
    main()
