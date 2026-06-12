#!/usr/bin/env python
"""CLI for the optional Schrodinger Phase baseline adapter."""

from __future__ import annotations

try:
    from .screening import run_schrodinger_phase_dataset_screening, run_schrodinger_phase_screening
    from ..core.command_baseline_cli import run_command_baseline_cli
except ImportError:
    from screening import run_schrodinger_phase_dataset_screening, run_schrodinger_phase_screening
    from pharmacophore.core.command_baseline_cli import run_command_baseline_cli


def main() -> None:
    run_command_baseline_cli(
        description="Run Schrodinger Phase through a command-template adapter.",
        run_screening=run_schrodinger_phase_screening,
        run_dataset_screening=run_schrodinger_phase_dataset_screening,
    )


if __name__ == "__main__":
    main()
