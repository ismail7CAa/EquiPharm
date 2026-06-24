#!/usr/bin/env python
"""CLI for EquiPharm Hungarian v5 hard coverage."""

from .screening import run_equipharm_hungarian_v5_hard_screening
from ..core.v5_cli import run_v5_cli


if __name__ == "__main__":
    run_v5_cli(run_equipharm_hungarian_v5_hard_screening, variant="hard coverage")
