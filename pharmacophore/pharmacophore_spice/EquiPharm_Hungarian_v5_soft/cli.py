#!/usr/bin/env python
"""CLI for the SPICE-backed EquiPharm_Hungarian_v5_soft pipeline."""

from .screening import run_equipharm_hungarian_v5_soft_spice_screening
from ..common import run_cli


if __name__ == "__main__":
    run_cli(run_equipharm_hungarian_v5_soft_spice_screening, "Run EquiPharm_Hungarian_v5_soft with a SPICE-pretrained encoder.")

