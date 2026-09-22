#!/usr/bin/env python
from .screening import run_equipharm_hungarian_v2_spice_qm9_screening
from ..common import run_cli


if __name__ == "__main__":
    run_cli(
        run_equipharm_hungarian_v2_spice_qm9_screening,
        "Run Hungarian v2 screening with a SPICE->QM9 fine-tuned encoder.",
    )
