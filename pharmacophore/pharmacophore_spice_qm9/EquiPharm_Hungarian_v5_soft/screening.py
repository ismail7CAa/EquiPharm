"""V5-soft screening with a SPICE->QM9 fine-tuned encoder."""

from ..common import run_matching


def run_equipharm_hungarian_v5_soft_spice_qm9_screening(**kwargs):
    return run_matching(**kwargs)
