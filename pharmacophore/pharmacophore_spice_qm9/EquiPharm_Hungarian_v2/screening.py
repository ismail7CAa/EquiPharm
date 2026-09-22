"""Hungarian v2 screening with a SPICE->QM9 fine-tuned encoder."""

from ..common import run_matching


def run_equipharm_hungarian_v2_spice_qm9_screening(**kwargs):
    return run_matching(**kwargs)
