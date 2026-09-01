"""SPICE-backed EquiPharm_Hungarian_Cosine screening wrapper."""

from ..common import run_matching


def run_equipharm_hungarian_cosine_spice_screening(**kwargs):
    return run_matching("EquiPharm_Hungarian_Cosine", "hungarian", "cosine", **kwargs)

