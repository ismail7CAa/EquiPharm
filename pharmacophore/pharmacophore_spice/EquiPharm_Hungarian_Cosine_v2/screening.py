"""SPICE-backed EquiPharm_Hungarian_Cosine_v2 screening wrapper."""

from ..common import run_matching


def run_equipharm_hungarian_cosine_v2_spice_screening(**kwargs):
    return run_matching("EquiPharm_Hungarian_Cosine_v2", "hungarian", "cosine_geometry", **kwargs)

