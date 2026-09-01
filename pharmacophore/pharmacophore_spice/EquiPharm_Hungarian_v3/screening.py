"""SPICE-backed EquiPharm_Hungarian_v3 screening wrapper."""

from ..common import run_matching


def run_equipharm_hungarian_v3_spice_screening(**kwargs):
    return run_matching("EquiPharm_Hungarian_v3", "hungarian_euclidean", "geometry_distance", **kwargs)

