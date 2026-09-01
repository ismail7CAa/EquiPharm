"""SPICE-backed EquiPharm_Hungarian_v2 screening wrapper."""

from ..common import run_matching


def run_equipharm_hungarian_v2_spice_screening(**kwargs):
    return run_matching("EquiPharm_Hungarian_v2", "hungarian_euclidean", "embedding_geometry_distance", **kwargs)

