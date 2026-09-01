"""SPICE-backed EquiPharm_Hungarian screening wrapper."""

from ..common import run_matching


def run_equipharm_hungarian_spice_screening(**kwargs):
    return run_matching("EquiPharm_Hungarian", "hungarian_euclidean", "embedding_distance", **kwargs)

