"""SPICE-backed EquiPharm_Hungarian_v4 screening wrapper."""

from ..common import run_matching


def run_equipharm_hungarian_v4_spice_screening(**kwargs):
    kwargs.setdefault("distance_sigma", 1)
    kwargs.setdefault("geometry_penalty_weight", 1)
    kwargs.setdefault("enforce_feature_family", True)
    return run_matching("EquiPharm_Hungarian_v4", "hungarian_gaussian", "tiered_distance_geometry", **kwargs)

