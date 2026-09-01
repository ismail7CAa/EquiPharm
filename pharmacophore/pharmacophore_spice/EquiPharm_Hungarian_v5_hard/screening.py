"""SPICE-backed EquiPharm_Hungarian_v5_hard screening wrapper."""

from ..common import run_matching


def run_equipharm_hungarian_v5_hard_spice_screening(**kwargs):
    kwargs.setdefault("embedding_weight", 0.4)
    kwargs.setdefault("spatial_weight", 0.6)
    kwargs.setdefault("spatial_tau", 2)
    kwargs.setdefault("geometry_penalty_weight", 0.3)
    kwargs.setdefault("require_full_query_coverage", True)
    kwargs.setdefault("enforce_feature_family", True)
    return run_matching("EquiPharm_Hungarian_v5_hard", "hungarian_cosine_quality", "hybrid_local_geometry", **kwargs)

