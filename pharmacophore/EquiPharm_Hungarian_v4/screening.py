"""EquiPharm Hungarian v4 tiered scoring wrapper."""

from __future__ import annotations

try:
    from ..core.matching_screening import screen_actives_decoys_matching
except ImportError:
    from pharmacophore.core.matching_screening import screen_actives_decoys_matching


def run_equipharm_hungarian_v4_screening(**kwargs):
    """Run embedding-distance assignment with tiered quality and geometry scoring."""
    kwargs.setdefault("model_module", "benchmarking.Methods.equiformer_encoder_matching")
    kwargs.setdefault("model_class", "EquiformerQM9")
    kwargs.setdefault("pipeline_name", "EquiPharm_Hungarian_v4")
    kwargs.setdefault("matching_method", "hungarian_gaussian")
    kwargs.setdefault("matching_score_mode", "tiered_distance_geometry")
    kwargs.setdefault("distance_sigma", 1.0)
    kwargs.setdefault("geometry_penalty_weight", 1.0)
    kwargs.setdefault("enforce_feature_family", True)
    kwargs.setdefault("rotatable_only", False)
    kwargs.setdefault("heavy_only", True)
    kwargs.setdefault("exclude_rings", True)
    kwargs.setdefault("one_per_bond", False)
    return screen_actives_decoys_matching(**kwargs)
