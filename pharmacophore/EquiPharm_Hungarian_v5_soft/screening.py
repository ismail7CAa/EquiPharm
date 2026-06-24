"""EquiPharm Hungarian v5 soft-coverage wrapper."""

from __future__ import annotations

try:
    from ..core.matching_screening import screen_actives_decoys_matching
except ImportError:
    from pharmacophore.core.matching_screening import screen_actives_decoys_matching


def run_equipharm_hungarian_v5_soft_screening(**kwargs):
    """Run V5 while assigning zero quality to unmatched query features."""
    kwargs.setdefault("model_module", "benchmarking.Methods.equiformer_encoder_matching")
    kwargs.setdefault("model_class", "EquiformerQM9")
    kwargs.setdefault("pipeline_name", "EquiPharm_Hungarian_v5_soft")
    kwargs.setdefault("matching_method", "hungarian_cosine_quality")
    kwargs.setdefault("matching_score_mode", "hybrid_local_geometry")
    kwargs.setdefault("embedding_weight", 0.4)
    kwargs.setdefault("spatial_weight", 0.6)
    kwargs.setdefault("spatial_tau", 2.0)
    kwargs.setdefault("geometry_penalty_weight", 0.3)
    kwargs.setdefault("require_full_query_coverage", False)
    kwargs.setdefault("enforce_feature_family", True)
    kwargs.setdefault("rotatable_only", False)
    kwargs.setdefault("heavy_only", True)
    kwargs.setdefault("exclude_rings", True)
    kwargs.setdefault("one_per_bond", False)
    return screen_actives_decoys_matching(**kwargs)
