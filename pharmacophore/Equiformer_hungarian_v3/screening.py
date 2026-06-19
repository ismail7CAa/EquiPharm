"""Equiformer Hungarian v3 hybrid feature-matching screening wrapper."""

from __future__ import annotations

try:
    from ..core.matching_screening import screen_actives_decoys_matching
except ImportError:
    from pharmacophore.core.matching_screening import screen_actives_decoys_matching


def run_equiformer_hungarian_v3_screening(**kwargs):
    """Run Euclidean embedding Hungarian assignment with 3D geometry scoring."""
    kwargs.setdefault("model_module", "benchmarking.Methods.equiformer_encoder_matching")
    kwargs.setdefault("model_class", "EquiformerQM9")
    kwargs.setdefault("pipeline_name", "Equiformer_hungarian_v3")
    kwargs.setdefault("matching_method", "hungarian_euclidean")
    kwargs.setdefault("matching_score_mode", "geometry_distance")
    kwargs.setdefault("rotatable_only", False)
    kwargs.setdefault("heavy_only", True)
    kwargs.setdefault("exclude_rings", True)
    kwargs.setdefault("one_per_bond", False)
    return screen_actives_decoys_matching(**kwargs)
