"""EquiPharm Hungarian feature-matching screening wrapper."""

from __future__ import annotations

try:
    from ..core.matching_screening import screen_actives_decoys_matching
except ImportError:
    from pharmacophore.core.matching_screening import screen_actives_decoys_matching


def run_equipharm_hungarian_screening(**kwargs):
    """Run pharmacophore-feature Equiformer screening with Hungarian matching."""
    kwargs.setdefault("model_module", "benchmarking.Methods.equiformer_encoder_hungarian")
    kwargs.setdefault("model_class", "EquiformerQM9")
    kwargs.setdefault("pipeline_name", "EquiPharm_Hungarian")
    kwargs.setdefault("matching_method", "hungarian")
    kwargs.setdefault("rotatable_only", False)
    kwargs.setdefault("heavy_only", True)
    kwargs.setdefault("exclude_rings", True)
    kwargs.setdefault("one_per_bond", False)
    return screen_actives_decoys_matching(**kwargs)
