"""EquiPharm screening wrapper."""

from __future__ import annotations

try:
    from ..core.screening import screen_actives_decoys
except ImportError:
    from pharmacophore.core.screening import screen_actives_decoys


def run_equipharm_screening(**kwargs):
    """Run pharmacophore-feature-aware Equiformer screening."""
    kwargs.setdefault("model_module", "benchmarking.Methods.equiformer_encoder_pharmaco_feat")
    kwargs.setdefault("model_class", "EquiformerQM9")
    kwargs.setdefault("pipeline_name", "EquiPharm")
    kwargs.setdefault("use_pharmacophore_features", True)
    kwargs.setdefault("rotatable_only", False)
    kwargs.setdefault("heavy_only", True)
    kwargs.setdefault("exclude_rings", True)
    kwargs.setdefault("one_per_bond", False)
    return screen_actives_decoys(**kwargs)
