"""Plain Equiformer with optimization wrapper."""

from __future__ import annotations

try:
    from ..core.screening import screen_actives_decoys
except ImportError:
    from pharmacophore.core.screening import screen_actives_decoys


def run_equiformer_optimization_screening(**kwargs):
    """Run plain Equiformer screening with optional torsion optimization."""
    kwargs.setdefault("model_module", "benchmarking.Methods.equiformer_architecture")
    kwargs.setdefault("model_class", "EquiformerQM9")
    kwargs.setdefault("pipeline_name", "Equiformer_with_optimization")
    kwargs.setdefault("use_pharmacophore_features", False)
    kwargs.setdefault("rotatable_only", False)
    kwargs.setdefault("heavy_only", True)
    kwargs.setdefault("exclude_rings", True)
    kwargs.setdefault("one_per_bond", False)
    return screen_actives_decoys(**kwargs)
