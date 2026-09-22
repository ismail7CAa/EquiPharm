"""Shared configuration for isolated SPICE->QM9 screening."""

from pharmacophore.pharmacophore_spice.common import run_cli as _run_cli
from pharmacophore.core.matching_screening import screen_actives_decoys_matching


MODEL = {
    "model_module": "pharm_training.equiformer_encoder_spice_qm9_pharmaco",
    "model_class": "SPICEQM9PharmacophoreEncoder",
}


def run_matching(**kwargs):
    for key, value in MODEL.items():
        kwargs.setdefault(key, value)
    kwargs.setdefault("pipeline_name", "EquiPharm_Hungarian_v5_soft_SPICE_QM9")
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


def run_cli(runner, description):
    return _run_cli(runner, description, result_family="pharmacophore_spice_qm9")
