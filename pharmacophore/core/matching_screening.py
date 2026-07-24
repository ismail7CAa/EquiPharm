"""Reusable screening workflow for feature-level EquiPharm matching variants."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Batch
from tqdm import tqdm

try:
    from .artifacts import initialize_artifacts, save_failure_artifact, save_molecule_artifact, save_query_artifact
    from .matching import matching_score
    from .metrics import write_outputs
    from .model_loading import model_kwargs_from_checkpoint
    from .molecule_io import (
        prepare_mol_for_pharmacophore,
        read_query_ligand,
        rdkit_mol_to_pyg_equiformer,
        read_sdf_mol,
    )
    from .resume import append_score_row, initialize_score_file, load_resume_rows
    from .screening import import_model_class, infer_target_name
    from .torsion import optimize_torsions
except ImportError:
    from pharmacophore.core.artifacts import initialize_artifacts, save_failure_artifact, save_molecule_artifact, save_query_artifact
    from pharmacophore.core.matching import matching_score
    from pharmacophore.core.metrics import write_outputs
    from pharmacophore.core.model_loading import model_kwargs_from_checkpoint
    from pharmacophore.core.molecule_io import (
        prepare_mol_for_pharmacophore,
        read_query_ligand,
        rdkit_mol_to_pyg_equiformer,
        read_sdf_mol,
    )
    from pharmacophore.core.resume import append_score_row, initialize_score_file, load_resume_rows
    from pharmacophore.core.screening import import_model_class, infer_target_name
    from pharmacophore.core.torsion import optimize_torsions


def load_matching_model(
    *,
    checkpoint_path: str | Path,
    device: torch.device | str,
    model_module: str,
    model_class: str,
):
    model_type = import_model_class(model_module, model_class)
    # Training checkpoints contain optimizer/NumPy metadata in addition to tensors.
    # PyTorch 2.6 changed the default to weights_only=True, which rejects these
    # trusted, locally produced checkpoints.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_kwargs = model_kwargs_from_checkpoint(model_type, checkpoint)
    print(f"Reconstructing {model_class} from checkpoint with {model_kwargs}")
    model = model_type(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if not hasattr(model, "encode_pharmacophore_features"):
        raise AttributeError("Matching model must expose encode_pharmacophore_features().")
    return model


def mol_to_matching_data(model, mol: Chem.Mol, *, name: str, idx: int):
    if not hasattr(model, "pharmaco_features"):
        raise AttributeError("Model does not expose pharmaco_features().")

    mol = prepare_mol_for_pharmacophore(mol)
    features = model.pharmaco_features(mol)
    return rdkit_mol_to_pyg_equiformer(
        mol,
        y=torch.zeros((1, 19), dtype=torch.float32),
        smiles=Chem.MolToSmiles(mol, isomericSmiles=True),
        name=name,
        idx=idx,
        pharmacophore_features=features,
    )


@torch.inference_mode()
def encode_feature_set(model, mol: Chem.Mol, device, *, name: str, idx: int) -> dict:
    data = mol_to_matching_data(model, mol, name=name, idx=idx)
    batch = Batch.from_data_list([data]).to(device)
    return model.encode_pharmacophore_features(batch)


def score_candidate_features(
    *,
    query_encoding: dict,
    candidate_encoding: dict,
    method: str,
    score_mode: str,
    distance_sigma: float = 1.0,
    geometry_penalty_weight: float = 1.0,
    enforce_feature_family: bool = True,
    embedding_weight: float = 0.4,
    spatial_weight: float = 0.6,
    spatial_tau: float = 2.0,
    require_full_query_coverage: bool = False,
) -> tuple[float, str, tuple[int, int], list[dict], dict]:
    query_features = query_encoding["feature_embeddings"]
    candidate_features = candidate_encoding["feature_embeddings"]

    if query_features.size(0) == 0 or candidate_features.size(0) == 0:
        score = F.cosine_similarity(
            query_encoding["global_embedding"].unsqueeze(0),
            candidate_encoding["global_embedding"].unsqueeze(0),
            dim=1,
        )
        fallback_score = float(score.squeeze(0).detach().cpu().item())
        is_tiered_score = score_mode == "tiered_distance_geometry"
        needs_pharmacophore_features = score_mode in {"tiered_distance_geometry", "hybrid_local_geometry"}
        returned_score = 0.0 if needs_pharmacophore_features else fallback_score
        score_source = "no_pharmacophore_features" if needs_pharmacophore_features else "global_fallback"
        return returned_score, score_source, (
            int(query_features.size(0)),
            int(candidate_features.size(0)),
        ), [], {
            "score_mode": "global_fallback",
            "strict_score": fallback_score,
            "balanced_score": fallback_score,
            "feature_distance_score": fallback_score,
            "geometry_distance_score": fallback_score,
            "embedding_distance_score": fallback_score,
            "embedding_geometry_distance_score": fallback_score,
            "matched_cosine_similarity_score": fallback_score,
            "cosine_geometry_score": fallback_score,
            "selected_similarity_total": fallback_score,
            "matched_feature_count": 0,
            "matched_feature_distance_sum": 0.0,
            "matched_feature_distance_count": 0,
            "average_feature_distance": 0.0,
            "geometry_distance_delta_sum": 0.0,
            "geometry_distance_pair_count": 0,
            "average_geometry_distance_delta": 0.0,
            "matched_embedding_distance_sum": 0.0,
            "matched_embedding_distance_count": 0,
            "average_embedding_distance": 0.0,
            "embedding_geometry_delta_sum": 0.0,
            "embedding_geometry_pair_count": 0,
            "average_embedding_geometry_delta": 0.0,
            "matched_cosine_similarity_sum": 0.0,
            "matched_cosine_similarity_count": 0,
            "cosine_geometry_delta_sum": 0.0,
            "cosine_geometry_pair_count": 0,
            "average_cosine_geometry_delta": 0.0,
            "matched_average_similarity": 0.0,
            "query_feature_coverage": 0.0,
            "candidate_feature_coverage": 0.0,
            "tier1_score": 0.0 if is_tiered_score else fallback_score,
            "tier1_distance_quality_sum": 0.0 if is_tiered_score else fallback_score,
            "tier1_distance_sigma": distance_sigma,
            "tier2_geometry_rmse": 0.0,
            "tier2_geometry_pair_count": 0,
            "tier2_geometry_available": False,
            "geometry_penalty_weight": geometry_penalty_weight,
            "geometry_penalty_factor": 1.0,
            "tiered_final_score": 0.0 if is_tiered_score else fallback_score,
            "v5_tier1_score": 0.0,
            "v5_embedding_quality_sum": 0.0,
            "v5_local_spatial_quality_sum": 0.0,
            "v5_hybrid_quality_sum": 0.0,
            "v5_embedding_weight": embedding_weight,
            "v5_spatial_weight": spatial_weight,
            "v5_spatial_tau": spatial_tau,
            "v5_geometry_rmse": 0.0,
            "v5_query_distance_scale": 0.0,
            "v5_normalized_geometry_error": 0.0,
            "v5_geometry_pair_count": 0,
            "v5_geometry_available": False,
            "v5_geometry_penalty_weight": geometry_penalty_weight,
            "v5_geometry_penalty_factor": 1.0,
            "v5_unmatched_query_count": int(query_features.size(0)),
            "v5_unmatched_candidate_count": int(candidate_features.size(0)),
            "v5_full_query_coverage": False,
            "v5_coverage_status": "no_pharmacophore_features",
            "v5_require_full_query_coverage": require_full_query_coverage,
            "v5_rejected_incomplete_coverage": require_full_query_coverage,
            "v5_final_score": 0.0,
        }

    score, similarity, match_details, score_components = matching_score(
        query_features,
        candidate_features,
        query_metadata=query_encoding.get("feature_metadata"),
        candidate_metadata=candidate_encoding.get("feature_metadata"),
        method=method,
        score_mode=score_mode,
        enforce_feature_family=enforce_feature_family,
        distance_sigma=distance_sigma,
        geometry_penalty_weight=geometry_penalty_weight,
        embedding_weight=embedding_weight,
        spatial_weight=spatial_weight,
        spatial_tau=spatial_tau,
        require_full_query_coverage=require_full_query_coverage,
    )
    return score, method, (int(similarity.size(0)), int(similarity.size(1))), match_details, score_components


def screen_actives_decoys_matching(
    *,
    checkpoint_path: str | Path,
    query_ligand: str | Path,
    actives_dir: str | Path,
    decoys_dir: str | Path,
    output_dir: str | Path,
    model_module: str,
    model_class: str = "EquiformerQM9",
    pipeline_name: str = "EquiPharm_Matching",
    matching_method: str,
    matching_score_mode: str = "feature_distance",
    target_name: str | None = None,
    device: str = "cuda",
    optimize: bool = True,
    maxiter: int = 3,
    popsize: int = 2,
    rotatable_only: bool = False,
    heavy_only: bool = True,
    exclude_rings: bool = True,
    one_per_bond: bool = False,
    limit: int | None = None,
    distance_sigma: float = 1.0,
    geometry_penalty_weight: float = 1.0,
    enforce_feature_family: bool = True,
    embedding_weight: float = 0.4,
    spatial_weight: float = 0.6,
    spatial_tau: float = 2.0,
    require_full_query_coverage: bool = False,
) -> dict:
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use a GPU node or pass device='cpu'.")

    target_name = target_name or infer_target_name(query_ligand, actives_dir, decoys_dir, output_dir)
    initialize_score_file(output_dir, matching_score_fieldnames())
    initialize_artifacts(
        output_dir,
        {
            "pipeline_name": pipeline_name,
            "target_name": target_name,
            "checkpoint_path": str(checkpoint_path),
            "query_ligand": str(query_ligand),
            "actives_dir": str(actives_dir),
            "decoys_dir": str(decoys_dir),
            "model_module": model_module,
            "model_class": model_class,
            "matching_method": matching_method,
            "matching_score_mode": matching_score_mode,
            "device": device,
            "optimize": optimize,
            "maxiter": maxiter,
            "popsize": popsize,
            "rotatable_only": rotatable_only,
            "heavy_only": heavy_only,
            "exclude_rings": exclude_rings,
            "one_per_bond": one_per_bond,
            "limit": limit,
            "distance_sigma": distance_sigma,
            "geometry_penalty_weight": geometry_penalty_weight,
            "enforce_feature_family": enforce_feature_family,
            "embedding_weight": embedding_weight,
            "spatial_weight": spatial_weight,
            "spatial_tau": spatial_tau,
            "require_full_query_coverage": require_full_query_coverage,
        },
    )

    model = load_matching_model(
        checkpoint_path=checkpoint_path,
        device=device_obj,
        model_module=model_module,
        model_class=model_class,
    )
    query_mol = read_query_ligand(query_ligand, sanitize=True, keep_hs=False)
    query_encoding = encode_feature_set(model, query_mol, device_obj, name="query_ligand", idx=0)
    save_query_artifact(output_dir, query_ligand=query_ligand, encoding=query_encoding)

    active_files = sorted(Path(actives_dir).glob("*.sdf"))
    decoy_files = sorted(Path(decoys_dir).glob("*.sdf"))
    candidates = [(path, 1) for path in active_files] + [(path, 0) for path in decoy_files]
    if limit is not None:
        candidates = candidates[:limit]

    rows, completed_paths = load_resume_rows(output_dir)
    for idx, (sdf_path, label) in enumerate(tqdm(candidates, desc=f"Screening {pipeline_name}")):
        if str(sdf_path) in completed_paths:
            continue
        tag = "active" if label == 1 else "decoy"
        try:
            mol = read_sdf_mol(sdf_path, sanitize=True, remove_hs=False, require_3d=True)
            mol.SetProp("_Name", f"{tag}_{idx:05d}")
            mol.SetProp("source", "DUD-E_SDF")
            mol.SetProp("sdf_path", str(sdf_path))

            def objective(candidate_mol):
                candidate_encoding = encode_feature_set(
                    model,
                    candidate_mol,
                    device_obj,
                    name=mol.GetProp("_Name"),
                    idx=idx,
                )
                score, score_source, matrix_shape, match_details, score_components = score_candidate_features(
                    query_encoding=query_encoding,
                    candidate_encoding=candidate_encoding,
                    method=matching_method,
                    score_mode=matching_score_mode,
                    distance_sigma=distance_sigma,
                    geometry_penalty_weight=geometry_penalty_weight,
                    enforce_feature_family=enforce_feature_family,
                    embedding_weight=embedding_weight,
                    spatial_weight=spatial_weight,
                    spatial_tau=spatial_tau,
                    require_full_query_coverage=require_full_query_coverage,
                )
                return score, score_source, matrix_shape, match_details, score_components, candidate_encoding

            if optimize:
                def scalar_objective(candidate_mol):
                    score, _, _, _, _, _ = objective(candidate_mol)
                    return score

                optimized_mol, score, opt_meta = optimize_torsions(
                    mol,
                    scalar_objective,
                    maxiter=maxiter,
                    popsize=popsize,
                    rotatable_only=rotatable_only,
                    heavy_only=heavy_only,
                    exclude_rings=exclude_rings,
                    one_per_bond=one_per_bond,
                    seed=idx,
                )
                _, score_source, matrix_shape, match_details, score_components, candidate_encoding = objective(optimized_mol)
            else:
                score, score_source, matrix_shape, match_details, score_components, candidate_encoding = objective(mol)
                opt_meta = {"torsion_count": 0, "theta": []}

            row = {
                "pipeline": pipeline_name,
                "target": target_name,
                "name": mol.GetProp("_Name"),
                "path": str(sdf_path),
                "label": label,
                "score": score,
                "score_source": score_source,
                "query_feature_count": matrix_shape[0],
                "candidate_feature_count": matrix_shape[1],
                "matching_score_mode": score_components.get("score_mode", matching_score_mode),
                "hungarian_score_strict": score_components.get("strict_score", score),
                "hungarian_score_balanced": score_components.get("balanced_score", score),
                "feature_distance_score": score_components.get("feature_distance_score", score),
                "geometry_distance_score": score_components.get("geometry_distance_score", score),
                "embedding_distance_score": score_components.get("embedding_distance_score", score),
                "embedding_geometry_distance_score": score_components.get("embedding_geometry_distance_score", score),
                "matched_cosine_similarity_score": score_components.get("matched_cosine_similarity_score", score),
                "cosine_geometry_score": score_components.get("cosine_geometry_score", score),
                "selected_similarity_total": score_components.get("selected_similarity_total", 0.0),
                "matched_feature_count": score_components.get(
                    "matched_feature_count",
                    sum(1 for match in match_details if match.get("status") == "matched"),
                ),
                "matched_feature_distance_sum": score_components.get("matched_feature_distance_sum", 0.0),
                "matched_feature_distance_count": score_components.get("matched_feature_distance_count", 0),
                "average_feature_distance": score_components.get("average_feature_distance", 0.0),
                "geometry_distance_delta_sum": score_components.get("geometry_distance_delta_sum", 0.0),
                "geometry_distance_pair_count": score_components.get("geometry_distance_pair_count", 0),
                "average_geometry_distance_delta": score_components.get("average_geometry_distance_delta", 0.0),
                "matched_embedding_distance_sum": score_components.get("matched_embedding_distance_sum", 0.0),
                "matched_embedding_distance_count": score_components.get("matched_embedding_distance_count", 0),
                "average_embedding_distance": score_components.get("average_embedding_distance", 0.0),
                "embedding_geometry_delta_sum": score_components.get("embedding_geometry_delta_sum", 0.0),
                "embedding_geometry_pair_count": score_components.get("embedding_geometry_pair_count", 0),
                "average_embedding_geometry_delta": score_components.get("average_embedding_geometry_delta", 0.0),
                "matched_cosine_similarity_sum": score_components.get("matched_cosine_similarity_sum", 0.0),
                "matched_cosine_similarity_count": score_components.get("matched_cosine_similarity_count", 0),
                "cosine_geometry_delta_sum": score_components.get("cosine_geometry_delta_sum", 0.0),
                "cosine_geometry_pair_count": score_components.get("cosine_geometry_pair_count", 0),
                "average_cosine_geometry_delta": score_components.get("average_cosine_geometry_delta", 0.0),
                "matched_average_similarity": score_components.get("matched_average_similarity", 0.0),
                "query_feature_coverage": score_components.get("query_feature_coverage", 0.0),
                "candidate_feature_coverage": score_components.get("candidate_feature_coverage", 0.0),
                "tier1_score": score_components.get("tier1_score", score),
                "tier1_distance_quality_sum": score_components.get("tier1_distance_quality_sum", 0.0),
                "tier1_distance_sigma": score_components.get("tier1_distance_sigma", distance_sigma),
                "tier2_geometry_rmse": score_components.get("tier2_geometry_rmse", 0.0),
                "tier2_geometry_pair_count": score_components.get("tier2_geometry_pair_count", 0),
                "tier2_geometry_available": score_components.get("tier2_geometry_available", False),
                "geometry_penalty_weight": score_components.get("geometry_penalty_weight", geometry_penalty_weight),
                "geometry_penalty_factor": score_components.get("geometry_penalty_factor", 1.0),
                "tiered_final_score": score_components.get("tiered_final_score", score),
                "v5_tier1_score": score_components.get("v5_tier1_score", score),
                "v5_embedding_quality_sum": score_components.get("v5_embedding_quality_sum", 0.0),
                "v5_local_spatial_quality_sum": score_components.get("v5_local_spatial_quality_sum", 0.0),
                "v5_hybrid_quality_sum": score_components.get("v5_hybrid_quality_sum", 0.0),
                "v5_embedding_weight": score_components.get("v5_embedding_weight", embedding_weight),
                "v5_spatial_weight": score_components.get("v5_spatial_weight", spatial_weight),
                "v5_spatial_tau": score_components.get("v5_spatial_tau", spatial_tau),
                "v5_geometry_rmse": score_components.get("v5_geometry_rmse", 0.0),
                "v5_query_distance_scale": score_components.get("v5_query_distance_scale", 0.0),
                "v5_normalized_geometry_error": score_components.get("v5_normalized_geometry_error", 0.0),
                "v5_geometry_pair_count": score_components.get("v5_geometry_pair_count", 0),
                "v5_geometry_available": score_components.get("v5_geometry_available", False),
                "v5_geometry_penalty_weight": score_components.get("v5_geometry_penalty_weight", geometry_penalty_weight),
                "v5_geometry_penalty_factor": score_components.get("v5_geometry_penalty_factor", 1.0),
                "v5_unmatched_query_count": score_components.get("v5_unmatched_query_count", 0),
                "v5_unmatched_candidate_count": score_components.get("v5_unmatched_candidate_count", 0),
                "v5_full_query_coverage": score_components.get("v5_full_query_coverage", False),
                "v5_coverage_status": score_components.get("v5_coverage_status", "unknown"),
                "v5_require_full_query_coverage": score_components.get("v5_require_full_query_coverage", require_full_query_coverage),
                "v5_rejected_incomplete_coverage": score_components.get("v5_rejected_incomplete_coverage", False),
                "v5_final_score": score_components.get("v5_final_score", score),
                "matching_details": json.dumps(match_details, sort_keys=True),
                "torsion_count": opt_meta["torsion_count"],
            }
            save_molecule_artifact(
                output_dir,
                row=row,
                encoding=candidate_encoding,
                opt_meta=opt_meta,
                match_details=match_details,
                score_components=score_components,
            )
        except Exception as exc:
            row = {
                "pipeline": pipeline_name,
                "target": target_name,
                "name": sdf_path.stem,
                "path": str(sdf_path),
                "label": label,
                "score": float("nan"),
                "torsion_count": 0,
                "error": str(exc),
            }
            save_failure_artifact(output_dir, row=row, error=str(exc))
        rows.append(row)
        append_score_row(output_dir, row)

    rows = [row for row in rows if row.get("score") == row.get("score")]
    return write_outputs(
        output_dir,
        rows,
        pipeline_name=pipeline_name,
        target_name=target_name,
        write_named_roc_curve=True,
    )


def matching_score_fieldnames() -> list[str]:
    return [
        "pipeline",
        "target",
        "name",
        "path",
        "label",
        "score",
        "score_source",
        "query_feature_count",
        "candidate_feature_count",
        "matching_score_mode",
        "hungarian_score_strict",
        "hungarian_score_balanced",
        "feature_distance_score",
        "geometry_distance_score",
        "embedding_distance_score",
        "embedding_geometry_distance_score",
        "matched_cosine_similarity_score",
        "cosine_geometry_score",
        "selected_similarity_total",
        "matched_feature_count",
        "matched_feature_distance_sum",
        "matched_feature_distance_count",
        "average_feature_distance",
        "geometry_distance_delta_sum",
        "geometry_distance_pair_count",
        "average_geometry_distance_delta",
        "matched_embedding_distance_sum",
        "matched_embedding_distance_count",
        "average_embedding_distance",
        "embedding_geometry_delta_sum",
        "embedding_geometry_pair_count",
        "average_embedding_geometry_delta",
        "matched_cosine_similarity_sum",
        "matched_cosine_similarity_count",
        "cosine_geometry_delta_sum",
        "cosine_geometry_pair_count",
        "average_cosine_geometry_delta",
        "matched_average_similarity",
        "query_feature_coverage",
        "candidate_feature_coverage",
        "tier1_score",
        "tier1_distance_quality_sum",
        "tier1_distance_sigma",
        "tier2_geometry_rmse",
        "tier2_geometry_pair_count",
        "tier2_geometry_available",
        "geometry_penalty_weight",
        "geometry_penalty_factor",
        "tiered_final_score",
        "v5_tier1_score",
        "v5_embedding_quality_sum",
        "v5_local_spatial_quality_sum",
        "v5_hybrid_quality_sum",
        "v5_embedding_weight",
        "v5_spatial_weight",
        "v5_spatial_tau",
        "v5_geometry_rmse",
        "v5_query_distance_scale",
        "v5_normalized_geometry_error",
        "v5_geometry_pair_count",
        "v5_geometry_available",
        "v5_geometry_penalty_weight",
        "v5_geometry_penalty_factor",
        "v5_unmatched_query_count",
        "v5_unmatched_candidate_count",
        "v5_full_query_coverage",
        "v5_coverage_status",
        "v5_require_full_query_coverage",
        "v5_rejected_incomplete_coverage",
        "v5_final_score",
        "matching_details",
        "torsion_count",
        "error",
    ]
