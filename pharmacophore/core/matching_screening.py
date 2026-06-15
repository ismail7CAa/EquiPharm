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
    from .matching import matching_score
    from .metrics import write_outputs
    from .molecule_io import (
        prepare_mol_for_pharmacophore,
        read_query_ligand,
        rdkit_mol_to_pyg_equiformer,
        read_sdf_mol,
    )
    from .resume import append_score_row, load_resume_rows
    from .screening import import_model_class, infer_target_name
    from .torsion import optimize_torsions
except ImportError:
    from pharmacophore.core.matching import matching_score
    from pharmacophore.core.metrics import write_outputs
    from pharmacophore.core.molecule_io import (
        prepare_mol_for_pharmacophore,
        read_query_ligand,
        rdkit_mol_to_pyg_equiformer,
        read_sdf_mol,
    )
    from pharmacophore.core.resume import append_score_row, load_resume_rows
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = model_type().to(device)
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
        return fallback_score, "global_fallback", (
            int(query_features.size(0)),
            int(candidate_features.size(0)),
        ), [], {
            "score_mode": "global_fallback",
            "strict_score": fallback_score,
            "balanced_score": fallback_score,
            "feature_distance_score": fallback_score,
            "geometry_distance_score": fallback_score,
            "selected_similarity_total": fallback_score,
            "matched_feature_count": 0,
            "matched_feature_distance_sum": 0.0,
            "matched_feature_distance_count": 0,
            "average_feature_distance": 0.0,
            "geometry_distance_delta_sum": 0.0,
            "geometry_distance_pair_count": 0,
            "average_geometry_distance_delta": 0.0,
            "matched_average_similarity": 0.0,
            "query_feature_coverage": 0.0,
            "candidate_feature_coverage": 0.0,
        }

    score, similarity, match_details, score_components = matching_score(
        query_features,
        candidate_features,
        query_metadata=query_encoding.get("feature_metadata"),
        candidate_metadata=candidate_encoding.get("feature_metadata"),
        method=method,
        score_mode=score_mode,
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
) -> dict:
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use a GPU node or pass device='cpu'.")

    target_name = target_name or infer_target_name(query_ligand, actives_dir, decoys_dir, output_dir)

    model = load_matching_model(
        checkpoint_path=checkpoint_path,
        device=device_obj,
        model_module=model_module,
        model_class=model_class,
    )
    query_mol = read_query_ligand(query_ligand, sanitize=True, keep_hs=False)
    query_encoding = encode_feature_set(model, query_mol, device_obj, name="query_ligand", idx=0)

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
                )
                return score, score_source, matrix_shape, match_details, score_components

            if optimize:
                def scalar_objective(candidate_mol):
                    score, _, _, _, _ = objective(candidate_mol)
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
                _, score_source, matrix_shape, match_details, score_components = objective(optimized_mol)
            else:
                score, score_source, matrix_shape, match_details, score_components = objective(mol)
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
                "matched_average_similarity": score_components.get("matched_average_similarity", 0.0),
                "query_feature_coverage": score_components.get("query_feature_coverage", 0.0),
                "candidate_feature_coverage": score_components.get("candidate_feature_coverage", 0.0),
                "matching_details": json.dumps(match_details, sort_keys=True),
                "torsion_count": opt_meta["torsion_count"],
            }
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
