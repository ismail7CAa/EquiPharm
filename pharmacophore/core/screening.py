"""Reusable virtual screening workflow for EquiPharm and Equiformer baselines."""

from __future__ import annotations

import importlib
from pathlib import Path

import torch
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Batch
from tqdm import tqdm

try:
    from .metrics import write_outputs
    from .molecule_io import (
        mol2_to_rdkit_mol,
        prepare_mol_for_pharmacophore,
        rdkit_mol_to_pyg_equiformer,
        read_sdf_mol,
    )
    from .torsion import optimize_torsions
except ImportError:
    from pharmacophore.core.metrics import write_outputs
    from pharmacophore.core.molecule_io import (
        mol2_to_rdkit_mol,
        prepare_mol_for_pharmacophore,
        rdkit_mol_to_pyg_equiformer,
        read_sdf_mol,
    )
    from pharmacophore.core.torsion import optimize_torsions


def import_model_class(model_module: str, model_class: str):
    module = importlib.import_module(model_module)
    return getattr(module, model_class)


def load_model(
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
    return model


def mol_to_data(
    model,
    mol: Chem.Mol,
    *,
    name: str,
    idx: int,
    use_pharmacophore_features: bool,
):
    features = None
    if use_pharmacophore_features:
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
def encode_molecule(
    model,
    mol: Chem.Mol,
    device,
    *,
    name: str,
    idx: int,
    use_pharmacophore_features: bool,
):
    data = mol_to_data(
        model,
        mol,
        name=name,
        idx=idx,
        use_pharmacophore_features=use_pharmacophore_features,
    )
    batch = Batch.from_data_list([data]).to(device)
    if hasattr(model, "encode"):
        return model.encode(batch)
    return model(batch)


def cosine_similarity(
    model,
    query_embedding,
    mol: Chem.Mol,
    device,
    *,
    name: str,
    idx: int,
    use_pharmacophore_features: bool,
) -> float:
    embedding = encode_molecule(
        model,
        mol,
        device,
        name=name,
        idx=idx,
        use_pharmacophore_features=use_pharmacophore_features,
    )
    score = F.cosine_similarity(query_embedding.to(device), embedding, dim=1)
    return float(score.squeeze(0).detach().cpu().item())


def screen_actives_decoys(
    *,
    checkpoint_path: str | Path,
    query_ligand: str | Path,
    actives_dir: str | Path,
    decoys_dir: str | Path,
    output_dir: str | Path,
    model_module: str,
    model_class: str = "EquiformerQM9",
    pipeline_name: str = "screening",
    target_name: str | None = None,
    use_pharmacophore_features: bool = False,
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

    model = load_model(
        checkpoint_path=checkpoint_path,
        device=device_obj,
        model_module=model_module,
        model_class=model_class,
    )
    query_mol = mol2_to_rdkit_mol(query_ligand, sanitize=True, keep_hs=False)
    query_embedding = encode_molecule(
        model,
        query_mol,
        device_obj,
        name="query_ligand",
        idx=0,
        use_pharmacophore_features=use_pharmacophore_features,
    )

    active_files = sorted(Path(actives_dir).glob("*.sdf"))
    decoy_files = sorted(Path(decoys_dir).glob("*.sdf"))
    candidates = [(path, 1) for path in active_files] + [(path, 0) for path in decoy_files]
    if limit is not None:
        candidates = candidates[:limit]

    rows = []
    for idx, (sdf_path, label) in enumerate(tqdm(candidates, desc="Screening")):
        tag = "active" if label == 1 else "decoy"
        try:
            mol = read_sdf_mol(sdf_path, sanitize=True, remove_hs=False, require_3d=True)
            mol.SetProp("_Name", f"{tag}_{idx:05d}")
            mol.SetProp("source", "DUD-E_SDF")
            mol.SetProp("sdf_path", str(sdf_path))

            def objective(candidate_mol):
                return cosine_similarity(
                    model,
                    query_embedding,
                    candidate_mol,
                    device_obj,
                    name=mol.GetProp("_Name"),
                    idx=idx,
                    use_pharmacophore_features=use_pharmacophore_features,
                )

            if optimize:
                _, score, opt_meta = optimize_torsions(
                    mol,
                    objective,
                    maxiter=maxiter,
                    popsize=popsize,
                    rotatable_only=rotatable_only,
                    heavy_only=heavy_only,
                    exclude_rings=exclude_rings,
                    one_per_bond=one_per_bond,
                    seed=idx,
                )
            else:
                score = objective(mol)
                opt_meta = {"torsion_count": 0, "theta": []}

            rows.append(
                {
                    "pipeline": pipeline_name,
                    "target": target_name,
                    "name": mol.GetProp("_Name"),
                    "path": str(sdf_path),
                    "label": label,
                    "score": score,
                    "torsion_count": opt_meta["torsion_count"],
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "pipeline": pipeline_name,
                    "target": target_name,
                    "name": sdf_path.stem,
                    "path": str(sdf_path),
                    "label": label,
                    "score": float("nan"),
                    "torsion_count": 0,
                    "error": str(exc),
                }
            )

    rows = [row for row in rows if row.get("score") == row.get("score")]
    return write_outputs(
        output_dir,
        rows,
        pipeline_name=pipeline_name,
        target_name=target_name,
    )


def infer_target_name(*paths) -> str:
    """Infer a target name from common DUD-E-style paths."""
    for raw_path in paths:
        path = Path(raw_path)
        parts = path.parts
        if "DUD-E" in parts:
            idx = parts.index("DUD-E")
            if idx + 1 < len(parts):
                return parts[idx + 1]
    output_name = Path(paths[-1]).name
    return output_name or "unknown_target"
