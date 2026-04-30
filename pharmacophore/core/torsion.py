"""Black-box torsion optimization for pharmacophore screening.

The public API is `optimize_torsions()`. Helper functions are kept internal so
downstream code does not need to manage torsion definitions directly.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdMolTransforms
from scipy.optimize import differential_evolution


def _extract_torsions(
    mol: Chem.Mol,
    *,
    unique: bool = True,
    heavy_only: bool = True,
    rotatable_only: bool = True,
    exclude_rings: bool = True,
    one_per_bond: bool = False,
) -> list[dict]:
    """Extract torsion definitions from an RDKit molecule with 3D coordinates."""
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no 3D coordinates")

    conf = mol.GetConformer()
    rot_bonds = None
    if rotatable_only:
        rot_bonds = set(
            tuple(sorted(bond))
            for bond in rdMolDescriptors.CalcRotatableBonds(
                mol,
                strict=True,
                includeTerminalBonds=False,
                returnBonds=True,
            )
        )

    torsions = []
    seen = set()
    for bond in mol.GetBonds():
        j = bond.GetBeginAtomIdx()
        k = bond.GetEndAtomIdx()
        if exclude_rings and bond.IsInRing():
            continue
        if rotatable_only and tuple(sorted((j, k))) not in rot_bonds:
            continue

        nbrs_j = [atom.GetIdx() for atom in mol.GetAtomWithIdx(j).GetNeighbors() if atom.GetIdx() != k]
        nbrs_k = [atom.GetIdx() for atom in mol.GetAtomWithIdx(k).GetNeighbors() if atom.GetIdx() != j]
        if heavy_only:
            nbrs_j = [i for i in nbrs_j if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
            nbrs_k = [l for l in nbrs_k if mol.GetAtomWithIdx(l).GetAtomicNum() > 1]
        if not nbrs_j or not nbrs_k:
            continue

        pairs = [(min(nbrs_j), min(nbrs_k))] if one_per_bond else [(i, l) for i in nbrs_j for l in nbrs_k if i != l]
        for i, l in pairs:
            atoms = (int(i), int(j), int(k), int(l))
            key = atoms if atoms < (atoms[3], atoms[2], atoms[1], atoms[0]) else (atoms[3], atoms[2], atoms[1], atoms[0])
            if unique and key in seen:
                continue
            seen.add(key)
            torsions.append(
                {
                    "atoms": atoms,
                    "bond": (int(j), int(k)),
                    "angle_deg": float(rdMolTransforms.GetDihedralDeg(conf, *atoms)),
                }
            )
    return torsions


def _set_torsion_angle(
    mol: Chem.Mol,
    i: int,
    j: int,
    k: int,
    l: int,
    new_angle: float,
    conf_id: int = 0,
    *,
    copy: bool = True,
    require_bond_jk: bool = True,
) -> Chem.Mol:
    """Set a dihedral angle in degrees on an RDKit conformer."""
    updated = Chem.Mol(mol) if copy else mol
    if require_bond_jk and updated.GetBondBetweenAtoms(int(j), int(k)) is None:
        raise ValueError(f"No bond between torsion atoms j={j} and k={k}")
    angle = float(new_angle)
    while angle <= -180.0:
        angle += 360.0
    while angle > 180.0:
        angle -= 360.0
    rdMolTransforms.SetDihedralDeg(updated.GetConformer(conf_id), int(i), int(j), int(k), int(l), angle)
    return updated


def _apply_torsion_vector(mol: Chem.Mol, torsions: list[dict], theta_vec) -> Chem.Mol:
    updated = Chem.Mol(mol)
    for torsion, theta in zip(torsions, np.mod(theta_vec, 360.0)):
        updated = _set_torsion_angle(updated, *torsion["atoms"], float(theta), conf_id=0)
    return updated


def optimize_torsions(
    mol: Chem.Mol,
    objective: Callable[[Chem.Mol], float],
    *,
    maxiter: int = 3,
    popsize: int = 2,
    rotatable_only: bool = False,
    heavy_only: bool = True,
    exclude_rings: bool = True,
    one_per_bond: bool = False,
    seed: int = 0,
):
    """Optimize molecular torsions to maximize an objective function.

    The objective receives an RDKit molecule and returns a scalar score.
    Torsion extraction, angle assignment, and differential evolution are handled
    internally.
    """
    torsions = _extract_torsions(
        mol,
        rotatable_only=rotatable_only,
        heavy_only=heavy_only,
        exclude_rings=exclude_rings,
        one_per_bond=one_per_bond,
    )
    if not torsions:
        score = objective(mol)
        return mol, score, {"torsion_count": 0, "theta": []}

    def minimize(theta_vec):
        return -objective(_apply_torsion_vector(mol, torsions, theta_vec))

    result = differential_evolution(
        minimize,
        bounds=[(0.0, 360.0)] * len(torsions),
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-3,
        polish=True,
        updating="deferred",
        workers=1,
        seed=seed,
    )
    theta = np.mod(result.x, 360.0)
    return _apply_torsion_vector(mol, torsions, theta), float(-result.fun), {
        "torsion_count": len(torsions),
        "theta": theta.tolist(),
    }
