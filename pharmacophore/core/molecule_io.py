"""Molecule readers and RDKit-to-PyG conversion helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Geometry import Point3D
from torch_geometric.data import Data


MOL2_BOND_TYPES = {
    "1": rdchem.BondType.SINGLE,
    "2": rdchem.BondType.DOUBLE,
    "3": rdchem.BondType.TRIPLE,
    "am": rdchem.BondType.SINGLE,
    "ar": rdchem.BondType.AROMATIC,
    "du": rdchem.BondType.SINGLE,
    "un": rdchem.BondType.SINGLE,
}


def parse_mol2_atoms_bonds(mol2_path: str | Path):
    atoms, bonds = [], []
    in_atoms = False
    in_bonds = False

    with Path(mol2_path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("@<TRIPOS>"):
                in_atoms = line == "@<TRIPOS>ATOM"
                in_bonds = line == "@<TRIPOS>BOND"
                continue

            if in_atoms:
                parts = line.split()
                if len(parts) < 6:
                    continue

                atoms.append(
                    {
                        "idx": int(parts[0]),
                        "name": parts[1],
                        "x": float(parts[2]),
                        "y": float(parts[3]),
                        "z": float(parts[4]),
                        "sybyl_type": parts[5],
                        "charge": float(parts[-1]) if len(parts) >= 9 else None,
                    }
                )
            elif in_bonds:
                parts = line.split()
                if len(parts) < 4:
                    continue
                bonds.append({"a1": int(parts[1]), "a2": int(parts[2]), "btype": parts[3]})

    if not atoms:
        raise ValueError(f"No MOL2 atoms parsed from {mol2_path}")
    return atoms, bonds


def element_from_sybyl(sybyl_type: str, atom_name: str) -> str:
    base = sybyl_type.split(".", 1)[0].strip()
    return base if base else atom_name[:2].strip().title()


def mol2_to_rdkit_mol(
    mol2_path: str | Path,
    *,
    sanitize: bool = True,
    keep_hs: bool = True,
    store_charges: bool = True,
) -> Chem.Mol:
    """Build an RDKit molecule with a 3D conformer from a Tripos MOL2 file."""
    atoms, bonds = parse_mol2_atoms_bonds(mol2_path)
    id_to_rd_idx = {}
    rw = Chem.RWMol()

    for atom_info in atoms:
        atom = Chem.Atom(element_from_sybyl(atom_info["sybyl_type"], atom_info["name"]))
        atom.SetProp("mol2_name", atom_info["name"])
        atom.SetProp("mol2_type", atom_info["sybyl_type"])
        if store_charges and atom_info["charge"] is not None:
            atom.SetDoubleProp("mol2_charge", float(atom_info["charge"]))
        id_to_rd_idx[atom_info["idx"]] = rw.AddAtom(atom)

    for bond_info in bonds:
        i = id_to_rd_idx[bond_info["a1"]]
        j = id_to_rd_idx[bond_info["a2"]]
        btype = bond_info["btype"].lower()
        rw.AddBond(i, j, MOL2_BOND_TYPES.get(btype, rdchem.BondType.SINGLE))
        if btype == "ar":
            rw.GetBondBetweenAtoms(i, j).SetIsAromatic(True)
            rw.GetAtomWithIdx(i).SetIsAromatic(True)
            rw.GetAtomWithIdx(j).SetIsAromatic(True)

    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom_info in atoms:
        i = id_to_rd_idx[atom_info["idx"]]
        conf.SetAtomPosition(i, Point3D(atom_info["x"], atom_info["y"], atom_info["z"]))

    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)

    if not keep_hs:
        mol = Chem.RemoveHs(mol, sanitize=sanitize)
    if sanitize:
        Chem.SanitizeMol(mol)
    return mol


def read_sdf_mol(
    sdf_path: str | Path,
    *,
    sanitize: bool = True,
    remove_hs: bool = False,
    largest_fragment: bool = True,
    require_3d: bool = True,
    mol_index: int = 0,
) -> Chem.Mol:
    """Read one molecule from an SDF file."""
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=sanitize, removeHs=False)
    if len(supplier) == 0:
        raise ValueError(f"No molecules found in SDF: {sdf_path}")

    mol = supplier[mol_index]
    if mol is None:
        raise ValueError(f"Failed to read molecule {mol_index} from {sdf_path}")

    if largest_fragment:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(frags) > 1:
            mol = max(frags, key=lambda frag: frag.GetNumAtoms())

    mol = Chem.RemoveHs(mol) if remove_hs else Chem.AddHs(mol, addCoords=True)
    if require_3d and mol.GetNumConformers() == 0:
        raise ValueError(f"Molecule has no 3D conformer: {sdf_path}")
    return mol


def prepare_mol_for_pharmacophore(mol: Chem.Mol) -> Chem.Mol:
    """Prepare an RDKit molecule for robust pharmacophore feature extraction."""
    prepared = Chem.Mol(mol)
    prepared.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(prepared)
    try:
        Chem.SanitizeMol(
            prepared,
            sanitizeOps=(
                Chem.SanitizeFlags.SANITIZE_SYMMRINGS
                | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
            ),
        )
    except Exception:
        pass
    return prepared


def rdkit_mol_to_pyg_equiformer(
    mol: Chem.Mol,
    *,
    conf_id: int = 0,
    y: Optional[torch.Tensor] = None,
    smiles: Optional[str] = None,
    name: Optional[str] = None,
    idx: Optional[int] = None,
    pharmacophore_features=None,
) -> Data:
    """Convert an RDKit conformer into the PyG Data layout expected by Equiformer."""
    if mol is None:
        raise ValueError("mol is None")
    if mol.GetNumConformers() == 0:
        raise ValueError("RDKit molecule has no conformer")

    conf = mol.GetConformer(conf_id)
    num_atoms = mol.GetNumAtoms()

    pos = torch.tensor(
        [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(num_atoms)
        ],
        dtype=torch.float32,
    )
    z = torch.tensor([mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(num_atoms)], dtype=torch.long)

    atom_types = [1, 6, 7, 8, 9]
    hyb_map = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP3: 2,
    }
    x = torch.zeros((num_atoms, 11), dtype=torch.float32)
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        atomic_num = atom.GetAtomicNum()
        if atomic_num in atom_types:
            x[i, atom_types.index(atomic_num)] = 1.0
        x[i, 5] = 1.0 if atom.GetIsAromatic() else 0.0
        if atom.GetHybridization() in hyb_map:
            x[i, 6 + hyb_map[atom.GetHybridization()]] = 1.0
        x[i, 9] = float(atom.GetFormalCharge())
        x[i, 10] = float(atom.GetTotalNumHs(includeNeighbors=True))

    edge_src, edge_dst, edge_attr = [], [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        attr = bond_onehot(bond)
        edge_src.extend([i, j])
        edge_dst.extend([j, i])
        edge_attr.extend([attr, attr])

    data = Data(
        x=x,
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=torch.zeros((1, 19), dtype=torch.float32) if y is None else torch.as_tensor(y).float().view(1, -1),
        pos=pos,
        z=z,
        smiles=smiles or Chem.MolToSmiles(mol, isomericSmiles=True),
    )
    if name is not None:
        data.name = name
    if idx is not None:
        data.idx = torch.tensor([idx], dtype=torch.long)
    if pharmacophore_features is not None:
        data.pharmacophore_features = pharmacophore_features
    return data


def bond_onehot(bond: Chem.Bond) -> list[float]:
    bond_type = bond.GetBondType()
    if bond.GetIsAromatic() or bond_type == Chem.rdchem.BondType.AROMATIC:
        return [0.0, 0.0, 0.0, 1.0]
    if bond_type == Chem.rdchem.BondType.SINGLE:
        return [1.0, 0.0, 0.0, 0.0]
    if bond_type == Chem.rdchem.BondType.DOUBLE:
        return [0.0, 1.0, 0.0, 0.0]
    if bond_type == Chem.rdchem.BondType.TRIPLE:
        return [0.0, 0.0, 1.0, 0.0]
    return [1.0, 0.0, 0.0, 0.0]
