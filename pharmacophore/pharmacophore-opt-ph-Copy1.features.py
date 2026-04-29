#!/usr/bin/env python
# coding: utf-8

# all your imports stay here
# all your function definitions stay here
# do not put execution code here
# Import Packages
import sys
sys.path.append('/project/IZZY/molecular-representation/benchmarking/Methods/')   
from torch_geometric.datasets import QM9
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdMolTransforms
from torch_geometric.data import Data

from torch_geometric.data import Batch
import torch.nn.functional as F
import numpy as np
import math
from rdkit.Chem import rdchem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdMolTransforms, rdMolDescriptors
from typing import Optional
import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score 
import torch
from scipy.optimize import differential_evolution
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import AllChem
import os 
import sys
sys.path.append('/project/IZZY/molecular-representation/benchmarking/Methods/') 
from equiformer_encoder_pharmaco_feat import EquiformerQM9
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures


# In[2]:


import torch, torch_geometric
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("No CUDA device available")


# In[3]:


import torch_scatter, torch_sparse
print(torch_scatter.__version__)
print(torch_sparse.__version__)


# In[4]:


# Device - Checkpoint - Calling the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("/project/IZZY/molecular-representation/benchmarking/Methods/checkpoints_Equiformernew1/best_model.pt", map_location=device)
print(checkpoint.keys())

# Equiformer 
model = EquiformerQM9().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# In[5]:


# Converting Pyg into RDKit mol
# MOL2 helpers
_MOL2_TO_RDKit_BOND = {
    "1": rdchem.BondType.SINGLE,
    "2": rdchem.BondType.DOUBLE,
    "3": rdchem.BondType.TRIPLE,
    "am": rdchem.BondType.SINGLE,   # amide bond
    "ar": rdchem.BondType.AROMATIC,
    "du": rdchem.BondType.SINGLE,    
    "un": rdchem.BondType.SINGLE,    
}


def _parse_mol2_atoms_bonds(mol2_path: str):
    """
    Minimal MOL2 parser for @<TRIPOS>ATOM and @<TRIPOS>BOND blocks.
    Returns:
      atoms: list of dicts with keys: idx(1-based), name, x,y,z, sybyl_type, charge(optional)
      bonds: list of dicts with keys: a1(1-based), a2(1-based), btype(str)
    """
    atoms, bonds = [], []
    in_atoms = False
    in_bonds = False

    with open(mol2_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("@<TRIPOS>"):
                in_atoms = (line == "@<TRIPOS>ATOM")
                in_bonds = (line == "@<TRIPOS>BOND")
                continue

            if in_atoms:
                # Typical MOL2 ATOM line (space-separated):
                # atom_id name x y z type subst_id subst_name charge
                parts = line.split()
                if len(parts) < 6:
                    continue

                atom_id = int(parts[0])
                name = parts[1]
                x, y, z = map(float, parts[2:5])
                sybyl_type = parts[5]
                charge = float(parts[-1]) if len(parts) >= 9 else None

                atoms.append(
                    dict(idx=atom_id, name=name, x=x, y=y, z=z, sybyl_type=sybyl_type, charge=charge)
                )

            elif in_bonds:
                parts = line.split()
                if len(parts) < 4:
                    continue
                a1 = int(parts[1])
                a2 = int(parts[2])
                btype = parts[3]
                bonds.append(dict(a1=a1, a2=a2, btype=btype))

    if not atoms:
        raise ValueError("No @<TRIPOS>ATOM block found / parsed.")
    return atoms, bonds


def _element_from_sybyl(sybyl_type: str, atom_name: str):
    """
    MOL2 SYBYL types look like: C.3, C.ar, N.am, O.2, H, Cl, Br, etc.
    We'll infer element from SYBYL prefix, falling back to atom_name.
    """
    base = sybyl_type.split(".", 1)[0].strip()
    if base:
        return base
    return atom_name[:2].strip().title()

# Main function
def mol2_to_rdkit_mol(
    mol2_path: str,
    sanitize: bool = True,
    keep_hs: bool = True,
    store_charges: bool = True,
):
    """
    Build an RDKit Mol from a Tripos MOL2 file and attach the 3D conformer from MOL2 coordinates.

    - keep_hs=True keeps explicit H atoms as provided by MOL2 (your example includes H1..H21).
    - store_charges=True writes per-atom 'mol2_charge' property.
    """
    atoms, bonds = _parse_mol2_atoms_bonds(mol2_path)

    # MOL2 atom ids are 1-based; map them to 0-based RDKit indices
    id_to_rd_idx = {}

    rw = Chem.RWMol()

    # Add atoms
    for a in atoms:
        elem = _element_from_sybyl(a["sybyl_type"], a["name"])
        atom = Chem.Atom(elem)

        # store original MOL2 metadata
        atom.SetProp("mol2_name", a["name"])
        atom.SetProp("mol2_type", a["sybyl_type"])
        if store_charges and a["charge"] is not None:
            atom.SetDoubleProp("mol2_charge", float(a["charge"]))

        rd_idx = rw.AddAtom(atom)
        id_to_rd_idx[a["idx"]] = rd_idx

    # Add bonds
    for b in bonds:
        i = id_to_rd_idx[b["a1"]]
        j = id_to_rd_idx[b["a2"]]
        btype = b["btype"].lower()

        rd_bond = _MOL2_TO_RDKit_BOND.get(btype, rdchem.BondType.SINGLE)
        rw.AddBond(i, j, rd_bond)
        if btype == "ar":
            rw.GetBondBetweenAtoms(i, j).SetIsAromatic(True)
            rw.GetAtomWithIdx(i).SetIsAromatic(True)
            rw.GetAtomWithIdx(j).SetIsAromatic(True)

    mol = rw.GetMol()
    # Attach coordinates as conformer
    conf = Chem.Conformer(mol.GetNumAtoms())
    for a in atoms:
        i = id_to_rd_idx[a["idx"]]
        conf.SetAtomPosition(i, Point3D(float(a["x"]), float(a["y"]), float(a["z"])))

    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    
    if not keep_hs:
        mol = Chem.RemoveHs(mol, sanitize=sanitize)

    # Sanitize 
    if sanitize:
        Chem.SanitizeMol(mol)
        
    return mol


# In[6]:


from rdkit import Chem

def read_sdf_mol(
    sdf_path: str,
    *,
    sanitize: bool = True,
    remove_hs: bool = False,
    largest_fragment: bool = True,
    require_3d: bool = False,
    mol_index: int = 0,
):
    """
    Read one molecule from an SDF file into an RDKit Mol.

    Parameters
    ----------
    sanitize : bool
        Whether to sanitize the molecule.
    remove_hs : bool
        If True, remove explicit hydrogens.
    largest_fragment : bool
        Keep only the largest fragment (recommended for ligands).
    require_3d : bool
        If True, raise error if no conformer present.
    mol_index : int
        Which molecule to read from the SDF (default: first).

    Returns
    -------
    RDKit Mol
    """

    suppl = Chem.SDMolSupplier(
        sdf_path,
        sanitize=sanitize,
        removeHs=False,  # we control H handling ourselves
    )

    if len(suppl) == 0:
        raise ValueError(f"No molecules found in SDF: {sdf_path}")

    mol = suppl[mol_index]

    if mol is None:
        raise ValueError(f"Failed to read molecule at index {mol_index} from {sdf_path}")

    if largest_fragment:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumAtoms())

    # Hydrogen handling
    if remove_hs:
        mol = Chem.RemoveHs(mol)
    else:
        mol = Chem.AddHs(mol, addCoords=True)

    # 3D check 
    if require_3d and mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no 3D conformer.")

    return mol


# In[7]:


def get_all_torsion_angles(
    mol: Chem.Mol,
    *,
    unique: bool = True,
    heavy_only: bool = True,
    rotatable_only: bool = True,
    exclude_rings: bool = True,
    one_per_bond: bool = False,
):
    """
    Extract torsion angles from an RDKit Mol with a 3D conformer.
    """

    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no 3D coordinates attached")

    conf = mol.GetConformer()

    rot_bonds = None
    if rotatable_only:
        # SMARTS for rotatable bonds
        rot_query = Chem.MolFromSmarts("[!$(*#*)&!D1]-!@[!$(*#*)&!D1]")
        matches = mol.GetSubstructMatches(rot_query)
        rot_bonds = set(tuple(sorted((a, b))) for a, b in matches)

    torsions = []
    seen = set()

    for bond in mol.GetBonds():
        j = bond.GetBeginAtomIdx()
        k = bond.GetEndAtomIdx()

        if exclude_rings and bond.IsInRing():
            continue

        if rotatable_only:
            if tuple(sorted((j, k))) not in rot_bonds:
                continue

        nbrs_j = [a.GetIdx() for a in mol.GetAtomWithIdx(j).GetNeighbors() if a.GetIdx() != k]
        nbrs_k = [a.GetIdx() for a in mol.GetAtomWithIdx(k).GetNeighbors() if a.GetIdx() != j]

        if heavy_only:
            nbrs_j = [i for i in nbrs_j if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
            nbrs_k = [l for l in nbrs_k if mol.GetAtomWithIdx(l).GetAtomicNum() > 1]

        if not nbrs_j or not nbrs_k:
            continue

        if one_per_bond:
            pairs = [(min(nbrs_j), min(nbrs_k))]
        else:
            pairs = [(i, l) for i in nbrs_j for l in nbrs_k if i != l]

        for i, l in pairs:
            angle = rdMolTransforms.GetDihedralDeg(conf, int(i), int(j), int(k), int(l))
            atoms = (int(i), int(j), int(k), int(l))

            if unique:
                rev = (atoms[3], atoms[2], atoms[1], atoms[0])
                key = atoms if atoms < rev else rev
                if key in seen:
                    continue
                seen.add(key)

            torsions.append({
                "atoms": atoms,
                "bond": (int(j), int(k)),
                "angle_deg": float(angle),
            })

    return torsions


# In[8]:


# Set the torsion angles 
def set_torsion_angle(
    mol: Chem.Mol,
    i: int, j: int, k: int, l: int,
    new_angle: float,
    conf_id: int = 0,
    *,
    copy: bool = True,
    require_bond_jk: bool = True,
    require_path: bool = True,
):
    """
    Set dihedral (i,j,k,l) in degrees on a given conformer.

    Notes for your pipeline:
    - This changes coordinates of many atoms (rotates a fragment around j-k).
    - You must use consistent (i,j,k,l) definitions (ideally one per rotatable bond).
    """
    m = Chem.Mol(mol) if copy else mol  

    n = m.GetNumAtoms()
    for idx in (i, j, k, l):
        if not (0 <= int(idx) < n):
            raise IndexError(f"Atom index {idx} out of range [0, {n-1}]")

    if m.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformer")
    if conf_id < 0 or conf_id >= m.GetNumConformers():
        raise IndexError(f"conf_id={conf_id} out of range (n_confs={m.GetNumConformers()})")

    if require_bond_jk and m.GetBondBetweenAtoms(int(j), int(k)) is None:
        raise ValueError(f"No bond between j={j} and k={k}. Dihedral must be around a bonded pair.")

    if require_path:
        nbrs_j = {a.GetIdx() for a in m.GetAtomWithIdx(int(j)).GetNeighbors()}
        nbrs_k = {a.GetIdx() for a in m.GetAtomWithIdx(int(k)).GetNeighbors()}
        if int(i) not in nbrs_j:
            raise ValueError(f"i={i} is not a neighbor of j={j}. Invalid dihedral definition.")
        if int(l) not in nbrs_k:
            raise ValueError(f"l={l} is not a neighbor of k={k}. Invalid dihedral definition.")

    conf = m.GetConformer(conf_id)

    # wrap angle to a conventional range 
    ang = float(new_angle)
    while ang <= -180.0:
        ang += 360.0
    while ang > 180.0:
        ang -= 360.0
    rdMolTransforms.SetDihedralDeg(conf, int(i), int(j), int(k), int(l), ang)

    return m


# In[9]:


# Converting the new Molecule from RDKit to Pyg 
def rdkit_mol_to_pyg_equiformer(
    mol: Chem.Mol,
    *,
    conf_id: int = 0,
    y: Optional[torch.Tensor] = None,
    smiles: Optional[str] = None,
    name: Optional[str] = None,
    idx: Optional[int] = None,
):
    """
    Convert an RDKit Mol with a conformer into a PyG Data that matches the
      x:        [N, 11]
      z:        [N]
      pos:      [N, 3]
      edge_index:[2, 2E]
      edge_attr:[2E, 4]  
      y:        [1, 19]  
    """

    if mol is None:
        raise ValueError("mol is None")
    if mol.GetNumConformers() == 0:
        raise ValueError("RDKit mol has no conformer.")
    if conf_id < 0 or conf_id >= mol.GetNumConformers():
        raise IndexError(f"conf_id={conf_id} out of range (n_confs={mol.GetNumConformers()})")

    conf = mol.GetConformer(conf_id)
    n = mol.GetNumAtoms()

    # pos 
    pos = torch.tensor(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(n)],
        dtype=torch.float32
    )

    # z 
    z = torch.tensor([mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n)], dtype=torch.long)

    # x 
    atom_types = [1, 6, 7, 8, 9]  # H,C,N,O,F
    hyb_map = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP3: 2,
    }

    x = torch.zeros((n, 11), dtype=torch.float32)
    for i in range(n):
        a = mol.GetAtomWithIdx(i)
        anum = a.GetAtomicNum()

        # one-hot H/C/N/O/F
        if anum in atom_types:
            x[i, atom_types.index(anum)] = 1.0

        # aromatic
        x[i, 5] = 1.0 if a.GetIsAromatic() else 0.0

        # hybridization 
        hyb = a.GetHybridization()
        if hyb in hyb_map:
            x[i, 6 + hyb_map[hyb]] = 1.0

        # charge & H count
        x[i, 9]  = float(a.GetFormalCharge())
        x[i, 10] = float(a.GetTotalNumHs(includeNeighbors=True))

    # edges: edge_index , edge_attr 
    edge_src = []
    edge_dst = []
    edge_attr = []

    def bond_onehot(b: Chem.Bond):
        bt = b.GetBondType()
        if b.GetIsAromatic() or bt == Chem.rdchem.BondType.AROMATIC:
            return [0.0, 0.0, 0.0, 1.0]
        if bt == Chem.rdchem.BondType.SINGLE:
            return [1.0, 0.0, 0.0, 0.0]
        if bt == Chem.rdchem.BondType.DOUBLE:
            return [0.0, 1.0, 0.0, 0.0]
        if bt == Chem.rdchem.BondType.TRIPLE:
            return [0.0, 0.0, 1.0, 0.0]
        return [1.0, 0.0, 0.0, 0.0]

    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        attr = bond_onehot(b)
        
        edge_src += [i, j]
        edge_dst += [j, i]
        edge_attr += [attr, attr]

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    # y 
    if y is None:
        y = torch.zeros((1, 19), dtype=torch.float32)
    else:
        y = torch.as_tensor(y).float()
        if y.ndim == 1:
            y = y.view(1, -1)
        if y.shape[0] != 1:
            raise ValueError(f"y must have shape [1, *], got {tuple(y.shape)}")
            
    # smiles 
    if smiles is None:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        pos=pos,
        z=z,
        smiles=smiles,
    )
    
    if name is not None:
        data.name = name
    if idx is not None:
        data.idx = torch.tensor([idx], dtype=torch.long)

    return data


# In[10]:


def prepare_mol_for_pharmacophore(mol: Chem.Mol) -> Chem.Mol:

    m = Chem.Mol(mol)
    m.UpdatePropertyCache(strict=False)

    Chem.FastFindRings(m)
    try:
        Chem.SanitizeMol(
            m,
            sanitizeOps=(
                Chem.SanitizeFlags.SANITIZE_SYMMRINGS |
                Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
            )
        )
    except Exception:
        pass

    return m

# Import data 
mol2_path = "/project/IZZY/molecular-representation/dataset/DUD-E/fa10/crystal_ligand.mol2"

if not os.path.exists(mol2_path):
    raise FileNotFoundError(f"Could not find MOL2 file at: {mol2_path}")

#  MOL2 to RDKit mol 
mol = mol2_to_rdkit_mol(
    mol2_path,
    sanitize=True,
    keep_hs=False,        
    store_charges=True
)

# print("RDKit atoms:", mol.GetNumAtoms(), "conformers:", mol.GetNumConformers())

# SMILES string from RDKit 
smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
# print("SMILES:", smiles)

# 2) RDKit to PyG Data 
data = rdkit_mol_to_pyg_equiformer(
    mol,
    conf_id=0,
    y=torch.zeros((1, 19), dtype=torch.float32),  
    smiles=smiles,
    name="DUD-E_kit_crystal_ligand",
    idx=0
)
# print(data)
# print("num atoms (PyG):", data.pos.size(0))
# print("edge_index:", tuple(data.edge_index.shape), "edge_attr:", tuple(data.edge_attr.shape))
# print("x:", tuple(data.x.shape), "z:", tuple(data.z.shape), "y:", tuple(data.y.shape))


# In[15]:


def encode_pharmacophore(model, data, mol, device):
    model.eval()
    data.pharmacophore_features = model.pharmaco_features(mol) #it mutates the data object in-place.
    batch = Batch.from_data_list([data]).to(device)
    with torch.no_grad():
        z = model.encode(batch)
    return z


# In[16]:


# Pharmacophore-based reference embedding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

zt = encode_pharmacophore(model, data, mol, device)
# print(zt)


# In[17]:


# Precompute torsions ONCE
TORSIONS = get_all_torsion_angles(
    mol,
    rotatable_only=True,
    heavy_only=True,
    exclude_rings=True,
    one_per_bond=True
)
BOUNDS = [(0.0, 360.0)] * len(TORSIONS)

def objective_theta(theta_vec, mol_current, torsions_current, idx_current, z_ref):
    theta_vec = np.mod(theta_vec, 360.0)

    mol_tmp = Chem.Mol(mol_current)
    for t, theta in zip(torsions_current, theta_vec):
        i, j, k, l = t["atoms"]
        mol_tmp = set_torsion_angle(mol_tmp, i, j, k, l, float(theta), conf_id=0)

    data = rdkit_mol_to_pyg_equiformer(
        mol_tmp,
        conf_id=0,
        y=torch.zeros((1, 19), dtype=torch.float32),
        smiles=Chem.MolToSmiles(mol_tmp, isomericSmiles=True),
        name=mol_tmp.GetProp("_Name") if mol_tmp.HasProp("_Name") else "mol",
        idx=idx_current,
    )
    # data.pharmacophore_features = model.pharmaco_features(mol)

    z1 = encode_pharmacophore(model, data, mol_tmp, device)
    cs = F.cosine_similarity(z_ref.to(device), z1, dim=1)

    return float(-cs.item())

def cosine_sim(mol, theta_set, idx_current=0):
    torsions = get_all_torsion_angles(mol, rotatable_only=False)

    # start from a copy and apply all torsions cumulatively
    mol_new = Chem.Mol(mol)

    for t, theta in zip(torsions, theta_set):
        i, j, k, l = t["atoms"]
        mol_new = set_torsion_angle(mol_new, i, j, k, l, theta, conf_id=0)

    # RDKit -> PyG
    mol_name = mol_new.GetProp("_Name") if mol_new.HasProp("_Name") else "mol"

    data = rdkit_mol_to_pyg_equiformer(
        mol_new,
        conf_id=0,
        y=torch.zeros((1, 19), dtype=torch.float32),
        smiles=Chem.MolToSmiles(mol_new, isomericSmiles=True),
        name=mol_name,
        idx=idx_current,
    )
    # data.pharmacophore_features = model.pharmaco_features(mol)
    z1 = encode_pharmacophore(model, data, mol_new, device)
    cs = F.cosine_similarity(zt.to(device), z1, dim=1)

    return cs.squeeze(0).item()


def main():
    import torch, torch_geometric
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    else:
        print("No CUDA device available")

    import torch_scatter, torch_sparse
    print(torch_scatter.__version__)
    print(torch_sparse.__version__)

    # Device - Checkpoint - Calling the model
    global device, checkpoint, model, zt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        "/project/IZZY/molecular-representation/benchmarking/Methods/checkpoints_Equiformernew1/best_model.pt",
        map_location=device
    )
    print(checkpoint.keys())

    model = EquiformerQM9().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Import data
    mol2_path = "/project/IZZY/molecular-representation/dataset/DUD-E/fa10/crystal_ligand.mol2"

    if not os.path.exists(mol2_path):
        raise FileNotFoundError(f"Could not find MOL2 file at: {mol2_path}")

    mol = mol2_to_rdkit_mol(
        mol2_path,
        sanitize=True,
        keep_hs=False,
        store_charges=True
    )

    print("RDKit atoms:", mol.GetNumAtoms(), "conformers:", mol.GetNumConformers())

    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    print("SMILES:", smiles)

    data = rdkit_mol_to_pyg_equiformer(
        mol,
        conf_id=0,
        y=torch.zeros((1, 19), dtype=torch.float32),
        smiles=smiles,
        name="DUD-E_kit_crystal_ligand",
        idx=0
    )
    print(data)
    print("num atoms (PyG):", data.pos.size(0))
    print("edge_index:", tuple(data.edge_index.shape), "edge_attr:", tuple(data.edge_attr.shape))
    print("x:", tuple(data.x.shape), "z:", tuple(data.z.shape), "y:", tuple(data.y.shape))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    zt = encode_pharmacophore(model, data, mol, device)
    print(zt)

    # Actives and Decoys
    folder_active = "/project/IZZY/molecular-representation/dataset/DUD-E/fa10/actives_sdf"
    folder_decoy  = "/project/IZZY/molecular-representation/dataset/DUD-E/fa10/decoys_sdf"

    sdf_files_active = sorted(glob(os.path.join(folder_active, "*.sdf")))
    sdf_files_decoy  = sorted(glob(os.path.join(folder_decoy, "*.sdf")))
    sdf_files = sdf_files_active + sdf_files_decoy

    labels = [1] * len(sdf_files_active) + [0] * len(sdf_files_decoy)

    cos_sims = []

    for idx, (sdf_path, label) in enumerate(
        tqdm(zip(sdf_files, labels), total=len(sdf_files), desc="Processing")
    ):
        suppl = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=False)
        mol = suppl[0] if len(suppl) > 0 else None
        if mol is None:
            continue

        tag = "active" if label == 1 else "decoy"
        mol.SetProp("_Name", f"{tag}_{idx:04d}")
        mol.SetProp("source", "DUD-E_SDF")
        mol.SetProp("sdf_path", sdf_path)

        mol = Chem.AddHs(mol, addCoords=True)
        if mol.GetNumConformers() == 0:
            continue

        TORSIONS = get_all_torsion_angles(mol, rotatable_only=False)

        if TORSIONS:
            BOUNDS = [(0.0, 360.0)] * len(TORSIONS)

            result = differential_evolution(
                lambda theta: objective_theta(theta, mol, TORSIONS, idx, zt),
                bounds=BOUNDS,
                strategy="best1bin",
                maxiter=3,
                popsize=2,
                tol=1e-3,
                polish=True,
                updating="deferred",
                workers=1
            )

            best_theta_set = np.mod(result.x, 360.0)
            best_cosine = -result.fun

        else:
            data = rdkit_mol_to_pyg_equiformer(
                mol,
                conf_id=0,
                y=torch.zeros((1, 19), dtype=torch.float32),
                smiles=Chem.MolToSmiles(mol, isomericSmiles=True),
                name=mol.GetProp("_Name"),
                idx=idx,
            )

            z1 = encode_pharmacophore(model, data, mol, device)
            cs = F.cosine_similarity(zt.to(device), z1, dim=1)
            best_cosine = cs.squeeze(0).detach().cpu().item()

        cos_sims.append(best_cosine)

    # Experiments
    cos_np = np.array(cos_sims)
    labels_np = np.array(labels)

    active_scores = cos_np[labels_np == 1]
    decoy_scores = cos_np[labels_np == 0]

    plt.figure(figsize=(6, 5))
    plt.boxplot([decoy_scores, active_scores], labels=["Decoys", "Actives"], showfliers=False)
    plt.ylabel("Cosine Similarity")
    plt.title("Cosine Similarity: Acitves vs Decoys")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("cosine_similarity_boxplot.png", dpi=300, bbox_inches="tight")
    plt.show()

    scores = np.array(cos_sims)
    y_true = np.array(labels)
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    auroc = roc_auc_score(y_true, scores)
    print(f"AUROC = {auroc:.4f}")

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Actives vs Decoys")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curve_actives_vs_decoys.png", dpi=100, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()