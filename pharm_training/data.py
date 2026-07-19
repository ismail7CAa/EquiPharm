"""Lazy HDF5 datasets backed by preparation manifests."""

from __future__ import annotations

import bisect
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903


class PotentialDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str, cutoff: float, limit: int = -1):
        self.manifest = json.loads(Path(manifest_path).read_text())
        self.records = [r for r in self.manifest["records"] if r["split"] == split]
        self.cutoff = cutoff
        self.elements = self.manifest["elements"]
        self.element_to_index = {z: index for index, z in enumerate(self.elements)}
        self.atomic_refs = dict(zip(self.elements, self.manifest["atomic_reference_energies_hartree"]))
        self.cumulative = []
        total = 0
        for record in self.records:
            total += record["conformations"]
            self.cumulative.append(total)
        self.length = total if limit < 0 else min(total, limit)
        self._handles = {}

    def __len__(self):
        return self.length

    def _h5(self, source):
        if source not in self._handles:
            self._handles[source] = h5py.File(source, "r")
        return self._handles[source]

    def __getitem__(self, index):
        record_index = bisect.bisect_right(self.cumulative, index)
        previous = self.cumulative[record_index - 1] if record_index else 0
        conformation_index = index - previous
        record = self.records[record_index]
        group = self._h5(record["source"])[record["group"]]
        atomic_numbers = record["atomic_numbers"]
        positions = np.asarray(group[record["coordinate_key"]][conformation_index], dtype=np.float32)
        if self.manifest["position_unit"] == "bohr":
            positions *= BOHR_TO_ANGSTROM
        energy = float(group[record["energy_key"]][conformation_index])
        energy -= sum(self.atomic_refs[z] for z in atomic_numbers)
        energy *= HARTREE_TO_EV

        force = None
        if record["force_key"] is not None:
            force = np.asarray(group[record["force_key"]][conformation_index], dtype=np.float32)
            if self.manifest["force_is_negative_gradient"]:
                force = -force
            force *= HARTREE_TO_EV
            if self.manifest["position_unit"] == "bohr":
                force /= BOHR_TO_ANGSTROM

        pos = torch.from_numpy(positions)
        distances = torch.cdist(pos, pos)
        adjacency = (distances < self.cutoff) & (distances > 0)
        edge_index = adjacency.nonzero(as_tuple=False).T.contiguous()
        data = Data(
            pos=pos,
            z=torch.tensor(atomic_numbers, dtype=torch.long),
            element_index=torch.tensor(
                [self.element_to_index[z] for z in atomic_numbers], dtype=torch.long
            ),
            edge_index=edge_index,
            y=torch.tensor([energy], dtype=torch.float32),
        )
        if force is not None:
            data.force = torch.from_numpy(force)
        return data
