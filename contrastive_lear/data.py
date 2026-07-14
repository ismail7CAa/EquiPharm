"""Loading and splitting the official preprocessed PharmacoMatch data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.data import InMemoryDataset


class PharmacoMatchTrainingDataset(InMemoryDataset):
    def __init__(self, root: str | Path) -> None:
        super().__init__(str(root))
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return ["chembl_data.pt"]

    def download(self) -> None:
        pass

    def process(self) -> None:
        raise FileNotFoundError(
            "chembl_data.pt is missing. Run scripts/download_datasets.sh pharmacomatch."
        )


def split_dataset(dataset, seed: int, limit: int | None = None):
    size = len(dataset) if limit is None or limit < 0 else min(limit, len(dataset))
    indices = np.random.default_rng(seed).permutation(len(dataset))[:size]
    outer_cut = int(size * 0.98)
    inner_indices = indices[:outer_cut]
    test_indices = indices[outer_cut:]
    train_cut = int(len(inner_indices) * 0.9)
    return (
        Subset(dataset, inner_indices[:train_cut].tolist()),
        Subset(dataset, inner_indices[train_cut:].tolist()),
        Subset(dataset, test_indices.tolist()),
    )
