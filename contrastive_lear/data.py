"""Loading and splitting the official preprocessed PharmacoMatch data."""

from __future__ import annotations

from pathlib import Path

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


def split_dataset(
    dataset,
    limit: int | None = None,
    graph_size_upper_bound: int | None = None,
):
    """Reproduce PharmacoMatch's ordered outer/inner split and curriculum filter."""
    size = len(dataset) if limit is None or limit < 0 else min(limit, len(dataset))
    outer_cut = int(len(dataset) * 0.98)
    inner_indices = list(range(outer_cut))
    test_indices = list(range(outer_cut, len(dataset)))
    if graph_size_upper_bound is not None:
        inner_indices = [
            index
            for index in inner_indices
            if dataset[index].num_nodes <= graph_size_upper_bound
        ]
    inner_indices = inner_indices[:size]
    train_cut = int(len(inner_indices) * 0.9)
    return (
        Subset(dataset, inner_indices[:train_cut]),
        Subset(dataset, inner_indices[train_cut:]),
        Subset(dataset, test_indices),
    )
