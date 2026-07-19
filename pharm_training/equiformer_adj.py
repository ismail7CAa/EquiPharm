"""Adjacency-aware Equiformer potential used for transferable encoder pretraining."""

from __future__ import annotations

import torch
import torch.nn as nn
from equiformer_pytorch import Equiformer
from torch_geometric.utils import to_dense_adj, to_dense_batch


class EquiformerAdjPotential(nn.Module):
    """Energy-conserving potential with a screening-compatible Equiformer core."""

    def __init__(self, num_elements: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.species_embedding = nn.Embedding(num_elements, hidden_dim)
        self.model = Equiformer(
            dim=hidden_dim,
            dim_in=hidden_dim,
            input_degrees=1,
            num_degrees=2,
            heads=4,
            dim_head=hidden_dim // 4,
            depth=6,
            attend_sparse_neighbors=True,
            num_neighbors=2,
            num_adj_degrees_embed=2,
            max_sparse_neighbors=16,
            valid_radius=10.0,
            reduce_dim_out=False,
            attend_self=True,
            l2_dist_attention=False,
        )
        self.atomic_energy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    @staticmethod
    def _type0(output):
        if hasattr(output, "type0"):
            return output.type0
        if isinstance(output, dict):
            return output.get(0, next(iter(output.values())))
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def encode_nodes(self, data):
        features, mask = to_dense_batch(self.species_embedding(data.element_index), data.batch)
        coordinates, _ = to_dense_batch(data.pos, data.batch)
        adjacency = to_dense_adj(data.edge_index, batch=data.batch).bool()
        return self._type0(self.model(features, coordinates, mask=mask, adj_mat=adjacency)), mask

    def forward(self, data):
        nodes, mask = self.encode_nodes(data)
        atomic = self.atomic_energy(nodes).squeeze(-1) * mask
        return atomic.sum(dim=1)

    def transferable_state_dict(self):
        """Weights transferable after replacing the dataset-specific input/head."""
        return self.model.state_dict()
