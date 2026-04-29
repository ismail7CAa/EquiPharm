"""Adjacency-aware Equiformer architecture for QM9 regression."""

from __future__ import annotations

import torch.nn as nn
from equiformer_pytorch import Equiformer
from torch_geometric.utils import to_dense_adj, to_dense_batch


class EquiformerQM9(nn.Module):
    """Equiformer model using molecular adjacency plus geometric neighbors."""

    def __init__(
        self,
        n_token: int = 11,
        n_out: int = 19,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(n_token, hidden_dim)
        self.model = Equiformer(
            dim=hidden_dim,
            dim_in=hidden_dim,
            input_degrees=1,
            num_degrees=2,
            heads=4,
            dim_head=hidden_dim // 4,
            depth=6,
            attend_sparse_neighbors=True,
            num_neighbors=4,
            num_adj_degrees_embed=2,
            max_sparse_neighbors=16,
            valid_radius=10.0,
            reduce_dim_out=False,
            attend_self=True,
            l2_dist_attention=False,
        )
        self.linear = nn.Linear(hidden_dim, n_out)

    def forward(self, data):
        return self.linear(self.encode(data))

    def encode(self, data):
        x = self.embedding(data.x)
        x, mask = to_dense_batch(x, data.batch)
        coords, _ = to_dense_batch(data.pos, data.batch)
        adj_mat = to_dense_adj(data.edge_index, batch=data.batch).bool()

        out = self.model(x, coords, mask=mask, adj_mat=adj_mat)
        x = self._extract_type0(out)

        mask_f = mask[:, : x.size(1)].float()
        x = (x * mask_f.unsqueeze(-1)).sum(dim=1)
        return x / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)

    @staticmethod
    def _extract_type0(out):
        if hasattr(out, "type0"):
            return out.type0
        if isinstance(out, dict):
            return out.get(0, next(iter(out.values())))
        if isinstance(out, (list, tuple)):
            return out[0]
        return out
