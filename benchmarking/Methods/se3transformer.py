#!/usr/bin/env python
"""Train an SE(3)-Transformer baseline on QM9."""

from __future__ import annotations

import torch.nn as nn
from se3_transformer_pytorch import SE3Transformer
from torch_geometric.utils import to_dense_adj, to_dense_batch

try:
    from .benchmark_utils import parse_benchmark_args, train_baseline
except ImportError:
    from benchmark_utils import parse_benchmark_args, train_baseline


class SE3EncoderDecoderQM9(nn.Module):
    """SE(3)-Transformer encoder with masked mean pooling for QM9 regression."""

    def __init__(
        self,
        in_dim: int = 11,
        hidden_dim: int = 8,
        dropout: float = 0.3,
        out_dim: int = 19,
        heads: int = 4,
        valid_radius: float = 10.0,
    ) -> None:
        super().__init__()
        del dropout

        self.embedding = nn.Linear(in_dim, hidden_dim)
        self.transformer = SE3Transformer(
            dim=hidden_dim,
            heads=heads,
            dim_head=hidden_dim // heads,
            depth=2,
            num_degrees=2,
            valid_radius=valid_radius,
            max_sparse_neighbors=14,
        )
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x = self.embedding(data.x)
        x, mask = to_dense_batch(x, data.batch)
        coords, _ = to_dense_batch(data.pos, data.batch)
        adj_mat = to_dense_adj(data.edge_index, batch=data.batch).bool()

        x = self.transformer(x, coords, mask=mask, adj_mat=adj_mat)
        mask_f = mask.float()
        x = (x * mask_f.unsqueeze(-1)).sum(dim=1)
        x = x / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.linear(x)


def main() -> None:
    config = parse_benchmark_args(
        model_name="SE3Transformer",
        default_epochs=1000,
        default_batch_size=8,
        default_eval_batch_size=16,
        default_hidden_dim=8,
        default_lr=3e-4,
    )
    train_baseline(config, SE3EncoderDecoderQM9)


if __name__ == "__main__":
    main()
