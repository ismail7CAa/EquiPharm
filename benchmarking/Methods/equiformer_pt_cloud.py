#!/usr/bin/env python
"""Train a point-cloud Equiformer baseline on QM9."""

from __future__ import annotations

import torch.nn as nn
from equiformer_pytorch import Equiformer
from torch_geometric.utils import to_dense_batch

try:
    from .benchmark_utils import parse_benchmark_args, train_baseline
except ImportError:
    from benchmark_utils import parse_benchmark_args, train_baseline


class EquiformerQM9PointCloud(nn.Module):
    """Equiformer model using 3D coordinates without molecular adjacency."""

    def __init__(
        self,
        in_dim: int = 11,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        out_dim: int = 19,
    ) -> None:
        super().__init__()
        del dropout

        self.embedding = nn.Linear(in_dim, hidden_dim)
        self.model = Equiformer(
            dim=hidden_dim,
            dim_in=hidden_dim,
            input_degrees=1,
            num_degrees=2,
            heads=4,
            dim_head=hidden_dim // 4,
            depth=1,
            attend_sparse_neighbors=False,
            num_neighbors=16,
            reduce_dim_out=False,
            attend_self=True,
            l2_dist_attention=False,
        )
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x = self.embedding(data.x)
        x, mask = to_dense_batch(x, data.batch)
        coords, _ = to_dense_batch(data.pos, data.batch)

        out = self.model(x, coords, mask=mask)
        x = self._extract_type0(out)

        if x.ndim == 2:
            return self.linear(x)

        if x.ndim == 3:
            mask_f = mask[:, : x.size(1)].float()
            x = (x * mask_f.unsqueeze(-1)).sum(dim=1)
            x = x / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            return self.linear(x)

        raise ValueError(f"Unexpected Equiformer output shape: {x.shape}")

    @staticmethod
    def _extract_type0(out):
        if hasattr(out, "type0"):
            return out.type0
        if isinstance(out, dict):
            return out.get(0, next(iter(out.values())))
        if isinstance(out, (list, tuple)):
            return out[0]
        return out


def main() -> None:
    config = parse_benchmark_args(
        model_name="EquiformerPointCloud",
        default_epochs=1500,
        default_batch_size=12,
        default_eval_batch_size=32,
        default_hidden_dim=128,
        default_lr=3e-6,
    )
    train_baseline(config, EquiformerQM9PointCloud)


if __name__ == "__main__":
    main()
