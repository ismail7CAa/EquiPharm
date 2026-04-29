#!/usr/bin/env python
"""Train a Graph Attention Network baseline on QM9."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

try:
    from .benchmark_utils import parse_benchmark_args, train_baseline
except ImportError:
    from benchmark_utils import parse_benchmark_args, train_baseline


class GATModel(nn.Module):
    """Three-layer GAT regressor for all 19 QM9 targets."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        out_dim: int = 19,
    ) -> None:
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        x = F.relu(self.conv1(batch.x, batch.edge_index))
        x = F.relu(self.conv2(x, batch.edge_index))
        x = F.relu(self.conv3(x, batch.edge_index))
        x = global_mean_pool(x, batch.batch)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def main() -> None:
    config = parse_benchmark_args(model_name="GAT", default_epochs=1000)
    train_baseline(config, GATModel)


if __name__ == "__main__":
    main()
