#!/usr/bin/env python
"""Train an adjacency-aware Equiformer baseline on QM9."""

from __future__ import annotations

import torch.nn as nn

try:
    from .benchmark_utils import parse_benchmark_args, train_baseline
    from .equiformer_architecture import EquiformerQM9
except ImportError:
    from benchmark_utils import parse_benchmark_args, train_baseline
    from equiformer_architecture import EquiformerQM9


def build_equiformer_adj(
    in_dim: int,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    out_dim: int = 19,
) -> nn.Module:
    """Build the adjacency-aware Equiformer architecture."""
    del dropout
    return EquiformerQM9(n_token=in_dim, n_out=out_dim, hidden_dim=hidden_dim)


def main() -> None:
    config = parse_benchmark_args(
        model_name="EquiformerAdj",
        default_epochs=1500,
        default_batch_size=12,
        default_eval_batch_size=32,
        default_hidden_dim=128,
        default_lr=3e-6,
    )
    train_baseline(config, build_equiformer_adj)


if __name__ == "__main__":
    main()
