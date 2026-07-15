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
    drop_path: float = 0.0,
) -> nn.Module:
    """Build the adjacency-aware Equiformer architecture."""
    del dropout
    return EquiformerQM9(
        n_token=in_dim,
        n_out=out_dim,
        hidden_dim=hidden_dim,
        drop_path=drop_path,
    )


def main() -> None:
    config = parse_benchmark_args(
        model_name="EquiformerAdj",
        default_epochs=300,
        default_batch_size=128,
        default_eval_batch_size=128,
        default_hidden_dim=128,
        default_dropout=0.0,
        default_lr=5e-4,
        default_weight_decay=5e-3,
        default_optimizer="adamw",
        default_opt_eps=1e-8,
        default_scheduler="cosine",
        default_loss="l1",
        default_model_ema=False,
        default_model_ema_decay=0.9999,
        default_drop_path=0.0,
        default_warmup_lr=1e-6,
        default_warmup_epochs=5,
        default_min_lr=1e-6,
        default_seed=0,
        default_split_seed=1,
        default_seeds=[1, 2, 3],
        default_vary_split_seed=False,
    )
    train_baseline(config, build_equiformer_adj)


if __name__ == "__main__":
    main()
