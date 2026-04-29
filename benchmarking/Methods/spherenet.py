#!/usr/bin/env python
"""Train a SphereNet baseline on QM9."""

from __future__ import annotations

import torch.nn as nn
from dig.threedgraph.method import SphereNet

try:
    from .benchmark_utils import parse_benchmark_args, train_baseline
except ImportError:
    from benchmark_utils import parse_benchmark_args, train_baseline


def build_spherenet(
    in_dim: int,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    out_dim: int = 19,
) -> nn.Module:
    """Build the SphereNet architecture used for QM9 benchmarking."""
    del in_dim, hidden_dim, dropout, out_dim
    return SphereNet(
        energy_and_force=False,
        cutoff=5.0,
        num_layers=4,
        hidden_channels=128,
        out_channels=19,
        int_emb_size=64,
        basis_emb_size_dist=8,
        basis_emb_size_angle=8,
        basis_emb_size_torsion=8,
        out_emb_channels=256,
        num_spherical=3,
        num_radial=6,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        use_node_features=True,
    )


def main() -> None:
    config = parse_benchmark_args(model_name="SphereNet", default_epochs=1000)
    train_baseline(config, build_spherenet)


if __name__ == "__main__":
    main()
