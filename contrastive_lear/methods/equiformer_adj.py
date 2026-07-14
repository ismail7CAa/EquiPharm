"""Adjacency-aware Equiformer pharmacophore encoder."""

from __future__ import annotations

import torch.nn as nn

from benchmarking.Methods.equiformer_architecture import EquiformerQM9

from .projector import PositiveProjector


class EquiformerAdjContrastive(nn.Module):
    def __init__(self, embedding_dim: int = 512) -> None:
        super().__init__()
        self.encoder = EquiformerQM9(n_token=7, n_out=embedding_dim, hidden_dim=128, drop_path=0.0)
        self.projector = PositiveProjector(128, output_dim=embedding_dim)

    def forward(self, data):
        return self.projector(self.encoder.encode(data))
