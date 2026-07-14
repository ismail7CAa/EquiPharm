"""Shared non-negative order-embedding projector."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositiveLinear(nn.Linear):
    def forward(self, inputs):
        return F.linear(inputs, self.weight.abs(), None)


class PositiveProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 1024, output_dim: int = 512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            PositiveLinear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, inputs):
        return self.layers(inputs)
