# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (dig_envi)
#     language: python
#     name: dig_envi
# ---

# %%
# Load Packages 

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from equiformer_pytorch import Equiformer
from torch_geometric.utils import to_dense_batch
import os 

# %%

class EquiformerQM9_PointCloud(nn.Module):
    def __init__(self, n_token=11, n_out=19, hidden_dim=128):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Atom feature embedding -> hidden_dim
        self.embedding = nn.Linear(n_token, hidden_dim)

        # Equiformer core
        # Pure geometric neighborhood attention: no adjacency matrix, only point cloud + nearest neighbors
        self.model = Equiformer(
            dim=hidden_dim,
            dim_in=hidden_dim,
            input_degrees=1,
            num_degrees=2,

            heads=4,
            dim_head=hidden_dim // 4,
            depth=1,
            # Point-cloud mode
            attend_sparse_neighbors=False,  
            num_neighbors=16,                
            #valid_radius=5.0,                
            reduce_dim_out=False,
            attend_self=True,
            l2_dist_attention=False
        )

        # Regression head
        self.linear = nn.Linear(hidden_dim, n_out)

    def forward(self, data):
        x, coords, batch = data.x, data.pos, data.batch

        # Embed node features
        x = self.embedding(x)

        # Dense batching
        x, mask = to_dense_batch(x, batch)         
        coords, _ = to_dense_batch(coords, batch)  

        # Forward through Equiformer using point cloud only
        out = self.model(x, coords, mask=mask)

        # Extract invariant degree-0 features
        if hasattr(out, "type0"):
            x = out.type0
        elif isinstance(out, dict):
            x = out.get(0, next(iter(out.values())))
        elif isinstance(out, (list, tuple)):
            x = out[0]
        else:
            x = out

        # Pool if needed and predict
        if x.ndim == 2:
            return self.linear(x)

        if x.ndim == 3:
            mask_f = mask[:, :x.size(1)].float()
            x = (x * mask_f.unsqueeze(-1)).sum(dim=1) / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            return self.linear(x)

        raise ValueError(f"Unexpected output shape from Equiformer: {x.shape}")
# # %%