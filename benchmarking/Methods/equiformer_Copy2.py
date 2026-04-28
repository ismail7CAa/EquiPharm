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
from torch_geometric.utils import to_dense_batch, to_dense_adj
import os 

# %%
# Build Equiformer model
# Define Equiformer 
class EquiformerQM9(nn.Module):
    def __init__(self, n_token=11, n_out=19, hidden_dim=128):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 1) Atom feature embedding 
        self.embedding = nn.Linear(n_token, hidden_dim)

        # 2) Equiformer core
        # input_degrees=1: inputs are scalar features
        # num_degrees=2: internal features include degree 0 and 1 (scalars + vectors)
        self.model = Equiformer(
            dim=hidden_dim,
            dim_in=hidden_dim,
            input_degrees=1,
            num_degrees=2,

            heads=4,
            dim_head=hidden_dim // 4,   # 32 when hidden_dim=128
            depth=6, 

            # --- key efficiency / "molecular graph" knobs ---
            attend_sparse_neighbors=True,  # requires adj_mat
            num_neighbors=4,               # 0 = bonds only; >0 adds closest geometric neighbors
            num_adj_degrees_embed=2,       # adds 2-hop connectivity embedding
            max_sparse_neighbors=16,       # cap total sparse neighbors

            # we generally don't need valid_radius if we pass adj_mat,
            valid_radius=10.0,

            reduce_dim_out=False,
            attend_self=True,
            l2_dist_attention=False
        )

        # 3) Regression head
        self.linear = nn.Linear(hidden_dim, n_out) # After we get a molecule-level representation (a vector per molecule), we predict 19 targets.

    
    def forward(self, data):
        
        x = self.encode(data)
        
        # # 6) Pool if needed and predict
        # if x.ndim == 2:
        #     # (B, F) already pooled
        #     return self.linear(x)

        # if x.ndim == 3:
        #     # (B, N, F) node features -> masked mean pooling
        #     mask_f = mask[:, :x.size(1)].float()
        #     x = (x * mask_f.unsqueeze(-1)).sum(dim=1) / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.linear(x)

        # raise ValueError(f"Unexpected output shape from Equiformer: {x.shape}")
        

    def encode(self, data, mask=None):
        x, coords, batch = data.x, data.pos, data.batch

        # 1) Embed node features
        x = self.embedding(x)

        # 2) Dense batching : This is necessary because attention is usually implemented on dense tensors.
        x, mask = to_dense_batch(x, batch)          # converts variable-length node lists into a padded dense tensor
        coords, _ = to_dense_batch(coords, batch)   # converts coordinates to dense padded form too

        # 3) Build adjacency from bond graph (PyG edge_index)
        #    to_dense_adj returns float 0/1, convert to bool
        """
        We pass the bond adjacency matrix to Equiformer to constrain attention to chemically meaningful neighbors.
        This ensures that information always propagates along covalent bonds,while geometric neighbors are added to capture non-bonded interactions.
        """
        adj_mat = to_dense_adj(data.edge_index, batch=batch).bool()  # (B, N, N)

        # 4) Forward through Equiformer using sparse neighbor attention
        out = self.model(x, coords, mask=mask, adj_mat=adj_mat)  # runs Equiformer on dense atom features, coordinates, mask, and adjacency

        # 5) Extract invariant (degree-0) features safely
        if hasattr(out, "type0"):
            x = out.type0
        elif isinstance(out, dict):
            x = out.get(0, next(iter(out.values())))
        elif isinstance(out, (list, tuple)):
            x = out[0]
        else:
            x = out

        # masked mean pooling
        mask_f = mask[:, :x.size(1)].float()
        x = (x * mask_f.unsqueeze(-1)).sum(dim=1) / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)

        return x

    
    #     def encode_nodes(self, data):
    #         x, coords, batch = data.x, data.pos, data.batch
        
    #         x = self.embedding(x)
    #         x, mask = to_dense_batch(x, batch)
    #         coords, _ = to_dense_batch(coords, batch)
    #         adj_mat = to_dense_adj(data.edge_index, batch=batch).bool()
        
    #         out = self.model(x, coords, mask=mask, adj_mat=adj_mat)
        
    #         if hasattr(out, "type0"):
    #             x = out.type0
    #         elif isinstance(out, dict):
    #             x = out.get(0, next(iter(out.values())))
    #         elif isinstance(out, (list, tuple)):
    #             x = out[0]
    #         else:
    #             x = out
        
    #         return x, mask
# # %%