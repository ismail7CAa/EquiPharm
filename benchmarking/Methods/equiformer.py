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
import numpy as np
from equiformer_pytorch import Equiformer
from torch_geometric.utils import to_dense_batch, to_dense_adj

from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures


# %%
# Build Equiformer model
class EquiformerQM9(nn.Module):
    def __init__(self, n_token=11, n_out=19, hidden_dim=128):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 1) Atom feature embedding
        self.embedding = nn.Linear(n_token, hidden_dim)

        # 2) Equiformer core
        self.model = Equiformer(
            dim=hidden_dim,
            dim_in=hidden_dim,
            input_degrees=1,
            num_degrees=2,

            heads=4,
            dim_head=hidden_dim // 4,
            depth=1,

            attend_sparse_neighbors=True,
            num_neighbors=0,
            num_adj_degrees_embed=2,
            max_sparse_neighbors=16,

            valid_radius=5.0,

            reduce_dim_out=False,
            attend_self=True,
            l2_dist_attention=False
        )

        # 3) Regression head
        self.linear = nn.Linear(hidden_dim, n_out)

        # 4) RDKit pharmacophore factory
        fdef_name = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        self.feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

    def forward(self, data):
        x = self.encode(data)
        return self.linear(x)

    def pharmaco_features(self, mol):
        """
        Extract pharmacophore features from an RDKit Mol.
        Returns a list of dicts like:
        [
            {"atom_ids": (1, 2)},
            {"atom_ids": (4,)},
            ...
        ]
        """
        if mol is None:
            return []

        features = []
        for feat in self.feature_factory.GetFeaturesForMol(mol):
            atom_ids = tuple(int(i) for i in feat.GetAtomIds())
            features.append({"atom_ids": atom_ids})

        return features

    def masked_mean_pool(self, atom_embeddings, mask):
        """
        atom_embeddings: (N, F)
        mask: (N,) boolean
        """
        mask_f = mask.float().unsqueeze(-1)  # (N, 1)  *Convert boolean mask to float*---> allows padded atoms to be zeroed out
        denom = mask_f.sum(dim=0).clamp_min(1.0)  # (1,) *Count how many valid atoms there are.* , clamp_min to avoid division by zero
        pooled = (atom_embeddings * mask_f).sum(dim=0) / denom # masked averaging : - invalid padded atoms becom zero, - sum valid atomo embeddings, -divide by numbeerof V.A
        return pooled

    def pharmacophore_pool(self, atom_embeddings, pharmacophore_features, mask=None):
        """
        atom_embeddings: (N, F) for one molecule
        pharmacophore_features: list of dicts with key 'atom_ids'
        mask: (N,) boolean, optional

        Strategy:
        - For each pharmacophore feature, average the embeddings of its atoms
        - Then average all pharmacophore feature embeddings
        - If no valid features exist, fallback to masked mean pooling
        """
        feat_embs = []

        num_atoms = atom_embeddings.size(0)

        for feat in pharmacophore_features:
            if isinstance(feat, list):
                # flatten one accidental nesting level
                for subfeat in feat:
                    if not isinstance(subfeat, dict):
                        continue
        
                    ids = list(subfeat.get("atom_ids", []))
                    if len(ids) == 0:
                        continue
        
                    valid_ids = [i for i in ids if 0 <= i < num_atoms]
                    if mask is not None:
                        valid_ids = [i for i in valid_ids if bool(mask[i].item())]
        
                    if len(valid_ids) == 0:
                        continue
        
                    feat_emb = atom_embeddings[valid_ids].mean(dim=0)
                    feat_embs.append(feat_emb)
                continue
        
            if not isinstance(feat, dict):
                continue
        
            ids = list(feat.get("atom_ids", []))
            if len(ids) == 0:
                continue
        
            valid_ids = [i for i in ids if 0 <= i < num_atoms]
            if mask is not None:
                valid_ids = [i for i in valid_ids if bool(mask[i].item())]
        
            if len(valid_ids) == 0:
                continue
        
            feat_emb = atom_embeddings[valid_ids].mean(dim=0)
            feat_embs.append(feat_emb)

        # fallback if no pharmacophore feature survived
        if len(feat_embs) == 0:
            if mask is not None:
                return self.masked_mean_pool(atom_embeddings, mask)
            return atom_embeddings.mean(dim=0)

        feat_embs = torch.stack(feat_embs, dim=0)  # (num_features, F)
        return feat_embs.mean(dim=0)               # (F,)

    def _extract_type0(self, out):
        """
        Safely extract invariant degree-0 features from Equiformer output.
        """
        if hasattr(out, "type0"):
            return out.type0
        elif isinstance(out, dict):
            return out.get(0, next(iter(out.values())))
        elif isinstance(out, (list, tuple)):
            return out[0]
        else:
            return out

    def encode(self, data, mask=None):
        x, coords, batch = data.x, data.pos, data.batch

        # 1) Embed node features
        x = self.embedding(x)

        # 2) Dense batching
        x, mask = to_dense_batch(x, batch)         # (B, N, F), (B, N)
        coords, _ = to_dense_batch(coords, batch)  # (B, N, 3)

        # 3) Build adjacency from bond graph
        adj_mat = to_dense_adj(data.edge_index, batch=batch).bool()  # (B, N, N)

        # 4) Forward through Equiformer
        out = self.model(x, coords, mask=mask, adj_mat=adj_mat)

        # 5) Extract invariant (degree-0) atom embeddings
        x = self._extract_type0(out)  # expected shape: (B, N, F)

        # 6) Pharmacophore-aware pooling per molecule
        pooled_list = []

        # data.pharmacophore_features is expected to be:
        # - for batch size 1: list[dict]
        # - for batch size B: list[list[dict]]
        batch_pharma = getattr(data, "pharmacophore_features", None)

        for b in range(x.size(0)):
            atom_emb_b = x[b]      # (N, F)
            mask_b = mask[b]       # (N,)

            # Fallback: no pharmacophore info at all
            if batch_pharma is None:
                pooled_b = self.masked_mean_pool(atom_emb_b, mask_b)
                pooled_list.append(pooled_b)
                continue

            # Handle single-graph case and batched case
            if x.size(0) == 1:
                pharma_feats_b = batch_pharma
            
                # unwrap nested single-item list if needed
                if (
                    isinstance(pharma_feats_b, list)
                    and len(pharma_feats_b) > 0
                    and isinstance(pharma_feats_b[0], list)
                ):
                    pharma_feats_b = pharma_feats_b[0]
            else:
                pharma_feats_b = batch_pharma[b]

            # If no features for this molecule -> fallback
            if pharma_feats_b is None or len(pharma_feats_b) == 0:
                pooled_b = self.masked_mean_pool(atom_emb_b, mask_b)
            else:
                pooled_b = self.pharmacophore_pool(
                    atom_embeddings=atom_emb_b,
                    pharmacophore_features=pharma_feats_b,
                    mask=mask_b
                )

            pooled_list.append(pooled_b)

        x = torch.stack(pooled_list, dim=0)  # (B, F)
        return x
    
# # %%