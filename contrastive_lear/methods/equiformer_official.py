"""Original Equiformer backbone adapted to seven pharmacophore feature types."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_scatter import scatter

from benchmarking.Methods.equiformer_official_benchmark import load_official_qm9_module

from .projector import PositiveProjector


class EquiformerOfficialContrastive(nn.Module):
    def __init__(self, embedding_dim: int = 512) -> None:
        super().__init__()
        official = load_official_qm9_module()
        self.official = official
        self.encoder = official.graph_attention_transformer_nonlinear_l2(
            irreps_in="7x0e",
            radius=5.0,
            num_basis=128,
            task_mean=None,
            task_std=None,
            atomref=None,
        )
        self.encoder.atom_embed = official.NodeEmbeddingNetwork(
            self.encoder.irreps_node_embedding, 7
        )
        self.projector = PositiveProjector(512, output_dim=embedding_dim)

    def forward(self, data):
        model = self.encoder
        edge_src, edge_dst = radius_graph(
            data.pos, r=model.max_radius, batch=data.batch, max_num_neighbors=1000
        )
        edge_vec = data.pos.index_select(0, edge_src) - data.pos.index_select(0, edge_dst)
        edge_sh = self.official.o3.spherical_harmonics(
            l=model.irreps_edge_attr,
            x=edge_vec,
            normalize=True,
            normalization="component",
        )
        feature_types = data.x.argmax(dim=1)
        node_features, _, _ = model.atom_embed(feature_types)
        edge_length_embedding = model.rbf(edge_vec.norm(dim=1))
        node_features = node_features + model.edge_deg_embed(
            node_features,
            edge_sh,
            edge_length_embedding,
            edge_src,
            edge_dst,
            data.batch,
        )
        node_attr = torch.ones_like(node_features[:, :1])
        for block in model.blocks:
            node_features = block(
                node_input=node_features,
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,
                edge_scalars=edge_length_embedding,
                batch=data.batch,
            )
        node_features = model.norm(node_features, batch=data.batch)
        pooled = scatter(node_features, data.batch, dim=0, reduce="sum")
        pooled = pooled / (model.scale_scatter.avg_aggregate_num ** 0.5)
        return self.projector(pooled)
