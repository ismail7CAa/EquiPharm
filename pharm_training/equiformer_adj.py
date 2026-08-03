"""Configurable EquiformerAdj encoder and energy-conserving SPICE potential."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn
from equiformer_pytorch import Equiformer
from torch_geometric.utils import to_dense_adj, to_dense_batch


@dataclass(frozen=True)
class EquiformerAdjConfig:
    """Architecture values saved with a checkpoint and reused downstream."""

    hidden_dim: int = 128
    depth: int = 6
    num_degrees: int = 2
    heads: int = 4
    dim_head: int | None = None
    num_neighbors: int = 0
    num_adj_degrees_embed: int = 2
    max_sparse_neighbors: int | None = 32
    valid_radius: float = 6.0
    attend_self: bool = True
    l2_dist_attention: bool = False

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None = None) -> "EquiformerAdjConfig":
        values = values or {}
        known = cls.__dataclass_fields__
        return cls(**{key: values[key] for key in known if key in values})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EquiformerAdjEncoder(nn.Module):
    """Reusable geometric core that returns invariant per-atom embeddings."""

    def __init__(self, num_elements: int, config: EquiformerAdjConfig | None = None) -> None:
        super().__init__()
        self.config = config or EquiformerAdjConfig()
        cfg = self.config
        if cfg.hidden_dim % cfg.heads:
            raise ValueError("hidden_dim must be divisible by heads when dim_head is omitted")
        dim_head = cfg.dim_head or cfg.hidden_dim // cfg.heads
        self.species_embedding = nn.Embedding(num_elements, cfg.hidden_dim)
        self.model = Equiformer(
            dim=cfg.hidden_dim,
            dim_in=cfg.hidden_dim,
            input_degrees=1,
            num_degrees=cfg.num_degrees,
            heads=cfg.heads,
            dim_head=dim_head,
            depth=cfg.depth,
            attend_sparse_neighbors=True,
            num_neighbors=cfg.num_neighbors,
            num_adj_degrees_embed=cfg.num_adj_degrees_embed,
            max_sparse_neighbors=cfg.max_sparse_neighbors,
            valid_radius=cfg.valid_radius,
            reduce_dim_out=False,
            attend_self=cfg.attend_self,
            l2_dist_attention=cfg.l2_dist_attention,
        )

    @staticmethod
    def _type0(output):
        if hasattr(output, "type0"):
            return output.type0
        if isinstance(output, dict):
            return output.get(0, next(iter(output.values())))
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def encode_embedded_nodes(self, data, embedded_features):
        """Encode already-projected node descriptors from any downstream modality."""
        if embedded_features.size(-1) != self.config.hidden_dim:
            raise ValueError(
                f"Expected projected feature width {self.config.hidden_dim}, "
                f"got {embedded_features.size(-1)}"
            )
        features, mask = to_dense_batch(embedded_features, data.batch)
        coordinates, _ = to_dense_batch(data.pos, data.batch)
        adjacency = to_dense_adj(data.edge_index, batch=data.batch).bool()
        output = self.model(features, coordinates, mask=mask, adj_mat=adjacency)
        return self._type0(output), mask

    def encode_nodes(self, data):
        """Encode SPICE nodes using the dataset-specific element embedding."""
        if not hasattr(data, "atom_type"):
            raise AttributeError(
                "SPICE batches must contain data.atom_type; regenerate batches with "
                "the current pharm_training.data implementation"
            )
        return self.encode_embedded_nodes(data, self.species_embedding(data.atom_type))

    @staticmethod
    def _valid_features(features: Iterable, num_atoms: int, mask=None):
        for feature in features or []:
            if isinstance(feature, list):
                yield from EquiformerAdjEncoder._valid_features(feature, num_atoms, mask)
                continue
            if not isinstance(feature, dict):
                continue
            atom_ids = [int(index) for index in feature.get("atom_ids", ())]
            atom_ids = [index for index in atom_ids if 0 <= index < num_atoms]
            if mask is not None:
                atom_ids = [index for index in atom_ids if bool(mask[index].item())]
            if atom_ids:
                yield feature, atom_ids

    @staticmethod
    def pharmacophore_feature_embeddings(atom_embeddings, features, mask=None, coordinates=None):
        """Pool atom embeddings into one vector per externally extracted feature."""
        embeddings, metadata = [], []
        for feature, atom_ids in EquiformerAdjEncoder._valid_features(
            features, atom_embeddings.size(0), mask
        ):
            embeddings.append(atom_embeddings[atom_ids].mean(dim=0))
            item = {
                "atom_ids": tuple(atom_ids),
                "family": feature.get("family"),
                "type": feature.get("type"),
            }
            if coordinates is not None:
                center = coordinates[atom_ids].mean(dim=0).detach().cpu().tolist()
                item["center"] = tuple(float(value) for value in center)
            metadata.append(item)
        if not embeddings:
            return atom_embeddings.new_zeros((0, atom_embeddings.size(-1))), []
        return torch.stack(embeddings), metadata

    def encode_pharmacophore_features(self, data, embedded_features=None):
        """Return feature embeddings/metadata ready for downstream Hungarian matching."""
        nodes, mask = (
            self.encode_nodes(data) if embedded_features is None
            else self.encode_embedded_nodes(data, embedded_features)
        )
        coordinates, _ = to_dense_batch(data.pos, data.batch)
        batch_features = getattr(data, "pharmacophore_features", None)
        outputs = []
        for batch_index in range(nodes.size(0)):
            if batch_features is None:
                features = []
            elif nodes.size(0) == 1:
                features = batch_features
                if features and isinstance(features[0], list):
                    features = features[0]
            else:
                features = batch_features[batch_index]
            embeddings, metadata = self.pharmacophore_feature_embeddings(
                nodes[batch_index], features, mask[batch_index], coordinates[batch_index]
            )
            outputs.append({"embeddings": embeddings, "metadata": metadata})
        return outputs


class EquiformerAdjPotential(nn.Module):
    """Energy-conserving potential wrapping the reusable EquiformerAdj encoder."""

    def __init__(
        self,
        num_elements: int,
        hidden_dim: int = 128,
        architecture: dict[str, Any] | EquiformerAdjConfig | None = None,
    ) -> None:
        super().__init__()
        if isinstance(architecture, EquiformerAdjConfig):
            config = architecture
        else:
            values = dict(architecture or {})
            values.setdefault("hidden_dim", hidden_dim)
            config = EquiformerAdjConfig.from_mapping(values)
        self.encoder = EquiformerAdjEncoder(num_elements, config)
        self.atomic_energy = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.SiLU(), nn.Linear(config.hidden_dim, 1)
        )

    @property
    def architecture_config(self) -> dict[str, Any]:
        return self.encoder.config.to_dict()

    @property
    def model(self):
        """Expose the geometric core for compatibility with transfer.py."""
        return self.encoder.model

    @property
    def species_embedding(self):
        return self.encoder.species_embedding

    def encode_nodes(self, data):
        return self.encoder.encode_nodes(data)

    def encode_pharmacophore_features(self, data, embedded_features=None):
        return self.encoder.encode_pharmacophore_features(data, embedded_features)

    def forward(self, data):
        nodes, mask = self.encode_nodes(data)
        return (self.atomic_energy(nodes).squeeze(-1) * mask).sum(dim=1)

    def transferable_state_dict(self):
        """Core weights compatible with the current pharmacophore encoders."""
        return self.model.state_dict()
