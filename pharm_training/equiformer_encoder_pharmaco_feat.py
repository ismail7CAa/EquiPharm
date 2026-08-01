"""Pharmacophore adapter for the EquiformerAdj core pretrained on SPICE.

This module belongs to the SPICE workflow. It does not import the QM9 benchmark
encoder and does not copy the SPICE-specific element embedding or energy head.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .equiformer_adj import EquiformerAdjConfig, EquiformerAdjEncoder


class SPICEPharmacophoreEncoder(nn.Module):
    """Adapt a SPICE-pretrained geometric core to descriptor-based screening."""

    def __init__(
        self,
        descriptor_dim: int,
        architecture: dict[str, Any] | EquiformerAdjConfig | None = None,
    ) -> None:
        super().__init__()
        config = (
            architecture if isinstance(architecture, EquiformerAdjConfig)
            else EquiformerAdjConfig.from_mapping(architecture)
        )
        self.descriptor_dim = descriptor_dim
        self.architecture = config
        self.input_projection = nn.Linear(descriptor_dim, config.hidden_dim)
        # The element embedding is unused here; downstream inputs are descriptors.
        self.encoder = EquiformerAdjEncoder(num_elements=1, config=config)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        descriptor_dim: int,
        map_location: str | torch.device = "cpu",
    ) -> "SPICEPharmacophoreEncoder":
        """Build the matching architecture and load its SPICE geometric weights."""
        payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
        architecture = payload.get("architecture") or payload.get("config", {}).get("architecture")
        if architecture is None:
            raise KeyError("Checkpoint does not contain Equiformer architecture metadata")
        if "encoder_state_dict" not in payload:
            raise KeyError("Checkpoint does not contain encoder_state_dict")
        model = cls(descriptor_dim=descriptor_dim, architecture=architecture)
        model.encoder.model.load_state_dict(payload["encoder_state_dict"], strict=True)
        return model

    @staticmethod
    def extract_pharmacophore_features(molecule):
        """Extract RDKit BaseFeatures metadata without coupling RDKit to SPICE training."""
        if molecule is None:
            return []
        from rdkit import RDConfig
        from rdkit.Chem import ChemicalFeatures

        factory = ChemicalFeatures.BuildFeatureFactory(
            str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
        )
        return [
            {
                "atom_ids": tuple(int(index) for index in feature.GetAtomIds()),
                "family": feature.GetFamily(),
                "type": feature.GetType(),
            }
            for feature in factory.GetFeaturesForMol(molecule)
        ]

    def encode_nodes(self, data):
        """Return invariant atom embeddings and the dense valid-atom mask."""
        if not hasattr(data, "x"):
            raise AttributeError("Pharmacophore data must contain node descriptors in data.x")
        projected = self.input_projection(data.x)
        return self.encoder.encode_embedded_nodes(data, projected)

    def encode_pharmacophore_features(self, data):
        """Return feature embeddings and metadata for downstream Hungarian matching."""
        if not hasattr(data, "pharmacophore_features"):
            raise AttributeError(
                "Attach RDKit feature dictionaries to data.pharmacophore_features first"
            )
        projected = self.input_projection(data.x)
        return self.encoder.encode_pharmacophore_features(data, projected)

    def forward(self, data):
        """Return one pooled molecular embedding, preferring pharmacophore features."""
        nodes, mask = self.encode_nodes(data)
        feature_outputs = None
        if hasattr(data, "pharmacophore_features"):
            feature_outputs = self.encode_pharmacophore_features(data)

        pooled = []
        for index in range(nodes.size(0)):
            if feature_outputs is not None and feature_outputs[index]["embeddings"].numel():
                pooled.append(feature_outputs[index]["embeddings"].mean(dim=0))
            else:
                pooled.append(nodes[index][mask[index]].mean(dim=0))
        return torch.stack(pooled)


__all__ = ["SPICEPharmacophoreEncoder"]
