"""Pharmacophore adapter for the EquiformerAdj core pretrained on SPICE.

This module belongs to the SPICE workflow. It does not import the QM9 benchmark
encoder and does not copy the SPICE-specific element embedding or energy head.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

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
        projection_state = payload.get("input_projection_state_dict")
        if projection_state is not None:
            model.input_projection.load_state_dict(projection_state, strict=True)
        elif model._initialize_projection_from_species(payload):
            pass
        else:
            # Keep this fallback reproducible. It is suitable for integration and
            # smoke tests, but a trained projection or a full SPICE best.pt is
            # preferable for scientific screening.
            generator = torch.Generator(device="cpu").manual_seed(0)
            with torch.no_grad():
                model.input_projection.weight.normal_(
                    mean=0.0,
                    std=model.architecture.hidden_dim ** -0.5,
                    generator=generator,
                )
                model.input_projection.bias.zero_()
            warnings.warn(
                "The checkpoint has no descriptor projection or SPICE species embedding; "
                "using a deterministic untrained projection. Prefer the trial best.pt "
                "checkpoint, or train and save input_projection_state_dict.",
                RuntimeWarning,
                stacklevel=2,
            )
        return model

    @classmethod
    def screening_from_checkpoint(
        cls,
        checkpoint: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "SPICEPharmacophoreEncoder":
        """Construct the adapter expected by the pharmacophore screening loaders."""
        return cls.from_pretrained(checkpoint, descriptor_dim=11, map_location=map_location)

    def _initialize_projection_from_species(self, payload: dict[str, Any]) -> bool:
        """Map the five element one-hot inputs onto learned SPICE element vectors."""
        state = payload.get("species_embedding_state_dict")
        if state is not None:
            species = state.get("weight")
        else:
            species = payload.get("model_state_dict", {}).get("encoder.species_embedding.weight")
        elements = payload.get("elements")
        if species is None or elements is None:
            return False

        element_rows = {int(atomic_number): index for index, atomic_number in enumerate(elements)}
        descriptor_elements = (1, 6, 7, 8, 9)  # molecule_io.py channels 0..4
        if any(atomic_number not in element_rows for atomic_number in descriptor_elements):
            return False
        with torch.no_grad():
            self.input_projection.weight.zero_()
            self.input_projection.bias.zero_()
            for column, atomic_number in enumerate(descriptor_elements):
                self.input_projection.weight[:, column].copy_(species[element_rows[atomic_number]])
        return True

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

    def pharmaco_features(self, molecule):
        """Compatibility alias used by the pharmacophore screening pipelines."""
        return self.extract_pharmacophore_features(molecule)

    def encode_nodes(self, data):
        """Return invariant atom embeddings and the dense valid-atom mask."""
        if not hasattr(data, "x"):
            raise AttributeError("Pharmacophore data must contain node descriptors in data.x")
        projected = self.input_projection(data.x)
        return self.encoder.encode_embedded_nodes(data, projected)

    def _feature_outputs(self, data):
        if not hasattr(data, "pharmacophore_features"):
            raise AttributeError(
                "Attach RDKit feature dictionaries to data.pharmacophore_features first"
            )
        projected = self.input_projection(data.x)
        return self.encoder.encode_pharmacophore_features(data, projected)

    def encode_pharmacophore_features(self, data):
        """Return the dictionary consumed by Hungarian screening (batch size one)."""
        outputs = self._feature_outputs(data)
        if len(outputs) != 1:
            raise ValueError("Pharmacophore matching currently expects a single molecule batch")
        output = outputs[0]
        nodes, mask = self.encode_nodes(data)
        global_embedding = nodes[0][mask[0]].mean(dim=0)
        return {
            "feature_embeddings": output["embeddings"],
            "feature_metadata": output["metadata"],
            "global_embedding": global_embedding,
        }

    def encode(self, data):
        """Compatibility method used by pooled cosine-similarity screening."""
        return self.forward(data)

    def forward(self, data):
        """Return one pooled molecular embedding, preferring pharmacophore features."""
        nodes, mask = self.encode_nodes(data)
        feature_outputs = None
        if hasattr(data, "pharmacophore_features"):
            feature_outputs = self._feature_outputs(data)

        pooled = []
        for index in range(nodes.size(0)):
            if feature_outputs is not None and feature_outputs[index]["embeddings"].numel():
                pooled.append(feature_outputs[index]["embeddings"].mean(dim=0))
            else:
                pooled.append(nodes[index][mask[index]].mean(dim=0))
        return torch.stack(pooled)


__all__ = ["SPICEPharmacophoreEncoder"]
