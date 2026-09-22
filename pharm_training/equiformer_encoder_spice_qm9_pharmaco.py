"""Pharmacophore adapter restricted to exported SPICE->QM9 encoders."""

from __future__ import annotations

from pathlib import Path

import torch

from .equiformer_encoder_pharmaco_feat import SPICEPharmacophoreEncoder


class SPICEQM9PharmacophoreEncoder(SPICEPharmacophoreEncoder):
    """Load a geometric core that was pretrained on SPICE then fine-tuned on QM9."""

    @classmethod
    def from_pretrained(cls, checkpoint, descriptor_dim, map_location="cpu"):
        payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
        if payload.get("dataset") != "spice_qm9_finetune":
            raise ValueError(
                "Expected an exported SPICE->QM9 encoder checkpoint. Create one with "
                "python -m pharm_training.export_spice_qm9_encoder."
            )
        return super().from_pretrained(checkpoint, descriptor_dim, map_location)


__all__ = ["SPICEQM9PharmacophoreEncoder"]
