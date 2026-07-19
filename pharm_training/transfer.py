"""Utilities for transferring a potential-pretrained core into screening models."""

from pathlib import Path

import torch


def load_pretrained_core(model, checkpoint: str | Path, map_location="cpu"):
    """Load only geometric Equiformer weights, leaving input and task heads untouched."""
    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    if not hasattr(model, "model"):
        raise AttributeError("Target model must expose its Equiformer core as `.model`.")
    model.model.load_state_dict(payload["encoder_state_dict"], strict=True)
    return model
