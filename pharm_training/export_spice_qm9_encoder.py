#!/usr/bin/env python
"""Export a SPICE->QM9 fine-tuned encoder for pharmacophore screening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


CORE_PREFIX = "encoder.model."
SPECIES_KEY = "encoder.species_embedding.weight"


def export_encoder(
    fine_tuned_checkpoint: str | Path,
    source_spice_checkpoint: str | Path | None,
    output: str | Path,
) -> Path:
    """Combine fine-tuned weights with architecture metadata from SPICE pretraining."""
    fine_tuned_checkpoint = Path(fine_tuned_checkpoint)
    if source_spice_checkpoint is None:
        provenance_path = fine_tuned_checkpoint.parent.parent / "transfer_provenance.json"
        if not provenance_path.is_file():
            raise FileNotFoundError(
                "Could not find transfer_provenance.json; pass --source-spice-checkpoint"
            )
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        source_spice_checkpoint = provenance["source_checkpoint"]
    source_spice_checkpoint = Path(source_spice_checkpoint)
    output = Path(output)
    tuned = torch.load(fine_tuned_checkpoint, map_location="cpu", weights_only=False)
    source = torch.load(source_spice_checkpoint, map_location="cpu", weights_only=False)
    model_state = tuned.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise KeyError("QM9 checkpoint does not contain model_state_dict")

    encoder_state = {
        key[len(CORE_PREFIX) :]: value
        for key, value in model_state.items()
        if key.startswith(CORE_PREFIX)
    }
    if not encoder_state:
        raise KeyError(
            f"QM9 checkpoint has no weights beginning with {CORE_PREFIX!r}; "
            "select best_model.pt or last_checkpoint.pt from the fine-tuning run"
        )
    species = model_state.get(SPECIES_KEY)
    if species is None:
        raise KeyError(f"QM9 checkpoint does not contain {SPECIES_KEY!r}")
    architecture = source.get("architecture") or source.get("config", {}).get("architecture")
    elements = source.get("elements")
    if architecture is None or not elements:
        raise KeyError("Source SPICE checkpoint must contain architecture and elements")

    payload = {
        "encoder_state_dict": encoder_state,
        "species_embedding_state_dict": {"weight": species},
        "architecture": architecture,
        "elements": elements,
        "dataset": "spice_qm9_finetune",
        "source_spice_checkpoint": str(source_spice_checkpoint),
        "source_qm9_checkpoint": str(fine_tuned_checkpoint),
        "qm9_epoch": tuned.get("epoch"),
        "qm9_config": tuned.get("config"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="QM9 best_model.pt")
    parser.add_argument(
        "--source-spice-checkpoint",
        type=Path,
        help="Original SPICE checkpoint (auto-detected from transfer_provenance.json when omitted)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = export_encoder(args.checkpoint, args.source_spice_checkpoint, args.output)
    print(path)


if __name__ == "__main__":
    main()
