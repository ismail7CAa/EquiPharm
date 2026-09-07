#!/usr/bin/env python
"""Fine-tune a SPICE-pretrained EquiformerAdj on nine QM9 properties.

The task is intentionally fixed to the six electronic QM9 targets and the
three rotational constants.  A very small learning rate is used for every
parameter so the transferred geometric representation changes gradually.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from benchmarking.Methods.benchmark_utils import BenchmarkConfig, train_baseline
from pharm_training.equiformer_adj import EquiformerAdjConfig, EquiformerAdjEncoder


TARGET_PRESET = "electronic_geometry"
DEFAULT_LEARNING_RATE = 1e-6
DEFAULT_MIN_LEARNING_RATE = 1e-7


class SPICEQM9FineTuner(nn.Module):
    """SPICE geometric encoder with a new nine-property QM9 regression head."""

    def __init__(self, checkpoint: str | Path, out_dim: int = 9, dropout: float = 0.0):
        super().__init__()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "encoder_state_dict" not in payload:
            raise KeyError("SPICE checkpoint does not contain encoder_state_dict")
        if "architecture" not in payload:
            raise KeyError("SPICE checkpoint does not contain its architecture")
        elements = [int(value) for value in payload.get("elements", [])]
        if not elements:
            raise KeyError("SPICE checkpoint does not contain its element list")

        config = EquiformerAdjConfig.from_mapping(payload["architecture"])
        self.encoder = EquiformerAdjEncoder(len(elements), config)
        self.encoder.model.load_state_dict(payload["encoder_state_dict"], strict=True)
        self._load_species_embedding(payload)

        lookup = torch.full((max(elements) + 1,), -1, dtype=torch.long)
        for species_index, atomic_number in enumerate(elements):
            lookup[atomic_number] = species_index
        self.register_buffer("element_lookup", lookup)
        self.dropout = nn.Dropout(dropout)
        self.regression_head = nn.Linear(config.hidden_dim, out_dim)

    def _load_species_embedding(self, payload: dict) -> None:
        state = payload.get("species_embedding_state_dict")
        if state is not None:
            self.encoder.species_embedding.load_state_dict(state, strict=True)
            return

        model_state = payload.get("model_state_dict", {})
        for key in ("encoder.species_embedding.weight", "species_embedding.weight"):
            if key in model_state:
                self.encoder.species_embedding.weight.data.copy_(model_state[key])
                return
        raise KeyError(
            "SPICE checkpoint has no species embedding. Use checkpoints/best.pt "
            "or checkpoints/trained_encoder.pt from pharm_training.train."
        )

    def forward(self, data):
        atomic_numbers = data.z.long()
        if atomic_numbers.numel() and int(atomic_numbers.max()) >= self.element_lookup.numel():
            raise ValueError("QM9 contains an element absent from the SPICE checkpoint")
        atom_type = self.element_lookup[atomic_numbers]
        if (atom_type < 0).any():
            missing = sorted(set(atomic_numbers[atom_type < 0].detach().cpu().tolist()))
            raise ValueError(f"QM9 element(s) absent from the SPICE checkpoint: {missing}")

        embedded = self.encoder.species_embedding(atom_type)
        nodes, mask = self.encoder.encode_embedded_nodes(data, embedded)
        mask_float = mask[:, : nodes.size(1)].to(nodes.dtype)
        pooled = (nodes * mask_float.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.regression_head(self.dropout(pooled))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a SPICE EquiformerAdj on nine electronic/geometry QM9 targets."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/QM9"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/pharm_training/spice_qm9_finetune")
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--min-learning-rate", type=float, default=DEFAULT_MIN_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--train-size", type=int, default=110_000)
    parser.add_argument("--valid-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--fixed-split", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--no-auto-resume", action="store_true")
    args = parser.parse_args()
    if args.learning_rate <= 0 or args.learning_rate > 1e-5:
        parser.error("--learning-rate must be greater than 0 and at most 1e-5")
    if args.min_learning_rate < 0 or args.min_learning_rate > args.learning_rate:
        parser.error("--min-learning-rate must be between 0 and --learning-rate")
    return args


def make_config(args: argparse.Namespace, hidden_dim: int) -> BenchmarkConfig:
    return BenchmarkConfig(
        model_name="SPICEQM9FineTune",
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        hidden_dim=hidden_dim,
        dropout=args.dropout,
        lr=args.learning_rate,
        lr_decay_step=50,
        lr_decay_factor=0.5,
        weight_decay=args.weight_decay,
        optimizer="adamw",
        opt_eps=1e-8,
        scheduler="cosine",
        loss="l1",
        model_ema=False,
        model_ema_decay=0.9999,
        drop_path=0.0,
        warmup_lr=0.0,
        warmup_epochs=0,
        min_lr=args.min_learning_rate,
        seed=args.seed,
        split_seed=args.split_seed,
        seeds=args.seeds,
        vary_split_seed=not args.fixed_split,
        train_size=args.train_size,
        valid_size=args.valid_size,
        device=args.device,
        resume_from=args.resume_from,
        auto_resume=not args.no_auto_resume,
        target_preset=TARGET_PRESET,
        target=None,
    )


def main() -> None:
    args = arguments()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = EquiformerAdjConfig.from_mapping(payload.get("architecture"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "source_checkpoint": str(args.checkpoint),
        "target_preset": TARGET_PRESET,
        "targets": [
            "dipole_moment",
            "polarizability",
            "homo",
            "lumo",
            "homo_lumo_gap",
            "electronic_spatial_extent",
            "rotational_constant_a",
            "rotational_constant_b",
            "rotational_constant_c",
        ],
        "encoder_learning_rate": args.learning_rate,
    }
    (args.output_dir / "transfer_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    def model_factory(in_dim, hidden_dim, dropout, out_dim):
        del in_dim, hidden_dim
        return SPICEQM9FineTuner(args.checkpoint, out_dim=out_dim, dropout=dropout)

    train_baseline(make_config(args, architecture.hidden_dim), model_factory)


if __name__ == "__main__":
    main()
