#!/usr/bin/env python
"""Reproducible energy/force pretraining for the EquiformerAdj encoder."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .data import PotentialDataset
from .equiformer_adj import EquiformerAdjPotential


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int)
    return parser.parse_args()


def scheduler_for(optimizer, config):
    if config.get("scheduler", "cosine") == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.get("lr_reduce_factor", 0.5),
            patience=config.get("lr_patience", 6),
            threshold=config.get("min_delta", 0.0),
            min_lr=config["min_learning_rate"],
        )
    warmup, epochs = config["warmup_epochs"], config["epochs"]
    minimum = config["min_learning_rate"] / config["learning_rate"]

    def scale(epoch):
        if warmup and epoch < warmup:
            return max(1e-3, (epoch + 1) / warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, scale)


def epoch_pass(model, loader, device, config, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "energy_loss": 0.0,
        "force_loss": 0.0,
        "energy_mae_ev_per_atom": 0.0,
        "force_mae_ev_per_angstrom": 0.0,
    }
    examples = 0
    skipped_nonfinite = 0
    for batch_number, batch in enumerate(
        tqdm(loader, desc="train" if training else "eval", leave=False), start=1
    ):
        if not hasattr(batch, "atom_type"):
            raise AttributeError("SPICE batch is missing the categorical atom_type field")
        if batch.atom_type.numel():
            minimum = int(batch.atom_type.min())
            maximum = int(batch.atom_type.max())
            embedding_size = model.species_embedding.num_embeddings
            if minimum < 0 or maximum >= embedding_size:
                raise IndexError(
                    f"atom_type range [{minimum}, {maximum}] is outside the element "
                    f"embedding range [0, {embedding_size - 1}]. This usually means "
                    "PyG incremented an attribute named element_index while batching."
                )
        batch = batch.to(device)
        force_mode = config.get("force_mode", "direct")
        batch.pos.requires_grad_(force_mode == "energy_gradient")
        if training:
            optimizer.zero_grad(set_to_none=True)
        if force_mode == "direct":
            predicted_energy, predicted_force = model.predict_energy_and_direct_forces(batch)
        elif force_mode == "energy_gradient":
            predicted_energy = model(batch)
            predicted_force = -torch.autograd.grad(
                predicted_energy.sum(), batch.pos, create_graph=training, retain_graph=training
            )[0]
        else:
            raise ValueError(f"Unknown force_mode: {force_mode}")
        atom_counts = torch.bincount(batch.batch, minlength=predicted_energy.numel()).clamp_min(1)
        energy_error = (predicted_energy - batch.y.view(-1)) / atom_counts
        energy_loss = F.smooth_l1_loss(energy_error, torch.zeros_like(energy_error))
        energy_mae = energy_error.abs().mean()
        if hasattr(batch, "force"):
            force_loss = F.smooth_l1_loss(predicted_force, batch.force)
            force_mae = (predicted_force - batch.force).abs().mean()
        else:
            force_loss = predicted_energy.new_zeros(())
            force_mae = predicted_energy.new_zeros(())
        loss = config["energy_weight"] * energy_loss + config["force_weight"] * force_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss encountered")
        if training:
            loss.backward()
            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["gradient_clip"], error_if_nonfinite=True
                )
            except RuntimeError as error:
                if "non-finite" not in str(error):
                    raise
                bad_gradients = [
                    name for name, parameter in model.named_parameters()
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                ]
                optimizer.zero_grad(set_to_none=True)
                skipped_nonfinite += 1
                maximum_skips = config.get("max_nonfinite_batches_per_epoch", 5)
                preview = ", ".join(bad_gradients[:8])
                print(
                    f"warning: skipped batch {batch_number} with non-finite gradients "
                    f"in {len(bad_gradients)} parameters: {preview}",
                    flush=True,
                )
                if skipped_nonfinite > maximum_skips:
                    raise FloatingPointError(
                        f"More than {maximum_skips} batches in this epoch produced "
                        f"non-finite gradients. Affected parameters include: {preview}. "
                        "This is systematic numerical instability, not a batch outlier."
                    )
                continue
            optimizer.step()
        count = predicted_energy.numel()
        examples += count
        totals["loss"] += loss.detach().item() * count
        totals["energy_loss"] += energy_loss.detach().item() * count
        totals["force_loss"] += force_loss.detach().item() * count
        totals["energy_mae_ev_per_atom"] += energy_mae.detach().item() * count
        totals["force_mae_ev_per_angstrom"] += force_mae.detach().item() * count
    if not examples:
        raise RuntimeError(
            "No usable batches were produced; reduce batch size, inspect the manifest, "
            "or review the non-finite-gradient diagnostics above."
        )
    if skipped_nonfinite:
        print(f"epoch skipped_nonfinite_batches={skipped_nonfinite}", flush=True)
    return {key: value / examples for key, value in totals.items()}


def validation_score(metrics, config):
    """A fixed, weight-independent score so hyperparameter trials are comparable."""
    energy_scale = config.get("selection_energy_scale_ev_per_atom", 0.043)
    force_scale = config.get("selection_force_scale_ev_per_angstrom", 0.1)
    energy_fraction = config.get("selection_energy_fraction", 0.5)
    if energy_scale <= 0 or force_scale <= 0 or not 0 <= energy_fraction <= 1:
        raise ValueError("Validation scales must be positive and energy fraction must be in [0, 1]")
    return (
        energy_fraction * metrics["energy_mae_ev_per_atom"] / energy_scale
        + (1 - energy_fraction) * metrics["force_mae_ev_per_angstrom"] / force_scale
    )


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    args = arguments()
    config = json.loads(args.config.read_text())
    if args.train_limit is not None:
        config["train_limit"] = args.train_limit
    if args.eval_limit is not None:
        config["eval_limit"] = args.eval_limit
    if config["dataset"].lower() != "spice" and config.get("spice_only", False):
        raise ValueError("This configuration is restricted to SPICE")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    manifest = Path(config["manifest"])
    dataset_options = {"sample_seed": config.get("subset_seed", seed)}
    train_data = PotentialDataset(
        manifest, "train", config["cutoff"], config["train_limit"], **dataset_options
    )
    val_data = PotentialDataset(
        manifest, "val", config["cutoff"], config["eval_limit"], **dataset_options
    )
    test_data = PotentialDataset(
        manifest, "test", config["cutoff"], config["eval_limit"], **dataset_options
    )
    loader_options = dict(batch_size=config["batch_size"], num_workers=config["num_workers"])
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)

    output = Path(config["output_dir"])
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    model = EquiformerAdjPotential(
        len(train_data.elements), architecture=config.get("architecture")
    ).to(device)
    optimizer = AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = scheduler_for(optimizer, config)
    start_epoch, best_score, best_val_loss, stale = 1, float("inf"), float("inf"), 0
    resume = args.resume or (checkpoints / "last.pt" if (checkpoints / "last.pt").exists() else None)
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = state["epoch"] + 1
        best_score = state.get("best_val_score", state.get("best_val_loss", float("inf")))
        best_val_loss = state.get("best_val_loss", float("inf"))
        stale = state["stale_epochs"]

    metrics_path = output / "metrics.csv"
    mode = "a" if start_epoch > 1 and metrics_path.exists() else "w"
    started = time.time()
    with metrics_path.open(mode, newline="") as handle, SummaryWriter(output / "logs") as writer:
        metric_names = ["loss", "energy_loss", "force_loss", "energy_mae_ev_per_atom",
                        "force_mae_ev_per_angstrom"]
        fields = (["epoch", "lr"] + [f"train_{key}" for key in metric_names]
                  + [f"val_{key}" for key in metric_names] + ["val_score"])
        csv_writer = csv.DictWriter(handle, fieldnames=fields)
        if mode == "w":
            csv_writer.writeheader()
        for epoch in range(start_epoch, config["epochs"] + 1):
            train_metrics = epoch_pass(model, train_loader, device, config, optimizer)
            val_metrics = epoch_pass(model, val_loader, device, config)
            row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            row["val_score"] = validation_score(val_metrics, config)
            csv_writer.writerow(row)
            handle.flush()
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)
            min_delta = config.get("min_delta", 0.0)
            improved = row["val_score"] < best_score - min_delta
            if improved:
                best_score, stale = row["val_score"], 0
                best_val_loss = val_metrics["loss"]
            else:
                stale += 1
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(row["val_score"])
            else:
                scheduler.step()
            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.transferable_state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_score": best_score,
                "stale_epochs": stale,
                "config": config,
                "architecture": model.architecture_config,
                "elements": train_data.elements,
            }
            checkpoint_interval = config.get("checkpoint_interval", 25)
            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                save(checkpoints / f"epoch_{epoch:04d}.pt", payload)
            save(checkpoints / "last.pt", payload)
            if improved:
                save(checkpoints / "best.pt", payload)
                save(checkpoints / "trained_encoder.pt", {
                    "encoder_state_dict": payload["encoder_state_dict"],
                    "source_checkpoint": str(checkpoints / "best.pt"),
                    "dataset": config["dataset"],
                    "architecture": payload["architecture"],
                    "config": config,
                })
            print(
                f"epoch={epoch} train={train_metrics['loss']:.6f} "
                f"val={val_metrics['loss']:.6f} score={row['val_score']:.6f} stale={stale}"
            )
            divergence_factor = config.get("divergence_factor", 4.0)
            minimum_epochs = config.get("minimum_epochs", 20)
            if epoch >= minimum_epochs and row["val_score"] > best_score * divergence_factor:
                print(f"Stopping: validation score diverged by more than {divergence_factor}x.")
                break
            if stale >= config["early_stopping_patience"]:
                print(f"Stopping: no validation improvement for {stale} epochs.")
                break

    best = torch.load(checkpoints / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    test_metrics = (
        epoch_pass(model, test_loader, device, config) if config.get("evaluate_test", True) else {}
    )
    results = {
        "dataset": config["dataset"], "best_epoch": best["epoch"],
        "best_val_loss": best["best_val_loss"], "best_val_score": best["best_val_score"],
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "training_seconds": time.time() - started,
        "trained_encoder": str(checkpoints / "trained_encoder.pt"),
    }
    with (output / "results.csv").open("w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(results))
        csv_writer.writeheader()
        csv_writer.writerow(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
