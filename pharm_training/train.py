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
from torch.optim.lr_scheduler import LambdaLR
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
    totals = {"loss": 0.0, "energy_mae_ev_per_atom": 0.0, "force_mae_ev_per_angstrom": 0.0}
    examples = 0
    for batch in tqdm(loader, desc="train" if training else "eval", leave=False):
        batch = batch.to(device)
        batch.pos.requires_grad_(True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        predicted_energy = model(batch)
        predicted_force = -torch.autograd.grad(
            predicted_energy.sum(), batch.pos, create_graph=training, retain_graph=training
        )[0]
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
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
        count = predicted_energy.numel()
        examples += count
        totals["loss"] += loss.detach().item() * count
        totals["energy_mae_ev_per_atom"] += energy_mae.detach().item() * count
        totals["force_mae_ev_per_angstrom"] += force_mae.detach().item() * count
    if not examples:
        raise RuntimeError("Data loader produced no batches; reduce batch size or inspect the manifest.")
    return {key: value / examples for key, value in totals.items()}


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
    train_data = PotentialDataset(manifest, "train", config["cutoff"], config["train_limit"])
    val_data = PotentialDataset(manifest, "val", config["cutoff"], config["eval_limit"])
    test_data = PotentialDataset(manifest, "test", config["cutoff"], config["eval_limit"])
    loader_options = dict(batch_size=config["batch_size"], num_workers=config["num_workers"])
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)

    output = Path(config["output_dir"])
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    model = EquiformerAdjPotential(len(train_data.elements)).to(device)
    optimizer = AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = scheduler_for(optimizer, config)
    start_epoch, best_val, stale = 1, float("inf"), 0
    resume = args.resume or (checkpoints / "last.pt" if (checkpoints / "last.pt").exists() else None)
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch, best_val, stale = state["epoch"] + 1, state["best_val_loss"], state["stale_epochs"]

    metrics_path = output / "metrics.csv"
    mode = "a" if start_epoch > 1 and metrics_path.exists() else "w"
    started = time.time()
    with metrics_path.open(mode, newline="") as handle, SummaryWriter(output / "logs") as writer:
        fields = ["epoch", "lr", "train_loss", "train_energy_mae_ev_per_atom",
                  "train_force_mae_ev_per_angstrom", "val_loss",
                  "val_energy_mae_ev_per_atom", "val_force_mae_ev_per_angstrom"]
        csv_writer = csv.DictWriter(handle, fieldnames=fields)
        if mode == "w":
            csv_writer.writeheader()
        for epoch in range(start_epoch, config["epochs"] + 1):
            train_metrics = epoch_pass(model, train_loader, device, config, optimizer)
            val_metrics = epoch_pass(model, val_loader, device, config)
            row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            csv_writer.writerow(row)
            handle.flush()
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)
            improved = val_metrics["loss"] < best_val
            if improved:
                best_val, stale = val_metrics["loss"], 0
            else:
                stale += 1
            scheduler.step()
            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.transferable_state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val,
                "stale_epochs": stale,
                "config": config,
                "elements": train_data.elements,
            }
            save(checkpoints / f"epoch_{epoch:04d}.pt", payload)
            save(checkpoints / "last.pt", payload)
            if improved:
                save(checkpoints / "best.pt", payload)
                save(checkpoints / "trained_encoder.pt", {
                    "encoder_state_dict": payload["encoder_state_dict"],
                    "source_checkpoint": str(checkpoints / "best.pt"),
                    "dataset": config["dataset"],
                    "config": config,
                })
            print(f"epoch={epoch} train={train_metrics['loss']:.6f} val={val_metrics['loss']:.6f}")
            if stale >= config["early_stopping_patience"]:
                break

    best = torch.load(checkpoints / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    test_metrics = epoch_pass(model, test_loader, device, config)
    results = {
        "dataset": config["dataset"], "best_epoch": best["epoch"],
        "best_val_loss": best["best_val_loss"], **{f"test_{k}": v for k, v in test_metrics.items()},
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
