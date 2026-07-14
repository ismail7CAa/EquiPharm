#!/usr/bin/env python
"""Train either Equiformer method on PharmacoMatch's contrastive objective."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .augment import add_complete_edges, make_views
from .data import PharmacoMatchTrainingDataset, split_dataset
from .loss import binary_auroc, order_embedding_loss
from .methods import METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contrastive Equiformer training on PharmacoMatch.")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/training_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/contrastive_lear"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=100.0)
    parser.add_argument("--radius", type=float, default=1.5)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    return parser.parse_args()


def prepare_views(batch, device, radius, method):
    batch = batch.to(device)
    views = make_views(batch, radius)
    for name, view in views.items():
        view = view.to(device)
        if method == "equiformer_adj":
            view = add_complete_edges(view)
        views[name] = view
    return views


def run_epoch(model, loader, device, args, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses, scores, labels = [], [], []
    positive_penalties, negative_penalties = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="train" if training else "eval", leave=False):
            views = prepare_views(batch, device, args.radius, args.method)
            if training:
                optimizer.zero_grad(set_to_none=True)
            embeddings = {name: model(view) for name, view in views.items()}
            loss, batch_scores, batch_labels, positive, negative = order_embedding_loss(
                embeddings, args.margin
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(loss.detach().cpu())
            scores.append(batch_scores.detach().cpu())
            labels.append(batch_labels.detach().cpu())
            positive_penalties.append(positive.detach().cpu())
            negative_penalties.append(negative.detach().cpu())
    all_scores, all_labels = torch.cat(scores), torch.cat(labels)
    return {
        "loss": torch.stack(losses).mean().item(),
        "auroc": binary_auroc(all_scores, all_labels),
        "positive_penalty": torch.stack(positive_penalties).mean().item(),
        "negative_penalty": torch.stack(negative_penalties).mean().item(),
    }


def save_checkpoint(path, epoch, model, optimizer, best_val, args):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_auroc": best_val,
            "config": vars(args),
        },
        temporary,
    )
    temporary.replace(path)


def train_seed(args, dataset, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_set, val_set, test_set = split_dataset(dataset, args.split_seed, args.limit)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False,
                             num_workers=args.num_workers)

    run_dir = args.output_dir / args.method / f"seed_{seed}"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as handle:
        json.dump({**vars(args), "seed": seed}, handle, indent=2, default=str)

    model = METHODS[args.method](embedding_dim=args.embedding_dim).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    history_path = run_dir / "metrics.csv"
    best_val = -1.0
    best_test = float("nan")
    start = time.time()
    with history_path.open("w", newline="") as history_file, SummaryWriter(run_dir / "logs") as writer:
        fieldnames = [
            "epoch", "train_loss", "train_auroc", "val_loss", "val_auroc",
            "test_loss", "test_auroc", "positive_penalty", "negative_penalty",
        ]
        csv_writer = csv.DictWriter(history_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, device, args, optimizer)
            val_metrics = run_epoch(model, val_loader, device, args)
            test_metrics = run_epoch(model, test_loader, device, args)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_auroc": train_metrics["auroc"],
                "val_loss": val_metrics["loss"],
                "val_auroc": val_metrics["auroc"],
                "test_loss": test_metrics["loss"],
                "test_auroc": test_metrics["auroc"],
                "positive_penalty": val_metrics["positive_penalty"],
                "negative_penalty": val_metrics["negative_penalty"],
            }
            csv_writer.writerow(row)
            history_file.flush()
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)
            if epoch % args.save_every == 0:
                save_checkpoint(checkpoint_dir / f"epoch_{epoch:04d}.pt", epoch, model, optimizer, best_val, args)
            save_checkpoint(checkpoint_dir / "last.pt", epoch, model, optimizer, best_val, args)
            if val_metrics["auroc"] > best_val:
                best_val = val_metrics["auroc"]
                best_test = test_metrics["auroc"]
                save_checkpoint(checkpoint_dir / "best.pt", epoch, model, optimizer, best_val, args)
            print(
                f"{args.method} seed={seed} epoch={epoch}: "
                f"train_loss={train_metrics['loss']:.4f} val_auc={val_metrics['auroc']:.4f} "
                f"test_auc={test_metrics['auroc']:.4f}"
            )
    return {
        "method": args.method,
        "seed": seed,
        "split_seed": args.split_seed,
        "best_val_auroc": best_val,
        "test_auroc_at_best_val": best_test,
        "training_seconds": time.time() - start,
        "output_dir": str(run_dir),
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    dataset = PharmacoMatchTrainingDataset(args.data_dir)
    summaries = [train_seed(args, dataset, seed, device) for seed in args.seeds]
    summary_path = args.output_dir / args.method / "seed_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
