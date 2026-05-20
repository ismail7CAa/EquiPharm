"""Shared utilities for QM9 2D GNN benchmark scripts."""

from __future__ import annotations

import argparse
import copy
import csv
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from ase.units import Ang, Bohr, Hartree, eV
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

try:
    from .train_eval import evaluate, train_epoch
except ImportError:
    from train_eval import evaluate, train_epoch


PROPERTY_NAMES = [
    "μ (D)",
    "α (Ang³)",
    "ε_HOMO (eV)",
    "ε_LUMO (eV)",
    "Δε (eV)",
    "⟨R²⟩ (Ang²)",
    "ZPVE (eV)",
    "U₀ (eV)",
    "U (eV)",
    "H (eV)",
    "G (eV)",
    "c_v",
    "U₀_atom",
    "U_atom",
    "H_atom",
    "G_atom",
    "A (GHz)",
    "B (GHz)",
    "C (GHz)",
]


@dataclass(frozen=True)
class BenchmarkConfig:
    model_name: str
    data_dir: Path
    output_dir: Path
    epochs: int
    batch_size: int
    eval_batch_size: int
    hidden_dim: int
    dropout: float
    lr: float
    lr_decay_step: int
    lr_decay_factor: float
    weight_decay: float
    optimizer: str
    opt_eps: float
    scheduler: str
    loss: str
    model_ema: bool
    model_ema_decay: float
    drop_path: float
    warmup_lr: float
    warmup_epochs: int
    min_lr: float
    seed: int
    train_size: int
    valid_size: int
    device: str


def get_qm9_conversions_tensor(device: torch.device | str) -> torch.Tensor:
    """Return conversion factors for QM9 targets in the benchmark units."""
    return torch.tensor(
        [
            1.0,  # 0 - μ (D)
            Bohr**3 / Ang**3,  # 1 - α (Bohr³ -> Ang³)
            Hartree / eV,  # 2 - ε_HOMO
            Hartree / eV,  # 3 - ε_LUMO
            Hartree / eV,  # 4 - Δε
            Bohr**2 / Ang**2,  # 5 - ⟨R²⟩
            Hartree / eV,  # 6 - ZPVE
            Hartree / eV,  # 7 - U₀
            Hartree / eV,  # 8 - U
            Hartree / eV,  # 9 - H
            Hartree / eV,  # 10 - G
            1.0,  # 11 - c_v
            1.0,  # 12 - U₀_atom
            Hartree / eV,  # 13 - U_atom
            Hartree / eV,  # 14 - H_atom
            Hartree / eV,  # 15 - G_atom
            1.0,  # 16 - A (GHz)
            1.0,  # 17 - B (GHz)
            1.0,  # 18 - C (GHz)
        ],
        dtype=torch.float,
        device=device,
    )


def load_qm9_splits(config: BenchmarkConfig):
    dataset = QM9(root=str(config.data_dir))
    num_mols = len(dataset)

    if config.train_size + config.valid_size >= num_mols:
        raise ValueError(
            "train_size + valid_size must be smaller than the QM9 dataset size "
            f"({num_mols})."
        )

    generator = torch.Generator().manual_seed(config.seed)
    perm = torch.randperm(num_mols, generator=generator)

    train_idx = perm[: config.train_size]
    valid_idx = perm[config.train_size : config.train_size + config.valid_size]
    test_idx = perm[config.train_size + config.valid_size :]

    y_raw_all = dataset.data.y.clone().cpu()
    conversions_cpu = get_qm9_conversions_tensor("cpu")
    y_conv_all = y_raw_all * conversions_cpu.unsqueeze(0)

    norm_stats = {"mean": [], "std": []}
    y_norm_all = torch.zeros_like(y_conv_all)

    for target_idx in range(y_conv_all.shape[1]):
        train_y = y_conv_all[train_idx.cpu(), target_idx]
        mean = float(train_y.mean().item())
        std = float(train_y.std().item())
        std = std if std != 0.0 else 1.0

        y_norm_all[:, target_idx] = (y_conv_all[:, target_idx] - mean) / std
        norm_stats["mean"].append(mean)
        norm_stats["std"].append(std)

    dataset.data.y = y_norm_all.to(torch.float)
    return dataset, dataset[train_idx], dataset[valid_idx], dataset[test_idx], norm_stats


def write_best_metrics(path: Path, best_val: list[float], best_test: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target", "val_MAE", "test_MAE"])
        writer.writerows(zip(PROPERTY_NAMES, best_val, best_test))


def write_run_config(path: Path, config: BenchmarkConfig) -> None:
    config_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(config).items()
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config_dict, handle, indent=2, sort_keys=True)
        handle.write("\n")


def safe_metric_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


class ModelEma:
    """Track an exponential moving average copy of model weights."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


def build_optimizer(config: BenchmarkConfig, model: nn.Module):
    if config.optimizer == "adam":
        return Adam(
            model.parameters(),
            lr=config.lr,
            eps=config.opt_eps,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adamw":
        return AdamW(
            model.parameters(),
            lr=config.lr,
            eps=config.opt_eps,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def build_scheduler(config: BenchmarkConfig, optimizer):
    if config.scheduler == "step":
        return StepLR(
            optimizer,
            step_size=config.lr_decay_step,
            gamma=config.lr_decay_factor,
        )

    if config.scheduler in {"cosine", "warmup_cosine"}:
        warmup_epochs = max(0, config.warmup_epochs)
        decay_epochs = max(1, config.epochs - warmup_epochs)
        min_lr_ratio = config.min_lr / config.lr if config.lr > 0 else 0.0
        warmup_lr_ratio = config.warmup_lr / config.lr if config.lr > 0 else 0.0

        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                if warmup_epochs == 1:
                    return 1.0
                progress = float(epoch) / float(warmup_epochs - 1)
                return warmup_lr_ratio + (1.0 - warmup_lr_ratio) * progress

            progress = min(1.0, max(0.0, float(epoch - warmup_epochs) / float(decay_epochs)))
            cosine_decay = 0.5 * (1.0 + math.cos(progress * math.pi))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def build_model(
    model_factory: Callable[..., nn.Module],
    in_dim: int,
    hidden_dim: int,
    dropout: float,
    out_dim: int,
    drop_path: float,
) -> nn.Module:
    factory_signature = inspect.signature(model_factory)
    if "drop_path" in factory_signature.parameters:
        return model_factory(in_dim, hidden_dim, dropout, out_dim, drop_path=drop_path)
    return model_factory(in_dim, hidden_dim, dropout, out_dim)


def train_baseline(
    config: BenchmarkConfig,
    model_factory: Callable[..., nn.Module],
) -> None:
    torch.manual_seed(config.seed)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but no CUDA device is available. "
            "Run on a GPU node or pass --device cpu for a local smoke test."
        )

    device = torch.device(
        config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    checkpoint_dir = config.output_dir / "checkpoints"
    log_dir = config.output_dir / "logs"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(config.output_dir / "config.json", config)

    dataset, train_dataset, valid_dataset, test_dataset, norm_stats = load_qm9_splits(config)
    print(f"Using device: {device}")
    print(f"Total QM9 molecules: {len(dataset)}")
    print(f"Node feature dim: {dataset.num_node_features}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=config.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False)

    model = build_model(
        model_factory,
        dataset.num_node_features,
        config.hidden_dim,
        config.dropout,
        len(PROPERTY_NAMES),
        config.drop_path,
    ).to(device)

    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    model_ema = ModelEma(model, config.model_ema_decay) if config.model_ema else None

    best_mean_val = float("inf")
    best_val = [float("inf")] * len(PROPERTY_NAMES)
    best_test = [float("inf")] * len(PROPERTY_NAMES)

    print(f"#params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training {config.model_name} for all 19 QM9 targets")
    print(
        "Optimization: "
        f"{config.optimizer}, opt_eps={config.opt_eps}, scheduler={config.scheduler}, lr={config.lr}, "
        f"weight_decay={config.weight_decay}, loss={config.loss}, "
        f"warmup_lr={config.warmup_lr}, warmup_epochs={config.warmup_epochs}, "
        f"min_lr={config.min_lr}, model_ema={config.model_ema}, "
        f"model_ema_decay={config.model_ema_decay}, drop_path={config.drop_path}"
    )

    with SummaryWriter(log_dir=str(log_dir)) as writer:
        for epoch in range(1, config.epochs + 1):
            print(f"\n=== Epoch {epoch} ===")
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                loss_fn=config.loss,
                model_ema=model_ema,
            )
            eval_model = model_ema.module if model_ema is not None else model
            val_mae = evaluate(eval_model, val_loader, device, norm_stats["mean"], norm_stats["std"])
            test_mae = evaluate(eval_model, test_loader, device, norm_stats["mean"], norm_stats["std"])

            writer.add_scalar(f"train/loss_{config.loss}", train_loss, epoch)
            print(f"Train loss ({config.loss.upper()}): {train_loss:.6f}")

            for target_idx, prop in enumerate(PROPERTY_NAMES):
                metric_name = safe_metric_name(prop)
                writer.add_scalar(f"val_mae/{metric_name}", val_mae[target_idx], epoch)
                writer.add_scalar(f"test_mae/{metric_name}", test_mae[target_idx], epoch)
                print(
                    f"  {prop:15s} | Val MAE: {val_mae[target_idx]:.6f} "
                    f"| Test MAE: {test_mae[target_idx]:.6f}"
                )

            mean_val_mae = sum(val_mae) / len(val_mae)
            if mean_val_mae < best_mean_val:
                best_mean_val = mean_val_mae
                best_val = val_mae
                best_test = test_mae

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "model_ema_state_dict": (
                            model_ema.module.state_dict() if model_ema is not None else None
                        ),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val": best_val,
                        "best_test": best_test,
                        "best_mean_val": best_mean_val,
                        "config": vars(config),
                    },
                    checkpoint_dir / "best_model.pt",
                )
                write_best_metrics(config.output_dir / "best_val_test_mae.csv", best_val, best_test)
                print(f"Saved best model with mean validation MAE {best_mean_val:.4f}")

            scheduler.step()

    print("\nFinished training.")
    print("Best validation and test MAEs per property:")
    for prop, val, test in zip(PROPERTY_NAMES, best_val, best_test):
        print(f"  {prop:15s} | Best validation MAE: {val:.6f} | Test MAE at best val: {test:.6f}")


def parse_benchmark_args(
    model_name: str,
    default_epochs: int,
    default_batch_size: int = 128,
    default_eval_batch_size: int = 256,
    default_hidden_dim: int = 128,
    default_dropout: float = 0.3,
    default_lr: float = 1e-5,
    default_lr_decay_step: int = 50,
    default_lr_decay_factor: float = 0.5,
    default_weight_decay: float = 1e-4,
    default_optimizer: str = "adam",
    default_opt_eps: float = 1e-8,
    default_scheduler: str = "step",
    default_loss: str = "mse",
    default_model_ema: bool = False,
    default_model_ema_decay: float = 0.9999,
    default_drop_path: float = 0.0,
    default_warmup_lr: float = 0.0,
    default_warmup_epochs: int = 0,
    default_min_lr: float = 0.0,
) -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description=f"Train a {model_name} baseline on QM9.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/QM9"))
    parser.add_argument("--output-dir", type=Path, default=Path(f"runs/{model_name}"))
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=default_eval_batch_size)
    parser.add_argument("--hidden-dim", type=int, default=default_hidden_dim)
    parser.add_argument("--dropout", type=float, default=default_dropout)
    parser.add_argument("--lr", type=float, default=default_lr)
    parser.add_argument("--lr-decay-step", type=int, default=default_lr_decay_step)
    parser.add_argument("--lr-decay-factor", type=float, default=default_lr_decay_factor)
    parser.add_argument("--weight-decay", type=float, default=default_weight_decay)
    parser.add_argument("--optimizer", "--opt", choices=["adam", "adamw"], default=default_optimizer)
    parser.add_argument("--opt-eps", type=float, default=default_opt_eps)
    parser.add_argument(
        "--scheduler",
        "--sched",
        choices=["step", "cosine", "warmup_cosine"],
        default=default_scheduler,
    )
    parser.add_argument("--loss", choices=["mse", "l1"], default=default_loss)
    parser.add_argument("--model-ema", action="store_true", default=default_model_ema)
    parser.add_argument("--no-model-ema", action="store_false", dest="model_ema")
    parser.add_argument("--model-ema-decay", type=float, default=default_model_ema_decay)
    parser.add_argument("--drop-path", type=float, default=default_drop_path)
    parser.add_argument("--warmup-lr", type=float, default=default_warmup_lr)
    parser.add_argument("--warmup-epochs", type=int, default=default_warmup_epochs)
    parser.add_argument("--min-lr", type=float, default=default_min_lr)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=110_000)
    parser.add_argument("--valid-size", type=int, default=10_000)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    return BenchmarkConfig(model_name=model_name, **vars(args))
