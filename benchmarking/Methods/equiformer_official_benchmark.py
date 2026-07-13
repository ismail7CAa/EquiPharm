#!/usr/bin/env python
"""Run the original Equiformer backbone in this repository's QM9 pipeline."""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

try:
    from .benchmark_utils import parse_benchmark_args, train_baseline
except ImportError:
    from benchmark_utils import parse_benchmark_args, train_baseline


OFFICIAL_REPO_ENV = "EQUIFORMER_OFFICIAL_REPO"
DEFAULT_OFFICIAL_REPO = Path("external/equiformer_official")


def _install_optional_ocp_stub() -> None:
    """Allow the authors' Gaussian QM9 model to import without OC20 installed.

    The official module imports the OC20 Bessel layer eagerly even when the
    Gaussian basis is selected.  Our adapted benchmark uses the authors'
    Gaussian 128-basis configuration, so the unavailable class must never be
    instantiated.
    """
    try:
        importlib.import_module("ocpmodels.models.gemnet.layers.radial_basis")
        return
    except ImportError:
        pass

    class UnavailableRadialBasis(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            raise ImportError(
                "The optional OC20 package is required for the Bessel Equiformer variant."
            )

    module_names = [
        "ocpmodels",
        "ocpmodels.models",
        "ocpmodels.models.gemnet",
        "ocpmodels.models.gemnet.layers",
        "ocpmodels.models.gemnet.layers.radial_basis",
    ]
    for name in module_names:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules[module_names[-1]].RadialBasis = UnavailableRadialBasis


def load_official_qm9_module():
    repo = Path(os.environ.get(OFFICIAL_REPO_ENV, DEFAULT_OFFICIAL_REPO)).resolve()
    nets_dir = repo / "nets"
    source = nets_dir / "graph_attention_transformer.py"
    if not source.is_file():
        raise FileNotFoundError(
            f"Official Equiformer source not found at {source}. Clone "
            "https://github.com/atomicarchitects/equiformer to "
            f"{DEFAULT_OFFICIAL_REPO}, or set {OFFICIAL_REPO_ENV}."
        )

    _install_optional_ocp_stub()
    package_name = "equiformer_official_nets"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(nets_dir)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.graph_attention_transformer")


class OfficialEquiformerQM9(nn.Module):
    """Author backbone with a benchmark-compatible invariant output head."""

    def __init__(
        self,
        in_dim: int = 11,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        out_dim: int = 19,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        del in_dim
        if hidden_dim != 128:
            raise ValueError("The official QM9 architecture requires --hidden-dim 128.")
        if dropout != 0.0:
            raise ValueError("The official Gaussian QM9 configuration requires --dropout 0.0.")
        if drop_path != 0.0:
            raise ValueError("The official QM9 configuration requires --drop-path 0.0.")

        official = load_official_qm9_module()
        self.model = official.graph_attention_transformer_nonlinear_l2(
            irreps_in="5x0e",
            radius=5.0,
            num_basis=128,
            task_mean=None,
            task_std=None,
            atomref=None,
        )
        feature_irreps = self.model.irreps_feature
        self.model.head = nn.Sequential(
            official.LinearRS(feature_irreps, feature_irreps),
            official.Activation(feature_irreps, acts=[torch.nn.SiLU()]),
            official.LinearRS(feature_irreps, official.o3.Irreps(f"{out_dim}x0e")),
        )

    def forward(self, data):
        return self.model(
            f_in=data.x,
            pos=data.pos,
            batch=data.batch,
            node_atom=data.z,
        )


def main() -> None:
    config = parse_benchmark_args(
        model_name="EquiformerOfficialBenchmark",
        default_epochs=300,
        default_batch_size=128,
        default_eval_batch_size=128,
        default_hidden_dim=128,
        default_dropout=0.0,
        default_lr=5e-4,
        default_weight_decay=5e-3,
        default_optimizer="adamw",
        default_opt_eps=1e-8,
        default_scheduler="cosine",
        default_loss="l1",
        default_model_ema=False,
        default_drop_path=0.0,
        default_warmup_lr=1e-6,
        default_warmup_epochs=5,
        default_min_lr=1e-6,
        default_seed=0,
        default_split_seed=1,
        default_seeds=[1, 2, 3],
    )
    train_baseline(config, OfficialEquiformerQM9)


if __name__ == "__main__":
    main()
