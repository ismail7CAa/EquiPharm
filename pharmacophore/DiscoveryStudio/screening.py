"""Discovery Studio pharmacophore command-template baseline wrapper."""

from __future__ import annotations

try:
    from ..core.command_baseline import run_command_baseline_dataset_screening, run_command_baseline_screening
except ImportError:
    from pharmacophore.core.command_baseline import run_command_baseline_dataset_screening, run_command_baseline_screening


PIPELINE_NAME = "DiscoveryStudio"


def run_discovery_studio_screening(**kwargs):
    kwargs.setdefault("pipeline_name", PIPELINE_NAME)
    return run_command_baseline_screening(**kwargs)


def run_discovery_studio_dataset_screening(**kwargs):
    kwargs.setdefault("pipeline_name", PIPELINE_NAME)
    return run_command_baseline_dataset_screening(**kwargs)
