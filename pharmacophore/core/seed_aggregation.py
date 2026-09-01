"""Run repeated screening seeds and aggregate their numeric metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path


def run_seeded(runner, config: dict, seeds=(1, 2, 3)) -> dict:
    output_root = Path(config["output_dir"])
    seed_results = []
    for seed in seeds:
        seed_config = dict(config)
        seed_config["seed"] = seed
        seed_config["output_dir"] = str(output_root / f"seed_{seed}")
        seed_results.append((seed, runner(**seed_config)))

    common_keys = set.intersection(*(set(metrics) for _, metrics in seed_results))
    means = {}
    for key in sorted(common_keys):
        values = [metrics[key] for _, metrics in seed_results]
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            finite_values = [float(value) for value in values if math.isfinite(float(value))]
            if finite_values:
                means[key] = sum(finite_values) / len(finite_values)

    result = {
        "aggregation": "arithmetic_mean_across_completed_seeds",
        "n_completed_seeds": len(seed_results),
        "seeds": [seed for seed, _ in seed_results],
        "mean": means,
        "per_seed": {str(seed): metrics for seed, metrics in seed_results},
    }
    mean_dir = output_root / "seed_mean"
    mean_dir.mkdir(parents=True, exist_ok=True)
    (mean_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
