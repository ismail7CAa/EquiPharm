#!/usr/bin/env python
"""Aggregate numeric metrics from multiple seeded benchmark runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("metrics", nargs="+", type=Path)
    args = parser.parse_args()

    rows = []
    for path in args.metrics:
        if not path.is_file():
            print(f"Skipping missing metrics file: {path}")
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        rows.append((path, row))

    if not rows:
        raise RuntimeError(f"No completed seed metrics found for {args.target}")

    common_keys = set.intersection(*(set(row) for _, row in rows))
    means = {}
    for key in sorted(common_keys):
        values = [row[key] for _, row in rows]
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            finite_values = [float(value) for value in values if math.isfinite(float(value))]
            if finite_values:
                means[key] = sum(finite_values) / len(finite_values)

    result = {
        "target": args.target,
        "aggregation": "arithmetic_mean_across_completed_seeds",
        "n_completed_seeds": len(rows),
        "source_metrics": [str(path) for path, _ in rows],
        "mean": means,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
