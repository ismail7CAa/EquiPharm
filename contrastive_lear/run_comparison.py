#!/usr/bin/env python
"""Train both methods and write a unified cross-model comparison CSV."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Compare both Equiformers on PharmacoMatch.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/training_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/contrastive_lear"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    common = [
        "--data-dir", str(args.data_dir),
        "--output-dir", str(args.output_dir),
        "--seeds", *map(str, args.seeds),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.eval_batch_size),
        "--limit", str(args.limit),
        "--save-every", str(args.save_every),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
    ]
    methods = ["equiformer_adj", "equiformer_official"]
    for method in methods:
        subprocess.run(
            [sys.executable, "-m", "contrastive_lear.train", "--method", method, *common],
            check=True,
        )

    rows = []
    for method in methods:
        with (args.output_dir / method / "seed_summary.csv").open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    output = args.output_dir / "model_comparison.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Comparison written to {output}")


if __name__ == "__main__":
    main()
