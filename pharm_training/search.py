#!/usr/bin/env python
"""Deterministic, resumable hyperparameter search for SPICE pretraining."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import subprocess
import sys
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def nested_set(mapping, dotted_key, value):
    parts = dotted_key.split(".")
    target = mapping
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def trial_id(parameters):
    canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:10]


def build_trials(search_config):
    parameters = search_config["parameters"]
    keys = sorted(parameters)
    combinations = [dict(zip(keys, values)) for values in itertools.product(
        *(parameters[key] for key in keys)
    )]
    random.Random(search_config.get("seed", 42)).shuffle(combinations)
    return combinations


def read_result(path):
    with path.open(newline="") as handle:
        return next(csv.DictReader(handle))


def write_summary(output, rows, base_config):
    ranked = sorted(
        rows,
        key=lambda row: float(row.get("best_val_score", "inf"))
        if row.get("status") == "complete" else float("inf"),
    )
    fields = sorted({key for row in ranked for key in row})
    with (output / "search_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ranked)
    complete = [row for row in ranked if row.get("status") == "complete"]
    if complete:
        best_config = json.loads(Path(complete[0]["config_path"]).read_text())
        (output / "best_config.json").write_text(json.dumps(best_config, indent=2))
        full_config = json.loads(json.dumps(best_config))
        for key in ("epochs", "early_stopping_patience", "minimum_epochs", "train_limit",
                    "eval_limit", "evaluate_test"):
            full_config[key] = base_config[key]
        full_config["output_dir"] = str(output / "final_full_run")
        (output / "best_full_config.json").write_text(json.dumps(full_config, indent=2))


def main():
    args = arguments()
    search = json.loads(args.config.read_text())
    base = json.loads(Path(search["base_config"]).read_text())
    output = Path(search["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    trials = build_trials(search)
    maximum = args.max_trials if args.max_trials is not None else search.get("max_trials")
    if maximum is not None:
        trials = trials[:maximum]

    rows = []
    for index, parameters in enumerate(trials, 1):
        identifier = trial_id(parameters)
        trial_dir = output / f"trial_{identifier}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        config = json.loads(json.dumps(base))
        config.update(search.get("overrides", {}))
        for key, value in parameters.items():
            nested_set(config, key, value)
        config["output_dir"] = str(trial_dir)
        config_path = trial_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2))
        result_path = trial_dir / "results.csv"
        row = {
            "trial": identifier,
            "status": "pending",
            "config_path": str(config_path.resolve()),
            **{f"parameter.{key}": value for key, value in parameters.items()},
        }
        if result_path.exists():
            row.update(read_result(result_path))
            row["status"] = "complete"
        elif args.dry_run:
            row["status"] = "dry_run"
        else:
            print(f"[{index}/{len(trials)}] trial={identifier} parameters={parameters}", flush=True)
            with (trial_dir / "console.log").open("a") as log:
                process = subprocess.run(
                    [sys.executable, "-m", "pharm_training.train", "--config", str(config_path),
                     "--device", args.device],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if process.returncode == 0 and result_path.exists():
                row.update(read_result(result_path))
                row["status"] = "complete"
            else:
                row["status"] = "failed"
                row["return_code"] = process.returncode
        rows.append(row)
        write_summary(output, rows, base)
    print(f"Search summary: {output / 'search_summary.csv'}")


if __name__ == "__main__":
    main()
