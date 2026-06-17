"""Helpers for resumable screening score tables."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pandas as pd


SCORES_FILENAME = "scores.csv"


def initialize_score_file(output_dir: str | Path, fieldnames: list[str]) -> Path:
    """Create an empty scores.csv with headers if it does not exist yet."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    score_path = output_path / SCORES_FILENAME
    if not score_path.exists():
        with score_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
    return score_path


def load_resume_rows(output_dir: str | Path) -> tuple[list[dict], set[str]]:
    """Return existing score rows and paths with completed finite scores."""
    score_path = Path(output_dir) / SCORES_FILENAME
    if not score_path.exists():
        return [], set()

    df = pd.read_csv(score_path)
    rows = df.to_dict("records")
    completed_paths = {
        str(row["path"])
        for row in rows
        if row.get("path") and score_is_finite(row.get("score"))
    }
    return rows, completed_paths


def append_score_row(output_dir: str | Path, row: dict) -> None:
    """Append one candidate row to scores.csv so interrupted runs can resume."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    score_path = output_path / SCORES_FILENAME

    if score_path.exists():
        with score_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        missing = [key for key in row if key not in fieldnames]
        if missing:
            fieldnames.extend(missing)
            with score_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    else:
        fieldnames = list(row.keys())

    with score_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if score_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def score_is_finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
