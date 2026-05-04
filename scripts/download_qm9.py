#!/usr/bin/env python
"""Download and process QM9 into the repository's default data path."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch_geometric.datasets import QM9


def main() -> None:
    parser = argparse.ArgumentParser(description="Download QM9 for the benchmark scripts.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/QM9"))
    args = parser.parse_args()

    dataset = QM9(root=str(args.output_dir))
    print(f"QM9 ready at {args.output_dir} ({len(dataset)} molecules).")


if __name__ == "__main__":
    main()
