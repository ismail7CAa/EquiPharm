#!/usr/bin/env python
"""Prepare a molecule-disjoint SPICE manifest without copying the HDF5 data."""

import argparse
from pathlib import Path

from .prepare_common import prepare_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/SPICE/SPICE-2.0.1.hdf5"))
    parser.add_argument("--output", type=Path, default=Path("data/SPICE/prepared/manifest.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_manifest("spice", args.source, args.output, args.seed)


if __name__ == "__main__":
    main()
