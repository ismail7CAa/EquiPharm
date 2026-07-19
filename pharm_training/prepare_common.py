"""Create molecule-level split manifests for ANI-2x and SPICE HDF5 files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ANI_ELEMENTS = [1, 6, 7, 8, 9, 16, 17]
SPICE_ELEMENTS = [1, 3, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 35, 53]
SYMBOL_TO_Z = {
    "H": 1, "Li": 3, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Na": 11, "Mg": 12, "Si": 14, "P": 15, "S": 16, "Cl": 17,
    "K": 19, "Ca": 20, "Br": 35, "I": 53,
}


def _find_groups(handle: h5py.File, required: tuple[str, ...]):
    found = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Group) and all(key in obj for key in required):
            found.append(name)

    handle.visititems(visitor)
    return found


def _split(name: str, seed: int) -> str:
    value = int.from_bytes(
        hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big"
    ) / 2**64
    return "train" if value < 0.90 else "val" if value < 0.95 else "test"


def _atomic_numbers(values) -> list[int]:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.number):
        return [int(value) for value in array.reshape(-1)]
    result = []
    for value in array.reshape(-1):
        symbol = value.decode() if isinstance(value, bytes) else str(value)
        result.append(SYMBOL_TO_Z[symbol])
    return result


def prepare_manifest(dataset: str, source: Path, output: Path, seed: int = 42) -> None:
    sources = sorted(source.rglob("*.h5")) + sorted(source.rglob("*.hdf5")) if source.is_dir() else [source]
    if not sources or any(not path.is_file() for path in sources):
        raise FileNotFoundError(f"No HDF5 dataset files found at: {source}")
    if dataset == "ani2x":
        coordinate_key, species_key = "coordinates", "species"
        energy_candidates = ("energies", "energy", "wb97x_dz.energy")
        force_candidates = ("forces", "force", "wb97x_dz.forces")
        elements = ANI_ELEMENTS
        position_unit, energy_unit = "angstrom", "hartree"
    elif dataset == "spice":
        coordinate_key, species_key = "conformations", "atomic_numbers"
        energy_candidates = ("formation_energy",)
        force_candidates = ("dft_total_gradient",)
        elements = SPICE_ELEMENTS
        position_unit, energy_unit = "bohr", "hartree"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    records = []
    regression_rows, regression_targets = [], []
    for source_file in sources:
        with h5py.File(source_file, "r") as handle:
            groups = _find_groups(handle, (coordinate_key, species_key))
            for name in groups:
                group = handle[name]
                energy_key = next((key for key in energy_candidates if key in group), None)
                force_key = next((key for key in force_candidates if key in group), None)
                if energy_key is None:
                    continue
                atomic_numbers = _atomic_numbers(group[species_key][...])
                conformations = int(group[coordinate_key].shape[0])
                identity = f"{source_file.name}:{name}"
                record = {
                    "source": str(source_file.resolve()),
                    "group": name,
                    "split": _split(identity, seed),
                    "conformations": conformations,
                    "atomic_numbers": atomic_numbers,
                    "coordinate_key": coordinate_key,
                    "energy_key": energy_key,
                    "force_key": force_key,
                }
                records.append(record)
                if dataset == "ani2x" and record["split"] == "train":
                    counts = [atomic_numbers.count(z) for z in elements]
                    regression_rows.append(counts)
                    regression_targets.append(float(np.asarray(group[energy_key]).mean()))
    if not records:
        raise ValueError(f"No compatible molecule groups found at {source}")

    atomic_references = [0.0] * len(elements)
    if regression_rows:
        atomic_references = np.linalg.lstsq(
            np.asarray(regression_rows, dtype=np.float64),
            np.asarray(regression_targets, dtype=np.float64),
            rcond=None,
        )[0].tolist()
    metadata = {
        "format_version": 1,
        "dataset": dataset,
        "source": str(source.resolve()),
        "split_seed": seed,
        "split_strategy": "molecule_hash_90_5_5",
        "elements": elements,
        "position_unit": position_unit,
        "energy_unit": energy_unit,
        "force_is_negative_gradient": dataset == "spice",
        "atomic_reference_energies_hartree": atomic_references,
        "records": records,
        "counts": {
            split: sum(r["conformations"] for r in records if r["split"] == split)
            for split in ("train", "val", "test")
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata["counts"], indent=2))
    print(f"Manifest written to {output}")
