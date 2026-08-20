import argparse
import json
import random
from pathlib import Path

from rdkit import Chem


def molecule_name(mol, fallback):
    if mol.HasProp("_Name"):
        name = mol.GetProp("_Name").strip()
        if name:
            return name

    for key in ("ID", "Id", "id", "MOL_ID", "mol_id", "NAME", "Name"):
        if mol.HasProp(key):
            value = mol.GetProp(key).strip()
            if value:
                return value

    return fallback


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle valid SDF records reproducibly and write a fixed-size subset."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--target", required=True)
    parser.add_argument("--class-label", required=True, choices=("active", "inactive"))
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    supplier = Chem.SDMolSupplier(str(input_path), sanitize=False, removeHs=False)

    valid_records = []
    invalid_indices = []

    for original_index, mol in enumerate(supplier):
        if mol is None:
            invalid_indices.append(original_index)
            continue

        valid_records.append(
            {
                "original_index_zero_based": original_index,
                "original_index_one_based": original_index + 1,
                "name": molecule_name(mol, f"record_{original_index + 1}"),
                "mol": mol,
            }
        )

    if len(valid_records) < args.count:
        raise RuntimeError(
            f"Requested {args.count} {args.class_label}s, but only "
            f"{len(valid_records)} valid molecules were found in {input_path}."
        )

    # DUD-E records may have a meaningful/sorted source order. Shuffle the full
    # valid record order immediately before taking the requested subset.
    shuffled = valid_records.copy()
    rng = random.Random(args.seed)
    rng.shuffle(shuffled)
    selected = shuffled[: args.count]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output_path))
    if writer is None:
        raise RuntimeError(f"Could not open output SDF: {output_path}")

    try:
        for item in selected:
            writer.write(item["mol"])
    finally:
        writer.close()

    manifest = {
        "target": args.target,
        "class": args.class_label,
        "seed": args.seed,
        "selection": "shuffle_all_valid_records_then_take_first_n",
        "input_sdf": str(input_path),
        "output_sdf": str(output_path),
        "requested_count": args.count,
        "available_valid_count": len(valid_records),
        "invalid_record_count": len(invalid_indices),
        "invalid_original_indices_zero_based": invalid_indices,
        "selected": [
            {
                "selection_rank_one_based": rank,
                "original_index_zero_based": item["original_index_zero_based"],
                "original_index_one_based": item["original_index_one_based"],
                "name": item["name"],
            }
            for rank, item in enumerate(selected, start=1)
        ],
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        json.dumps(
            {
                "target": args.target,
                "class": args.class_label,
                "seed": args.seed,
                "selected": len(selected),
                "available_valid": len(valid_records),
                "invalid": len(invalid_indices),
                "output": str(output_path),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
