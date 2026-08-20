import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch

repo = Path("external/PharmacoMatch").resolve()
sys.path.insert(0, str(repo))

import CDPL.Pharm as Pharm


def stable_seed(base_seed: int, target: str, dataset_name: str) -> int:
    """Return a deterministic seed for each target/class pair."""
    key = f"{target}:{dataset_name}".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
    return (base_seed + offset) % (2**32)


class SubsetPharmacophoreAlignment:
    def __init__(
        self,
        target_root: Path,
        output_dir: Path,
        n_actives: int,
        n_inactives: int,
        seed: int,
        selection_mode: str,
    ) -> None:
        self.target_root = target_root
        self.output_dir = output_dir
        self.n_actives = n_actives
        self.n_inactives = n_inactives
        self.seed = seed
        self.selection_mode = selection_mode
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.selection_manifest = {
            "target": self.target_root.name,
            "selection_mode": selection_mode,
            "base_seed": seed,
            "datasets": {},
        }

    def run(self) -> None:
        tic = time.perf_counter()
        self._alignment("actives", self.n_actives)
        self._alignment("inactives", self.n_inactives)
        self.alignment_time = time.perf_counter() - tic

        manifest_path = self.output_dir / "selected_molecule_ids.json"
        manifest_path.write_text(
            json.dumps(self.selection_manifest, indent=2), encoding="utf-8"
        )
        print(f"Selection manifest: {manifest_path}")
        print(f"Total alignment time: {self.alignment_time:.3f} s")

    def _alignment(self, dataset_name: str, requested_molecules: int) -> None:
        ref_ph4_file = self.target_root / "raw" / "query.pml"
        in_file = self.target_root / "raw" / f"{dataset_name}.psd"
        out_file = self.output_dir / f"all_{dataset_name}_aligned.pt"

        if not ref_ph4_file.exists():
            raise FileNotFoundError(ref_ph4_file)
        if not in_file.exists():
            raise FileNotFoundError(in_file)

        ref_ph4 = self._read_ref_pharmacophore(ref_ph4_file)
        db_accessor = Pharm.PSDScreeningDBAccessor(str(in_file))
        num_ph4s = int(db_accessor.getNumPharmacophores())

        # Select compounds, not individual conformers/pharmacophores.
        all_molecule_ids = sorted(
            {int(db_accessor.getMoleculeIndex(i)) for i in range(num_ph4s)}
        )

        if requested_molecules > len(all_molecule_ids):
            raise ValueError(
                f"Requested {requested_molecules} {dataset_name}, but only "
                f"{len(all_molecule_ids)} unique molecule IDs exist in {in_file}"
            )

        class_seed = stable_seed(
            self.seed, self.target_root.name, dataset_name
        )

        if self.selection_mode == "first":
            selected_ids = all_molecule_ids[:requested_molecules]
        else:
            rng = random.Random(class_seed)
            selected_ids = sorted(
                rng.sample(all_molecule_ids, requested_molecules)
            )

        selected_set = set(selected_ids)

        mol_ph4 = Pharm.BasicPharmacophore()
        alignment = Pharm.PharmacophoreAlignment(True)
        self._clear_feature_orientations(ref_ph4)
        alignment.addFeatures(ref_ph4, True)
        alignment.performExhaustiveSearch(False)

        fit_score = Pharm.PharmacophoreFitScore(
            match_cnt_weight=1.0,
            pos_match_weight=0.9,
            geom_match_weight=0.0,
        )

        alignment_scores = []
        scored_molecule_ids = set()
        selected_pharmacophore_count = 0

        for i in range(num_ph4s):
            mol_idx = int(db_accessor.getMoleculeIndex(i))
            if mol_idx not in selected_set:
                continue

            db_accessor.getPharmacophore(i, mol_ph4)
            conf_idx = int(db_accessor.getConformationIndex(i))
            selected_pharmacophore_count += 1

            if mol_ph4.getNumFeatures() == 0:
                continue

            self._clear_feature_orientations(mol_ph4)
            alignment.clearEntities(False)
            alignment.addFeatures(mol_ph4, False)

            solutions = []
            while alignment.nextAlignment():
                score = float(
                    fit_score(ref_ph4, mol_ph4, alignment.getTransform())
                )
                solutions.append(score)

            if solutions:
                solution = max(solutions)
                row = [
                    int(solution),
                    solution % 1,
                    mol_ph4.getNumFeatures(),
                    mol_idx,
                    conf_idx,
                ]
            else:
                row = [
                    0,
                    0.0,
                    mol_ph4.getNumFeatures(),
                    mol_idx,
                    conf_idx,
                ]

            alignment_scores.append(row)
            scored_molecule_ids.add(mol_idx)

        if not alignment_scores:
            raise RuntimeError(
                f"No alignment rows were produced for {dataset_name}"
            )

        scores_tensor = torch.tensor(alignment_scores, dtype=torch.float32)
        torch.save(scores_tensor, out_file)

        missing_ids = sorted(selected_set - scored_molecule_ids)
        self.selection_manifest["datasets"][dataset_name] = {
            "input_psd": str(in_file),
            "available_unique_molecules": len(all_molecule_ids),
            "requested_molecules": requested_molecules,
            "class_seed": class_seed,
            "selected_molecule_ids": selected_ids,
            "selected_pharmacophores": selected_pharmacophore_count,
            "scored_unique_molecules": len(scored_molecule_ids),
            "missing_after_alignment": missing_ids,
            "output_tensor": str(out_file),
        }

        print(
            f"{dataset_name}: selected {len(selected_ids)} molecules, "
            f"processed {selected_pharmacophore_count} pharmacophores, "
            f"scored {len(scored_molecule_ids)} unique molecules, "
            f"saved {scores_tensor.shape[0]} rows to {out_file}"
        )
        if missing_ids:
            print(
                f"WARNING: {len(missing_ids)} selected {dataset_name} molecules "
                "had no scored pharmacophore rows. See the manifest."
            )

    @staticmethod
    def _read_ref_pharmacophore(filename: Path):
        reader = Pharm.PharmacophoreReader(str(filename))
        ph4 = Pharm.BasicPharmacophore()
        if not reader.read(ph4):
            raise RuntimeError(
                f"Reading reference pharmacophore failed: {filename}"
            )
        return ph4

    @staticmethod
    def _clear_feature_orientations(ph4) -> None:
        for feature in ph4:
            Pharm.clearOrientation(feature)
            Pharm.setGeometry(feature, Pharm.FeatureGeometry.SPHERE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-actives", type=int, default=50)
    parser.add_argument("--n-inactives", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection-mode",
        choices=("random", "first"),
        default="random",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = SubsetPharmacophoreAlignment(
        target_root=args.target_root.resolve(),
        output_dir=args.output_dir.resolve(),
        n_actives=args.n_actives,
        n_inactives=args.n_inactives,
        seed=args.seed,
        selection_mode=args.selection_mode,
    )
    runner.run()
