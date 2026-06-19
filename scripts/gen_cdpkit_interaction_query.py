#!/usr/bin/env python

from __future__ import annotations

import argparse
import re
from pathlib import Path

import CDPL.Chem as Chem
import CDPL.Biomol as Biomol
import CDPL.Pharm as Pharm
import CDPL.MolProp as MolProp


def read_receptor(receptor_file: str, strip_patterns: list[str]) -> Chem.BasicMolecule:
    reader = Chem.MoleculeReader(receptor_file)
    rec = Chem.BasicMolecule()

    if not reader.read(rec):
        raise RuntimeError(f"Could not read receptor: {receptor_file}")

    # Optional: remove residues such as water or the co-crystallized ligand if needed.
    if strip_patterns:
        atoms_to_remove = Chem.Fragment()
        compiled = [re.compile(p) for p in strip_patterns]

        for atom in rec.atoms:
            try:
                res_id = Biomol.getResidueCode(atom)
                chain_id = Biomol.getChainID(atom)
                res_num = str(Biomol.getResidueSequenceNumber(atom))
                full_id = f"{chain_id}_{res_id}_{res_num}"
            except Exception:
                full_id = ""

            if any(p.search(full_id) or p.search(res_id) for p in compiled):
                atoms_to_remove.addAtom(atom)

        if atoms_to_remove.numAtoms > 0:
            rec -= atoms_to_remove

    # Same preparation logic as CDPKit cookbook.
    Chem.perceiveSSSR(rec, True)
    Chem.setRingFlags(rec, True)
    Chem.calcImplicitHydrogenCounts(rec, True)
    Chem.perceiveHybridizationStates(rec, True)
    Chem.setAromaticityFlags(rec, True)
    Chem.calcAtomCIPConfigurations(rec, True)
    Chem.calcBondCIPConfigurations(rec, True)
    MolProp.calcAtomHydrophobicities(rec, False)

    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--receptor", required=True)
    parser.add_argument("-l", "--ligand", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-i", "--interaction-info", default=None)
    parser.add_argument(
        "-s",
        "--strip",
        nargs="*",
        default=["HOH"],
        help="Residue regex patterns to remove from receptor, e.g. HOH.",
    )
    parser.add_argument(
        "-x",
        "--exclusion-volumes",
        action="store_true",
        help="Generate exclusion volumes. Start without this for DUD-E screening.",
    )
    args = parser.parse_args()

    receptor = read_receptor(args.receptor, args.strip)

    lig_reader = Chem.MoleculeReader(args.ligand)
    writer = Pharm.FeatureContainerWriter(args.output)

    info_fh = open(args.interaction_info, "w") if args.interaction_info else None
    if info_fh:
        info_fh.write(
            "Input Ligand Index\tPharm Feature Index\tFeature Type\tLigand Atom Indices\tPocket Residues\n"
        )

    generator = Pharm.InteractionPharmacophoreGenerator()
    generator.addExclusionVolumes(args.exclusion_volumes)

    ligand = Chem.BasicMolecule()
    ph4 = Pharm.BasicPharmacophore()

    n_written = 0
    ligand_idx = 0

    while lig_reader.read(ligand):
        ligand_idx += 1
        ph4.clear()

        Chem.calcImplicitHydrogenCounts(ligand, True)
        Chem.perceiveHybridizationStates(ligand, True)
        Chem.setAromaticityFlags(ligand, True)
        Chem.calcAtomCIPConfigurations(ligand, True)
        Chem.calcBondCIPConfigurations(ligand, True)
        MolProp.calcAtomHydrophobicities(ligand, False)

        generator.generate(ligand, receptor, ph4, True)

        name = Chem.getName(ligand).strip() or Path(args.ligand).stem
        Pharm.setName(ph4, name)

        if ph4.getNumFeatures() == 0:
            print(f"[WARN] No interaction pharmacophore features generated for ligand {ligand_idx}: {name}")
            continue

        print(f"[INFO] Generated {ph4.getNumFeatures()} interaction features for {name}")

        if not writer.write(ph4):
            raise RuntimeError(f"Could not write pharmacophore to {args.output}")

        if info_fh:
            for ftr in ph4:
                if Pharm.getType(ftr) == Pharm.FeatureType.EXCLUSION_VOLUME:
                    continue

                lig_atom_inds = []
                try:
                    substruct = Pharm.getSubstructure(ftr)
                    lig_atom_inds = sorted(ligand.getAtomIndex(atom) for atom in substruct.atoms)
                except Exception:
                    pass

                try:
                    env = Pharm.getEnvironmentResidueInfo(ftr)
                except Exception:
                    env = ""

                info_fh.write(
                    f"{ligand_idx}\t{ph4.getFeatureIndex(ftr)}\t{Pharm.getType(ftr)}\t"
                    f"{','.join(map(str, lig_atom_inds))}\t{env}\n"
                )

        n_written += 1

    if info_fh:
        info_fh.close()

    if n_written == 0:
        raise RuntimeError("No pharmacophore query was written.")

    print(f"[OK] Wrote {n_written} pharmacophore query/query queries to {args.output}")


if __name__ == "__main__":
    main()
