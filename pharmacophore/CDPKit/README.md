# CDPKit Baseline

Optional adapter for CDPKit's `psdcreate` and `psdscreen` command-line tools.

This wrapper is intentionally isolated from the EquiPharm implementation. It prepares a combined active/decoy SDF, builds a CDPKit pharmacophore-screening database, screens it with a supplied query pharmacophore, then writes the same `scores.csv`, `ranked_hits.csv`, `metrics.json`, and plots used by the other pipelines.

Important distinction: this wrapper is the simple command-line CDPKit baseline. It uses `psdscreen` hit output, so non-hit molecules are not reported by CDPKit and are assigned a default miss score by the wrapper. This is not the same scoring path as the CDPKit comparison used around PharmacoMatch.

For the PharmacoMatch-style CDPKit comparison, use the CDPL pharmacophore-alignment scripts in `scripts/`, especially:

```text
scripts/run_cdpkit_pharmacomatch_all_dude.sh
scripts/run_pharmacomatch_cdpkit_alignment.py
scripts/eval_pharmacomatch_cdpkit_alignment.py
```

That workflow uses the CDPL pharmacophore alignment API to compute an alignment score for every ligand pharmacophore, then ranks molecules by their best alignment score. See `challenges_and_limitations/00_PharmacoMatch_CDPKit.md` for the detailed limitation.

The candidate data layout is the same as EquiPharm:

```text
data/DUD-E/<target>/
  actives_sdf/
  decoys_sdf/
```

CDPKit additionally needs a pharmacophore query file supported by `psdscreen`: `.cdf`, `.pml`, or `.psd`. When `--target-dir` is used, the CLI will look for `query.cdf`, `query.pml`, `query.psd`, `crystal_ligand.cdf`, or `crystal_ligand.pml` inside that target directory. If none exists, it generates `query.pml` from `crystal_ligand.mol2` or `crystal_ligand.sdf` using the Python CDPL API.

Example:

```bash
python -m pharmacophore.CDPKit.cli \
  --target-dir data/DUD-E/<target> \
  --output-dir pharmacophore/results/CDPKit/<target>
```

Run every target in a dataset:

```bash
python -m pharmacophore.CDPKit.cli \
  --dataset-dir data/DUD-E \
  --output-dir pharmacophore/results/CDPKit
```

This writes per-target files under:

```text
pharmacophore/results/CDPKit/<target>/
  scores.csv
  ranked_hits.csv
  metrics.json
```

It also writes:

```text
pharmacophore/results/CDPKit/dataset_metrics.csv
```

You need CDPKit installed and `psdcreate`/`psdscreen` available on `PATH`, or pass explicit executable paths in the config.

The large external installations are expected locally on the Linux/Jupyter machine under `external/` and are not committed to this repository.
