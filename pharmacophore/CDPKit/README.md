# CDPKit / CDPL Alignment Baseline

Optional adapter for the CDPKit/CDPL pharmacophore-alignment baseline used in
the PharmacoMatch comparison.

This wrapper prepares active and decoy SDF bundles, creates CDPL pharmacophore
screening databases (`.psd`), aligns every ligand pharmacophore to `query.pml`,
and ranks molecules by their best CDPL pharmacophore fit score. It writes the
same `scores.csv`, `ranked_hits.csv`, `metrics.json`, and plots used by the
other pipelines.

This is not the old `psdscreen` hit-output workflow. `psdscreen` reports only
hits, whereas this alignment path gives every generated ligand pharmacophore a
continuous score and max-pools scores by molecule.

The candidate data layout is the same as EquiPharm:

```text
data/DUD-E/<target>/
  crystal_ligand.mol2
  actives_sdf/
  decoys_sdf/
```

CDPL alignment needs `query.pml`. When `--target-dir` is used, the CLI looks for
`query.pml` or `crystal_ligand.pml`; if neither exists, it generates `query.pml`
from the query ligand with Python CDPL.

Run one target:

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
  raw/query.pml
  raw/actives.psd
  raw/inactives.psd
  vs/all_actives_aligned.pt
  vs/all_inactives_aligned.pt
  scores.csv
  ranked_hits.csv
  metrics.json
```

You need Python CDPKit/CDPL available. If `psdcreate` is available on `PATH`, or
passed with `--psdcreate-bin`, it is used to create PSD databases; otherwise the
wrapper falls back to Python CDPL PSD creation.

## Python Function API

For another project, import the CDPL alignment scorer directly:

```python
from pharmacophore.CDPKit import score_cdpkit_alignment

score = score_cdpkit_alignment(
    query_pharmacophore="query.pml",
    candidate_sdf="ligand.sdf",
    work_dir="cdpkit_work",
)
```

For a batch of SDF files:

```python
from pharmacophore.CDPKit import score_cdpkit_alignment_batch

scores = score_cdpkit_alignment_batch(
    query_pharmacophore="query.pml",
    candidate_sdfs=["ligand_a.sdf", "ligand_b.sdf"],
    work_dir="cdpkit_work",
)
```

`scores` is a dictionary keyed by input file stem. The score is the best CDPL
pharmacophore fit score over generated pharmacophores/conformations for that
molecule.
