# CDPKit/CDPL PharmacoMatch-Style Alignment Handoff

This document is the external handoff for using the CDPKit baseline as the
CDPL pharmacophore-alignment version used in the PharmacoMatch comparison.

It is intended for another project or an AI agent that needs a ready function
or command-line workflow.

## What This Is

This is not the old `psdscreen` hit-only workflow.

The current wrapper follows the PharmacoMatch-style CDPL alignment path:

1. Use a CDPL query pharmacophore file, usually `query.pml`.
2. Generate ligand pharmacophores from candidate SDF molecules.
3. Store them in a CDPL pharmacophore screening database, `.psd`.
4. Align every generated candidate pharmacophore to the query.
5. Score every alignment with the CDPL fit score.
6. Max-pool scores per input molecule.
7. Return one continuous score per molecule.

The code lives in:

```text
pharmacophore/CDPKit/screening.py
pharmacophore/CDPKit/cli.py
```

The importable API is exported from:

```text
pharmacophore/CDPKit/__init__.py
```

## Environment Setup

### Fresh Machine Setup

For the boss or an external AI agent, the required code is this repository plus
the CDPKit/CDPL Python package. The PharmacoMatch repository is not required for
the function API below; this project already wraps the same CDPL alignment idea.

Clone this project:

```bash
git clone <EQUIPHARM_REPOSITORY_URL> equipharm
cd equipharm
```

Create the environment:

```bash
conda env create -f environment.yml
conda activate equipharm
```

If the environment is not created from `environment.yml`, install the minimum
CDPKit/CDPL dependency manually:

```bash
python -m pip install CDPKit
```

Check that Python CDPL imports:

```bash
python - <<'PY'
from CDPL import Chem, Pharm
print("CDPL Python modules are available")
PY
```

Check that this project's scoring function imports:

```bash
python - <<'PY'
from pharmacophore.CDPKit import score_cdpkit_alignment, score_cdpkit_alignment_batch
print("EquiPharm CDPKit/CDPL scoring API is available")
PY
```

Optional: check whether the CDPKit command-line PSD creator is available:

```bash
which psdcreate || true
```

`psdcreate` is helpful but not mandatory. If it is missing, the wrapper uses
Python CDPL to create `.psd` files.

### Existing Project Environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate equipharm
```

If the environment already exists and only needs updates:

```bash
conda env update -f environment.yml --prune
conda activate equipharm
```

The required CDPL/CDPKit dependency is listed in `environment.yml`:

```text
CDPKit
```

Quick import check:

```bash
python - <<'PY'
from pharmacophore.CDPKit import score_cdpkit_alignment, score_cdpkit_alignment_batch
print("CDPKit/CDPL function API is importable")
PY
```

Optional command check:

```bash
which psdcreate || true
```

If `psdcreate` is available, the wrapper can use it. If it is not available,
the wrapper falls back to Python CDPL PSD creation.

### Optional CDPKit Source Checkout

The function API does not require a separate CDPKit source checkout when the
`CDPKit` Python package is installed successfully.

Only do this if the external project specifically needs the CDPKit source tree
for auditing, patching, or building command-line binaries:

```bash
mkdir -p external
cd external
git clone <OFFICIAL_CDPKIT_REPOSITORY_URL> CDPKit
cd CDPKit
```

Then follow the official CDPKit build/install instructions for that source
tree. After installation, rerun:

```bash
python - <<'PY'
from CDPL import Chem, Pharm
print("CDPL source/build installation works")
PY
```

Do not clone CDPKit just for scoring if `python -m pip install CDPKit` already
works.

## Required Inputs

For a single molecule score:

```text
query.pml
ligand.sdf
work_dir/
```

For one DUD-E-style target:

```text
data/DUD-E/<target>/
  crystal_ligand.mol2
  actives_sdf/
  decoys_sdf/
```

The CLI prefers a query pharmacophore:

```text
query.pml
```

When `--target-dir` is used, the CLI looks for:

```text
data/DUD-E/<target>/query.pml
data/DUD-E/<target>/crystal_ligand.pml
```

If neither exists, it tries to generate `query.pml` from the query ligand using
Python CDPL.

## Python Function API

Use this when another AI agent or external Python project needs CDPKit/CDPL as a
function.

### Score One SDF File

```python
from pharmacophore.CDPKit import score_cdpkit_alignment

score = score_cdpkit_alignment(
    query_pharmacophore="query.pml",
    candidate_sdf="ligand.sdf",
    work_dir="cdpkit_work",
)

print(score)
```

### Score Many SDF Files

```python
from pharmacophore.CDPKit import score_cdpkit_alignment_batch

scores = score_cdpkit_alignment_batch(
    query_pharmacophore="query.pml",
    candidate_sdfs=[
        "ligand_a.sdf",
        "ligand_b.sdf",
        "ligand_c.sdf",
    ],
    work_dir="cdpkit_work",
)

print(scores)
```

Return type:

```python
dict[str, float]
```

The dictionary key is the input SDF file stem:

```text
ligand_a.sdf -> "ligand_a"
```

The value is the best CDPL pharmacophore alignment score found for that input
molecule.

### Function Parameters

Single molecule:

```python
score_cdpkit_alignment(
    *,
    query_pharmacophore: str | Path,
    candidate_sdf: str | Path,
    work_dir: str | Path,
    psdcreate_bin: str | Path | None = "psdcreate",
    num_threads: int | None = None,
    default_score: float = 0.0,
) -> float
```

Batch:

```python
score_cdpkit_alignment_batch(
    *,
    query_pharmacophore: str | Path,
    candidate_sdfs: Iterable[str | Path],
    work_dir: str | Path,
    psdcreate_bin: str | Path | None = "psdcreate",
    num_threads: int | None = None,
    default_score: float = 0.0,
) -> dict[str, float]
```

Use `default_score=0.0` for missing or unscored molecules unless the external
project needs a different fallback.

## CLI Commands

### Run One Target

```bash
python -m pharmacophore.CDPKit.cli \
  --target-dir data/DUD-E/<target> \
  --output-dir pharmacophore/results/CDPKit/<target>
```

Example:

```bash
python -m pharmacophore.CDPKit.cli \
  --target-dir data/DUD-E/aces \
  --output-dir pharmacophore/results/CDPKit/aces
```

### Run One Target With Explicit Query

```bash
python -m pharmacophore.CDPKit.cli \
  --query-pharmacophore data/DUD-E/<target>/query.pml \
  --actives-dir data/DUD-E/<target>/actives_sdf \
  --decoys-dir data/DUD-E/<target>/decoys_sdf \
  --output-dir pharmacophore/results/CDPKit/<target> \
  --target-name <target>
```

### Run One Target With `psdcreate`

```bash
python -m pharmacophore.CDPKit.cli \
  --target-dir data/DUD-E/<target> \
  --output-dir pharmacophore/results/CDPKit/<target> \
  --psdcreate-bin psdcreate \
  --num-threads 8
```

### Run Every DUD-E Target

```bash
python -m pharmacophore.CDPKit.cli \
  --dataset-dir data/DUD-E \
  --output-dir pharmacophore/results/CDPKit
```

### Run Every Target And Skip Missing Queries

```bash
python -m pharmacophore.CDPKit.cli \
  --dataset-dir data/DUD-E \
  --output-dir pharmacophore/results/CDPKit \
  --skip-missing-queries
```

### Run A Small Debug Limit

```bash
python -m pharmacophore.CDPKit.cli \
  --target-dir data/DUD-E/<target> \
  --output-dir pharmacophore/results/CDPKit_debug/<target> \
  --limit 20
```

## Output Files

For one target:

```text
pharmacophore/results/CDPKit/<target>/
  raw/query.pml
  raw/actives.psd
  raw/inactives.psd
  preprocessing/actives.sdf
  preprocessing/inactives.sdf
  vs/all_actives_aligned.pt
  vs/all_inactives_aligned.pt
  scores.csv
  ranked_hits.csv
  metrics.json
  screening_performance_summary.csv
```

Important score table columns:

```text
pipeline
target
name
path
label
score
```

`label` is:

```text
1 = active
0 = decoy
```

Higher `score` is better.

## Minimal AI Agent Contract

If an AI agent needs to call CDPKit/CDPL scoring, use this contract.

Input:

```json
{
  "query_pharmacophore": "query.pml",
  "candidate_sdfs": ["ligand_a.sdf", "ligand_b.sdf"],
  "work_dir": "cdpkit_work"
}
```

Python call:

```python
from pharmacophore.CDPKit import score_cdpkit_alignment_batch

scores = score_cdpkit_alignment_batch(
    query_pharmacophore=payload["query_pharmacophore"],
    candidate_sdfs=payload["candidate_sdfs"],
    work_dir=payload["work_dir"],
)
```

Output:

```json
{
  "ligand_a": 1.73,
  "ligand_b": 0.42
}
```

The numeric values above are examples. Real values come from CDPL alignment.

## Troubleshooting

### `CDPL alignment requires query_pharmacophore to be a .pml file`

Pass a CDPL pharmacophore query:

```bash
--query-pharmacophore path/to/query.pml
```

or use:

```bash
--target-dir data/DUD-E/<target>
```

so the wrapper can find or generate `query.pml`.

### `No CDPL query pharmacophore found`

Make sure the target directory contains one of:

```text
query.pml
crystal_ligand.pml
crystal_ligand.mol2
```

If only `crystal_ligand.mol2` exists, Python CDPL must be importable so the
query pharmacophore can be generated.

### `psdcreate` Not Found

This is acceptable if Python CDPL is installed. The wrapper will use Python CDPL
to create `.psd` files.

If an external system requires the command-line tool, install CDPKit with
command-line binaries and pass:

```bash
--psdcreate-bin /path/to/psdcreate
```

### Empty Or Missing Scores

Check:

```text
work_dir/
pharmacophore/results/CDPKit/<target>/raw/
pharmacophore/results/CDPKit/<target>/vs/
```

The common causes are:

1. invalid SDF molecules,
2. empty generated pharmacophores,
3. missing 3D coordinates,
4. invalid or missing `query.pml`,
5. CDPL import or binary installation problems.

## Validation Commands

Smoke-test import:

```bash
python - <<'PY'
from pharmacophore.CDPKit import score_cdpkit_alignment_batch
print(score_cdpkit_alignment_batch)
PY
```

Run project tests that cover the wrapper:

```bash
python -m pytest pharmacophore/tests/test_cli_smoke.py -q
```

## Notes For The Boss

The boss-facing function is:

```python
from pharmacophore.CDPKit import score_cdpkit_alignment_batch
```

This is the recommended function for AI-agent integration because it:

1. accepts a query `.pml`,
2. accepts one or many candidate SDF files,
3. creates CDPL pharmacophores,
4. aligns them to the query,
5. returns continuous max-pooled CDPL fit scores,
6. does not depend on the old `psdscreen` hit-only output.

For one molecule, use:

```python
from pharmacophore.CDPKit import score_cdpkit_alignment
```

For production or agent workflows, prefer the batch function because it avoids
recreating setup work separately for every candidate.
