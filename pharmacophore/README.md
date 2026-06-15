# Pharmacophore Screening

<p align="center">
  <img src="../figures/pharmacophore_screening_pipeline_overview.png" alt="Pharmacophore screening pipeline overview" width="850"/>
</p>

This directory contains reproducible screening pipelines plus the original exploratory notebooks/scripts.
DUD-E data and model checkpoints are not committed to the repository; paths are supplied through config files or CLI arguments.

## Pipelines

### EquiPharm

Folder: `pharmacophore/EquiPharm/`

This is the pharmacophore-feature-aware pipeline, cleaned from the `legacy/pharmacophore-opt-ph-Copy2.features.py` style workflow.
Before encoding a molecule, it extracts RDKit pharmacophore features and attaches them to the PyG graph:

```python
data.pharmacophore_features = model.pharmaco_features(mol)
```

Run:

```bash
python -m pharmacophore.EquiPharm.cli \
  --target-dir data/DUD-E/<target> \
  --target-name <target> \
  --checkpoint models_checkpt/checkpoint_02-05-26/best_model.pt \
  --output-dir pharmacophore/results/EquiPharm/<target>
```

Or run from a config file:

```bash
python -m pharmacophore.EquiPharm.cli \
  --config pharmacophore/EquiPharm/configs/target.example.json
```

### EquiPharm Hungarian Matching

Folder:

```text
pharmacophore/EquiPharm_Hungarian/
pharmacophore/EquiPharm_Hungarian_v2/
```

This is a copy of EquiPharm that keeps the extracted RDKit pharmacophore features as a feature set instead of immediately averaging them into one global vector. For each query-candidate pair, the screening layer builds a query-feature by candidate-feature cosine similarity matrix.

`EquiPharm_Hungarian` uses hard one-to-one Hungarian assignment through the shared encoder module `benchmarking.Methods.equiformer_encoder_matching`.
The matching cost matrix is constrained by pharmacophore family: donor-to-donor, acceptor-to-acceptor, aromatic-to-aromatic, and related same-family matches are allowed, while incompatible pairs are assigned `Inf` and cannot be selected.

After matching, `EquiPharm_Hungarian` uses `score = -mean(d_i)`, where `d_i` is the 3D distance between a matched query pharmacophore feature and candidate pharmacophore feature.
`EquiPharm_Hungarian_v2` uses `score = -mean(|d_g - d_g'|)`, where `d_g` is the distance between two matched query features and `d_g'` is the distance between their corresponding candidate features.

Run Hungarian matching:

```bash
python -m pharmacophore.EquiPharm_Hungarian.cli \
  --target-dir data/DUD-E/<target> \
  --target-name <target> \
  --checkpoint models_checkpt/checkpoint_02-05-26/best_model.pt \
  --output-dir pharmacophore/results/EquiPharm_Hungarian/<target>
```

Run geometry-distance Hungarian matching:

```bash
python -m pharmacophore.EquiPharm_Hungarian_v2.cli \
  --target-dir data/DUD-E/<target> \
  --target-name <target> \
  --checkpoint models_checkpt/checkpoint_02-05-26/best_model.pt \
  --output-dir pharmacophore/results/EquiPharm_Hungarian_v2/<target>
```

The Hungarian variant writes named AUROC plots such as:

```text
pharmacophore/results/EquiPharm_Hungarian/<target>/EquiPharm_Hungarian_<target>_auroc_curve.png
pharmacophore/results/EquiPharm_Hungarian_v2/<target>/EquiPharm_Hungarian_v2_<target>_auroc_curve.png
```

### Optional External Baselines

External baselines are scaffolded but kept isolated from the main EquiPharm code path:

- `pharmacophore/CDPKit/` wraps CDPKit `psdcreate`/`psdscreen` for pharmacophore database screening.
- `pharmacophore/PharmacoMatch/` wraps a configurable PharmacoMatch command and parses a score from JSON or text output.
- `pharmacophore/SchrodingerPhase/` wraps a local Schrodinger Phase screening command.
- `pharmacophore/OpenPharmaco/` wraps a local OpenPharmaco-style screening command.
- `pharmacophore/Pharmit/` wraps a local Pharmit screening command or API script.
- `pharmacophore/DiscoveryStudio/` wraps a local Discovery Studio pharmacophore screening command.

These adapters are ready to configure, but they do not run unless their CLIs are called explicitly:

```bash
python -m pharmacophore.CDPKit.cli \
  --config pharmacophore/CDPKit/configs/target.example.json

python -m pharmacophore.PharmacoMatch.cli \
  --config pharmacophore/PharmacoMatch/configs/target.example.json

python -m pharmacophore.SchrodingerPhase.cli \
  --config pharmacophore/SchrodingerPhase/configs/target.example.json

python -m pharmacophore.OpenPharmaco.cli \
  --config pharmacophore/OpenPharmaco/configs/target.example.json

python -m pharmacophore.Pharmit.cli \
  --config pharmacophore/Pharmit/configs/target.example.json

python -m pharmacophore.DiscoveryStudio.cli \
  --config pharmacophore/DiscoveryStudio/configs/target.example.json
```

These adapters write the same result files as the maintained pipelines under `pharmacophore/results/<pipeline>/<target>/`.
For full datasets, pass `--dataset-dir data/DUD-E` and the adapters also write `pharmacophore/results/<pipeline>/dataset_metrics.csv`.

### Run All Screening Methods

Use the all-method runner to execute:

```text
CDPKit
PharmacoMatch
SchrodingerPhase
OpenPharmaco
Pharmit
DiscoveryStudio
EquiPharm
EquiPharm_Hungarian
EquiPharm_Hungarian_v2
```

Example for one target smoke run:

```bash
python -m pharmacophore.run_all_screening \
  --dataset-dir data/DUD-E \
  --target aces \
  --checkpoint models_checkpt/checkpoint_02-05-26/best_model.pt \
  --output-dir pharmacophore/results \
  --pharmacomatch-command-template "python screen.py --query {query_ligand} --candidate {candidate}" \
  --pharmacomatch-score-json-key score \
  --phase-command-template "phase_screen --query {query_ligand} --candidate {candidate} --json" \
  --phase-score-json-key score \
  --openpharmaco-command-template "openpharmaco_screen --query {query_ligand} --candidate {candidate}" \
  --openpharmaco-score-regex "score[:=]\\s*([-+0-9.eE]+)" \
  --pharmit-command-template "pharmit_screen --query {query_ligand} --candidate {candidate} --json" \
  --pharmit-score-json-key score \
  --discoverystudio-command-template "discovery_studio_pharmacophore_screen --query {query_ligand} --candidate {candidate} --json" \
  --discoverystudio-score-json-key score \
  --limit 100
```

EquiPharm-family target runs resume automatically from `scores.csv`. If a target run is interrupted, rerun the same command with the same `--output-dir`; molecules whose paths already have finite scores in `scores.csv` are skipped, and new rows are appended as they complete.

To force a fresh run, remove the target output directory or its `scores.csv` first.

Example for all DUD-E targets:

```bash
python -m pharmacophore.run_all_screening \
  --dataset-dir data/DUD-E \
  --checkpoint models_checkpt/checkpoint_02-05-26/best_model.pt \
  --output-dir pharmacophore/results \
  --pharmacomatch-command-template "python screen.py --query {query_ligand} --candidate {candidate}" \
  --pharmacomatch-score-json-key score
```

The same command works for the normalized LIT-PCBA, DEKOIS2, and BayesBind roots:

Include the same `--phase-*`, `--openpharmaco-*`, `--pharmit-*`, and `--discoverystudio-*` command-template options when you want those external baselines included for the dataset.

```bash
python -m pharmacophore.run_all_screening \
  --dataset-dir data/LIT-PCBA \
  --checkpoint models_checkpt/checkpoint_02-05-26/best_model.pt \
  --output-dir pharmacophore/results \
  --pharmacomatch-command-template "python screen.py --query {query_ligand} --candidate {candidate}" \
  --pharmacomatch-score-json-key score

python -m pharmacophore.run_all_screening \
  --dataset-dir data/DEKOIS2 \
  --checkpoint models_checkpt/checkpoint_02-05-26/best_model.pt \
  --output-dir pharmacophore/results \
  --pharmacomatch-command-template "python screen.py --query {query_ligand} --candidate {candidate}" \
  --pharmacomatch-score-json-key score

python -m pharmacophore.run_all_screening \
  --dataset-dir data/BayesBind \
  --checkpoint models_checkpt/checkpoint_02-05-26/best_model.pt \
  --output-dir pharmacophore/results \
  --pharmacomatch-command-template "python screen.py --query {query_ligand} --candidate {candidate}" \
  --pharmacomatch-score-json-key score
```

The runner writes:

```text
pharmacophore/results/<dataset>/all_screening_metrics.csv
pharmacophore/results/<dataset>/all_screening_scores.csv
```

`all_screening_metrics.csv` contains AUROC, PR-AUC, EF1%, and BEDROC(alpha=20) for every completed method-target run. `all_screening_scores.csv` combines the per-candidate score rows from every completed method-target run.

### Equiformer With Optimization

Folder: `pharmacophore/Equiformer_with_optimization/`

This is the plain Equiformer screening pipeline with the same torsion optimization and active/decoy evaluation flow.
It does not attach pharmacophore features before encoding and is useful as a direct baseline.

Run:

```bash
python -m pharmacophore.Equiformer_with_optimization.cli \
  --target-dir data/DUD-E/<target> \
  --target-name <target> \
  --checkpoint checkpoints/equiformer/best_model.pt \
  --output-dir pharmacophore/results/Equiformer_with_optimization/<target>
```

Or run from a config file:

```bash
python -m pharmacophore.Equiformer_with_optimization.cli \
  --config pharmacophore/Equiformer_with_optimization/configs/target.example.json
```

## Expected Data Layout

Keep DUD-E locally outside git:

```text
data/DUD-E/<target>/
  crystal_ligand.mol2
  actives_sdf/
  decoys_sdf/
```

From the repository root, download and prepare DUD-E into this layout with:

```bash
bash scripts/download_datasets.sh dude
```

Prepare additional screening benchmarks with the same layout:

```bash
LIT_PCBA_URL="<official-lit-pcba-archive-url>" bash scripts/download_datasets.sh lit-pcba
DEKOIS2_URL="<official-dekois2-archive-url>" bash scripts/download_datasets.sh dekois2
BAYESBIND_URL="<official-bayesbind-archive-url>" bash scripts/download_datasets.sh bayesbind
```

Or normalize an already-downloaded archive/extracted folder:

```bash
python scripts/prepare_screening_dataset.py \
  --source-dir path/to/LIT-PCBA \
  --output-dir data/LIT-PCBA

python scripts/prepare_screening_dataset.py \
  --source-dir path/to/DEKOIS2 \
  --output-dir data/DEKOIS2

python scripts/prepare_screening_dataset.py \
  --source-dir path/to/BayesBind \
  --output-dir data/BayesBind
```

Examples:

```text
data/DUD-E/aces/
data/DUD-E/urok/
data/DUD-E/egfr/
data/LIT-PCBA/ADRB2/
data/DEKOIS2/<target>/
data/BayesBind/<target>/
```

## Configs

Each pipeline has an example target config:

```text
pharmacophore/EquiPharm/configs/target.example.json
pharmacophore/EquiPharm_Hungarian/configs/target.example.json
pharmacophore/EquiPharm_Hungarian_v2/configs/target.example.json
pharmacophore/Equiformer_with_optimization/configs/target.example.json
pharmacophore/SchrodingerPhase/configs/target.example.json
pharmacophore/OpenPharmaco/configs/target.example.json
pharmacophore/Pharmit/configs/target.example.json
pharmacophore/DiscoveryStudio/configs/target.example.json
```

Replace `<target>` with the DUD-E target name or override paths from the command line.

## Outputs

Single-pipeline runs write:

```text
pharmacophore/results/<pipeline>/<target>/
  scores.csv
  ranked_hits.csv
  metrics.json
  screening_performance_summary.csv
  auroc_curve_coordinates.csv
  cosine_similarity_boxplot.png
  roc_curve_actives_vs_decoys.png       # EquiPharm-family pipelines
  <pipeline>_<target>_auroc_curve.png   # EquiPharm, EquiPharm_Hungarian, EquiPharm_Hungarian_v2
```

All-method dataset runs write:

```text
pharmacophore/results/<dataset>/
  all_screening_metrics.csv
  all_screening_scores.csv
  <pipeline>/
    <target>/
      scores.csv
      ranked_hits.csv
      metrics.json
      screening_performance_summary.csv
      auroc_curve_coordinates.csv
      cosine_similarity_boxplot.png
      roc_curve_actives_vs_decoys.png       # EquiPharm-family pipelines
      <pipeline>_<target>_auroc_curve.png   # EquiPharm, EquiPharm_Hungarian, EquiPharm_Hungarian_v2
```

Examples:

```text
pharmacophore/results/DUD-E/all_screening_metrics.csv
pharmacophore/results/LIT-PCBA/all_screening_metrics.csv
pharmacophore/results/DEKOIS2/all_screening_metrics.csv
pharmacophore/results/BayesBind/all_screening_metrics.csv
```

`metrics.json` and `screening_performance_summary.csv` include AUROC, PR-AUC, EF1%, and BEDROC(alpha=20), plus the pipeline name and protein target name.
`auroc_curve_coordinates.csv` stores the false-positive-rate, true-positive-rate, and threshold values used to draw the ROC curve.
For `EquiPharm_Hungarian` and `EquiPharm_Hungarian_v2`, `scores.csv` also includes `feature_distance_score`, `geometry_distance_score`, raw average-distance columns, `matched_feature_count`, coverage columns, and `matching_details`, where `matching_details` is JSON describing the selected query-candidate pharmacophore feature matches and unmatched query features.
If `--target-name` is omitted, the target is inferred from paths like `data/DUD-E/<target>/...`.

Existing reference plots and CSV exports from the exploratory workflow are kept in:

```text
pharmacophore/results/
  results_pharmaco_AUROC.csv
  cosine_similarity_boxplot.png
  roc_curve_actives_vs_decoys.png
```

## Smoke Tests

Use `--limit` before launching full DUD-E screening:

```bash
python -m pharmacophore.EquiPharm.cli \
  --target-dir data/DUD-E/aces \
  --checkpoint models_checkpt/checkpoint_02-05-26/best_model.pt \
  --output-dir pharmacophore/results/EquiPharm/aces_smoke \
  --limit 100
```

Run the lightweight software smoke tests without DUD-E data or checkpoints:

```bash
python -m unittest pharmacophore.tests.test_cli_smoke
```

## Shared Utilities

The maintained pipelines share:

- `core/molecule_io.py` - MOL2/SDF readers, molecule preparation, RDKit-to-PyG conversion.
- `core/torsion.py` - black-box torsion optimization.
- `core/screening.py` - generic screening engine used by embedding-cosine pipelines.
- `core/matching.py` and `core/matching_screening.py` - feature-level matching scorers and screening engine.
- `core/metrics.py` - metrics and plot generation.

## Legacy Scripts

Older exploratory Python scripts are preserved in `legacy/` for traceability.
They are useful for comparison while reviewing the migration, but the maintained software entry points are the pipeline CLIs.

## Reference Notebooks

The original notebooks and copied experiment scripts are kept as references during migration, including:

- `notebooks/pharmacophore-opt.ipynb`
- `notebooks/pharmacophore-opt-expt.ipynb`
- `notebooks/pharmacophore-opt-ph.features.ipynb`
- `notebooks/pharmacophore-without-opt.ipynb`
- `notebooks/visualization-pharmaphore.ipynb`
- `legacy/pharmacophore-opt-ph.features.py`
- `legacy/pharmacophore-opt-ph-Copy*.features.py`
- `legacy/tm_calculate.py`

They are not the recommended reproducible entry points.
