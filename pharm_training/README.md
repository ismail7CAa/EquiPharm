# Pharmacophore Encoder Pretraining

We use this pipeline to pretrain the adjacency-aware Equiformer core with SPICE
formation energies. The experiment remains separate from our QM9 benchmarking
pipeline.

We chose SPICE because it provides atomic identities, 3D conformations,
quantum-mechanical formation energies, and DFT gradients for drug-like
molecules, peptides, and interacting molecular systems. The current workflow
uses formation energy only; gradient/force arrays are not loaded. SPICE also
covers many of the atoms
that later define hydrogen-bond, aromatic, charged, hydrophobic, and halogen
pharmacophore environments.

SPICE does not include pharmacophore labels. We extract those later with RDKit.
At this stage, our aim is to learn a physically informed geometric encoder that
can be adapted to pharmacophore feature representation and Hungarian matching.

## Data preparation

Create the environment, download SPICE, and prepare its manifest:

```bash
conda env create -f environment.yml
conda activate equipharm

bash scripts/download_datasets.sh spice
python -m pharm_training.prepare_spice
```

The downloader writes to `data/` by default. Set `DATA_DIR=/absolute/path` when
the dataset must be stored on another disk. In that case, pass the same location
to the preparation command:

```bash
DATA_DIR=/absolute/path bash scripts/download_datasets.sh spice
python -m pharm_training.prepare_spice \
  --source /absolute/path/SPICE/SPICE-2.0.1.hdf5 \
  --output data/SPICE/prepared/manifest.json
```

Preparation validates the HDF5 structure and creates deterministic,
molecule-disjoint 90%/5%/5% splits. It writes only a manifest and does not copy
the large dataset. Keeping every conformation of a molecule in one split
prevents closely related conformations from leaking into validation or test
data.

Internally, element categories are stored as `atom_type`. This name is
intentional: PyTorch Geometric increments batched fields containing `index`, so
an attribute such as `element_index` would corrupt categorical element IDs.

## Training

The model predicts one total energy from its invariant degree-0 features. The
atomic identities, 3D coordinates, and 6 Å adjacency still shape every internal
atom representation, but no atomic force target is read and no force head is
created. The documented configuration uses `force_mode: energy_only`.

Our baseline uses a 6 Å graph, an energy Smooth-L1 loss, AdamW, gradient
clipping, and a validation-driven learning-rate scheduler. The configuration
allows at most 700 epochs, but this is a safety ceiling rather than a target.
Training normally stops earlier when validation performance no longer improves.
NaN/Inf losses fail immediately, and severe validation divergence also stops
the run.

Rare coordinate-basis singularities may produce non-finite gradients for an
individual conformation batch. We discard the affected update and allow at most
five such batches per epoch. If the limit is exceeded, training stops and names
the affected parameters. This prevents a numerical problem from contaminating
the optimizer while also distinguishing an isolated geometry from systematic
model instability.

## SPICE hyperparameter search

We select training hyperparameters from validation performance rather than
assuming one manually chosen configuration is best. Preview the deterministic
pilot trials without starting training:

```bash
python -m pharm_training.search \
  --config pharm_training/configs/spice_search.json \
  --device cuda \
  --dry-run
```

Start or resume the search:

```bash
python -m pharm_training.search \
  --config pharm_training/configs/spice_search.json \
  --device cuda
```

Pilot trials use the same deterministic samples of 50,000 training and 10,000
validation conformations. They compare learning rate, weight decay, and neighbor
cap (16, 24, or 32), with a maximum of 120 epochs per trial. The resulting
3 × 2 × 3 grid contains 18 trials. We rank every trial by normalized validation
energy MAE.

The test split is not evaluated during this search. Interrupted trials resume
from `last.pt`, completed trials are skipped, and failures keep `console.log`.

Afterward, `best_config.json` describes the winning pilot and
`best_full_config.json` promotes its selected parameters to the full dataset and
700-epoch maximum. Review the ranking, then run the full confirmation:

```bash
python -m pharm_training.train \
  --config runs/pharm_training/spice_search/best_full_config.json \
  --device cuda
```

Only this final run evaluates the held-out test set. For final reported results,
we should repeat the winning full configuration with multiple seeds.

## Downstream pharmacophore interface

`EquiformerAdjEncoder.encode_nodes()` returns invariant per-atom embeddings.
`encode_pharmacophore_features()` accepts externally extracted feature metadata
containing atom IDs and returns one embedding plus family/type/3D-center metadata
per feature. Those outputs are ready for the repository's Hungarian matchers.
For descriptor-based screening, project descriptors to the checkpoint's
`hidden_dim` and pass them through `encode_embedded_nodes()`; the SPICE-specific
element embedding is intentionally not transferred.
We deliberately keep RDKit extraction and Hungarian assignment outside the
potential. SPICE pretraining learns the molecular geometry; feature definition
and matching remain downstream screening tasks.

Each run produces:

```text
runs/pharm_training/<dataset>/
  config.json
  metrics.csv
  results.csv
  logs/
  checkpoints/
    epoch_0025.pt
    epoch_0050.pt
    ...
    last.pt
    best.pt
    trained_encoder.pt
```

`trained_encoder.pt` contains only the transferable Equiformer geometric core.
The atomic-species input layer and potential head are deliberately excluded.
SPICE atomic identities and pharmacophore-screening descriptors are different
modalities, so copying those layers would be invalid.

The dedicated downstream adapter is
`pharm_training/equiformer_encoder_pharmaco_feat.py`. It reconstructs the exact
architecture saved by the SPICE run, creates a new descriptor projection, and
loads only the pretrained geometric weights:

```python
from pharm_training.equiformer_encoder_pharmaco_feat import SPICEPharmacophoreEncoder

model = SPICEPharmacophoreEncoder.from_pretrained(
    checkpoint="runs/pharm_training/spice/checkpoints/trained_encoder.pt",
    descriptor_dim=11,
)
```

The descriptor dimension must match `data.x` in the later screening workflow.
RDKit feature dictionaries are attached to `data.pharmacophore_features`, after
which `model.encode_pharmacophore_features(data)` returns the embeddings and
metadata used by the Hungarian matcher.

We must still fine-tune or calibrate the transferred encoder for the
pharmacophore objective. The pretraining checkpoint alone is not a calibrated
pharmacophore matcher or a ready-to-use screening checkpoint.

The full scientific and technical record is in
[`project_documentation/02_SPICE_EquiformerAdj_Pretraining.txt`](../project_documentation/02_SPICE_EquiformerAdj_Pretraining.txt).

Sources: [ANI-2x](https://doi.org/10.5281/zenodo.10108942),
[SPICE 2.0.1](https://doi.org/10.5281/zenodo.10975225), and the
[Equiformer energy/force training design](https://github.com/atomicarchitects/equiformer/tree/main/scripts/train/md17/equiformer).
