# Pharmacophore Encoder Pretraining

This pipeline pretrains a copy of the adjacency-aware Equiformer core as an
energy-conserving neural potential. ANI-2x and SPICE are separate scientific
experiments: each has its own configuration, output directory, and loss balance.

## Data preparation

The repository downloader writes into the repository's `data/` directory by
default, regardless of the directory from which the script is called. Set
`DATA_DIR=/absolute/path` when a machine stores large datasets elsewhere.

```bash
bash scripts/download_datasets.sh ani2x
bash scripts/download_datasets.sh spice

python -m pharm_training.prepare_ani2x
python -m pharm_training.prepare_spice
```

Preparation does not duplicate the large HDF5 contents. It validates their
schema, creates deterministic molecule-disjoint 90%/5%/5% splits, and writes a
manifest. Splitting by molecule prevents conformations of the same molecule from
leaking across training and evaluation. ANI-2x total energies are residualized
against least-squares per-element reference energies fitted on the training split;
SPICE uses its published formation energies directly.

## Training

```bash
python -m pharm_training.train \
  --config pharm_training/configs/ani2x.json \
  --device cuda

python -m pharm_training.train \
  --config pharm_training/configs/spice.json \
  --device cuda
```

ANI-2x uses a larger batch and lower force weight because it contains smaller
organic molecules at the ωB97X/6-31G(d) level. SPICE uses a smaller batch, wider
6 Å neighborhood, stronger force supervision, and shorter schedule because it is
larger, contains systems up to 110 atoms, and targets noncovalent biomolecular
interactions at ωB97M-D3(BJ)/def2-TZVPPD. These are documented starting recipes,
not claims of dataset-optimal hyperparameters; final studies should report an
ablation or validation-based search around them.

Each run produces:

```text
runs/pharm_training/<dataset>/
  config.json
  metrics.csv
  results.csv
  logs/
  checkpoints/
    epoch_0001.pt
    ...
    last.pt
    best.pt
    trained_encoder.pt
```

`trained_encoder.pt` contains only the transferable Equiformer geometric core.
The atomic-species input layer and potential head are deliberately excluded.
ANI/SPICE atomic identities and pharmacophore-screening RDKit descriptors are
different modalities, so copying those layers would be invalid. Load the core
before pharmacophore-specific fine-tuning with:

```python
from benchmarking.Methods.equiformer_encoder_matching import EquiformerQM9
from pharm_training.transfer import load_pretrained_core

model = EquiformerQM9()
load_pretrained_core(
    model,
    "runs/pharm_training/spice/checkpoints/trained_encoder.pt",
)
```

The resulting model must be fine-tuned on the pharmacophore objective before it
is used as a screening checkpoint. This pretraining checkpoint alone is not a
calibrated pharmacophore matcher.

Sources: [ANI-2x](https://doi.org/10.5281/zenodo.10108942),
[SPICE 2.0.1](https://doi.org/10.5281/zenodo.10975225), and the
[Equiformer energy/force training design](https://github.com/atomicarchitects/equiformer/tree/main/scripts/train/md17/equiformer).
