# EquiPharm

EquiPharm is the pharmacophore-feature-aware screening pipeline.
It mirrors the cleaned version of the `legacy/pharmacophore-opt-ph-Copy2.features.py` experiment, but all target-specific paths are configurable.

## What Makes This Pipeline Different

Before encoding a molecule, EquiPharm extracts RDKit pharmacophore features and attaches them to the PyG object:

```python
data.pharmacophore_features = model.pharmaco_features(mol)
```

The pharmacophore-aware Equiformer then pools atom embeddings around those features.

## Expected Data Layout

DUD-E data is not committed to the repository. Keep it locally:

```text
data/DUD-E/<target>/
  crystal_ligand.mol2
  actives_sdf/
  decoys_sdf/
```

## Run Any Target

Using a target directory:

```bash
python -m pharmacophore.EquiPharm.cli \
  --target-dir data/DUD-E/aces \
  --target-name aces \
  --checkpoint checkpoints/equipharm/best_model.pt \
  --output-dir pharmacophore/results/EquiPharm/aces
```

Using a config file:

```bash
python -m pharmacophore.EquiPharm.cli \
  --config pharmacophore/EquiPharm/configs/target.example.json
```

For a smoke test:

```bash
python -m pharmacophore.EquiPharm.cli \
  --target-dir data/DUD-E/aces \
  --target-name aces \
  --checkpoint checkpoints/equipharm/best_model.pt \
  --output-dir pharmacophore/results/EquiPharm/aces_smoke \
  --limit 100
```

Run the pipeline smoke test without DUD-E data or checkpoints:

```bash
python -m unittest pharmacophore.tests.test_cli_smoke
```
