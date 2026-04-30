# Equiformer With Optimization

This pipeline is the plain Equiformer screening workflow with torsion optimization.
It does not attach pharmacophore features to the PyG object before encoding.

Use this when you want to compare against the pharmacophore-aware EquiPharm pipeline.

## Expected Data Layout

DUD-E data is not committed to the repository. Keep it locally:

```text
data/DUD-E/<target>/
  crystal_ligand.mol2
  actives_sdf/
  decoys_sdf/
```

## Run Any Target

```bash
python -m pharmacophore.Equiformer_with_optimization.cli \
  --target-dir data/DUD-E/aces \
  --target-name aces \
  --checkpoint checkpoints/equiformer/best_model.pt \
  --output-dir pharmacophore/results/Equiformer_with_optimization/aces
```

Using a config file:

```bash
python -m pharmacophore.Equiformer_with_optimization.cli \
  --config pharmacophore/Equiformer_with_optimization/configs/target.example.json
```

For a smoke test:

```bash
python -m pharmacophore.Equiformer_with_optimization.cli \
  --target-dir data/DUD-E/aces \
  --target-name aces \
  --checkpoint checkpoints/equiformer/best_model.pt \
  --output-dir pharmacophore/results/Equiformer_with_optimization/aces_smoke \
  --limit 100
```

Run the pipeline smoke test without DUD-E data or checkpoints:

```bash
python -m unittest pharmacophore.tests.test_cli_smoke
```
