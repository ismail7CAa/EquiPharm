# Contrastive Equiformer Learning

This pipeline compares the repository's adjacency-aware Equiformer and the
original Equiformer backbone on PharmacoMatch's self-supervised 3D
pharmacophore-matching objective.

## Data

Download and extract the official PharmacoMatch archive into the repository's
existing `data/` directory:

```bash
bash scripts/download_datasets.sh pharmacomatch
```

The default training file is:

```text
data/training_data/processed/chembl_data.pt
```

## Train and compare both models

```bash
python -m contrastive_lear.run_comparison \
  --seeds 1 2 3 \
  --epochs 500 \
  --device cuda
```

Use `--limit 10000 --epochs 1` for a smoke test. Each epoch is checkpointed by
default; increase `--save-every` when disk usage matters.

## Train one method

```bash
python -m contrastive_lear.train \
  --method equiformer_adj \
  --seeds 1 2 3 \
  --device cuda

python -m contrastive_lear.train \
  --method equiformer_official \
  --seeds 1 2 3 \
  --device cuda
```

## Outputs

```text
runs/contrastive_lear/
  model_comparison.csv
  equiformer_adj/
    seed_summary.csv
    seed_<seed>/
      config.json
      metrics.csv
      checkpoints/
        best.pt
        last.pt
        epoch_0001.pt
        ...
      logs/
  equiformer_official/
    ...
```

`metrics.csv` records train, validation, and test loss/AUROC for every epoch.
`model_comparison.csv` compares the best-validation result and corresponding
test AUROC for every method and seed.
