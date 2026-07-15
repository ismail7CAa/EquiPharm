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
  --device cuda
```

The defaults reproduce PharmacoMatch's published training setup for both
Equiformer encoders:

- seed 42, 500 epochs, and float32 training;
- batch size 256 for training and evaluation;
- Adam with learning rate `1e-3` and no scheduler or weight decay;
- summed order-embedding loss with margin 100 and gradient clipping at 1;
- spherical displacement radius 1.5 Å;
- a three-layer, 1024-wide projector with dropout 0.2 and 512 outputs;
- the ordered 98% inner/2% outer split, followed by a 90%/10% train/inner-validation split;
- curriculum learning starting with pharmacophores of at most four features and
  increasing the limit after 10 epochs without a validation-loss improvement of 0.1.

The values come from the authors'
[`config.yaml`](https://github.com/molinfo-vienna/PharmacoMatch/blob/7a5c7fa0869a24e6a4a044bbd5affc6821f9352d/scripts/config.yaml)
and
[`training.py`](https://github.com/molinfo-vienna/PharmacoMatch/blob/7a5c7fa0869a24e6a4a044bbd5affc6821f9352d/scripts/training.py).
The Equiformer encoder dimensions remain architecture-specific; PharmacoMatch used
an NNConv encoder rather than either Equiformer model compared here.

Use `--limit 10000 --epochs 1` for a smoke test. Each epoch is checkpointed by
default; increase `--save-every` when disk usage matters.

## Train one method

```bash
python -m contrastive_lear.train \
  --method equiformer_adj \
  --device cuda

python -m contrastive_lear.train \
  --method equiformer_official \
  --device cuda
```

Pass `--seeds 1 2 3` when you want a repeated-seed comparison beyond the
authors' seed-42 run. Use `--no-curriculum` only for an ablation.

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
