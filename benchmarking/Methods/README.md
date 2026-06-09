# Benchmarking Methods

<p align="center">
  <img src="../../figures/benchmarking_pipeline_overview.png" alt="Benchmarking pipeline overview" width="850"/>
</p>

This directory contains the GPU-oriented QM9 benchmark models and shared training utilities.

## Shared pipeline

- `benchmark_utils.py` - QM9 loading, target conversion, normalization, training, logging, checkpointing, and result export.
- `train_eval.py` - one-epoch training and evaluation helpers.

## Dependencies

These benchmarks are GPU-first and expect a CUDA-enabled PyTorch environment.
Core packages:

- `torch`
- `torch-geometric`
- `torch-scatter`
- `ase`
- `tensorboard`
- `dig`
- `equiformer-pytorch`
- `se3-transformer-pytorch`

## 2D GNN baselines

- `GCN.py` - Graph Convolutional Network.
- `GAT.py` - Graph Attention Network.
- `GIN.py` - Graph Isomorphism Network.
- `SAGE.py` - GraphSAGE.

## 3D / equivariant baselines

- `spherenet.py` - SphereNet baseline.
- `se3transformer.py` - SE(3)-Transformer baseline.
- `equiformer_adj.py` - adjacency-aware Equiformer training entry point.
- `equiformer_architecture.py` - adjacency-aware Equiformer architecture.
- `equiformer_pt_cloud.py` - point-cloud Equiformer baseline.

## Pharmacophore-oriented model

- `equiformer_encoder_pharmaco_feat.py` - Equiformer variant used by the pharmacophore workflow.
- `equiformer.py` - legacy pharmacophore-aware Equiformer variant kept for compatibility.

## GPU Commands

Run training through the entry point scripts:

```bash
python benchmarking/Methods/GCN.py --epochs 10 --device cuda
python benchmarking/Methods/GAT.py --epochs 10 --device cuda
python benchmarking/Methods/GIN.py --epochs 10 --device cuda
python benchmarking/Methods/SAGE.py --epochs 10 --device cuda
python benchmarking/Methods/spherenet.py --epochs 10 --device cuda
python benchmarking/Methods/se3transformer.py --epochs 10 --device cuda
python benchmarking/Methods/equiformer_pt_cloud.py --epochs 10 --device cuda
```

Each run writes a reproducible output directory:

```text
runs/<model>/
  config.json
  best_val_test_mae.csv
  checkpoints/best_model.pt
  checkpoints/last_checkpoint.pt
  logs/
```

Training automatically writes `checkpoints/last_checkpoint.pt` after every completed epoch.
If a run is interrupted, launch the same command again with the same `--output-dir`; the
benchmark will resume from that checkpoint and continue at the next epoch. To resume from
a specific file, pass:

```bash
python benchmarking/Methods/GCN.py \
  --resume-from runs/GCN/checkpoints/last_checkpoint.pt \
  --device cuda
```

Use `--no-auto-resume` when you intentionally want to ignore an existing recovery
checkpoint and start a fresh run in the same output directory.

The adjacency-aware Equiformer entry point uses Equiformer-style QM9 defaults: AdamW, cosine warmup scheduling, EMA, 110k/10k train/validation split sizes, and repeated seeds. Run the full comparison with:

```bash
python benchmarking/Methods/equiformer_adj.py --device cuda
```

By default this runs seeds `1 2 3`, writes per-seed outputs under `runs/EquiformerAdj/seed_<seed>/`, and writes `runs/EquiformerAdj/seed_summary.csv`. To compare fewer seeds:

```bash
python benchmarking/Methods/equiformer_adj.py --seeds 1 2 --device cuda
```

## Smoke Test

For a quick GPU smoke test, reduce the split sizes:

```bash
python benchmarking/Methods/GAT.py \
  --epochs 1 \
  --train-size 256 \
  --valid-size 64 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --device cuda
```

For an EquiformerAdj smoke test, also override the multi-seed default:

```bash
python benchmarking/Methods/equiformer_adj.py \
  --epochs 1 \
  --seeds 1 \
  --train-size 256 \
  --valid-size 64 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --device cuda
```

Use `--device cpu` only for local syntax or data-pipeline checks when a GPU is unavailable.

## Import Safety

Benchmark entry points should be import-safe: importing a model file must not load QM9 or start training.
Use the model classes/builders directly in downstream pipelines:

```python
from benchmarking.Methods.GAT import GATModel
from benchmarking.Methods.equiformer_architecture import EquiformerQM9
from benchmarking.Methods.equiformer_pt_cloud import EquiformerQM9PointCloud
```
