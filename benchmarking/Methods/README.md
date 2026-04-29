# Benchmarking Methods

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
python benchmarking/Methods/equiformer_adj.py --epochs 10 --device cuda
python benchmarking/Methods/equiformer_pt_cloud.py --epochs 10 --device cuda
```

Each run writes a reproducible output directory:

```text
runs/<model>/
  config.json
  best_val_test_mae.csv
  checkpoints/best_model.pt
  logs/
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

Use `--device cpu` only for local syntax or data-pipeline checks when a GPU is unavailable.

## Import Safety

Benchmark entry points should be import-safe: importing a model file must not load QM9 or start training.
Use the model classes/builders directly in downstream pipelines:

```python
from benchmarking.Methods.GAT import GATModel
from benchmarking.Methods.equiformer_architecture import EquiformerQM9
from benchmarking.Methods.equiformer_pt_cloud import EquiformerQM9PointCloud
```
