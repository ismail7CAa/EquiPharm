# EquiPharm Sinkhorn

EquiPharm Sinkhorn keeps the RDKit pharmacophore feature extraction from EquiPharm, but scores molecules with soft optimal-transport matching between query and candidate pharmacophore feature embeddings.

The encoder returns feature-level embeddings, the screening layer builds a query-feature by candidate-feature cosine similarity matrix, and Sinkhorn normalization converts that matrix into a soft matching score.

## Run

```bash
python -m pharmacophore.EquiPharm_Sinkhorn.cli \
  --target-dir data/DUD-E/<target> \
  --target-name <target> \
  --checkpoint models_checkpt/checkpoint_02-05-26/best_model.pt \
  --output-dir pharmacophore/results/EquiPharm_Sinkhorn/<target>
```

Config example:

```bash
python -m pharmacophore.EquiPharm_Sinkhorn.cli \
  --config pharmacophore/EquiPharm_Sinkhorn/configs/target.example.json
```

Useful tuning arguments are `--mismatch-penalty`, `--sinkhorn-temperature`, and `--sinkhorn-iterations`.

Each run writes `scores.csv`, `ranked_hits.csv`, `metrics.json`, `screening_performance_summary.csv`, `auroc_curve_coordinates.csv`, and a named AUROC plot such as `EquiPharm_Sinkhorn_aces_auroc_curve.png`.
