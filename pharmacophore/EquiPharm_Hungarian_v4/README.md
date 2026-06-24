# EquiPharm Hungarian v4

V4 uses two explicit scoring tiers after feature-family-constrained Hungarian matching.

## Scoring

The cost matrix is built from Gaussian distance quality between learned feature
embeddings, so Hungarian assignment directly maximizes the Tier 1 objective.
For every matched pair with embedding distance `d`, Tier 1 uses:

```text
quality(d) = exp(-0.5 * (d / distance_sigma)^2)
tier1 = sum(quality) / max(query_feature_count, candidate_feature_count)
```

The normalization penalizes unmatched query and candidate features. Tier 2 compares
all pairwise 3D distances among the matched feature centers:

```text
geometry_rmse = RMSE(abs(query_pair_distance - candidate_pair_distance))
geometry_factor = exp(-geometry_penalty_weight * geometry_rmse)
final_score = tier1 * geometry_factor
```

With fewer than two usable matched centers, geometry is marked unavailable and its
factor is neutral (`1.0`). Scores remain in `[0, 1]`, where larger is better.

## Run

```bash
python -m pharmacophore.EquiPharm_Hungarian_v4.cli \
  --target-dir data/DUD-E/<target> \
  --target-name <target> \
  --checkpoint models_checkpt/checkpoint_02-05-26/best_model.pt \
  --output-dir pharmacophore/results/EquiPharm_Hungarian_v4/<target> \
  --distance-sigma 1.0 \
  --geometry-penalty-weight 1.0
```

Tune `distance_sigma`, `geometry_penalty_weight`, and (only for experiments)
`enforce_feature_family`. The score table exports Tier 1, Tier 2, coverage, pair
counts, penalty factor, final score, and per-match JSON diagnostics.
