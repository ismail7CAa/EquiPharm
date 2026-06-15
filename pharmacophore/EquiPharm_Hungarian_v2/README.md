# EquiPharm Hungarian v2

EquiPharm Hungarian v2 keeps the RDKit pharmacophore feature extraction from EquiPharm, but scores molecules by hard one-to-one matching between query and candidate pharmacophore feature embeddings with an internal geometry-distance score.

The encoder returns feature-level embeddings, the screening layer builds a query-feature by candidate-feature matrix, and Hungarian assignment selects compatible feature pairs.

The assignment is chemically constrained: compatible pharmacophore families such as donor-to-donor, acceptor-to-acceptor, and aromatic-to-aromatic receive cosine-distance costs, while incompatible family pairs are set to `Inf` and cannot be selected. Internal dummy columns let unmatched query features contribute zero instead of forcing an invalid match.

The final ranking score is:

```text
score = -mean(|d_g - d_g'|)
```

where `d_g` is the distance between two matched pharmacophore features in the query and `d_g'` is the distance between the corresponding matched features in the candidate. If only one feature is matched, v2 falls back to the v1 `-mean(d_i)` feature-distance score.

## Run

```bash
python -m pharmacophore.EquiPharm_Hungarian_v2.cli \
  --target-dir data/DUD-E/<target> \
  --target-name <target> \
  --checkpoint models_checkpt/checkpoint_02-05-26/best_model.pt \
  --output-dir pharmacophore/results/EquiPharm_Hungarian_v2/<target>
```

Config example:

```bash
python -m pharmacophore.EquiPharm_Hungarian_v2.cli \
  --config pharmacophore/EquiPharm_Hungarian_v2/configs/target.example.json
```

Each run writes `scores.csv`, `ranked_hits.csv`, `metrics.json`, `screening_performance_summary.csv`, `auroc_curve_coordinates.csv`, and a named AUROC plot such as `EquiPharm_Hungarian_v2_aces_auroc_curve.png`.

For interpretability, `scores.csv` also includes `feature_distance_score`, `geometry_distance_score`, raw average-distance columns, `matched_feature_count`, coverage columns, and `matching_details`. The `matching_details` column is JSON with query feature family/type/atom IDs, candidate feature family/type/atom IDs, similarity, feature distance, and whether each query feature was matched or left unmatched.
