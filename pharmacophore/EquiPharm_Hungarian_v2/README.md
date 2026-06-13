# EquiPharm Hungarian v2

EquiPharm Hungarian v2 keeps the RDKit pharmacophore feature extraction from EquiPharm, but scores molecules by hard one-to-one matching between query and candidate pharmacophore feature embeddings with the balanced score.

The encoder returns feature-level embeddings, the screening layer builds a query-feature by candidate-feature matrix, and Hungarian assignment selects compatible feature pairs.

The assignment is chemically constrained: compatible pharmacophore families such as donor-to-donor, acceptor-to-acceptor, and aromatic-to-aromatic receive cosine-distance costs, while incompatible family pairs are set to `Inf` and cannot be selected. Internal dummy columns let unmatched query features contribute zero instead of forcing an invalid match.

The final score is:

```text
matched_average * query_coverage
```

where `matched_average = sum(matches) / num_matched_features` and `query_coverage = num_matched_features / num_query_features`.

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

For interpretability, `scores.csv` also includes `hungarian_score_strict`, `hungarian_score_balanced`, `matched_feature_count`, coverage columns, and `matching_details`. The `matching_details` column is JSON with query feature family/type/atom IDs, candidate feature family/type/atom IDs, similarity, and whether each query feature was matched or left unmatched.
