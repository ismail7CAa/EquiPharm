# EquiPharm Hungarian

EquiPharm Hungarian keeps the RDKit pharmacophore feature extraction from EquiPharm, but matches molecules by hard one-to-one assignment between query and candidate pharmacophore feature embeddings.

The encoder returns feature-level embeddings, the screening layer builds a query-feature by candidate-feature matrix, and Hungarian assignment selects compatible feature pairs. After matching, the ranking score is `-mean(d_i)`, where `d_i` is the 3D distance between each matched query and candidate pharmacophore feature center.

The assignment is chemically constrained: compatible pharmacophore families such as donor-to-donor, acceptor-to-acceptor, and aromatic-to-aromatic receive cosine-distance costs, while incompatible family pairs are set to `Inf` and cannot be selected. Internal dummy columns let unmatched query features contribute zero instead of forcing an invalid match.

## Run

```bash
python -m pharmacophore.EquiPharm_Hungarian.cli \
  --target-dir data/DUD-E/<target> \
  --target-name <target> \
  --checkpoint models_checkpt/checkpoint_02-05-26/best_model.pt \
  --output-dir pharmacophore/results/EquiPharm_Hungarian/<target>
```

Config example:

```bash
python -m pharmacophore.EquiPharm_Hungarian.cli \
  --config pharmacophore/EquiPharm_Hungarian/configs/target.example.json
```

Each run writes `scores.csv`, `ranked_hits.csv`, `metrics.json`, `screening_performance_summary.csv`, `auroc_curve_coordinates.csv`, and a named AUROC plot such as `EquiPharm_Hungarian_aces_auroc_curve.png`.

For interpretability, `scores.csv` also includes `feature_distance_score`, `average_feature_distance`, `matched_feature_count`, and `matching_details`. The `matching_details` column is JSON with query feature family/type/atom IDs, candidate feature family/type/atom IDs, similarity, feature distance, and whether each query feature was matched or left unmatched.
