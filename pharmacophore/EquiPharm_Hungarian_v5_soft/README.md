# EquiPharm Hungarian v5 soft

V5 performs feature-family-constrained Hungarian matching with normalized cosine
embedding quality. Each match combines 40% embedding quality with 60% alignment-free
local spatial quality. Tier 1 is normalized by the query feature count, so unmatched
candidate features are ignored while missing query features contribute zero.

Global Tier 2 geometry is the matched pair-distance RMSE divided by the median query
pair distance. The final score is:

```text
final = tier1 * exp(-geometry_penalty_weight * normalized_geometry_error)
```

Run with:

```bash
python -m pharmacophore.EquiPharm_Hungarian_v5_soft.cli \
  --target-dir data/DUD-E/<target> --target-name <target> \
  --checkpoint models_checkpt/checkpoint_02-05-26/best_model.pt \
  --output-dir pharmacophore/results/EquiPharm_Hungarian_v5_soft/<target>
```
