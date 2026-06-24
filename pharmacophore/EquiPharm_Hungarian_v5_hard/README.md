# EquiPharm Hungarian v5 hard

This pipeline uses the same hybrid cosine, local spatial, and global geometry score
as V5 soft. It requires every query feature to receive a compatible one-to-one match.
Incomplete candidates receive a finite score of zero and are exported with
`v5_rejected_incomplete_coverage=true`. Extra candidate features remain allowed.

Run with:

```bash
python -m pharmacophore.EquiPharm_Hungarian_v5_hard.cli \
  --target-dir data/DUD-E/<target> --target-name <target> \
  --checkpoint models_checkpt/checkpoint_02-05-26/best_model.pt \
  --output-dir pharmacophore/results/EquiPharm_Hungarian_v5_hard/<target>
```
