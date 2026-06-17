# EquiPharm Hungarian 3D

Hungarian assignment performed directly on RDKit/PyG pharmacophore feature-center
coordinates with same-family compatibility constraints.

The score is the negative mean pairwise 3D geometry-distance error between matched
query and candidate feature pairs:

```text
score = -mean(|d_query(i, j) - d_candidate(i', j')|)
```

```bash
python -m pharmacophore.EquiPharm_Hungarian_3D.cli \
  --checkpoint models_checkpt/checkpoint_02-05-26/best_model.pt \
  --target-dir data/DUD-E/<target> \
  --output-dir pharmacophore/results/EquiPharm_Hungarian_3D/<target>
```
