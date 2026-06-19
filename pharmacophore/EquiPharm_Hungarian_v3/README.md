# EquiPharm Hungarian v3

EquiPharm Hungarian v3 is a hybrid feature-matching variant:

- Hungarian assignment uses Euclidean distance between learned pharmacophore feature embeddings.
- Final scoring uses 3D geometry preservation from RDKit/PyG pharmacophore feature centers.

The assignment is:

```text
M* = argmin_M sum ||q_i - c_j||_2
```

where `q_i` and `c_j` are learned query and candidate feature embeddings.

The final ranking score is:

```text
score = -mean(|d_q - d_c|)
```

where `d_q` is the 3D Euclidean distance between two matched query feature centers and `d_c` is the 3D Euclidean distance between the corresponding matched candidate feature centers.

## Run

```bash
python -m pharmacophore.EquiPharm_Hungarian_v3.cli \
  --target-dir data/DUD-E/<target> \
  --target-name <target> \
  --checkpoint models_checkpt/checkpoint_02-05-26/best_model.pt \
  --output-dir pharmacophore/results/EquiPharm_Hungarian_v3/<target>
```

Config example:

```bash
python -m pharmacophore.EquiPharm_Hungarian_v3.cli \
  --config pharmacophore/EquiPharm_Hungarian_v3/configs/target.example.json
```

Runs resume automatically from `scores.csv`. If the server interrupts screening, rerun the same command with the same `--output-dir`; paths with finite scores are skipped and missing molecules continue.
