# OpenPharmaco Baseline

Optional command-template adapter for OpenPharmaco-style pharmacophore screening.

Provide the local screening command and score parser for your installation:

```bash
python -m pharmacophore.OpenPharmaco.cli \
  --target-dir data/DUD-E/<target> \
  --output-dir pharmacophore/results/OpenPharmaco/<target> \
  --command-template "openpharmaco_screen --query {query_ligand} --candidate {candidate}" \
  --score-regex "score[:=]\\s*([-+0-9.eE]+)"
```

The command template can use `{query_ligand}`, `{candidate}`, `{candidate_name}`, `{output_dir}`, `{target_name}`, `{label}`, and `{idx}` placeholders.
