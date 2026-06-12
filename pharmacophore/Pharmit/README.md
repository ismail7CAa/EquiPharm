# Pharmit Baseline

Optional command-template adapter for Pharmit pharmacophore screening.

Use this wrapper when you have a local/API command that screens one candidate against a query and returns one numeric score:

```bash
python -m pharmacophore.Pharmit.cli \
  --target-dir data/DUD-E/<target> \
  --output-dir pharmacophore/results/Pharmit/<target> \
  --command-template "pharmit_screen --query {query_ligand} --candidate {candidate} --json" \
  --score-json-key score
```

The command template can use `{query_ligand}`, `{candidate}`, `{candidate_name}`, `{output_dir}`, `{target_name}`, `{label}`, and `{idx}` placeholders.
