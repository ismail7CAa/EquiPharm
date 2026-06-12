# Discovery Studio Pharmacophore Baseline

Optional command-template adapter for Discovery Studio pharmacophore screening.

Discovery Studio is proprietary software, so this wrapper expects you to provide the local command or script that performs one query-candidate screen and emits one score:

```bash
python -m pharmacophore.DiscoveryStudio.cli \
  --target-dir data/DUD-E/<target> \
  --output-dir pharmacophore/results/DiscoveryStudio/<target> \
  --command-template "discovery_studio_pharmacophore_screen --query {query_ligand} --candidate {candidate} --json" \
  --score-json-key score
```

The command template can use `{query_ligand}`, `{candidate}`, `{candidate_name}`, `{output_dir}`, `{target_name}`, `{label}`, and `{idx}` placeholders.
