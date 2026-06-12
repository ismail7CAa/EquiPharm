# Schrodinger Phase Baseline

Optional command-template adapter for Schrodinger Phase pharmacophore screening.

Phase is proprietary software, so this wrapper does not assume one executable path or output format. Provide a local command template and a score parser, and the adapter writes the same `scores.csv`, `ranked_hits.csv`, `metrics.json`, and summary files as the other baselines.

```bash
python -m pharmacophore.SchrodingerPhase.cli \
  --target-dir data/DUD-E/<target> \
  --output-dir pharmacophore/results/SchrodingerPhase/<target> \
  --command-template "phase_screen --query {query_ligand} --candidate {candidate} --json" \
  --score-json-key score
```

The command template can use `{query_ligand}`, `{candidate}`, `{candidate_name}`, `{output_dir}`, `{target_name}`, `{label}`, and `{idx}` placeholders.
