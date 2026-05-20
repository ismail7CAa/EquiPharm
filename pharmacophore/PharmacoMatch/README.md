# PharmacoMatch Baseline

Optional command-template adapter for PharmacoMatch-style screening.

The public method is described as a neural 3D pharmacophore/subgraph-matching approach, but projects can expose different scripts, checkpoint names, and output formats. This wrapper therefore does not assume one concrete package API. Instead, it runs a configurable command per candidate molecule and parses one numeric score from JSON or text output.

The data layout is the same as EquiPharm:

```text
data/DUD-E/<target>/
  crystal_ligand.mol2
  actives_sdf/
  decoys_sdf/
```

Example:

```bash
python -m pharmacophore.PharmacoMatch.cli \
  --target-dir data/DUD-E/<target> \
  --output-dir pharmacophore/results/PharmacoMatch/<target> \
  --command-template "python path/to/pharmacomatch_screen.py --query {query_ligand} --candidate {candidate} --json" \
  --score-json-key score
```

Run every target in a dataset:

```bash
python -m pharmacophore.PharmacoMatch.cli \
  --dataset-dir data/DUD-E \
  --output-dir pharmacophore/results/PharmacoMatch \
  --command-template "python path/to/pharmacomatch_screen.py --query {query_ligand} --candidate {candidate} --json" \
  --score-json-key score
```

This writes per-target files under:

```text
pharmacophore/results/PharmacoMatch/<target>/
  scores.csv
  ranked_hits.csv
  metrics.json
```

It also writes:

```text
pharmacophore/results/PharmacoMatch/dataset_metrics.csv
```

The command template can use `{query_ligand}`, `{candidate}`, `{candidate_name}`, `{output_dir}`, `{target_name}`, `{label}`, and `{idx}` placeholders.
