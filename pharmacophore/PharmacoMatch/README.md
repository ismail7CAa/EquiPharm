# PharmacoMatch Baseline

Optional command-template adapter for PharmacoMatch-style screening.

The public method is described as a neural 3D pharmacophore/subgraph-matching approach, but projects can expose different scripts, checkpoint names, and output formats. This wrapper therefore does not assume one concrete package API. Instead, it runs a configurable command per candidate molecule and parses one numeric score from JSON or text output.

This folder contains the reusable PharmacoMatch adapter and official-runner integration. The separate CDPKit comparison used around PharmacoMatch is documented as a script-based CDPL alignment workflow, not as the simple `pharmacophore/CDPKit` `psdscreen` wrapper.

For that alignment-style comparison, use:

```text
scripts/run_cdpkit_pharmacomatch_all_dude.sh
scripts/run_pharmacomatch_cdpkit_alignment.py
scripts/eval_pharmacomatch_cdpkit_alignment.py
```

Those scripts assume the PharmacoMatch repository and generated data are available locally under `external/PharmacoMatch/` on the Linux/Jupyter machine. The `external/` directory is intentionally not committed because it can be large and machine-specific.

See `challenges_and_limitations/00_PharmacoMatch_CDPKit.md` for why `psdscreen` hit output and CDPL alignment scores are not equivalent.

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
