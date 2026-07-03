# PharmacoMatch and CDPKit Scoring Difference

## Problem

The first CDPKit implementation in this repository used the simpler CDPKit command-line screening workflow:

```text
psdcreate -> psdscreen -> hit SDF output
```

That does not reproduce the CDPKit comparison used around PharmacoMatch.

The reason is that PharmacoMatch compares against CDPKit through the CDPL pharmacophore-alignment API, not only through the simpler `psdscreen` command-line hit output.

## Why The Scores Differ

`psdscreen` is mainly a screening/filtering command. It reports molecules that pass the pharmacophore query constraints. Molecules that do not pass are absent from the hit output, so the wrapper has to assign a default miss score to non-hits.

The PharmacoMatch-style CDPKit comparison uses CDPL pharmacophore alignment instead. It aligns each ligand pharmacophore to the query pharmacophore and computes an alignment score for every ligand pharmacophore. Molecules can then be ranked by their best alignment score.

So the two approaches answer different questions:

```text
CDPKit psdscreen wrapper:
Did this molecule pass the pharmacophore screen, and what score/property was reported for hits?

CDPL alignment workflow:
How well can this ligand pharmacophore align to the query pharmacophore?
```

This explains why the CDPKit wrapper scores did not match the PharmacoMatch comparison scores.

## Current Repository Interpretation

The maintained CDPKit adapter now uses the PharmacoMatch-style CDPL alignment
workflow:

```text
pharmacophore/CDPKit/
```

It creates active/decoy pharmacophore databases, aligns every ligand
pharmacophore to `query.pml`, and ranks molecules by their best alignment score.
This is the workflow to use when comparing against PharmacoMatch's CDPKit-style
alignment scores.

```text
scripts/run_cdpkit_pharmacomatch_all_dude.sh
scripts/run_pharmacomatch_cdpkit_alignment.py
scripts/eval_pharmacomatch_cdpkit_alignment.py
```

These scripts are kept as legacy/reproducibility entry points for the original
server workflow.

## External Repositories

The required external repositories and large generated data are expected under:

```text
external/
```

on the Linux/Jupyter server. They are intentionally not committed to this repository because they are large and machine-specific.

For example:

```text
external/PharmacoMatch/
external/CDPKit/
```

The scripts assume this local layout on the execution machine.

## Practical Note

For paper or report wording, distinguish the historical implementation from the
current maintained adapter. The precise statement is:

```text
The original standalone CDPKit wrapper used psdcreate/psdscreen hit-based screening. The maintained CDPKit adapter now uses CDPL pharmacophore alignment and ranks ligands by their best alignment score, matching the PharmacoMatch-style CDPKit comparison.
```
