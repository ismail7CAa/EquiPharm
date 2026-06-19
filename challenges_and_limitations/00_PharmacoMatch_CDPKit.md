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

There are now two CDPKit-related workflows:

```text
pharmacophore/CDPKit/
```

This remains the reusable command-line CDPKit baseline adapter. It uses `psdcreate` and `psdscreen`, writes the same result files as the other wrappers, and is useful as a hit-based CDPKit screening baseline.

```text
scripts/run_cdpkit_pharmacomatch_all_dude.sh
scripts/run_pharmacomatch_cdpkit_alignment.py
scripts/eval_pharmacomatch_cdpkit_alignment.py
```

These scripts implement the PharmacoMatch-aligned CDPL pharmacophore alignment workflow. This is the workflow to use when comparing against PharmacoMatch's CDPKit-style alignment scores.

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

For paper or report wording, avoid saying that the simple `pharmacophore/CDPKit` wrapper reproduces the PharmacoMatch CDPKit baseline. The precise statement is:

```text
The standalone CDPKit wrapper uses psdcreate/psdscreen hit-based screening, whereas the PharmacoMatch comparison uses CDPL pharmacophore alignment and ranks ligands by their best alignment score.
```
