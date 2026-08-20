#!/usr/bin/env python

import argparse
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def run_cmd(cmd, env=None):
    print("[CMD]", " ".join(map(str, cmd)))
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env=env,
    )
    if result.returncode != 0:
        print("[STDOUT]")
        print(result.stdout)
        print("[STDERR]")
        print(result.stderr)
        raise RuntimeError(f"Command failed with return code {result.returncode}")
    return result


def make_openpharmaco_env(openpharmaco_root: Path):
    env = os.environ.copy()
    modules = openpharmaco_root / "modules"
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{modules}:{old_pythonpath}" if old_pythonpath else str(modules)
    return env


def clean_receptor_with_obabel(input_pdb: Path, output_pdb: Path):
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    if output_pdb.exists():
        return output_pdb

    run_cmd([
        "obabel",
        str(input_pdb),
        "-O",
        str(output_pdb),
    ])

    return output_pdb


def split_sdf_with_obabel(input_sdf: Path, output_dir: Path, prefix: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob(f"{prefix}_*.sdf"))
    if existing:
        return existing

    output_pattern = output_dir / f"{prefix}_.sdf"

    run_cmd([
        "obabel",
        str(input_sdf),
        "-O",
        str(output_pattern),
        "-m",
    ])

    files = sorted(output_dir.glob(f"{prefix}_*.sdf"))

    if not files:
        raise RuntimeError(f"No split molecules created from {input_sdf}")

    return files


def parse_score(stdout: str):
    for line in stdout.splitlines():
        line = line.strip()
        if line.lower().startswith("score"):
            value = line.replace("=", ":").split(":", 1)[1].strip()
            return float(value)

    raise RuntimeError(f"No score found in output:\n{stdout}")


def create_model(
    target_name: str,
    target_dir: Path,
    model_path: Path,
    openpharmaco_root: Path,
    work_root: Path,
):
    if model_path.exists():
        print(f"[SKIP] Model already exists: {model_path}")
        return model_path

    receptor_clean_in_target = target_dir / "receptor_clean.pdb"
    receptor_raw = target_dir / "receptor.pdb"
    ligand = target_dir / "crystal_ligand.mol2"

    if receptor_clean_in_target.exists():
        receptor = receptor_clean_in_target
    elif receptor_raw.exists():
        receptor = clean_receptor_with_obabel(
            receptor_raw,
            work_root / "clean_receptors" / target_name / "receptor_clean.pdb",
        )
    else:
        raise FileNotFoundError(f"No receptor_clean.pdb or receptor.pdb found for {target_name}")

    if not ligand.exists():
        raise FileNotFoundError(f"No crystal_ligand.mol2 found for {target_name}")

    model_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_openpharmaco_env(openpharmaco_root)

    cmd = [
        sys.executable,
        str(openpharmaco_root / "create_model.py"),
        "--protein",
        str(receptor),
        "--ligand",
        str(ligand),
        "--output",
        str(model_path),
    ]

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env=env,
    )

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Model creation failed for {target_name}")

    return model_path


def score_candidate(model_path: Path, candidate_path: Path, openpharmaco_root: Path):
    env = make_openpharmaco_env(openpharmaco_root)

    cmd = [
        sys.executable,
        str(openpharmaco_root / "batch_score.py"),
        "--model",
        str(model_path),
        "--candidate",
        str(candidate_path),
    ]

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env=env,
    )

    if result.returncode != 0:
        return None, result.stderr.strip()

    try:
        return parse_score(result.stdout), None
    except Exception as e:
        return None, str(e)


def enrichment_factor_at_fraction(y_true, scores, fraction=0.01):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    n = len(y_true)
    n_actives = int(y_true.sum())

    if n == 0 or n_actives == 0:
        return float("nan")

    k = max(1, int(math.ceil(n * fraction)))
    order = np.argsort(-scores)
    top_k = order[:k]

    actives_top_k = int(y_true[top_k].sum())

    return (actives_top_k / k) / (n_actives / n)


def bedroc(y_true, scores, alpha=20.0):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    n = len(y_true)
    n_actives = int(y_true.sum())

    if n == 0 or n_actives == 0 or n_actives == n:
        return float("nan")

    order = np.argsort(-scores)
    ranked_y = y_true[order]

    active_ranks = np.where(ranked_y == 1)[0] + 1

    rie_num = np.sum(np.exp(-alpha * active_ranks / n))
    rie_den = n_actives * (1.0 - np.exp(-alpha)) / (alpha * n / n)
    rie = rie_num / rie_den if rie_den != 0 else float("nan")

    max_ranks = np.arange(1, n_actives + 1)
    min_ranks = np.arange(n - n_actives + 1, n + 1)

    rie_max = np.sum(np.exp(-alpha * max_ranks / n)) / rie_den
    rie_min = np.sum(np.exp(-alpha * min_ranks / n)) / rie_den

    if rie_max == rie_min:
        return float("nan")

    return float((rie - rie_min) / (rie_max - rie_min))


def compute_metrics(rows):
    df = pd.DataFrame(rows)
    df_valid = df.dropna(subset=["score"]).copy()

    if df_valid.empty:
        raise RuntimeError("No valid scores available for metrics.")

    y_true = df_valid["label"].astype(int).to_numpy()
    scores = df_valid["score"].astype(float).to_numpy()

    metrics = {
        "n_total": len(df_valid),
        "n_actives": int(y_true.sum()),
        "n_decoys": int((1 - y_true).sum()),
        "n_failed": int(df["score"].isna().sum()),
        "roc_auc": roc_auc_score(y_true, scores) if len(set(y_true)) == 2 else float("nan"),
        "pr_auc": average_precision_score(y_true, scores) if len(set(y_true)) == 2 else float("nan"),
        "ef1": enrichment_factor_at_fraction(y_true, scores, fraction=0.01),
        "bedroc20": bedroc(y_true, scores, alpha=20.0),
    }

    ranked_df = df_valid.sort_values("score", ascending=False)

    return metrics, ranked_df


def run_target_seed(
    target_name: str,
    model_path: Path,
    active_files: list[Path],
    decoy_files: list[Path],
    seed: int,
    openpharmaco_root: Path,
    output_root: Path,
    n_actives: int,
    n_decoys: int,
):
    rng = random.Random(seed)

    active_subset = sorted(active_files)
    decoy_subset = sorted(decoy_files)

    rng.shuffle(active_subset)
    rng.shuffle(decoy_subset)

    active_subset = active_subset[:n_actives]
    decoy_subset = decoy_subset[:n_decoys]

    if len(active_subset) < n_actives:
        raise RuntimeError(
            f"{target_name} seed {seed}: only {len(active_subset)} actives available, need {n_actives}"
        )

    if len(decoy_subset) < n_decoys:
        raise RuntimeError(
            f"{target_name} seed {seed}: only {len(decoy_subset)} decoys available, need {n_decoys}"
        )

    candidates = [(p, 1) for p in active_subset] + [(p, 0) for p in decoy_subset]

    print(
        f"[{target_name}][seed={seed}] "
        f"Scoring {len(active_subset)} actives + {len(decoy_subset)} decoys"
    )

    rows = []
    failures = []

    for i, (candidate, label) in enumerate(candidates, start=1):
        if i % 50 == 0:
            print(f"[{target_name}][seed={seed}] scored {i}/{len(candidates)}")

        score, error = score_candidate(model_path, candidate, openpharmaco_root)

        row = {
            "target": target_name,
            "seed": seed,
            "compound_file": str(candidate),
            "compound_id": candidate.stem,
            "label": label,
            "score": score,
        }

        rows.append(row)

        if error is not None:
            failures.append({
                "target": target_name,
                "seed": seed,
                "compound_file": str(candidate),
                "compound_id": candidate.stem,
                "label": label,
                "error": error,
            })

    seed_output = output_root / target_name / f"seed_{seed}"
    seed_output.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(seed_output / "scores_raw.csv", index=False)

    if failures:
        pd.DataFrame(failures).to_csv(seed_output / "failures.csv", index=False)

    metrics, ranked_df = compute_metrics(rows)

    ranked_df.to_csv(seed_output / "scores_ranked.csv", index=False)

    metrics_row = {
        "target": target_name,
        "seed": seed,
        **metrics,
    }

    pd.DataFrame([metrics_row]).to_csv(seed_output / "metrics.csv", index=False)

    print(f"[DONE] {target_name} seed {seed}")
    print(metrics_row)

    return metrics_row


def summarize_target(target_name: str, seed_metrics: list[dict], output_root: Path):
    df = pd.DataFrame(seed_metrics)

    metrics_cols = ["roc_auc", "pr_auc", "ef1","ef5", "ef10", "bedroc20"]

    summary = {
        "target": target_name,
        "n_seeds": len(df),
    }

    for col in metrics_cols:
        summary[f"{col}_mean"] = df[col].mean()
        summary[f"{col}_std"] = df[col].std(ddof=1)

    target_dir = output_root / target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(target_dir / "seed_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(target_dir / "summary_mean_std.csv", index=False)

    return summary


def run_target(
    target_dir: Path,
    openpharmaco_root: Path,
    model_root: Path,
    output_root: Path,
    work_root: Path,
    seeds: list[int],
    n_actives: int,
    n_decoys: int,
):
    target_name = target_dir.name
    print(f"\n========== TARGET: {target_name} ==========")

    active_sdf = target_dir / "actives_final.sdf"
    decoy_sdf = target_dir / "decoys_final.sdf"

    if not active_sdf.exists() or not decoy_sdf.exists():
        print(f"[SKIP] Missing actives_final.sdf or decoys_final.sdf for {target_name}")
        return None

    model_path = model_root / f"{target_name}.pm"

    create_model(
        target_name=target_name,
        target_dir=target_dir,
        model_path=model_path,
        openpharmaco_root=openpharmaco_root,
        work_root=work_root,
    )

    split_dir = work_root / "split_sdf" / target_name

    active_files = split_sdf_with_obabel(
        active_sdf,
        split_dir / "actives",
        "active",
    )

    decoy_files = split_sdf_with_obabel(
        decoy_sdf,
        split_dir / "decoys",
        "decoy",
    )

    seed_metrics = []

    for seed in seeds:
        result = run_target_seed(
            target_name=target_name,
            model_path=model_path,
            active_files=active_files,
            decoy_files=decoy_files,
            seed=seed,
            openpharmaco_root=openpharmaco_root,
            output_root=output_root,
            n_actives=n_actives,
            n_decoys=n_decoys,
        )
        seed_metrics.append(result)

    summary = summarize_target(target_name, seed_metrics, output_root)

    print(f"[SUMMARY] {target_name}")
    print(summary)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dude-root", default="data/DUD-E")
    parser.add_argument("--openpharmaco-root", default="external/OpenPharmaco")
    parser.add_argument("--model-root", default="pharmacophore/models/OpenPharmaco")
    parser.add_argument("--output-root", default="pharmacophore/results/OpenPharmaco_subsets")
    parser.add_argument("--work-root", default="pharmacophore/work/OpenPharmaco")
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--n-actives", type=int, default=50)
    parser.add_argument("--n-decoys", type=int, default=500)
    args = parser.parse_args()

    dude_root = Path(args.dude_root)
    openpharmaco_root = Path(args.openpharmaco_root)
    model_root = Path(args.model_root)
    output_root = Path(args.output_root)
    work_root = Path(args.work_root)

    if args.targets:
        target_dirs = [dude_root / t for t in args.targets]
    else:
        target_dirs = sorted([p for p in dude_root.iterdir() if p.is_dir()])

    all_summaries = []

    for target_dir in target_dirs:
        try:
            summary = run_target(
                target_dir=target_dir,
                openpharmaco_root=openpharmaco_root,
                model_root=model_root,
                output_root=output_root,
                work_root=work_root,
                seeds=args.seeds,
                n_actives=args.n_actives,
                n_decoys=args.n_decoys,
            )
            if summary is not None:
                all_summaries.append(summary)
        except Exception as e:
            print(f"[FAILED] {target_dir.name}: {e}")
            fail_dir = output_root / target_dir.name
            fail_dir.mkdir(parents=True, exist_ok=True)
            with open(fail_dir / "target_error.txt", "w") as f:
                f.write(str(e))

    if all_summaries:
        output_root.mkdir(parents=True, exist_ok=True)
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(output_root / "summary_metrics_mean_std.csv", index=False)

        print("\n========== GLOBAL SUMMARY ==========")
        print(summary_df)


if __name__ == "__main__":
    main()
