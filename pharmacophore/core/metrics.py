"""Screening metrics and plot generation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve


def compute_metrics(scores, labels, *, pipeline_name: str, target_name: str) -> dict:
    y_true = np.asarray(labels, dtype=int)
    y_score = np.asarray(scores, dtype=float)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return {
        "pipeline": pipeline_name,
        "target": target_name,
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(auc(recall, precision)),
        "n_total": int(len(y_true)),
        "n_actives": int(y_true.sum()),
        "n_decoys": int((1 - y_true).sum()),
    }


def write_outputs(
    output_dir: str | Path,
    rows: list[dict],
    *,
    pipeline_name: str,
    target_name: str,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    df.to_csv(output_path / "scores.csv", index=False)
    df.to_csv(output_path / "ranked_hits.csv", index=False)

    metrics = compute_metrics(
        df["score"].to_numpy(),
        df["label"].to_numpy(),
        pipeline_name=pipeline_name,
        target_name=target_name,
    )
    with (output_path / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")

    title_prefix = f"{pipeline_name} - {target_name}"
    plot_score_boxplot(df, output_path / "cosine_similarity_boxplot.png", title_prefix=title_prefix)
    plot_roc_curve(
        df,
        metrics["roc_auc"],
        output_path / "roc_curve_actives_vs_decoys.png",
        title_prefix=title_prefix,
    )
    return metrics


def plot_score_boxplot(df: pd.DataFrame, path: Path, *, title_prefix: str) -> None:
    active_scores = df.loc[df["label"] == 1, "score"].to_numpy()
    decoy_scores = df.loc[df["label"] == 0, "score"].to_numpy()
    plt.figure(figsize=(6, 5))
    plt.boxplot([decoy_scores, active_scores], tick_labels=["Decoys", "Actives"], showfliers=False)
    plt.ylabel("Cosine Similarity")
    plt.title(f"{title_prefix}\nCosine Similarity: Actives vs Decoys")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_roc_curve(df: pd.DataFrame, roc_auc: float, path: Path, *, title_prefix: str) -> None:
    fpr, tpr, _ = roc_curve(df["label"].to_numpy(), df["score"].to_numpy())
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUROC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix}\nROC Curve: Actives vs Decoys")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
