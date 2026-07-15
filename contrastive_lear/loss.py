"""Order-embedding objective and metrics."""

from __future__ import annotations

import torch


def penalty(query: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (query - target).clamp_min(0).square().sum(dim=1)


def order_embedding_loss(embeddings: dict[str, torch.Tensor], margin: float = 100.0):
    queries = embeddings["queries"]
    targets = embeddings["targets"]
    references = embeddings["references"]
    negative_queries = embeddings["negative_queries"]
    negative_targets = embeddings["negative_targets"]
    shift = max(1, len(targets) // 2)

    positive = penalty(queries, targets)
    reference_positive = penalty(references, targets)
    negatives = torch.cat(
        [
            penalty(negative_queries, targets),
            penalty(references, negative_targets),
            penalty(queries, torch.roll(targets, shift, dims=0)),
            penalty(targets, torch.roll(targets, shift, dims=0)),
        ]
    )
    # PharmacoMatch optimizes the summed order-embedding penalties, not their means.
    loss = positive.sum() + (margin - negatives).clamp_min(0).sum()

    scores = torch.cat([-reference_positive, -negatives])
    labels = torch.cat(
        [torch.ones_like(reference_positive), torch.zeros_like(negatives)]
    ).long()
    return loss, scores, labels, positive.mean(), negatives.mean()


def binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    # Equivalent to the Mann-Whitney U statistic, evaluated in manageable chunks.
    wins = 0.0
    comparisons = 0
    for chunk in positive.split(1024):
        relation = chunk[:, None] - negative[None, :]
        wins += (relation > 0).sum().item() + 0.5 * (relation == 0).sum().item()
        comparisons += relation.numel()
    return wins / comparisons
