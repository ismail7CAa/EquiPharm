"""Feature-level pharmacophore matching scores."""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F


def cosine_similarity_matrix(query_features: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
    """Return the query-feature by candidate-feature cosine similarity matrix."""
    if query_features.numel() == 0 or candidate_features.numel() == 0:
        return query_features.new_zeros((query_features.size(0), candidate_features.size(0)))

    query_norm = F.normalize(query_features, dim=1)
    candidate_norm = F.normalize(candidate_features, dim=1)
    return query_norm @ candidate_norm.transpose(0, 1)


def apply_feature_family_penalty(
    similarity: torch.Tensor,
    query_metadata: list[dict],
    candidate_metadata: list[dict],
    *,
    mismatch_penalty: float = 0.5,
) -> torch.Tensor:
    """Penalize feature-family mismatches without forbidding imperfect matches."""
    if similarity.numel() == 0 or mismatch_penalty <= 0:
        return similarity

    adjusted = similarity.clone()
    for i, query_feature in enumerate(query_metadata):
        query_family = query_feature.get("family")
        if query_family is None:
            continue
        for j, candidate_feature in enumerate(candidate_metadata):
            candidate_family = candidate_feature.get("family")
            if candidate_family is not None and candidate_family != query_family:
                adjusted[i, j] = adjusted[i, j] - mismatch_penalty
    return adjusted


def hungarian_matching_score(similarity: torch.Tensor) -> float:
    """Return a normalized hard one-to-one matching score."""
    if similarity.numel() == 0:
        return 0.0

    matrix = similarity.detach().cpu()
    n_query, n_candidate = matrix.shape
    if n_query == 0 or n_candidate == 0:
        return 0.0

    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment((-matrix).numpy())
        selected = matrix[row_ind, col_ind]
        total = float(selected.sum().item())
    except Exception:
        total = _bruteforce_assignment_sum(matrix)

    return total / max(n_query, n_candidate)


def sinkhorn_matching_score(
    similarity: torch.Tensor,
    *,
    temperature: float = 0.1,
    iterations: int = 50,
) -> float:
    """Return a soft optimal-transport matching score from a similarity matrix."""
    if similarity.numel() == 0:
        return 0.0

    n_query, n_candidate = similarity.shape
    if n_query == 0 or n_candidate == 0:
        return 0.0
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    transport = torch.exp(similarity / temperature)
    transport = transport.clamp_min(torch.finfo(transport.dtype).tiny)

    query_mass = similarity.new_full((n_query,), 1.0 / n_query)
    candidate_mass = similarity.new_full((n_candidate,), 1.0 / n_candidate)

    for _ in range(iterations):
        transport = transport * (query_mass / transport.sum(dim=1).clamp_min(1e-12)).unsqueeze(1)
        transport = transport * (candidate_mass / transport.sum(dim=0).clamp_min(1e-12)).unsqueeze(0)

    matched_similarity = (transport * similarity).sum()
    return float(matched_similarity.detach().cpu().item())


def matching_score(
    query_features: torch.Tensor,
    candidate_features: torch.Tensor,
    *,
    query_metadata: list[dict] | None = None,
    candidate_metadata: list[dict] | None = None,
    method: str,
    mismatch_penalty: float = 0.5,
    sinkhorn_temperature: float = 0.1,
    sinkhorn_iterations: int = 50,
) -> tuple[float, torch.Tensor]:
    """Build the similarity matrix and return a score for the selected method."""
    similarity = cosine_similarity_matrix(query_features, candidate_features)
    if query_metadata is not None and candidate_metadata is not None:
        similarity = apply_feature_family_penalty(
            similarity,
            query_metadata,
            candidate_metadata,
            mismatch_penalty=mismatch_penalty,
        )

    if method == "hungarian":
        score = hungarian_matching_score(similarity)
    elif method == "sinkhorn":
        score = sinkhorn_matching_score(
            similarity,
            temperature=sinkhorn_temperature,
            iterations=sinkhorn_iterations,
        )
    else:
        raise ValueError(f"Unknown matching method: {method}")

    return score, similarity


def _bruteforce_assignment_sum(matrix: torch.Tensor) -> float:
    """Small fallback for environments without SciPy."""
    n_query, n_candidate = matrix.shape
    if min(n_query, n_candidate) > 8:
        values, _ = matrix.max(dim=1 if n_candidate >= n_query else 0)
        return float(values.sum().item())

    best = None
    if n_query <= n_candidate:
        for cols in itertools.permutations(range(n_candidate), n_query):
            total = sum(float(matrix[row, col].item()) for row, col in enumerate(cols))
            best = total if best is None else max(best, total)
    else:
        for rows in itertools.permutations(range(n_query), n_candidate):
            total = sum(float(matrix[row, col].item()) for col, row in enumerate(rows))
            best = total if best is None else max(best, total)
    return float(best or 0.0)
