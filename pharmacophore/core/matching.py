"""Feature-level pharmacophore matching scores."""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F


PHARMACOPHORE_FAMILY_GROUPS = {
    "Donor": "Donor",
    "Acceptor": "Acceptor",
    "Aromatic": "Aromatic",
    "Hydrophobe": "Hydrophobe",
    "LumpedHydrophobe": "Hydrophobe",
    "PosIonizable": "PosIonizable",
    "NegIonizable": "NegIonizable",
    "ZnBinder": "ZnBinder",
}


def cosine_similarity_matrix(query_features: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
    """Return the query-feature by candidate-feature cosine similarity matrix."""
    if query_features.numel() == 0 or candidate_features.numel() == 0:
        return query_features.new_zeros((query_features.size(0), candidate_features.size(0)))

    query_norm = F.normalize(query_features, dim=1)
    candidate_norm = F.normalize(candidate_features, dim=1)
    return query_norm @ candidate_norm.transpose(0, 1)


def feature_family_compatibility_mask(
    query_metadata: list[dict],
    candidate_metadata: list[dict],
) -> torch.Tensor:
    """Return True where query and candidate pharmacophore families are compatible."""
    mask = torch.zeros((len(query_metadata), len(candidate_metadata)), dtype=torch.bool)
    for i, query_feature in enumerate(query_metadata):
        for j, candidate_feature in enumerate(candidate_metadata):
            mask[i, j] = pharmacophore_families_compatible(
                query_feature.get("family"),
                candidate_feature.get("family"),
            )
    return mask


def pharmacophore_families_compatible(query_family: str | None, candidate_family: str | None) -> bool:
    """Return whether two RDKit pharmacophore families may be matched."""
    if query_family is None or candidate_family is None:
        return True
    query_group = PHARMACOPHORE_FAMILY_GROUPS.get(query_family, query_family)
    candidate_group = PHARMACOPHORE_FAMILY_GROUPS.get(candidate_family, candidate_family)
    return query_group == candidate_group


def hungarian_matching_score(
    similarity: torch.Tensor,
    *,
    compatibility_mask: torch.Tensor | None = None,
    unmatched_similarity: float = 0.0,
) -> float:
    """Return a normalized hard one-to-one pharmacophore matching score."""
    if similarity.numel() == 0:
        return 0.0

    similarity_cpu = similarity.detach().cpu()
    if compatibility_mask is not None:
        compatibility_mask = compatibility_mask.detach().cpu().bool()
    n_query, n_candidate = similarity_cpu.shape
    if n_query == 0 or n_candidate == 0:
        return 0.0

    cost = build_hungarian_cost_matrix(
        similarity_cpu,
        compatibility_mask=compatibility_mask,
        unmatched_similarity=unmatched_similarity,
    )

    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost.numpy())
        total = selected_similarity_sum(similarity_cpu, row_ind, col_ind)
    except Exception:
        total = _bruteforce_assignment_sum(similarity_cpu, compatibility_mask=compatibility_mask)

    return total / max(n_query, n_candidate)


def build_hungarian_cost_matrix(
    similarity: torch.Tensor,
    *,
    compatibility_mask: torch.Tensor | None = None,
    unmatched_similarity: float = 0.0,
) -> torch.Tensor:
    """
    Build the constrained Hungarian cost matrix.

    Real feature-to-feature entries use cosine distance (1 - similarity).
    Incompatible entries are set to Inf. Dummy columns allow query features
    with no compatible candidate to remain unmatched with zero contribution.
    """
    n_query, _ = similarity.shape
    cost = 1.0 - similarity
    if compatibility_mask is not None:
        # Real query-candidate pharmacophore pairs with incompatible families
        cost = cost.masked_fill(~compatibility_mask, float("inf"))

    # Dummy columns are not real feature matches; they let a query feature stay
    dummy_cost = similarity.new_full((n_query, n_query), 1.0 - float(unmatched_similarity))
    return torch.cat([cost, dummy_cost], dim=1)


def selected_similarity_sum(similarity: torch.Tensor, row_ind, col_ind) -> float:
    total = 0.0
    n_candidate = similarity.shape[1]
    for row, col in zip(row_ind, col_ind):
        if int(col) < n_candidate:
            total += float(similarity[int(row), int(col)].item())
    return total


def matching_score(
    query_features: torch.Tensor,
    candidate_features: torch.Tensor,
    *,
    query_metadata: list[dict] | None = None,
    candidate_metadata: list[dict] | None = None,
    method: str,
    enforce_feature_family: bool = True,
) -> tuple[float, torch.Tensor]:
    """Build the similarity matrix and return a score for the selected method."""
    similarity = cosine_similarity_matrix(query_features, candidate_features)

    if method != "hungarian":
        raise ValueError(f"Unknown matching method: {method}")

    compatibility_mask = None
    if enforce_feature_family and query_metadata is not None and candidate_metadata is not None:
        compatibility_mask = feature_family_compatibility_mask(query_metadata, candidate_metadata).to(similarity.device)

    score = hungarian_matching_score(similarity, compatibility_mask=compatibility_mask)
    return score, similarity


def _bruteforce_assignment_sum(matrix: torch.Tensor, *, compatibility_mask: torch.Tensor | None = None) -> float:
    """Small fallback for environments without SciPy."""
    n_query, n_candidate = matrix.shape
    if min(n_query, n_candidate) > 8:
        if compatibility_mask is not None:
            matrix = matrix.masked_fill(~compatibility_mask, float("-inf"))
        values, _ = matrix.max(dim=1)
        values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
        return float(values.sum().item())

    best = None
    if n_query <= n_candidate:
        col_options = range(n_candidate + n_query)
        for cols in itertools.permutations(col_options, n_query):
            total = _assignment_similarity_sum(matrix, cols, compatibility_mask=compatibility_mask)
            best = total if best is None else max(best, total)
    else:
        for cols in itertools.permutations(range(n_candidate + n_query), n_query):
            total = _assignment_similarity_sum(matrix, cols, compatibility_mask=compatibility_mask)
            best = total if best is None else max(best, total)
    return float(best or 0.0)


def _assignment_similarity_sum(
    matrix: torch.Tensor,
    cols,
    *,
    compatibility_mask: torch.Tensor | None = None,
) -> float:
    total = 0.0
    n_candidate = matrix.shape[1]
    for row, col in enumerate(cols):
        if col >= n_candidate:
            continue
        if compatibility_mask is not None and not bool(compatibility_mask[row, col].item()):
            return float("-inf")
        total += float(matrix[row, col].item())
    return total
