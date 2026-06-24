"""Feature-level pharmacophore matching scores."""

from __future__ import annotations

import itertools
import math

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

STRICT_SCORE_MODE = "strict"
BALANCED_SCORE_MODE = "balanced"
FEATURE_DISTANCE_SCORE_MODE = "feature_distance"
GEOMETRY_DISTANCE_SCORE_MODE = "geometry_distance"
EMBEDDING_DISTANCE_SCORE_MODE = "embedding_distance"
EMBEDDING_GEOMETRY_DISTANCE_SCORE_MODE = "embedding_geometry_distance"
COSINE_SCORE_MODE = "cosine"
COSINE_GEOMETRY_SCORE_MODE = "cosine_geometry"
TIERED_DISTANCE_GEOMETRY_SCORE_MODE = "tiered_distance_geometry"
HYBRID_LOCAL_GEOMETRY_SCORE_MODE = "hybrid_local_geometry"
NO_MATCH_DISTANCE = 1_000_000.0
HUNGARIAN_METHOD = "hungarian"
HUNGARIAN_EUCLIDEAN_METHOD = "hungarian_euclidean"
HUNGARIAN_GAUSSIAN_METHOD = "hungarian_gaussian"
HUNGARIAN_COSINE_QUALITY_METHOD = "hungarian_cosine_quality"
HUNGARIAN_3D_METHOD = "hungarian_3d"


def cosine_similarity_matrix(query_features: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
    """Return the query-feature by candidate-feature cosine similarity matrix."""
    if query_features.numel() == 0 or candidate_features.numel() == 0:
        return query_features.new_zeros((query_features.size(0), candidate_features.size(0)))

    query_norm = F.normalize(query_features, dim=1)
    candidate_norm = F.normalize(candidate_features, dim=1)
    return query_norm @ candidate_norm.transpose(0, 1)


def feature_center_distance_similarity_matrix(
    query_metadata: list[dict],
    candidate_metadata: list[dict],
    *,
    device=None,
) -> torch.Tensor:
    """Return negative 3D feature-center distances for Hungarian maximization."""
    rows = len(query_metadata)
    cols = len(candidate_metadata)
    if rows == 0 or cols == 0:
        return torch.zeros((rows, cols), dtype=torch.float32, device=device)

    similarity = torch.full((rows, cols), -NO_MATCH_DISTANCE, dtype=torch.float32, device=device)
    for i, query_feature in enumerate(query_metadata):
        query_center = query_feature.get("center")
        if query_center is None:
            continue
        query_center = torch.as_tensor(query_center, dtype=torch.float32, device=device)
        for j, candidate_feature in enumerate(candidate_metadata):
            candidate_center = candidate_feature.get("center")
            if candidate_center is None:
                continue
            candidate_center = torch.as_tensor(candidate_center, dtype=torch.float32, device=device)
            similarity[i, j] = -torch.linalg.vector_norm(query_center - candidate_center)
    return similarity


def embedding_distance_similarity_matrix(query_features: torch.Tensor, candidate_features: torch.Tensor) -> torch.Tensor:
    """Return negative embedding-space Euclidean distances for Hungarian maximization."""
    if query_features.numel() == 0 or candidate_features.numel() == 0:
        return query_features.new_zeros((query_features.size(0), candidate_features.size(0)))
    return -torch.cdist(query_features, candidate_features, p=2)


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
) -> tuple[float, list[tuple[int, int | None]], dict]:
    """Return a normalized hard one-to-one pharmacophore matching score and assignments."""
    if similarity.numel() == 0:
        return 0.0, [], empty_score_components()

    similarity_cpu = similarity.detach().cpu()
    if compatibility_mask is not None:
        compatibility_mask = compatibility_mask.detach().cpu().bool()
    n_query, n_candidate = similarity_cpu.shape
    if n_query == 0 or n_candidate == 0:
        return 0.0, [], empty_score_components()

    cost = build_hungarian_cost_matrix(
        similarity_cpu,
        compatibility_mask=compatibility_mask,
        unmatched_similarity=unmatched_similarity,
    )

    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost.numpy())
        assignments = assignment_pairs(row_ind, col_ind, n_candidate)
        total = selected_similarity_sum(similarity_cpu, assignments)
    except Exception:
        total, assignments = _bruteforce_assignment(similarity_cpu, compatibility_mask=compatibility_mask)

    components = score_components(total, assignments, n_query, n_candidate)
    return components["strict_score"], assignments, components


def score_components(
    selected_similarity_total: float,
    assignments: list[tuple[int, int | None]],
    n_query: int,
    n_candidate: int,
) -> dict:
    matched_count = sum(1 for _, col in assignments if col is not None)
    matched_average = selected_similarity_total / matched_count if matched_count else 0.0
    query_coverage = matched_count / n_query if n_query else 0.0
    candidate_coverage = matched_count / n_candidate if n_candidate else 0.0
    balanced_score = matched_average * query_coverage
    strict_score = selected_similarity_total / max(n_query, n_candidate) if max(n_query, n_candidate) else 0.0
    return {
        "selected_similarity_total": float(selected_similarity_total),
        "matched_feature_count": int(matched_count),
        "matched_average_similarity": float(matched_average),
        "query_feature_coverage": float(query_coverage),
        "candidate_feature_coverage": float(candidate_coverage),
        "balanced_score": float(balanced_score),
        "strict_score": float(strict_score),
    }


def empty_score_components() -> dict:
    return {
        "selected_similarity_total": 0.0,
        "matched_feature_count": 0,
        "matched_average_similarity": 0.0,
        "query_feature_coverage": 0.0,
        "candidate_feature_coverage": 0.0,
        "balanced_score": 0.0,
        "strict_score": 0.0,
    }


def distance_score_components(match_details: list[dict]) -> dict:
    matched = [match for match in match_details if match.get("status") == "matched"]
    feature_distances = [
        float(match["feature_distance"])
        for match in matched
        if match.get("feature_distance") is not None
    ]
    feature_average = sum(feature_distances) / len(feature_distances) if feature_distances else NO_MATCH_DISTANCE

    geometry_deltas = []
    for left, right in itertools.combinations(matched, 2):
        left_query_center = left.get("query_center")
        right_query_center = right.get("query_center")
        left_candidate_center = left.get("candidate_center")
        right_candidate_center = right.get("candidate_center")
        if not all((left_query_center, right_query_center, left_candidate_center, right_candidate_center)):
            continue
        query_distance = euclidean_distance(left_query_center, right_query_center)
        candidate_distance = euclidean_distance(left_candidate_center, right_candidate_center)
        geometry_deltas.append(abs(query_distance - candidate_distance))

    geometry_average = sum(geometry_deltas) / len(geometry_deltas) if geometry_deltas else feature_average
    return {
        "matched_feature_distance_sum": float(sum(feature_distances)),
        "matched_feature_distance_count": int(len(feature_distances)),
        "average_feature_distance": float(feature_average),
        "feature_distance_score": float(-feature_average),
        "geometry_distance_delta_sum": float(sum(geometry_deltas)),
        "geometry_distance_pair_count": int(len(geometry_deltas)),
        "average_geometry_distance_delta": float(geometry_average),
        "geometry_distance_score": float(-geometry_average),
    }


def cosine_score_components(
    query_features: torch.Tensor,
    candidate_features: torch.Tensor,
    assignments: list[tuple[int, int | None]],
) -> dict:
    matched = [(row, col) for row, col in assignments if col is not None]
    similarity = cosine_similarity_matrix(query_features, candidate_features)
    matched_values = [float(similarity[row, col].detach().cpu().item()) for row, col in matched]
    matched_average = sum(matched_values) / len(matched_values) if matched_values else 0.0

    query_internal = cosine_similarity_matrix(query_features, query_features)
    candidate_internal = cosine_similarity_matrix(candidate_features, candidate_features)
    geometry_deltas = []
    for (left_query, left_candidate), (right_query, right_candidate) in itertools.combinations(matched, 2):
        query_distance = 1.0 - float(query_internal[left_query, right_query].detach().cpu().item())
        candidate_distance = 1.0 - float(candidate_internal[left_candidate, right_candidate].detach().cpu().item())
        geometry_deltas.append(abs(query_distance - candidate_distance))

    geometry_average = sum(geometry_deltas) / len(geometry_deltas) if geometry_deltas else 0.0
    geometry_score = -geometry_average if geometry_deltas else matched_average
    return {
        "matched_cosine_similarity_sum": float(sum(matched_values)),
        "matched_cosine_similarity_count": int(len(matched_values)),
        "matched_cosine_similarity_score": float(matched_average),
        "cosine_geometry_delta_sum": float(sum(geometry_deltas)),
        "cosine_geometry_pair_count": int(len(geometry_deltas)),
        "average_cosine_geometry_delta": float(geometry_average),
        "cosine_geometry_score": float(geometry_score),
    }


def embedding_distance_score_components(
    query_features: torch.Tensor,
    candidate_features: torch.Tensor,
    assignments: list[tuple[int, int | None]],
) -> dict:
    matched = [(row, col) for row, col in assignments if col is not None]
    matched_distances = [
        float(torch.linalg.vector_norm(query_features[row] - candidate_features[col]).detach().cpu().item())
        for row, col in matched
    ]
    matched_average = sum(matched_distances) / len(matched_distances) if matched_distances else NO_MATCH_DISTANCE

    geometry_deltas = []
    for (left_query, left_candidate), (right_query, right_candidate) in itertools.combinations(matched, 2):
        query_distance = float(
            torch.linalg.vector_norm(query_features[left_query] - query_features[right_query]).detach().cpu().item()
        )
        candidate_distance = float(
            torch.linalg.vector_norm(candidate_features[left_candidate] - candidate_features[right_candidate]).detach().cpu().item()
        )
        geometry_deltas.append(abs(query_distance - candidate_distance))

    geometry_average = sum(geometry_deltas) / len(geometry_deltas) if geometry_deltas else matched_average
    return {
        "matched_embedding_distance_sum": float(sum(matched_distances)),
        "matched_embedding_distance_count": int(len(matched_distances)),
        "average_embedding_distance": float(matched_average),
        "embedding_distance_score": float(-matched_average),
        "embedding_geometry_delta_sum": float(sum(geometry_deltas)),
        "embedding_geometry_pair_count": int(len(geometry_deltas)),
        "average_embedding_geometry_delta": float(geometry_average),
        "embedding_geometry_distance_score": float(-geometry_average),
    }


def tiered_distance_geometry_score_components(
    query_features: torch.Tensor,
    candidate_features: torch.Tensor,
    assignments: list[tuple[int, int | None]],
    match_details: list[dict],
    *,
    distance_sigma: float = 1.0,
    geometry_penalty_weight: float = 1.0,
) -> dict:
    """Score matched embedding distances, then penalize distorted 3D geometry.

    Tier 1 uses a Gaussian distance-quality function and normalizes by the
    larger feature-set size, so unmatched features reduce the score. Tier 2
    compares all pairwise distances between matched 3D feature centers. Its
    RMSE is converted to a bounded multiplicative penalty.
    """
    if distance_sigma <= 0:
        raise ValueError("distance_sigma must be greater than zero.")
    if geometry_penalty_weight < 0:
        raise ValueError("geometry_penalty_weight must be non-negative.")

    matched_assignments = [(row, col) for row, col in assignments if col is not None]
    embedding_distances = [
        float(torch.linalg.vector_norm(query_features[row] - candidate_features[col]).detach().cpu().item())
        for row, col in matched_assignments
    ]
    qualities = [math.exp(-0.5 * (distance / distance_sigma) ** 2) for distance in embedding_distances]
    normalization = max(int(query_features.size(0)), int(candidate_features.size(0)))
    tier1_score = sum(qualities) / normalization if normalization else 0.0

    matched_details = [match for match in match_details if match.get("status") == "matched"]
    geometry_deltas = []
    for left, right in itertools.combinations(matched_details, 2):
        centers = (
            left.get("query_center"),
            right.get("query_center"),
            left.get("candidate_center"),
            right.get("candidate_center"),
        )
        if any(center is None for center in centers):
            continue
        query_distance = euclidean_distance(centers[0], centers[1])
        candidate_distance = euclidean_distance(centers[2], centers[3])
        geometry_deltas.append(abs(query_distance - candidate_distance))

    geometry_rmse = (
        math.sqrt(sum(delta * delta for delta in geometry_deltas) / len(geometry_deltas))
        if geometry_deltas
        else 0.0
    )
    geometry_available = bool(geometry_deltas)
    geometry_factor = math.exp(-geometry_penalty_weight * geometry_rmse) if geometry_available else 1.0
    final_score = tier1_score * geometry_factor

    for detail, distance, quality in zip(matched_details, embedding_distances, qualities):
        detail["embedding_distance"] = distance
        detail["distance_quality"] = quality

    return {
        "tier1_score": float(tier1_score),
        "tier1_distance_quality_sum": float(sum(qualities)),
        "tier1_distance_sigma": float(distance_sigma),
        "tier2_geometry_rmse": float(geometry_rmse),
        "tier2_geometry_pair_count": int(len(geometry_deltas)),
        "tier2_geometry_available": bool(geometry_available),
        "geometry_penalty_weight": float(geometry_penalty_weight),
        "geometry_penalty_factor": float(geometry_factor),
        "tiered_final_score": float(final_score),
    }


def hybrid_local_geometry_score_components(
    query_features: torch.Tensor,
    candidate_features: torch.Tensor,
    assignments: list[tuple[int, int | None]],
    match_details: list[dict],
    *,
    embedding_weight: float = 0.4,
    spatial_weight: float = 0.6,
    spatial_tau: float = 2.0,
    geometry_penalty_weight: float = 0.3,
    require_full_query_coverage: bool = False,
) -> dict:
    """Compute V5 alignment-free local-quality and global-geometry scoring."""
    if embedding_weight < 0 or spatial_weight < 0 or embedding_weight + spatial_weight <= 0:
        raise ValueError("V5 embedding and spatial weights must be non-negative with a positive sum.")
    if spatial_tau <= 0:
        raise ValueError("spatial_tau must be greater than zero.")
    if geometry_penalty_weight < 0:
        raise ValueError("geometry_penalty_weight must be non-negative.")

    weight_total = embedding_weight + spatial_weight
    embedding_weight /= weight_total
    spatial_weight /= weight_total
    matched_assignments = [(row, col) for row, col in assignments if col is not None]
    matched_details = [match for match in match_details if match.get("status") == "matched"]
    cosine = cosine_similarity_matrix(query_features, candidate_features)
    embedding_qualities = [
        max(0.0, min(1.0, (float(cosine[row, col].detach().cpu().item()) + 1.0) / 2.0))
        for row, col in matched_assignments
    ]

    pair_deltas: dict[tuple[int, int], float] = {}
    query_pair_distances = []
    for left_index, right_index in itertools.combinations(range(len(matched_details)), 2):
        left = matched_details[left_index]
        right = matched_details[right_index]
        centers = (
            left.get("query_center"),
            right.get("query_center"),
            left.get("candidate_center"),
            right.get("candidate_center"),
        )
        if any(center is None for center in centers):
            continue
        query_distance = euclidean_distance(centers[0], centers[1])
        candidate_distance = euclidean_distance(centers[2], centers[3])
        pair_deltas[(left_index, right_index)] = abs(query_distance - candidate_distance)
        query_pair_distances.append(query_distance)

    local_errors = []
    local_qualities = []
    local_available = []
    hybrid_qualities = []
    for index, embedding_quality in enumerate(embedding_qualities):
        deltas = [delta for pair, delta in pair_deltas.items() if index in pair]
        available = bool(deltas)
        local_error = math.sqrt(sum(delta * delta for delta in deltas) / len(deltas)) if deltas else 0.0
        local_quality = math.exp(-local_error / spatial_tau) if available else 1.0
        hybrid_quality = embedding_weight * embedding_quality + spatial_weight * local_quality
        local_errors.append(local_error)
        local_qualities.append(local_quality)
        local_available.append(available)
        hybrid_qualities.append(hybrid_quality)

    n_query = int(query_features.size(0))
    n_candidate = int(candidate_features.size(0))
    matched_count = len(matched_assignments)
    full_query_coverage = matched_count == n_query
    tier1_score = sum(hybrid_qualities) / n_query if n_query else 0.0

    geometry_rmse = (
        math.sqrt(sum(delta * delta for delta in pair_deltas.values()) / len(pair_deltas))
        if pair_deltas
        else 0.0
    )
    sorted_query_distances = sorted(query_pair_distances)
    middle = len(sorted_query_distances) // 2
    if not sorted_query_distances:
        query_distance_scale = 0.0
    elif len(sorted_query_distances) % 2:
        query_distance_scale = sorted_query_distances[middle]
    else:
        query_distance_scale = (sorted_query_distances[middle - 1] + sorted_query_distances[middle]) / 2.0
    normalized_geometry_error = geometry_rmse / max(query_distance_scale, 1e-8) if pair_deltas else 0.0
    geometry_factor = math.exp(-geometry_penalty_weight * normalized_geometry_error) if pair_deltas else 1.0
    coverage_status = "complete" if full_query_coverage else "incomplete"
    rejected = require_full_query_coverage and not full_query_coverage
    final_score = 0.0 if rejected else tier1_score * geometry_factor

    for detail, embedding_quality, local_error, local_quality, available, hybrid_quality in zip(
        matched_details,
        embedding_qualities,
        local_errors,
        local_qualities,
        local_available,
        hybrid_qualities,
    ):
        detail["embedding_quality"] = embedding_quality
        detail["local_spatial_error"] = local_error
        detail["local_spatial_quality"] = local_quality
        detail["local_spatial_available"] = available
        detail["hybrid_quality"] = hybrid_quality

    return {
        "v5_tier1_score": float(tier1_score),
        "v5_embedding_quality_sum": float(sum(embedding_qualities)),
        "v5_local_spatial_quality_sum": float(sum(local_qualities)),
        "v5_hybrid_quality_sum": float(sum(hybrid_qualities)),
        "v5_embedding_weight": float(embedding_weight),
        "v5_spatial_weight": float(spatial_weight),
        "v5_spatial_tau": float(spatial_tau),
        "v5_geometry_rmse": float(geometry_rmse),
        "v5_query_distance_scale": float(query_distance_scale),
        "v5_normalized_geometry_error": float(normalized_geometry_error),
        "v5_geometry_pair_count": int(len(pair_deltas)),
        "v5_geometry_available": bool(pair_deltas),
        "v5_geometry_penalty_weight": float(geometry_penalty_weight),
        "v5_geometry_penalty_factor": float(geometry_factor),
        "v5_query_feature_count": n_query,
        "v5_candidate_feature_count": n_candidate,
        "v5_unmatched_query_count": int(n_query - matched_count),
        "v5_unmatched_candidate_count": int(n_candidate - matched_count),
        "v5_full_query_coverage": bool(full_query_coverage),
        "v5_coverage_status": coverage_status,
        "v5_require_full_query_coverage": bool(require_full_query_coverage),
        "v5_rejected_incomplete_coverage": bool(rejected),
        "v5_final_score": float(final_score),
    }


def select_score(components: dict, score_mode: str) -> float:
    if score_mode == COSINE_SCORE_MODE:
        return float(components["matched_cosine_similarity_score"])
    if score_mode == COSINE_GEOMETRY_SCORE_MODE:
        return float(components["cosine_geometry_score"])
    if score_mode == TIERED_DISTANCE_GEOMETRY_SCORE_MODE:
        return float(components["tiered_final_score"])
    if score_mode == HYBRID_LOCAL_GEOMETRY_SCORE_MODE:
        return float(components["v5_final_score"])
    if score_mode == EMBEDDING_DISTANCE_SCORE_MODE:
        return float(components["embedding_distance_score"])
    if score_mode == EMBEDDING_GEOMETRY_DISTANCE_SCORE_MODE:
        return float(components["embedding_geometry_distance_score"])
    if score_mode == FEATURE_DISTANCE_SCORE_MODE:
        return float(components["feature_distance_score"])
    if score_mode == GEOMETRY_DISTANCE_SCORE_MODE:
        return float(components["geometry_distance_score"])
    if score_mode == STRICT_SCORE_MODE:
        return float(components["strict_score"])
    if score_mode == BALANCED_SCORE_MODE:
        return float(components["balanced_score"])
    raise ValueError(f"Unknown Hungarian score mode: {score_mode}")


def euclidean_distance(left, right) -> float:
    left_tensor = torch.as_tensor(left, dtype=torch.float32)
    right_tensor = torch.as_tensor(right, dtype=torch.float32)
    return float(torch.linalg.vector_norm(left_tensor - right_tensor).item())


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
    # Prefer an explicit unmatched assignment when a real pair has exactly the
    # same quality. This avoids reporting zero-similarity pairs as matches.
    dummy_cost = similarity.new_full((n_query, n_query), 1.0 - float(unmatched_similarity) - 1e-6)
    return torch.cat([cost, dummy_cost], dim=1)


def assignment_pairs(row_ind, col_ind, n_candidate: int) -> list[tuple[int, int | None]]:
    assignments = []
    for row, col in zip(row_ind, col_ind):
        row_idx = int(row)
        col_idx = int(col)
        assignments.append((row_idx, col_idx if col_idx < n_candidate else None))
    return assignments


def selected_similarity_sum(similarity: torch.Tensor, assignments: list[tuple[int, int | None]]) -> float:
    total = 0.0
    for row, col in assignments:
        if col is not None:
            total += float(similarity[int(row), int(col)].item())
    return total


def build_match_details(
    similarity: torch.Tensor,
    assignments: list[tuple[int, int | None]],
    query_metadata: list[dict] | None,
    candidate_metadata: list[dict] | None,
) -> list[dict]:
    details = []
    query_metadata = query_metadata or []
    candidate_metadata = candidate_metadata or []
    for query_index, candidate_index in sorted(assignments, key=lambda pair: pair[0]):
        query_feature = query_metadata[query_index] if query_index < len(query_metadata) else {}
        query_center = query_feature.get("center")
        if candidate_index is None:
            details.append(
                {
                    "query_index": query_index,
                    "candidate_index": None,
                    "query_family": query_feature.get("family"),
                    "query_type": query_feature.get("type"),
                    "query_atom_ids": query_feature.get("atom_ids", ()),
                    "query_center": query_center,
                    "candidate_family": None,
                    "candidate_type": None,
                    "candidate_atom_ids": (),
                    "candidate_center": None,
                    "feature_distance": None,
                    "similarity": 0.0,
                    "status": "unmatched",
                }
            )
            continue

        candidate_feature = candidate_metadata[candidate_index] if candidate_index < len(candidate_metadata) else {}
        candidate_center = candidate_feature.get("center")
        feature_distance = (
            euclidean_distance(query_center, candidate_center)
            if query_center is not None and candidate_center is not None
            else None
        )
        details.append(
            {
                "query_index": query_index,
                "candidate_index": candidate_index,
                "query_family": query_feature.get("family"),
                "query_type": query_feature.get("type"),
                "query_atom_ids": query_feature.get("atom_ids", ()),
                "query_center": query_center,
                "candidate_family": candidate_feature.get("family"),
                "candidate_type": candidate_feature.get("type"),
                "candidate_atom_ids": candidate_feature.get("atom_ids", ()),
                "candidate_center": candidate_center,
                "feature_distance": feature_distance,
                "similarity": float(similarity[query_index, candidate_index].detach().cpu().item()),
                "status": "matched",
            }
        )
    return details


def matching_score(
    query_features: torch.Tensor,
    candidate_features: torch.Tensor,
    *,
    query_metadata: list[dict] | None = None,
    candidate_metadata: list[dict] | None = None,
    method: str,
    enforce_feature_family: bool = True,
    score_mode: str = STRICT_SCORE_MODE,
    distance_sigma: float = 1.0,
    geometry_penalty_weight: float = 1.0,
    embedding_weight: float = 0.4,
    spatial_weight: float = 0.6,
    spatial_tau: float = 2.0,
    require_full_query_coverage: bool = False,
) -> tuple[float, torch.Tensor, list[dict], dict]:
    """Build the similarity matrix and return a score plus selected matches."""
    if method == HUNGARIAN_METHOD:
        similarity = cosine_similarity_matrix(query_features, candidate_features)
        unmatched_similarity = 0.0
    elif method == HUNGARIAN_EUCLIDEAN_METHOD:
        similarity = embedding_distance_similarity_matrix(query_features, candidate_features)
        unmatched_similarity = -NO_MATCH_DISTANCE
    elif method == HUNGARIAN_GAUSSIAN_METHOD:
        if distance_sigma <= 0:
            raise ValueError("distance_sigma must be greater than zero.")
        distances = torch.cdist(query_features, candidate_features, p=2)
        similarity = torch.exp(-0.5 * (distances / distance_sigma) ** 2)
        unmatched_similarity = 0.0
    elif method == HUNGARIAN_COSINE_QUALITY_METHOD:
        similarity = (cosine_similarity_matrix(query_features, candidate_features) + 1.0) / 2.0
        similarity = similarity.clamp(0.0, 1.0)
        # In V5, any family-compatible candidate is a feasible match; even a
        # zero-quality pair must beat the dummy so hard coverage is well-defined.
        unmatched_similarity = -2e-6
    elif method == HUNGARIAN_3D_METHOD:
        if query_metadata is None or candidate_metadata is None:
            raise ValueError("hungarian_3d matching requires query and candidate feature metadata.")
        similarity = feature_center_distance_similarity_matrix(
            query_metadata,
            candidate_metadata,
            device=query_features.device,
        )
        unmatched_similarity = -NO_MATCH_DISTANCE
    else:
        raise ValueError(f"Unknown matching method: {method}")

    compatibility_mask = None
    if enforce_feature_family and query_metadata is not None and candidate_metadata is not None:
        compatibility_mask = feature_family_compatibility_mask(query_metadata, candidate_metadata).to(similarity.device)

    _, assignments, components = hungarian_matching_score(
        similarity,
        compatibility_mask=compatibility_mask,
        unmatched_similarity=unmatched_similarity,
    )
    match_details = build_match_details(similarity, assignments, query_metadata, candidate_metadata)
    components = dict(components)
    components.update(distance_score_components(match_details))
    components.update(cosine_score_components(query_features, candidate_features, assignments))
    components.update(embedding_distance_score_components(query_features, candidate_features, assignments))
    components.update(
        tiered_distance_geometry_score_components(
            query_features,
            candidate_features,
            assignments,
            match_details,
            distance_sigma=distance_sigma,
            geometry_penalty_weight=geometry_penalty_weight,
        )
    )
    components.update(
        hybrid_local_geometry_score_components(
            query_features,
            candidate_features,
            assignments,
            match_details,
            embedding_weight=embedding_weight,
            spatial_weight=spatial_weight,
            spatial_tau=spatial_tau,
            geometry_penalty_weight=geometry_penalty_weight,
            require_full_query_coverage=require_full_query_coverage,
        )
    )
    components["score_mode"] = score_mode
    score = select_score(components, score_mode)
    return score, similarity, match_details, components


def _bruteforce_assignment(
    matrix: torch.Tensor,
    *,
    compatibility_mask: torch.Tensor | None = None,
) -> tuple[float, list[tuple[int, int | None]]]:
    """Small fallback for environments without SciPy."""
    n_query, n_candidate = matrix.shape
    if min(n_query, n_candidate) > 8:
        if compatibility_mask is not None:
            matrix = matrix.masked_fill(~compatibility_mask, float("-inf"))
        values, _ = matrix.max(dim=1)
        values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
        cols = []
        for row in range(n_query):
            col = int(torch.argmax(matrix[row]).item())
            if not torch.isfinite(matrix[row, col]):
                cols.append((row, None))
            else:
                cols.append((row, col))
        return float(values.sum().item()), cols

    best = None
    best_cols = None
    if n_query <= n_candidate:
        col_options = range(n_candidate + n_query)
        for cols in itertools.permutations(col_options, n_query):
            total = _assignment_similarity_sum(matrix, cols, compatibility_mask=compatibility_mask)
            if best is None or total > best:
                best = total
                best_cols = cols
    else:
        for cols in itertools.permutations(range(n_candidate + n_query), n_query):
            total = _assignment_similarity_sum(matrix, cols, compatibility_mask=compatibility_mask)
            if best is None or total > best:
                best = total
                best_cols = cols
    assignments = [
        (row, col if col < n_candidate else None)
        for row, col in enumerate(best_cols or [])
    ]
    return float(best or 0.0), assignments


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
