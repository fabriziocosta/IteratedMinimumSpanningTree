"""Validation, initialization, and sampling helpers for the estimator."""

from __future__ import annotations

import numbers

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.decomposition import PCA


def _positive_integer(name: str, value: object, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}."
        )
    result = int(value)
    if result < minimum:
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}."
        )
    return result

def _finite_nonnegative(
    name: str, value: object, *, strictly_positive: bool = False
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number.") from exc
    if not np.isfinite(result) or result < 0 or (strictly_positive and result == 0):
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ValueError(f"{name} must be a finite {qualifier} number.")
    return result


def _label_edge_multipliers(
    labels: object | None,
    edges: np.ndarray,
    strength: float,
    n_samples: int,
) -> np.ndarray:
    """Return stable same/different-class multipliers for graph edges."""
    multipliers = np.ones(len(edges), dtype=np.float64)
    if strength == 0.0:
        return multipliers
    if labels is None:
        raise ValueError("y is required when label_strength is greater than zero.")
    values = np.asarray(labels, dtype=object)
    if values.ndim != 1 or values.shape[0] != n_samples:
        raise ValueError("y must be one-dimensional with one label per sample.")

    def is_missing(value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, (float, np.floating)):
            return bool(np.isnan(value))
        return False

    missing = np.fromiter((is_missing(value) for value in values), bool, len(values))
    if np.all(missing):
        raise ValueError("y must contain at least one known label.")

    log_multipliers = np.zeros(len(edges), dtype=np.float64)
    if not len(edges):
        return multipliers
    for index, (left, right) in enumerate(edges):
        if missing[left] or missing[right]:
            continue
        try:
            same_class = bool(values[left] == values[right])
        except (TypeError, ValueError) as error:
            raise ValueError("y must contain scalar categorical labels.") from error
        log_multipliers[index] = strength / 2.0 if same_class else -strength / 2.0

    # A common scale cancels in the normalized attraction objective. Shifting
    # the largest log multiplier to zero avoids overflow for large strengths.
    log_multipliers -= np.max(log_multipliers)
    lower_bound = np.log(np.finfo(np.float32).tiny)
    return np.exp(np.maximum(log_multipliers, lower_bound))


def _normalized_edge_weights(ranks: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    """Combine inverse ranks with label multipliers and normalize to unit sum."""
    weights = np.asarray(multipliers, dtype=np.float64) / np.asarray(ranks)
    total = float(np.sum(weights))
    if total > 0.0:
        weights /= total
    return weights

def _standardized_pca_initialization(
    X: np.ndarray | csr_matrix,
    n_components: int,
    random_state: int | None,
) -> np.ndarray:
    """Return PCA scores after feature-wise standardization.

    Each feature is centered and divided by its population standard deviation;
    constant features use a scale of one. If more embedding dimensions are
    requested than PCA can provide, the remaining score columns are zero.

    Sparse inputs stay sparse. Their centered, standardized matrix is exposed
    to the SVD solver as a linear operator, avoiding a dense copy of ``X``.
    """
    n_samples, n_features = X.shape
    available_components = min(n_components, n_samples, n_features)
    scores = np.zeros((n_samples, n_components), dtype=np.float64)
    if available_components == 0:
        return scores

    if issparse(X):
        X = csr_matrix(X, dtype=np.float64)
        mean = np.asarray(X.mean(axis=0)).ravel()
        second_moment = np.asarray(X.power(2).mean(axis=0)).ravel()
        scale = np.sqrt(np.maximum(second_moment - mean**2, 0.0))
        scale[scale == 0.0] = 1.0
        scaled = X.multiply(1.0 / scale).tocsr()
        scaled_mean = mean / scale

        from scipy.sparse.linalg import LinearOperator, svds

        # The standardized matrix is scaled - 1 * scaled_mean. These products
        # apply that centering algebraically without storing the dense matrix.
        def matvec(vector):
            return np.asarray(scaled @ vector).ravel() - scaled_mean @ vector

        def rmatvec(vector):
            return np.asarray(scaled.T @ vector).ravel() - scaled_mean * np.sum(vector)

        def matmat(matrix):
            return (
                np.asarray(scaled @ matrix)
                - np.ones((n_samples, 1)) * (scaled_mean @ matrix)[None, :]
            )

        def rmatmat(matrix):
            return (
                np.asarray(scaled.T @ matrix)
                - scaled_mean[:, None] * np.sum(matrix, axis=0)[None, :]
            )

        centered_standardized = LinearOperator(
            shape=(n_samples, n_features),
            matvec=matvec,
            rmatvec=rmatvec,
            matmat=matmat,
            rmatmat=rmatmat,
            dtype=np.float64,
        )
        left, singular_values, _ = svds(
            centered_standardized,
            k=available_components,
            which="LM",
            solver="propack",
            random_state=random_state,
        )
        # PROPACK returns singular values in solver order; PCA orders axes by
        # descending variance. Canonicalize signs so repeated seeded fits also
        # return the same orientation for each score axis.
        order = np.argsort(singular_values)[::-1]
        left = left[:, order]
        singular_values = singular_values[order]
        pivot_rows = np.argmax(np.abs(left), axis=0)
        signs = np.sign(left[pivot_rows, np.arange(available_components)])
        signs[signs == 0.0] = 1.0
        left *= signs
        # PCA coordinates are left singular vectors multiplied by singular
        # values. The right singular vectors are not needed for initialization.
        pca_scores = left * singular_values
    else:
        dense = np.asarray(X, dtype=np.float64)
        mean = dense.mean(axis=0)
        scale = dense.std(axis=0)
        scale[scale == 0.0] = 1.0
        standardized = (dense - mean) / scale
        pca_scores = PCA(
            n_components=available_components,
            svd_solver="full",
        ).fit_transform(standardized)

    # Keep the requested output shape even when PCA has fewer valid axes.
    scores[:, :available_components] = pca_scores
    return scores

def _scale_initialization_coordinates(
    coordinates: np.ndarray, scaling: str
) -> np.ndarray:
    """Apply one shared PCA-coordinate scale so scores fit the unit box."""
    if scaling == "none":
        return coordinates
    if scaling != "global":
        raise ValueError("pca_init_scaling must be either 'global' or 'none'.")

    max_abs = float(np.max(np.abs(coordinates)))
    if max_abs == 0.0:
        return coordinates
    return coordinates / max_abs

def _append_candidate_attraction_edges(
    edges: np.ndarray,
    ranks: np.ndarray,
    candidate_edges: np.ndarray,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Append non-tree FAMST neighbor edges as rank-one attractions."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    ranks = np.asarray(ranks, dtype=np.int32).reshape(-1)
    candidates = np.asarray(candidate_edges, dtype=np.int64).reshape(-1, 2)
    if not candidates.size:
        return edges, ranks
    candidates = np.sort(candidates, axis=1)
    candidate_keys = np.unique(candidates[:, 0] * n_samples + candidates[:, 1])
    tree_edges = np.sort(edges, axis=1)
    tree_keys = np.unique(tree_edges[:, 0] * n_samples + tree_edges[:, 1])
    additional_keys = np.setdiff1d(candidate_keys, tree_keys, assume_unique=True)
    if not additional_keys.size:
        return edges, ranks
    additional_edges = np.column_stack(
        (additional_keys // n_samples, additional_keys % n_samples)
    ).astype(np.int64, copy=False)
    return (
        np.vstack((edges, additional_edges)),
        np.concatenate((ranks, np.ones(len(additional_edges), dtype=np.int32))),
    )

def _sample_negative_targets(
    rng: np.random.RandomState,
    packed_adjacency: np.ndarray,
    negative_sources: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Sample non-neighbors with rejection and an exact dense-graph fallback."""
    targets = rng.randint(n_samples, size=negative_sources.size)
    invalid = np.ones(targets.size, dtype=bool)
    for _ in range(8):
        invalid_indices = np.flatnonzero(invalid)
        if invalid_indices.size == 0:
            break
        candidate_targets = targets[invalid_indices]
        blocked = packed_adjacency[
            negative_sources[invalid_indices], candidate_targets >> 3
        ]
        bit_mask = 1 << (candidate_targets & 7)
        accepted = (blocked & bit_mask) == 0
        invalid[invalid_indices[accepted]] = False
        invalid_indices = invalid_indices[~accepted]
        targets[invalid_indices] = rng.randint(n_samples, size=invalid_indices.size)

    for source in np.unique(negative_sources[invalid]):
        source_indices = np.flatnonzero(invalid & (negative_sources == source))
        blocked = np.unpackbits(
            packed_adjacency[source], bitorder="little", count=n_samples
        )
        candidates = np.flatnonzero(blocked == 0)
        if candidates.size == 0:
            raise ValueError(
                "Cannot sample a negative pair for a fully connected node."
            )
        targets[source_indices] = rng.choice(candidates, size=source_indices.size)
    return targets

def _sample_sparse_negative_targets(
    rng: np.random.RandomState,
    adjacency_keys: np.ndarray,
    negative_sources: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Vectorized rejection-sample non-neighbors without a dense bit matrix."""
    targets = rng.randint(n_samples, size=negative_sources.size).astype(np.intp)
    invalid = np.ones(targets.size, dtype=bool)
    for _ in range(8):
        invalid_indices = np.flatnonzero(invalid)
        if invalid_indices.size == 0:
            break
        sources = negative_sources[invalid_indices]
        candidates = targets[invalid_indices]
        keys = sources.astype(np.int64) * n_samples + candidates
        locations = np.searchsorted(adjacency_keys, keys)
        in_bounds = locations < adjacency_keys.size
        blocked = np.zeros(keys.size, dtype=bool)
        blocked[in_bounds] = adjacency_keys[locations[in_bounds]] == keys[in_bounds]
        accepted = (sources != candidates) & ~blocked
        invalid[invalid_indices[accepted]] = False
        retry_indices = invalid_indices[~accepted]
        targets[retry_indices] = rng.randint(n_samples, size=retry_indices.size)

    # Rare for sparse graphs; handle unusually dense nodes exactly.
    for source in np.unique(negative_sources[invalid]):
        source_indices = np.flatnonzero(invalid & (negative_sources == source))
        start = int(source) * n_samples
        first = np.searchsorted(adjacency_keys, start)
        last = np.searchsorted(adjacency_keys, start + n_samples)
        blocked = adjacency_keys[first:last] - start
        candidates = np.ones(n_samples, dtype=bool)
        candidates[int(source)] = False
        candidates[blocked] = False
        available = np.flatnonzero(candidates)
        if available.size == 0:
            raise ValueError(
                "Cannot sample a negative pair for a fully connected node."
            )
        targets[source_indices] = rng.choice(available, size=source_indices.size)
    return targets
