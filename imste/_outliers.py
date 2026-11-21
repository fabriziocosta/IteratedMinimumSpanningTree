"""Outlier detection using an IMSTE embedding and local density ratios."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np
if TYPE_CHECKING:
    from ._estimator import IteratedMinimumSpanningTreeEmbedder


@dataclass(frozen=True)
class OutlierDetectionResult:
    """Embedding and local outlier diagnostics for the fitted samples.

    ``lof_scores`` follows the standard LOF convention: larger values indicate
    samples whose local density is lower than that of their neighbors.
    ``outlier_indices`` contains samples with no reciprocal links among their
    own ``k`` nearest neighbors, ordered from highest to lowest LOF score.
    All indices refer to rows in the input matrix.

    When several dimensions are evaluated, ``ranked_indices`` sorts by mean
    LOF rank across dimensions and ``outlier_indices`` contains samples with
    zero reciprocal neighbors in every evaluated dimension. The score and
    embedding fields, including ``estimator``, describe the first dimension in
    ``components``. The estimator on which ``outliers`` was called remains
    unchanged.
    """

    estimator: "IteratedMinimumSpanningTreeEmbedder"
    embedding: np.ndarray
    lof_scores: np.ndarray
    mutual_neighbor_counts: np.ndarray
    ranked_indices: np.ndarray
    outlier_indices: np.ndarray
    components: tuple[int, ...] = ()
    average_ranks: np.ndarray | None = None
    outlier_frequency: np.ndarray | None = None
    component_rankings: dict[int, np.ndarray] = field(default_factory=dict)
    outliers_by_component: dict[int, np.ndarray] = field(default_factory=dict)


def _score_embedding(
    embedding: object, k: int, *, random_state: int | None = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return LOF, reciprocal counts, all-row ranking, and zero-reciprocal rows."""
    coordinates = np.asarray(embedding, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[0] < 2:
        raise ValueError("embedding must be a 2D array with at least two rows.")
    if not np.isfinite(coordinates).all():
        raise ValueError("embedding must contain only finite values.")
    n_samples = coordinates.shape[0]
    k = _validate_k(k, n_samples=n_samples)

    distances, indices = _fast_embedding_neighbors(
        coordinates, int(k), random_state=random_state
    )

    k_distances = distances[:, -1]
    reachability = np.maximum(distances, k_distances[indices])
    mean_reachability = reachability.mean(axis=1)
    safe_mean = np.maximum(mean_reachability, np.finfo(np.float64).eps)
    local_reachability_density = 1.0 / safe_mean
    lof_scores = (
        local_reachability_density[indices].mean(axis=1)
        / local_reachability_density
    )

    neighbor_sets = [set(row.tolist()) for row in indices]
    mutual_counts = np.fromiter(
        (
            sum(sample_index in neighbor_sets[neighbor] for neighbor in row)
            for sample_index, row in enumerate(indices)
        ),
        dtype=np.int64,
        count=n_samples,
    )
    ranked_indices = np.argsort(-lof_scores, kind="stable")
    outlier_indices = np.asarray(
        [index for index in ranked_indices if mutual_counts[index] == 0],
        dtype=np.int64,
    )
    return lof_scores, mutual_counts, ranked_indices, outlier_indices


def _validate_k(k: int, *, n_samples: int | None = None) -> int:
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be a positive integer.")
    if n_samples is not None and k >= n_samples:
        raise ValueError(f"k must be smaller than the number of samples ({n_samples}).")
    return int(k)


def _normalize_components(
    n_components: int | Iterable[int] | None, *, current: int
) -> tuple[int, ...]:
    if n_components is None:
        values = (current,)
    elif isinstance(n_components, Integral) and not isinstance(
        n_components, (bool, np.bool_)
    ):
        values = (int(n_components),)
    else:
        if isinstance(n_components, (str, bytes)):
            raise ValueError("n_components must be a positive integer or a non-empty sequence.")
        try:
            values = tuple(n_components)
        except TypeError as exc:
            raise ValueError(
                "n_components must be a positive integer or a non-empty sequence."
            ) from exc
    if not values:
        raise ValueError("n_components must contain at least one dimension.")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < 1
        for value in values
    ):
        raise ValueError("Every n_components value must be a positive integer.")
    dimensions = tuple(int(value) for value in values)
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("n_components values must be unique.")
    return dimensions


def _fast_embedding_neighbors(
    coordinates: np.ndarray, k: int, *, random_state: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Build a fast Euclidean neighbor graph, preferring IMSTE's RINND backend."""
    from .mst.famst import _rinnd_class

    n_samples = len(coordinates)
    features = np.ascontiguousarray(coordinates, dtype=np.float32)
    RINND = _rinnd_class()
    if RINND is not None:
        n_neighbors = min(k, n_samples - 1)
        search = RINND(
            features,
            metric="euclidean",
            n_neighbors=n_neighbors,
            random_state=random_state,
            n_trees=max(4, min(8, 5 + int(round(n_samples**0.5 / 20.0)))),
            n_iters=max(5, int(round(np.log2(n_samples)))),
            max_candidates=max(20, min(60, int(1.5 * n_neighbors))),
            n_jobs=-1,
            verbose=False,
        )
        raw_indices, raw_distances = search.neighbor_graph
    else:
        try:
            from pynndescent import NNDescent
        except ImportError as exc:
            raise ImportError(
                "Outlier detection needs RINND or PyNNDescent for fast embedding "
                "neighbor search. Install the project dependencies with "
                "`python -m pip install -e .`."
            ) from exc
        n_neighbors = min(k + 1, n_samples)
        search = NNDescent(
            features,
            n_neighbors=n_neighbors,
            metric="euclidean",
            random_state=random_state,
            n_jobs=-1,
        )
        raw_indices, raw_distances = search.neighbor_graph

    raw_indices = np.asarray(raw_indices, dtype=np.int64)
    raw_distances = np.asarray(raw_distances, dtype=np.float64)
    distances = np.empty((n_samples, k), dtype=np.float64)
    indices = np.empty((n_samples, k), dtype=np.int64)
    for row in range(n_samples):
        valid = (
            (raw_indices[row] >= 0)
            & (raw_indices[row] < n_samples)
            & (raw_indices[row] != row)
            & np.isfinite(raw_distances[row])
        )
        row_indices = raw_indices[row, valid]
        row_distances = raw_distances[row, valid]
        order = np.argsort(row_distances, kind="stable")
        row_indices = row_indices[order]
        row_distances = row_distances[order]
        _, first_positions = np.unique(row_indices, return_index=True)
        first_positions.sort()
        row_indices = row_indices[first_positions]
        row_distances = row_distances[first_positions]
        if len(row_indices) < k:
            raise RuntimeError(
                "Fast neighbor search returned fewer than k non-self neighbors "
                f"for sample {row}."
            )
        indices[row] = row_indices[:k]
        distances[row] = row_distances[:k]
    return distances, indices


def detect_outliers(
    X: object,
    *,
    k: int = 20,
    estimator_params: Mapping[str, Any] | None = None,
    y: object | None = None,
) -> OutlierDetectionResult:
    """Fit IMSTE and identify local outlier candidates in its embedding.

    The embedding defaults to ten dimensions, which preserves more local
    structure for anomaly scoring than IMSTE's usual two-dimensional layout.
    LOF scores are computed from Euclidean distances in that embedding. Rows
    with no reciprocal links among their ``k`` nearest neighbors are returned
    as the binary outlier set, ordered by descending LOF score.

    Parameters
    ----------
    X : array-like or sparse matrix of shape (n_samples, n_features)
        Input samples to embed.
    k : int, default=20
        Number of neighbors used for LOF and reciprocal-neighbor counts.
        Must be between 1 and ``n_samples - 1``.
    estimator_params : mapping, default=None
        Keyword parameters passed to ``IteratedMinimumSpanningTreeEmbedder``.
        ``n_components`` defaults to 10 and can be overridden here.
    y : array-like, default=None
        Optional labels passed to the estimator. Required when
        ``label_strength`` is positive.

    Returns
    -------
    OutlierDetectionResult
        The fitted estimator, its embedding, LOF score per row, reciprocal
        neighbor count per row, all row indices ranked by LOF, and indices
        whose reciprocal-neighbor count is zero.
    """
    _validate_k(k)
    if estimator_params is None:
        params: dict[str, Any] = {}
    elif isinstance(estimator_params, Mapping):
        params = dict(estimator_params)
    else:
        raise ValueError("estimator_params must be a mapping or None.")

    params.setdefault("n_components", 10)
    from ._estimator import IteratedMinimumSpanningTreeEmbedder

    estimator = IteratedMinimumSpanningTreeEmbedder(**params)
    _fit_estimator(estimator, X, y=y)
    return _outlier_result(estimator, k=k)


def _fit_estimator(
    estimator: "IteratedMinimumSpanningTreeEmbedder",
    X: object,
    y: object | None = None,
) -> None:
    """Fit the supplied estimator, optionally passing labels."""
    if y is None:
        estimator.fit(X)
    else:
        estimator.fit(X, y=y)


def _outlier_result(
    estimator: "IteratedMinimumSpanningTreeEmbedder",
    *,
    k: int,
    n_components: int | Iterable[int] | None = None,
    y: object | None = None,
) -> OutlierDetectionResult:
    """Score one or more embedding dimensions without mutating the estimator."""
    _validate_k(k)
    if not hasattr(estimator, "embedding_"):
        raise ValueError("Fit the estimator before calling outliers().")
    dimensions = _normalize_components(n_components, current=estimator.n_components)

    from sklearn.base import clone

    component_results: dict[int, OutlierDetectionResult] = {}
    for dimension in dimensions:
        if dimension == estimator.n_components:
            component_estimator = estimator
        else:
            if not hasattr(estimator, "X_fit_"):
                raise ValueError(
                    "Comparing n_components requires a feature-based fitted estimator."
                )
            if estimator.label_strength > 0 and y is None:
                raise ValueError(
                    "Pass y when refitting multiple dimensions with positive "
                    "label_strength."
                )
            component_estimator = clone(estimator).set_params(
                n_components=dimension,
                init_coordinates=None,
                input_preprocessor=None,
            )
            if y is None:
                component_estimator.fit(estimator.X_fit_)
            else:
                component_estimator.fit(estimator.X_fit_, y=y)
        component_results[dimension] = _single_outlier_result(
            component_estimator, k=k
        )

    component_rankings = {
        dimension: result.ranked_indices.copy()
        for dimension, result in component_results.items()
    }
    ranks = np.empty((len(dimensions), len(estimator.embedding_)), dtype=np.float64)
    outlier_frequency = np.zeros(len(estimator.embedding_), dtype=np.int64)
    outliers_by_component = {}
    for row, dimension in enumerate(dimensions):
        result = component_results[dimension]
        ranks[row, result.ranked_indices] = np.arange(1, len(result.ranked_indices) + 1)
        outliers_by_component[dimension] = result.outlier_indices.copy()
        outlier_frequency[result.outlier_indices] += 1
    average_ranks = ranks.mean(axis=0)
    ranked_indices = np.argsort(average_ranks, kind="stable")
    consensus_outliers = np.asarray(
        [index for index in ranked_indices if outlier_frequency[index] == len(dimensions)],
        dtype=np.int64,
    )
    primary = component_results[dimensions[0]]
    return replace(
        primary,
        ranked_indices=ranked_indices,
        outlier_indices=consensus_outliers,
        components=dimensions,
        average_ranks=average_ranks,
        outlier_frequency=outlier_frequency,
        component_rankings=component_rankings,
        outliers_by_component=outliers_by_component,
    )


def _single_outlier_result(
    estimator: "IteratedMinimumSpanningTreeEmbedder", *, k: int
) -> OutlierDetectionResult:
    embedding = estimator.embedding_
    lof_scores, mutual_counts, ranked_indices, outlier_indices = _score_embedding(
        embedding, k, random_state=estimator.random_state
    )
    return OutlierDetectionResult(
        estimator=estimator,
        embedding=np.asarray(embedding).copy(),
        lof_scores=lof_scores,
        mutual_neighbor_counts=mutual_counts,
        ranked_indices=ranked_indices,
        outlier_indices=outlier_indices,
        components=(estimator.n_components,),
        average_ranks=_rank_vector(ranked_indices),
        outlier_frequency=(mutual_counts == 0).astype(np.int64),
        component_rankings={estimator.n_components: ranked_indices.copy()},
        outliers_by_component={estimator.n_components: outlier_indices.copy()},
    )


def _rank_vector(ranked_indices: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(ranked_indices), dtype=np.float64)
    ranks[ranked_indices] = np.arange(1, len(ranked_indices) + 1)
    return ranks
