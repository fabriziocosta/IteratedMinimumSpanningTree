"""Scikit-learn estimator for IMSTE embeddings."""

from __future__ import annotations

import numbers
import time
import warnings
from collections.abc import Callable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from ._optimizer import optimize_embedding
from .mst import build_mst_edges
from .mst.prim import iterated_prim_distance_edges as _iterated_mst_edges


def _positive_integer(name: str, value: object, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")
    return result


def _finite_nonnegative(name: str, value: object, *, strictly_positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number.") from exc
    if not np.isfinite(result) or result < 0 or (strictly_positive and result == 0):
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ValueError(f"{name} must be a finite {qualifier} number.")
    return result


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
            raise ValueError("Cannot sample a negative pair for a fully connected node.")
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
            raise ValueError("Cannot sample a negative pair for a fully connected node.")
        targets[source_indices] = rng.choice(available, size=source_indices.size)
    return targets


class IteratedMinimumSpanningTreeEmbedder(TransformerMixin, BaseEstimator):
    """Learn training-row coordinates from edge-disjoint minimum spanning trees.

    This is the transductive estimator. ``transform`` returns the learned
    coordinates only for the exact training matrix in its original row order.
    Use :class:`imste.inductive.InductiveIMSTE` to project new rows.

    Parameters
    ----------
    n_msts : int, default=15
        Number of edge-disjoint minimum spanning trees to construct.
    n_components : int, default=2
        Number of embedding coordinates per sample.
    n_epochs : int, default=200
        Number of optimization epochs.
    batch_size : int, default=4096
        Deprecated compatibility parameter. Sequential SGD no longer batches edges.
    learning_rate : float, default=0.05
        Base SGD step size; updates are sample-scaled and decay linearly by epoch.
    negative_ratio : int, default=5
        Number of sampled non-neighbors per endpoint of each positive edge.
    lambda_rep : float, default=0.5
        Repulsion share in the objective, from zero to one.
    logistic_margin : float, default=1.0
        Squared-distance margin used by logistic repulsion.
    logistic_temperature : float, default=0.5
        Positive temperature controlling logistic repulsion softness.
    random_state : int or None, default=42
        Seed controlling initialization, edge ordering, and negative sampling.
    epsilon : float, default=1e-4
        Smoothing added to squared distances in the logarithmic attraction.
    device : {'auto', 'cpu'}, default='auto'
        Numba optimizer device. Both values select CPU; ``'mps'`` is unsupported.
    mst_method : {'prim', 'famst'}, default='famst'
        ``'prim'`` constructs exact edge-disjoint trees using dense pairwise
        distances. ``'famst'`` uses a sparse approximate-neighbor graph and
        FAMST-style component connection and refinement; uses PyNNDescent.
    mst_neighbors : int, default=15
        Minimum initial approximate-neighbor count for ``mst_method='famst'``.
        FAMST targets the larger of this value and ``2 * n_msts``, subject to
        ``mst_max_neighbors``.
    mst_inter_component_edges : int, default=5
        FAMST candidate edges retained between each component pair selected by
        the representative MST.
    mst_representatives_per_component : int, default=10
        Maximum points sampled from each ANN component to select bridge-search
        component pairs with a representative MST.
    mst_max_neighbors : int or None, default=None
        Maximum approximate-neighbor count. ``None`` allows the FAMST builder to
        increase the count up to four times its starting neighbor count to obtain
        the requested number of edge-disjoint trees.
    progress_callback : callable or None, default=None
        Optional function called with ``(completed_epochs, total_epochs)`` after
        each optimization epoch.
    mst_progress_callback : callable or None, default=None
        Optional function called with ``(completed_trees, total_trees, detail)``
        during MST construction. FAMST's detail includes its current neighbor
        count.
    """

    def __init__(
        self,
        n_msts: int = 15,
        n_components: int = 2,
        n_epochs: int = 200,
        batch_size: int = 4096,
        learning_rate: float = 0.05,
        negative_ratio: int = 5,
        lambda_rep: float = 0.5,
        random_state: int | None = 42,
        epsilon: float = 1e-4,
        device: str = "auto",
        logistic_margin: float = 1.0,
        logistic_temperature: float = 0.5,
        mst_method: str = "famst",
        mst_neighbors: int = 15,
        mst_inter_component_edges: int = 5,
        mst_max_neighbors: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        mst_progress_callback: Callable[[int, int, str], None] | None = None,
        mst_representatives_per_component: int = 10,
    ) -> None:
        self.n_msts = n_msts
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.negative_ratio = negative_ratio
        self.lambda_rep = lambda_rep
        self.random_state = random_state
        self.epsilon = epsilon
        self.device = device
        self.logistic_margin = logistic_margin
        self.logistic_temperature = logistic_temperature
        self.mst_method = mst_method
        self.mst_neighbors = mst_neighbors
        self.mst_inter_component_edges = mst_inter_component_edges
        self.mst_max_neighbors = mst_max_neighbors
        self.progress_callback = progress_callback
        self.mst_progress_callback = mst_progress_callback
        self.mst_representatives_per_component = mst_representatives_per_component

    def fit(self, X: object, y: object = None) -> "IteratedMinimumSpanningTreeEmbedder":
        """Fit the embedding and store coordinates for the input rows.

        Successful fits expose stage timings in seconds: `graph_construction_time_`,
        `embedding_optimization_time_`, `optimizer_negative_sampling_time_`,
        `optimizer_coordinate_update_time_`, and total `fit_time_`.
        """
        fit_started = time.perf_counter()
        del y  # Unsupervised: labels are intentionally never used.
        n_msts = _positive_integer("n_msts", self.n_msts)
        n_components = _positive_integer("n_components", self.n_components)
        n_epochs = _positive_integer("n_epochs", self.n_epochs, allow_zero=True)
        _positive_integer("batch_size", self.batch_size)
        warnings.warn(
            "batch_size is deprecated and ignored by the sequential Numba optimizer.",
            DeprecationWarning,
            stacklevel=2,
        )
        negative_ratio = _positive_integer(
            "negative_ratio", self.negative_ratio, allow_zero=True
        )
        mst_neighbors = _positive_integer("mst_neighbors", self.mst_neighbors)
        inter_component_edges = _positive_integer(
            "mst_inter_component_edges", self.mst_inter_component_edges
        )
        representatives_per_component = _positive_integer(
            "mst_representatives_per_component",
            self.mst_representatives_per_component,
        )
        max_neighbors = (
            None
            if self.mst_max_neighbors is None
            else _positive_integer("mst_max_neighbors", self.mst_max_neighbors)
        )
        if self.mst_method not in {"prim", "famst"}:
            raise ValueError("mst_method must be either 'prim' or 'famst'.")
        learning_rate = _finite_nonnegative(
            "learning_rate", self.learning_rate, strictly_positive=True
        )
        lambda_rep = _finite_nonnegative("lambda_rep", self.lambda_rep)
        if lambda_rep > 1.0:
            raise ValueError("lambda_rep must be between 0 and 1 inclusive.")
        epsilon = _finite_nonnegative("epsilon", self.epsilon, strictly_positive=True)
        logistic_margin = _finite_nonnegative("logistic_margin", self.logistic_margin)
        logistic_temperature = _finite_nonnegative(
            "logistic_temperature", self.logistic_temperature, strictly_positive=True
        )
        X_checked = check_array(X, dtype=np.float64, ensure_2d=True)
        n_samples = X_checked.shape[0]
        if n_samples < 2:
            raise ValueError("X must contain at least two samples to construct an MST.")
        if self.device not in {"auto", "cpu"}:
            if self.device == "mps":
                raise ValueError("device='mps' is unsupported by the CPU Numba optimizer.")
            raise ValueError("device must be one of 'auto' or 'cpu'.")

        graph_started = time.perf_counter()
        edges, tree_ranks = build_mst_edges(
            X_checked,
            n_msts=n_msts,
            random_state=self.random_state,
            method=self.mst_method,
            neighbors=mst_neighbors,
            inter_component_edges=inter_component_edges,
            max_neighbors=max_neighbors,
            representatives_per_component=representatives_per_component,
            progress_callback=self.mst_progress_callback,
            return_ranks=True,
        )
        tree_ranks = np.asarray(tree_ranks, dtype=np.int32)
        self.graph_construction_time_ = time.perf_counter() - graph_started

        embedding_started = time.perf_counter()
        rng = np.random.RandomState(self.random_state)
        seed = int(rng.randint(0, np.iinfo(np.int32).max))
        coordinates = (rng.standard_normal((n_samples, n_components)) * 1e-3).astype(
            np.float32
        )
        optimizer_profile = {"negative_sampling": 0.0, "coordinate_updates": 0.0}

        def record_optimizer_timing(
            sampling_seconds: float, update_seconds: float
        ) -> None:
            optimizer_profile["negative_sampling"] += sampling_seconds
            optimizer_profile["coordinate_updates"] += update_seconds

        optimize_embedding(
            coordinates,
            edges,
            tree_ranks,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            negative_ratio=negative_ratio,
            lambda_rep=lambda_rep,
            epsilon=epsilon,
            margin=logistic_margin,
            temperature=logistic_temperature,
            random_state=seed,
            schedule_by_rank=self.mst_method != "prim",
            progress_callback=self.progress_callback,
            timing_callback=record_optimizer_timing,
        )
        self.embedding_optimization_time_ = time.perf_counter() - embedding_started
        self.optimizer_negative_sampling_time_ = optimizer_profile["negative_sampling"]
        self.optimizer_coordinate_update_time_ = optimizer_profile["coordinate_updates"]
        self.n_features_in_ = X_checked.shape[1]
        self.device_ = "cpu"
        if hasattr(X, "columns") and all(isinstance(c, str) for c in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        self.X_fit_ = X_checked.copy()
        self.embedding_ = coordinates.copy()
        self.graph_edges_ = edges.copy()
        self.graph_ranks_ = tree_ranks.copy()
        self.graph_weights_ = 1.0 / tree_ranks.astype(np.float64)
        self.fit_time_ = time.perf_counter() - fit_started
        return self

    def transform(self, X: object) -> np.ndarray:
        """Return fitted coordinates for the exact training rows only."""
        check_is_fitted(self, attributes=["embedding_", "X_fit_"])
        self._check_feature_names(X)
        X_checked = check_array(X, dtype=np.float64, ensure_2d=True)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features, but this estimator was fitted "
                f"with {self.n_features_in_} features."
            )
        if X_checked.shape == self.X_fit_.shape and np.array_equal(
            X_checked, self.X_fit_
        ):
            return self.embedding_.copy()
        raise ValueError(
            "transform is available only for the original training rows in the "
            "same order. Use imste.inductive.InductiveIMSTE to project new rows."
        )

    def _check_feature_names(self, X: object) -> None:
        fitted_names = getattr(self, "feature_names_in_", None)
        if fitted_names is None:
            return
        input_columns = getattr(X, "columns", None)
        if input_columns is None:
            raise ValueError(
                "X must be a DataFrame with the same named columns used during fit."
            )
        input_names = np.asarray(input_columns, dtype=object)
        if not np.array_equal(input_names, fitted_names):
            raise ValueError(
                "X must have the same feature names in the same order as during fit."
            )
