"""Scikit-learn estimator for IMSTE embeddings."""

from __future__ import annotations

import numbers
import time
from collections.abc import Callable

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array, check_is_fitted

from ._optimizer import optimize_embedding
from ._multilevel import (
    _canonical_edges,
    build_multilevel_hierarchy,
    expand_coordinates,
)
from .mst import build_mst_edges, iterated_graph_edges
from .mst.prim import iterated_prim_distance_edges as _iterated_mst_edges


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


class IteratedMinimumSpanningTreeEmbedder(TransformerMixin, BaseEstimator):
    """Learn training-row coordinates from ranked iterated spanning trees.

    Trees use unused edges where possible. When a residual graph has a local
    bottleneck, only the edges needed to reconnect its components are reused;
    each reuse increments that edge's rank and lowers its attraction weight.

    This is the transductive estimator. ``transform`` returns the learned
    coordinates only for the exact training matrix in its original row order.
    Use :class:`imste.inductive.InductiveIMSTE` to project new rows.
    Dense arrays and SciPy CSR matrices are supported.

    Parameters
    ----------
    n_msts : int, default=15
        Number of spanning-tree iterations to construct. Bottleneck edges may
        be reused, with their rank incremented each time.
    n_components : int, default=2
        Number of embedding coordinates per sample.
    pca_init : bool, default=True
        Initialize ``fit(X)`` coordinates from PCA on standardized features.
        ``fit_graph`` also uses PCA when ``init_features`` is supplied. Extra
        coordinates beyond the available PCA dimensions are zero. The existing
        random initialization is used when false or when graph features are absent.
    pca_init_scaling : {'global', 'none'}, default='global'
        Apply one uniform scale so all PCA coordinates fit in ``[-1, 1]``, or
        preserve the raw PCA score magnitudes. Does not affect random initialization.
    init_coordinates : array-like of shape (n_samples, n_components), default=None
        Explicit starting coordinates for optimization. These take precedence
        over PCA and random initialization and must follow the input row order.
    multilevel_init : bool, default=False
        Initialize from a rank-only multilevel coarsening of the fitted graph.
        Explicit ``init_coordinates`` take precedence. If coarsening makes no
        progress, the configured PCA or seeded random initialization is used.
    multilevel_reduction_ratio : int, default=4
        Approximate reduction factor for each coarsening level.
    multilevel_base_size : int, default=1024
        Stop coarsening once a level has at most this many nodes.
    multilevel_epochs : int, default=10
        Refinement epochs at each coarsened level.
    n_epochs : int, default=200
        Number of optimization epochs.
    batch_size : int, default=32768
        Maximum active edges per MPS update batch. Ignored by the CPU optimizer.
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
    device : {'auto', 'cpu', 'mps'}, default='auto'
        Optimizer device. ``'auto'`` and ``'cpu'`` select the Numba CPU backend;
        ``'mps'`` selects batched PyTorch updates on an available Apple GPU.
    mst_method : {'prim', 'famst'}, default='famst'
        ``'prim'`` constructs exact trees; CSR inputs avoid an all-pairs
        distance matrix but still take quadratic time.
        ``'famst'`` uses a sparse neighbor graph and FAMST-style component
        connection and refinement. Both modes reuse an edge only when required
        to connect a residual bottleneck, incrementing its rank for each reuse.
        FAMST uses PyNNDescent on CPU and exact chunked nearest-neighbor search
        on MPS when ``device='mps'``.
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
        increase the count up to four times its starting neighbor count before
        reusing residual bottleneck edges.
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
        batch_size: int = 32768,
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
        pca_init: bool = True,
        pca_init_scaling: str = "global",
        init_coordinates: object | None = None,
        multilevel_init: bool = False,
        multilevel_reduction_ratio: int = 4,
        multilevel_base_size: int = 1024,
        multilevel_epochs: int = 10,
    ) -> None:
        self.n_msts = n_msts
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.negative_ratio = negative_ratio
        self.lambda_rep = lambda_rep
        self.random_state = random_state
        self.pca_init = pca_init
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
        self.pca_init_scaling = pca_init_scaling
        self.init_coordinates = init_coordinates
        self.multilevel_init = multilevel_init
        self.multilevel_reduction_ratio = multilevel_reduction_ratio
        self.multilevel_base_size = multilevel_base_size
        self.multilevel_epochs = multilevel_epochs

    def _validated_fit_options(self) -> dict[str, object]:
        n_msts = _positive_integer("n_msts", self.n_msts)
        n_components = _positive_integer("n_components", self.n_components)
        n_epochs = _positive_integer("n_epochs", self.n_epochs, allow_zero=True)
        batch_size = _positive_integer("batch_size", self.batch_size)
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
        multilevel_reduction_ratio = _positive_integer(
            "multilevel_reduction_ratio", self.multilevel_reduction_ratio
        )
        multilevel_base_size = _positive_integer(
            "multilevel_base_size", self.multilevel_base_size
        )
        multilevel_epochs = _positive_integer(
            "multilevel_epochs", self.multilevel_epochs, allow_zero=True
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
        if not isinstance(self.pca_init, (bool, np.bool_)):
            raise ValueError("pca_init must be a boolean.")
        if not isinstance(self.multilevel_init, (bool, np.bool_)):
            raise ValueError("multilevel_init must be a boolean.")
        if not isinstance(self.pca_init_scaling, str) or self.pca_init_scaling not in {
            "global",
            "none",
        }:
            raise ValueError("pca_init_scaling must be either 'global' or 'none'.")
        if self.device not in {"auto", "cpu", "mps"}:
            raise ValueError("device must be one of 'auto', 'cpu', or 'mps'.")
        resolved_device = "cpu"
        if self.device == "mps":
            import torch

            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "device='mps' requires an MPS-capable PyTorch installation on a "
                    "Mac with a supported Apple GPU."
                )
            resolved_device = "mps"
        return {
            "n_msts": n_msts,
            "n_components": n_components,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "negative_ratio": negative_ratio,
            "mst_neighbors": mst_neighbors,
            "inter_component_edges": inter_component_edges,
            "representatives_per_component": representatives_per_component,
            "max_neighbors": max_neighbors,
            "learning_rate": learning_rate,
            "lambda_rep": lambda_rep,
            "epsilon": epsilon,
            "logistic_margin": logistic_margin,
            "logistic_temperature": logistic_temperature,
            "pca_init": bool(self.pca_init),
            "multilevel_init": bool(self.multilevel_init),
            "multilevel_reduction_ratio": multilevel_reduction_ratio,
            "multilevel_base_size": multilevel_base_size,
            "multilevel_epochs": multilevel_epochs,
            "pca_init_scaling": self.pca_init_scaling,
            "resolved_device": resolved_device,
        }

    def _fit_edges(
        self,
        *,
        n_samples: int,
        n_components: int,
        edges: np.ndarray,
        tree_ranks: np.ndarray,
        fit_started: float,
        resolved_device: str,
        schedule_by_rank: bool,
        X_checked: object | None = None,
        X_original: object | None = None,
        initialization_features: object | None = None,
        initial_coordinates: object | None = None,
        **options,
    ) -> "IteratedMinimumSpanningTreeEmbedder":
        """Optimize coordinates from a graph and attach fitted diagnostics.

        ``X_checked`` distinguishes feature-based fits from ``fit_graph``.
        Feature inputs can provide PCA initialization, while
        ``initial_coordinates`` directly supplies the optimizer's starting
        coordinates when present.
        """
        embedding_started = time.perf_counter()
        rng = np.random.RandomState(self.random_state)
        seed = int(rng.randint(0, np.iinfo(np.int32).max))
        pca_features = (
            X_checked if initialization_features is None else initialization_features
        )
        if initial_coordinates is not None:
            coordinates = check_array(
                initial_coordinates, dtype=np.float64, ensure_2d=True
            )
            if coordinates.shape != (n_samples, n_components):
                raise ValueError(
                    "init_coordinates must have shape "
                    f"({n_samples}, {n_components}); got {coordinates.shape}."
                )
            coordinates = coordinates.astype(np.float32, copy=True)
        elif options["multilevel_init"]:
            coordinates = self._multilevel_initialization(
                n_samples,
                n_components,
                edges,
                tree_ranks,
                resolved_device,
                options,
            )
            if coordinates is None:
                coordinates = self._default_initialization(
                    rng, pca_features, n_samples, n_components, options
                )
        elif pca_features is not None and options["pca_init"]:
            coordinates = _standardized_pca_initialization(
                pca_features, n_components, self.random_state
            )
            coordinates = _scale_initialization_coordinates(
                coordinates, options["pca_init_scaling"]
            ).astype(np.float32)
        else:
            coordinates = (
                rng.standard_normal((n_samples, n_components)) * 1e-3
            ).astype(np.float32)
        optimizer_profile = {"negative_sampling": 0.0, "coordinate_updates": 0.0}

        def record_optimizer_timing(
            sampling_seconds: float, update_seconds: float
        ) -> None:
            optimizer_profile["negative_sampling"] += sampling_seconds
            optimizer_profile["coordinate_updates"] += update_seconds

        optimizer = optimize_embedding
        if resolved_device == "mps":
            from ._optimizer_mps import optimize_embedding_mps

            optimizer = optimize_embedding_mps
            optimizer_batch_size = options["batch_size"]
        else:
            optimizer_batch_size = None
        optimizer_kwargs = dict(
            n_epochs=options["n_epochs"],
            learning_rate=options["learning_rate"],
            negative_ratio=options["negative_ratio"],
            lambda_rep=options["lambda_rep"],
            epsilon=options["epsilon"],
            margin=options["logistic_margin"],
            temperature=options["logistic_temperature"],
            random_state=seed,
            schedule_by_rank=schedule_by_rank,
            progress_callback=self.progress_callback,
            timing_callback=record_optimizer_timing,
        )
        if optimizer_batch_size is not None:
            optimizer_kwargs["batch_size"] = optimizer_batch_size
        optimizer(coordinates, edges, tree_ranks, **optimizer_kwargs)

        self.embedding_optimization_time_ = time.perf_counter() - embedding_started
        self.optimizer_negative_sampling_time_ = optimizer_profile["negative_sampling"]
        self.optimizer_coordinate_update_time_ = optimizer_profile["coordinate_updates"]
        self.device_ = resolved_device
        self.embedding_ = coordinates.copy()
        self.graph_edges_ = edges.copy()
        self.graph_ranks_ = tree_ranks.copy()
        self.graph_weights_ = 1.0 / tree_ranks.astype(np.float64)
        if X_checked is not None:
            if hasattr(self, "n_samples_fit_"):
                del self.n_samples_fit_
            self.n_features_in_ = X_checked.shape[1]
            if hasattr(X_original, "columns") and all(
                isinstance(column, str) for column in X_original.columns
            ):
                self.feature_names_in_ = np.asarray(X_original.columns, dtype=object)
            elif hasattr(self, "feature_names_in_"):
                del self.feature_names_in_
            self.X_fit_ = X_checked.copy()
            self.graph_input_ = False
        else:
            for name in ("n_features_in_", "feature_names_in_", "X_fit_"):
                if hasattr(self, name):
                    delattr(self, name)
            self.n_samples_fit_ = n_samples
            self.graph_input_ = True
        self.fit_time_ = time.perf_counter() - fit_started
        return self

    def _default_initialization(
        self, rng, pca_features, n_samples, n_components, options
    ):
        if pca_features is not None and options["pca_init"]:
            coordinates = _standardized_pca_initialization(
                pca_features, n_components, self.random_state
            )
            return _scale_initialization_coordinates(
                coordinates, options["pca_init_scaling"]
            ).astype(np.float32)
        return (rng.standard_normal((n_samples, n_components)) * 1e-3).astype(
            np.float32
        )

    def _multilevel_initialization(
        self, n_samples, n_components, edges, ranks, resolved_device, options
    ):
        levels = build_multilevel_hierarchy(
            n_samples,
            edges,
            ranks,
            reduction_ratio=options["multilevel_reduction_ratio"],
            base_size=options["multilevel_base_size"],
        )
        if not levels:
            return None

        quotient_edges = np.asarray(edges, dtype=np.int64)
        quotient_weights = 1.0 / np.asarray(ranks, dtype=np.float64)
        n_nodes = n_samples
        for labels, _, _ in levels:
            n_nodes = int(labels.max()) + 1
            if quotient_edges.size:
                left, right = labels[quotient_edges[:, 0]], labels[quotient_edges[:, 1]]
                crossing = left != right
                quotient_edges, quotient_weights = _canonical_edges(
                    np.column_stack((left[crossing], right[crossing])),
                    quotient_weights[crossing],
                )
            else:
                quotient_edges = np.empty((0, 2), dtype=np.int64)
                quotient_weights = np.empty(0, dtype=np.float64)

        seed_rng = np.random.RandomState(self.random_state)
        coordinates = (seed_rng.standard_normal((n_nodes, n_components)) * 1e-3).astype(
            np.float32
        )
        optimizer = optimize_embedding
        if resolved_device == "mps":
            from ._optimizer_mps import optimize_embedding_mps

            optimizer = optimize_embedding_mps
        for level_index in range(len(levels), -1, -1):
            level_edges = quotient_edges if level_index == len(levels) else None
            level_weights = quotient_weights if level_index == len(levels) else None
            if level_index != len(levels):
                labels, level_edges, level_weights = levels[level_index]
                coordinates = expand_coordinates(
                    labels,
                    level_edges,
                    level_weights,
                    coordinates,
                    random_state=None
                    if self.random_state is None
                    else self.random_state + level_index,
                )
            if level_edges.size and options["multilevel_epochs"]:
                kwargs = dict(
                    n_epochs=options["multilevel_epochs"],
                    learning_rate=options["learning_rate"],
                    negative_ratio=options["negative_ratio"],
                    lambda_rep=options["lambda_rep"],
                    epsilon=options["epsilon"],
                    margin=options["logistic_margin"],
                    temperature=options["logistic_temperature"],
                    random_state=None
                    if self.random_state is None
                    else self.random_state + level_index,
                    schedule_by_rank=False,
                    edge_weights=level_weights,
                )
                if resolved_device == "mps":
                    kwargs["batch_size"] = options["batch_size"]
                ranks_at_level = np.ones(len(level_edges), dtype=np.int32)
                optimizer(coordinates, level_edges, ranks_at_level, **kwargs)
        return coordinates

    def fit(self, X: object, y: object = None) -> "IteratedMinimumSpanningTreeEmbedder":
        """Fit the embedding and store coordinates for the input rows.

        Dense and CSR feature matrices are accepted. By default, standardized
        PCA scores are uniformly scaled by their global maximum absolute value
        so every coordinate fits in ``[-1, 1]``. Set
        ``pca_init_scaling="none"`` to preserve raw magnitudes, or
        ``pca_init=False`` to use the seeded random start. Supplying
        ``init_coordinates`` starts from those coordinates instead and takes
        precedence over either setting.

        Successful fits expose stage timings in seconds: `graph_construction_time_`,
        `embedding_optimization_time_`, `optimizer_negative_sampling_time_`,
        `optimizer_coordinate_update_time_`, and total `fit_time_`.
        """
        fit_started = time.perf_counter()
        del y  # Unsupervised: labels are intentionally never used.
        options = self._validated_fit_options()
        X_checked = check_array(
            X, dtype=np.float64, ensure_2d=True, accept_sparse="csr"
        )
        if issparse(X_checked):
            X_checked = csr_matrix(X_checked)
        n_samples = X_checked.shape[0]
        if n_samples < 2:
            raise ValueError("X must contain at least two samples to construct an MST.")
        graph_started = time.perf_counter()
        edges, tree_ranks = build_mst_edges(
            X_checked,
            n_msts=options["n_msts"],
            random_state=self.random_state,
            method=self.mst_method,
            neighbors=options["mst_neighbors"],
            inter_component_edges=options["inter_component_edges"],
            max_neighbors=options["max_neighbors"],
            representatives_per_component=options["representatives_per_component"],
            neighbor_device=options["resolved_device"],
            progress_callback=self.mst_progress_callback,
            return_ranks=True,
        )
        tree_ranks = np.asarray(tree_ranks, dtype=np.int32)
        self.graph_construction_time_ = time.perf_counter() - graph_started
        return self._fit_edges(
            n_samples=n_samples,
            n_components=options["n_components"],
            edges=edges,
            tree_ranks=tree_ranks,
            fit_started=fit_started,
            resolved_device=options["resolved_device"],
            schedule_by_rank=self.mst_method != "prim",
            X_checked=X_checked,
            X_original=X,
            initialization_features=X_checked,
            initial_coordinates=self.init_coordinates,
            **{
                name: value
                for name, value in options.items()
                if name not in {"n_components", "n_msts", "resolved_device"}
            },
        )

    def fit_graph(
        self,
        distance_graph: sparse.spmatrix,
        *,
        init_features: object | None = None,
    ) -> "IteratedMinimumSpanningTreeEmbedder":
        """Fit directly from a sparse graph whose stored values are distances.

        Every explicitly stored off-diagonal entry is a candidate edge; an
        implicit sparse zero means that no edge is available. Edges are treated
        as undirected, reciprocal entries use their smaller distance, and
        explicit zero-distance edges are retained. The graph's connected
        components define separate spanning forests. Later iterations reuse
        short edges when necessary and increase their ranks, which reduces
        their attraction weights.

        ``init_features`` optionally supplies the sample-by-feature matrix used
        for standardized PCA initialization. ``init_coordinates`` can instead
        supply a complete initial embedding with one row per graph node and one
        column per output dimension; it takes precedence over PCA. All rows
        must follow the graph's order. Initialization affects coordinates only;
        graph edges still determine the optimization objective. Without either
        form of initialization, the seeded random start is used. The fitted
        coordinates are available as ``embedding_``; feature-based ``transform``
        is unavailable after ``fit_graph``.
        """
        fit_started = time.perf_counter()
        options = self._validated_fit_options()
        if not sparse.issparse(distance_graph):
            raise ValueError("distance_graph must be a SciPy sparse matrix.")
        if (
            distance_graph.ndim != 2
            or distance_graph.shape[0] != distance_graph.shape[1]
        ):
            raise ValueError("distance_graph must be square.")
        if distance_graph.shape[0] < 2:
            raise ValueError("distance_graph must contain at least two samples.")

        initialization_features = None
        if init_features is not None:
            initialization_features = check_array(
                init_features, dtype=np.float64, ensure_2d=True, accept_sparse="csr"
            )
            if issparse(initialization_features):
                initialization_features = csr_matrix(initialization_features)
            if initialization_features.shape[0] != distance_graph.shape[0]:
                raise ValueError(
                    "init_features must have one row per distance_graph node."
                )

        graph_started = time.perf_counter()
        edges, tree_ranks = iterated_graph_edges(
            distance_graph,
            options["n_msts"],
            progress_callback=self.mst_progress_callback,
            return_ranks=True,
        )
        self.graph_construction_time_ = time.perf_counter() - graph_started
        tree_ranks = np.asarray(tree_ranks, dtype=np.int32)
        return self._fit_edges(
            n_samples=distance_graph.shape[0],
            n_components=options["n_components"],
            edges=edges,
            tree_ranks=tree_ranks,
            fit_started=fit_started,
            resolved_device=options["resolved_device"],
            schedule_by_rank=True,
            initialization_features=initialization_features,
            initial_coordinates=self.init_coordinates,
            **{
                name: value
                for name, value in options.items()
                if name not in {"n_components", "n_msts", "resolved_device"}
            },
        )

    def transform(self, X: object) -> np.ndarray:
        """Return fitted coordinates for the exact training rows only."""
        if getattr(self, "graph_input_", False):
            raise ValueError(
                "transform(X) is unavailable after fit_graph(); use embedding_."
            )
        check_is_fitted(self, attributes=["embedding_", "X_fit_"])
        self._check_feature_names(X)
        X_checked = check_array(
            X, dtype=np.float64, ensure_2d=True, accept_sparse="csr"
        )
        if issparse(X_checked):
            X_checked = csr_matrix(X_checked)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features, but this estimator was fitted "
                f"with {self.n_features_in_} features."
            )
        same_input = False
        if X_checked.shape == self.X_fit_.shape:
            if hasattr(X_checked, "tocsr") or hasattr(self.X_fit_, "tocsr"):
                left = csr_matrix(X_checked).copy()
                right = csr_matrix(self.X_fit_).copy()
                left.sum_duplicates()
                right.sum_duplicates()
                left.eliminate_zeros()
                right.eliminate_zeros()
                same_input = (left != right).nnz == 0
            else:
                same_input = np.array_equal(X_checked, self.X_fit_)
        if same_input:
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
