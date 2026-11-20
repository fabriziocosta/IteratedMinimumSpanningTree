"""Scikit-learn estimator for IMSTE embeddings."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

from ._estimator_graph import _EstimatorGraphMethods
from ._estimator_utils import (
    _append_candidate_attraction_edges,
    _finite_nonnegative,
    _positive_integer,
    _sample_negative_targets,
    _sample_sparse_negative_targets,
    _scale_initialization_coordinates,
    _standardized_pca_initialization,
)
from ._optimizer import optimize_embedding
from ._multilevel import (
    _canonical_edges,
    aggregate_features_by_group,
    build_multilevel_hierarchy,
    expand_coordinates,
)
from .mst import build_mst_edges
from .mst.prim import (
    _cosine_safe_normalize,
    iterated_prim_distance_edges as _iterated_mst_edges,
)


class IteratedMinimumSpanningTreeEmbedder(
    _EstimatorGraphMethods, TransformerMixin, BaseEstimator
):
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
    multilevel_init : bool, default=True
        Initialize from a rank-only multilevel coarsening of the fitted graph.
        Explicit ``init_coordinates`` take precedence. If coarsening makes no
        progress, the configured PCA or seeded random initialization is used.
        When ``pca_init=True`` and features are available, initialize the
        coarsest graph from PCA of its pooled group features; otherwise use a
        seeded random start at the coarsest level.
    multilevel_reduction_ratio : int, default=32
        Approximate reduction factor for each coarsening level.
    multilevel_base_size : int, default=512
        Stop coarsening once a level has at most this many nodes.
    multilevel_epochs : int, default=1
        Refinement epochs at each coarsened level.
    n_epochs : int, default=200
        Number of optimization epochs.
    batch_size : int, default=131072
        Maximum active edges per GPU update batch. Ignored by the CPU optimizer.
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
    device : {'auto', 'cpu', 'mps', 'cuda'}, default='auto'
        ``'auto'`` selects MPS on Apple Silicon, CUDA when available, and CPU
        otherwise. ``'cpu'`` selects the Numba optimizer; ``'mps'`` and
        ``'cuda'`` select their corresponding PyTorch GPU backends.
    mst_method : {'prim', 'famst'}, default='famst'
        ``'prim'`` constructs exact trees; CSR inputs avoid an all-pairs
        distance matrix but still take quadratic time.
        ``'famst'`` uses a sparse neighbor graph and FAMST-style component
        connection and refinement. Both modes reuse an edge only when required
        to connect a residual bottleneck, incrementing its rank for each reuse.
        FAMST uses the backend selected by ``mst_neighbor_backend``.
    mst_neighbor_backend : {'auto', 'rinnd', 'pynndescent', 'mps_exact'}, default='auto'
        Neighbor search backend for dense Euclidean FAMST inputs. ``'auto'``
        prefers RINND when available, otherwise it uses the existing device-based
        choice. RINND search does not change the optimizer selected by ``device``.
    metric : {'euclidean', 'cosine'}, default='euclidean'
        Feature distance used to construct the spanning-tree graph. Cosine
        distance with FAMST uses the exact Prim backend.
    mst_neighbors : int, default=15
        Minimum initial approximate-neighbor count for ``mst_method='famst'``.
        FAMST targets the larger of this value and ``2 * n_msts``, subject to
        ``mst_max_neighbors``.
    include_knn_attraction : bool, default=False
        Include FAMST's approximate candidate-neighbor edges as additional
        rank-one attractive edges alongside the ranked spanning trees. This
        makes local neighborhoods denser and is available for Euclidean FAMST
        fits.
    gravity_strength : float, default=0.2
        Attraction toward density-weighted region cores after IMSTE. Set to zero
        to disable gravity. Gravity requires dense Euclidean features and at
        least two embedding dimensions; it refines the first two. It is skipped
        automatically when ``metric='cosine'``.
    gravity_min_dist : float, default=0.0
        Optional local-distance floor after gravity, expressed as a
        fraction of the median pre-gravity nearest-neighbor distance among
        feature-space neighbors. Zero disables repulsion. Values above 1 request
        stronger expansion.
    gravity_neighbors : int, default=15
        Feature-space neighbors used to build gravity regions.
    gravity_resolution : float, default=1.0
        Leiden resolution used to identify gravity regions.
    gravity_core_quantile : float, default=0.95
        Within-region density quantile defining the core points.
    gravity_mass_power : float, default=0.5
        Exponent controlling how region and point density affect gravity.
    gravity_pack_regions : bool, default=False
        After gravity, translate each Leiden region as a rigid group onto a
        compact ring. This preserves within-region distances but changes
        distances between regions.
    gravity_pack_radius_quantile : float, default=0.9
        Within-region radial quantile used to estimate each region's ring size.
    gravity_pack_gap_fraction : float, default=0.25
        Gap between ring regions as a fraction of their median estimated radius.
    mean_shift_neighbors : int, default=100
        Number of neighbors from the starting 2D layout used for Gaussian mean shift.
    mean_shift_bandwidth_factor : float, default=5.0
        Gaussian bandwidth as a multiple of the median current 2D distance to
        the selected neighbor count.
    mean_shift_step_size : float, default=0.5
        Interpolation from each coordinate toward its kernel-weighted mean, in
        the interval [0, 1].
    mean_shift_iterations : int, default=1
        Mean-shift iterations applied after gravity. Zero disables mean shift.
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
        each optimization epoch, including multilevel refinement epochs.
    mst_progress_callback : callable or None, default=None
        Optional function called with ``(completed, total, detail)`` during MST
        construction and multilevel coarsening/refinement. FAMST's detail
        includes its current neighbor count.
    verbose : bool, default=False
        Print timestamped fit stages and display progress bars for graph
        construction, multilevel initialization, and optimization.
    input_preprocessor : {'text', 'image', 'graph'} or transformer, default=None
        Optional raw-input transformer to fit before IMSTE. Use
        :meth:`preprocessor` to select a modality preset with its options.
    """

    def __init__(
        self,
        n_msts: int = 15,
        n_components: int = 2,
        n_epochs: int = 200,
        batch_size: int = 131072,
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
        verbose: bool = False,
        mst_representatives_per_component: int = 10,
        pca_init: bool = True,
        pca_init_scaling: str = "global",
        init_coordinates: object | None = None,
        multilevel_init: bool = True,
        multilevel_reduction_ratio: int = 32,
        multilevel_base_size: int = 512,
        multilevel_epochs: int = 1,
        metric: str = "euclidean",
        mst_neighbor_backend: str = "auto",
        include_knn_attraction: bool = False,
        gravity_neighbors: int = 15,
        gravity_resolution: float = 1.0,
        gravity_strength: float = 0.2,
        gravity_min_dist: float = 0.0,
        gravity_core_quantile: float = 0.95,
        gravity_mass_power: float = 0.5,
        gravity_pack_regions: bool = False,
        gravity_pack_radius_quantile: float = 0.9,
        gravity_pack_gap_fraction: float = 0.25,
        mean_shift_neighbors: int = 100,
        mean_shift_bandwidth_factor: float = 5.0,
        mean_shift_step_size: float = 0.5,
        mean_shift_iterations: int = 1,
        input_preprocessor: object | str | None = None,
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
        self.verbose = verbose
        self._active_progress_callback = progress_callback
        self._active_mst_progress_callback = mst_progress_callback
        self._progress_reporter = None
        self.mst_representatives_per_component = mst_representatives_per_component
        self.pca_init_scaling = pca_init_scaling
        self.init_coordinates = init_coordinates
        self.multilevel_init = multilevel_init
        self.multilevel_reduction_ratio = multilevel_reduction_ratio
        self.multilevel_base_size = multilevel_base_size
        self.multilevel_epochs = multilevel_epochs
        self.metric = metric
        self.mst_neighbor_backend = mst_neighbor_backend
        self.include_knn_attraction = include_knn_attraction
        self.gravity_neighbors = gravity_neighbors
        self.gravity_resolution = gravity_resolution
        self.gravity_strength = gravity_strength
        self.gravity_min_dist = gravity_min_dist
        self.gravity_core_quantile = gravity_core_quantile
        self.gravity_mass_power = gravity_mass_power
        self.gravity_pack_regions = gravity_pack_regions
        self.gravity_pack_radius_quantile = gravity_pack_radius_quantile
        self.gravity_pack_gap_fraction = gravity_pack_gap_fraction
        self.mean_shift_neighbors = mean_shift_neighbors
        self.mean_shift_bandwidth_factor = mean_shift_bandwidth_factor
        self.mean_shift_step_size = mean_shift_step_size
        self.mean_shift_iterations = mean_shift_iterations
        self.input_preprocessor = input_preprocessor

    def _validated_fit_options(self) -> dict[str, object]:
        if not isinstance(self.verbose, (bool, np.bool_)):
            raise ValueError("verbose must be a boolean.")
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
        if self.mst_neighbor_backend not in {
            "auto", "rinnd", "pynndescent", "mps_exact"
        }:
            raise ValueError(
                "mst_neighbor_backend must be one of 'auto', 'rinnd', "
                "'pynndescent', or 'mps_exact'."
            )
        if self.metric not in {"euclidean", "cosine"}:
            raise ValueError("metric must be either 'euclidean' or 'cosine'.")
        if self.mst_method == "famst" and self.metric != "euclidean":
            if self.mst_neighbor_backend in {"rinnd", "mps_exact"}:
                raise ValueError(
                    f"mst_neighbor_backend={self.mst_neighbor_backend!r} "
                    "requires dense Euclidean FAMST input."
                )
        if self.mst_method == "prim" and self.mst_neighbor_backend != "auto":
            raise ValueError(
                "mst_neighbor_backend applies only when mst_method='famst'."
            )
        if not isinstance(self.include_knn_attraction, (bool, np.bool_)):
            raise ValueError("include_knn_attraction must be a boolean.")
        if self.include_knn_attraction and self.mst_method != "famst":
            raise ValueError("include_knn_attraction requires mst_method='famst'.")
        if self.include_knn_attraction and self.metric != "euclidean":
            raise ValueError("include_knn_attraction requires metric='euclidean'.")
        gravity_strength = _finite_nonnegative(
            "gravity_strength", self.gravity_strength
        )
        gravity_min_dist = _finite_nonnegative(
            "gravity_min_dist", self.gravity_min_dist
        )
        gravity_enabled = (
            gravity_strength > 0.0 or gravity_min_dist > 0.0
        ) and self.metric == "euclidean"
        if not gravity_enabled:
            gravity_strength = 0.0
            gravity_min_dist = 0.0
        if gravity_enabled and n_components < 2:
            raise ValueError("gravity refinement requires n_components >= 2.")
        gravity_neighbors = _positive_integer(
            "gravity_neighbors", self.gravity_neighbors
        )
        gravity_resolution = _finite_nonnegative(
            "gravity_resolution", self.gravity_resolution, strictly_positive=True
        )
        gravity_core_quantile = _finite_nonnegative(
            "gravity_core_quantile", self.gravity_core_quantile
        )
        if gravity_core_quantile > 1.0:
            raise ValueError("gravity_core_quantile must be in [0, 1].")
        gravity_mass_power = _finite_nonnegative(
            "gravity_mass_power", self.gravity_mass_power
        )
        if not isinstance(self.gravity_pack_regions, (bool, np.bool_)):
            raise ValueError("gravity_pack_regions must be a boolean.")
        gravity_pack_radius_quantile = _finite_nonnegative(
            "gravity_pack_radius_quantile", self.gravity_pack_radius_quantile
        )
        if not 0.0 < gravity_pack_radius_quantile < 1.0:
            raise ValueError("gravity_pack_radius_quantile must be between 0 and 1.")
        gravity_pack_gap_fraction = _finite_nonnegative(
            "gravity_pack_gap_fraction", self.gravity_pack_gap_fraction
        )
        mean_shift_neighbors = _positive_integer(
            "mean_shift_neighbors", self.mean_shift_neighbors
        )
        mean_shift_bandwidth_factor = _finite_nonnegative(
            "mean_shift_bandwidth_factor", self.mean_shift_bandwidth_factor,
            strictly_positive=True,
        )
        mean_shift_step_size = _finite_nonnegative(
            "mean_shift_step_size", self.mean_shift_step_size
        )
        if mean_shift_step_size > 1.0:
            raise ValueError("mean_shift_step_size must be in [0, 1].")
        mean_shift_iterations = _positive_integer(
            "mean_shift_iterations", self.mean_shift_iterations, allow_zero=True
        )
        if mean_shift_iterations and n_components < 2:
            raise ValueError("mean shift requires n_components >= 2.")
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
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device must be one of 'auto', 'cpu', 'mps', or 'cuda'.")
        resolved_device = "cpu"
        if self.device != "cpu":
            try:
                import torch
            except ImportError:
                if self.device != "auto":
                    raise RuntimeError(
                        f"device={self.device!r} requires PyTorch with a GPU backend. "
                        "Install it with `python -m pip install 'imste[gpu]'`."
                    )
            else:
                mps_backend = getattr(torch.backends, "mps", None)
                mps_available = (
                    mps_backend is not None and mps_backend.is_available()
                )
                cuda_available = torch.cuda.is_available()
                if self.device == "mps":
                    if not mps_available:
                        raise RuntimeError(
                            "device='mps' requires an MPS-capable PyTorch installation "
                            "on a Mac with a supported Apple GPU."
                        )
                    resolved_device = "mps"
                elif self.device == "cuda":
                    if not cuda_available:
                        raise RuntimeError(
                            "device='cuda' requires an available CUDA-capable GPU."
                        )
                    resolved_device = "cuda"
                elif mps_available:
                    resolved_device = "mps"
                elif cuda_available:
                    resolved_device = "cuda"
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
            "mst_neighbor_backend": self.mst_neighbor_backend,
            "include_knn_attraction": bool(self.include_knn_attraction),
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
            "metric": self.metric,
            "gravity_enabled": gravity_enabled,
            "gravity_neighbors": gravity_neighbors,
            "gravity_resolution": gravity_resolution,
            "gravity_strength": gravity_strength,
            "gravity_min_dist": gravity_min_dist,
            "gravity_core_quantile": gravity_core_quantile,
            "gravity_mass_power": gravity_mass_power,
            "gravity_pack_regions": bool(self.gravity_pack_regions),
            "gravity_pack_radius_quantile": gravity_pack_radius_quantile,
            "gravity_pack_gap_fraction": gravity_pack_gap_fraction,
            "mean_shift_neighbors": mean_shift_neighbors,
            "mean_shift_bandwidth_factor": mean_shift_bandwidth_factor,
            "mean_shift_step_size": mean_shift_step_size,
            "mean_shift_iterations": mean_shift_iterations,
        }

    def _activate_progress_reporting(self) -> None:
        """Combine user callbacks with the built-in verbose progress display."""
        self._active_progress_callback = self.progress_callback
        self._active_mst_progress_callback = self.mst_progress_callback
        self._progress_reporter = None
        if not self.verbose:
            return

        from ._progress import IMSTEProgressReporter

        reporter = IMSTEProgressReporter(self)
        reporter.start()
        self._progress_reporter = reporter

        def report_epochs(completed: int, total: int) -> None:
            if self.progress_callback is not None:
                self.progress_callback(completed, total)
            reporter.epoch_callback(completed, total)

        def report_stages(completed: int, total: int, detail: str) -> None:
            if self.mst_progress_callback is not None:
                self.mst_progress_callback(completed, total, detail)
            reporter.mst_callback(completed, total, detail)

        self._active_progress_callback = report_epochs
        self._active_mst_progress_callback = report_stages

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
        self.multilevel_initialization_time_ = 0.0
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
            multilevel_started = time.perf_counter()
            coordinates = self._multilevel_initialization(
                n_samples,
                n_components,
                edges,
                tree_ranks,
                resolved_device,
                pca_features,
                options,
            )
            self.multilevel_initialization_time_ = (
                time.perf_counter() - multilevel_started
            )
            if self._progress_reporter is not None:
                self._progress_reporter.multilevel_finished()
                self._progress_reporter.phase_finished(
                    "Multilevel initialization",
                    self.multilevel_initialization_time_,
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
        if resolved_device in {"mps", "cuda"}:
            from functools import partial

            from ._optimizer_torch import optimize_embedding_torch

            optimizer = partial(
                optimize_embedding_torch, device_type=resolved_device
            )
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
            progress_callback=self._active_progress_callback,
            timing_callback=record_optimizer_timing,
        )
        if optimizer_batch_size is not None:
            optimizer_kwargs["batch_size"] = optimizer_batch_size
        if self._active_mst_progress_callback is not None:
            self._active_mst_progress_callback(
                0,
                max(options["n_epochs"], 1),
                "Final embedding optimization",
            )
        final_optimization_started = time.perf_counter()
        optimizer(coordinates, edges, tree_ranks, **optimizer_kwargs)
        self.final_embedding_optimization_time_ = (
            time.perf_counter() - final_optimization_started
        )
        if self._progress_reporter is not None:
            self._progress_reporter.phase_finished(
                "Final embedding optimization",
                self.final_embedding_optimization_time_,
            )
        if self._active_mst_progress_callback is not None:
            self._active_mst_progress_callback(
                options["n_epochs"],
                max(options["n_epochs"], 1),
                "Final embedding optimization",
            )

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
        self,
        n_samples,
        n_components,
        edges,
        ranks,
        resolved_device,
        pca_features,
        options,
    ):
        levels = build_multilevel_hierarchy(
            n_samples,
            edges,
            ranks,
            reduction_ratio=options["multilevel_reduction_ratio"],
            base_size=options["multilevel_base_size"],
            progress_callback=self._active_mst_progress_callback,
        )
        if not levels:
            return None
        if self._progress_reporter is not None:
            self._progress_reporter.begin_multilevel_refinement(
                len(levels), options["multilevel_epochs"]
            )

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

        if pca_features is not None and options["pca_init"]:
            if n_nodes < 2:
                coordinates = np.zeros((n_nodes, n_components), dtype=np.float32)
            else:
                coarse_labels = np.arange(n_samples, dtype=np.int64)
                for labels, _, _ in levels:
                    coarse_labels = labels[coarse_labels]
                coarse_features = aggregate_features_by_group(
                    pca_features, coarse_labels
                )
                coordinates = _standardized_pca_initialization(
                    coarse_features, n_components, self.random_state
                )
                coordinates = _scale_initialization_coordinates(
                    coordinates, options["pca_init_scaling"]
                ).astype(np.float32)
        else:
            seed_rng = np.random.RandomState(self.random_state)
            coordinates = (
                seed_rng.standard_normal((n_nodes, n_components)) * 1e-3
            ).astype(np.float32)
        optimizer = optimize_embedding
        if resolved_device in {"mps", "cuda"}:
            from functools import partial

            from ._optimizer_torch import optimize_embedding_torch

            optimizer = partial(
                optimize_embedding_torch, device_type=resolved_device
            )
        for level_index in range(len(levels), -1, -1):
            level_edges = quotient_edges if level_index == len(levels) else None
            level_weights = quotient_weights if level_index == len(levels) else None
            if level_index != len(levels):
                labels, level_edges, level_weights = levels[level_index]
                if self._active_mst_progress_callback is not None:
                    self._active_mst_progress_callback(
                        0,
                        max(n_samples, 1),
                        f"Expanding level {len(levels) - level_index}",
                    )
                coordinates = expand_coordinates(
                    labels,
                    level_edges,
                    level_weights,
                    coordinates,
                    random_state=None
                    if self.random_state is None
                    else self.random_state + level_index,
                )
                if self._active_mst_progress_callback is not None:
                    self._active_mst_progress_callback(
                        n_samples,
                        max(n_samples, 1),
                        f"Expanded level {len(levels) - level_index}",
                    )
            if level_edges.size and options["multilevel_epochs"]:
                if self._active_mst_progress_callback is not None:
                    self._active_mst_progress_callback(
                        0,
                        options["multilevel_epochs"],
                        f"Refining level {level_index + 1}",
                    )
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
                    progress_callback=self._active_progress_callback,
                )
                if resolved_device in {"mps", "cuda"}:
                    kwargs["batch_size"] = options["batch_size"]
                ranks_at_level = np.ones(len(level_edges), dtype=np.int32)
                optimizer(coordinates, level_edges, ranks_at_level, **kwargs)
                if self._active_mst_progress_callback is not None:
                    self._active_mst_progress_callback(
                        options["multilevel_epochs"],
                        options["multilevel_epochs"],
                        f"Refined level {level_index + 1}",
                    )
        return coordinates

    @staticmethod
    def _feature_neighbor_graph(
        features: object,
        neighbors: int,
        metric: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the original feature-space KNN graph used by refinements."""
        from sklearn.neighbors import NearestNeighbors

        n_samples = features.shape[0]
        k = min(int(neighbors), n_samples - 1)
        index = NearestNeighbors(
            n_neighbors=min(k + 1, n_samples), metric=metric, n_jobs=-1
        ).fit(features)
        distances, indices = index.kneighbors(features, return_distance=True)
        result_indices = np.empty((n_samples, k), dtype=np.int64)
        result_distances = np.empty((n_samples, k), dtype=np.float64)
        for row in range(n_samples):
            keep = indices[row] != row
            row_indices = indices[row, keep][:k]
            row_distances = distances[row, keep][:k]
            order = np.argsort(row_distances, kind="stable")
            result_indices[row] = row_indices[order]
            result_distances[row] = row_distances[order]
        return result_indices, result_distances

    def _apply_mean_shift(
        self,
        options: dict[str, object],
    ) -> None:
        """Apply mean shift after gravity using base-layout neighbors."""
        self.embedding_before_mean_shift_ = self.embedding_.copy()
        self.mean_shift_neighbor_search_time_ = 0.0
        self.mean_shift_time_ = 0.0
        self.mean_shift_iterations_ = 0
        iterations = int(options["mean_shift_iterations"])
        step_size = float(options["mean_shift_step_size"])
        self.mean_shift_neighbor_graph_reused_ = False
        if iterations == 0 or step_size == 0.0:
            return

        from ._mean_shift import embedding_neighbor_indices, gaussian_mean_shift_layout

        started = time.perf_counter()
        base_layout = self.embedding_before_gravity_[:, :2]
        neighbor_indices = embedding_neighbor_indices(
            base_layout,
            min(int(options["mean_shift_neighbors"]), len(self.embedding_) - 1),
        )
        self.mean_shift_neighbor_search_time_ = time.perf_counter() - started
        self.mean_shift_neighbor_graph_reused_ = True
        refined, diagnostics = gaussian_mean_shift_layout(
            self.embedding_[:, :2],
            neighbor_indices=neighbor_indices,
            neighbors=min(int(options["mean_shift_neighbors"]), len(self.embedding_) - 1),
            bandwidth_factor=float(options["mean_shift_bandwidth_factor"]),
            step_size=step_size,
            iterations=iterations,
        )
        self.embedding_[:, :2] = refined
        self.mean_shift_time_ = time.perf_counter() - started
        self.mean_shift_iterations_ = int(diagnostics["iterations"])

    def _apply_gravity_min_dist_floor(
        self,
        options: dict[str, object],
        feature_neighbor_graph: tuple[np.ndarray, np.ndarray] | None,
    ) -> None:
        """Reapply the local spacing floor after all layout refinements."""
        fraction = float(options["gravity_min_dist"])
        if (
            not options["gravity_enabled"]
            or fraction == 0.0
            or feature_neighbor_graph is None
        ):
            return

        from ._gravity import _apply_local_distance_floor

        started = time.perf_counter()
        refined, diagnostics = _apply_local_distance_floor(
            self.embedding_[:, :2],
            feature_neighbor_graph[0],
            fraction,
            reference_embedding=self.embedding_before_gravity_[:, :2],
        )
        self.embedding_[:, :2] = refined
        self.gravity_min_dist_used_ = float(diagnostics["min_dist_threshold"])
        self.gravity_min_dist_reference_ = float(diagnostics["min_dist_reference"])
        self.gravity_min_dist_pairs_adjusted_ = int(
            diagnostics["min_dist_pairs_adjusted"]
        )
        self.gravity_time_ += time.perf_counter() - started

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
        `multilevel_initialization_time_`, `final_embedding_optimization_time_`,
        `embedding_optimization_time_`, `optimizer_negative_sampling_time_`,
        `optimizer_coordinate_update_time_`, and total `fit_time_`. Gravity fits
        also expose `gravity_neighbor_search_time_` and
        `gravity_neighbor_graph_reused_`. Feature based FAMST gravity fits expose
        the high-dimensional graph used by gravity as
        `feature_neighbor_graph_`. Mean shift runs after gravity and exposes
        its stage time and iteration count. GPU backends report negative
        sampling and coordinate updates as one combined duration to avoid
        per-batch device event recording on long fits.
        """
        fit_started = time.perf_counter()
        if self.input_preprocessor is None:
            if hasattr(self, "preprocessor_"):
                del self.preprocessor_
        else:
            from sklearn.base import clone

            from ._preprocessing import fit_transform_input, make_preprocessor

            configured = self.input_preprocessor
            if isinstance(configured, str):
                configured = make_preprocessor(configured)
            self.preprocessor_ = clone(configured)
            X = fit_transform_input(self.preprocessor_, X, y)
        del y  # Labels are available to an optional supervised preprocessor only.
        options = self._validated_fit_options()
        for name in ("feature_neighbor_graph_", "feature_neighbor_graph_reused_"):
            if hasattr(self, name):
                delattr(self, name)
        X_checked = check_array(
            X, dtype=np.float64, ensure_2d=True, accept_sparse="csr"
        )
        if issparse(X_checked):
            X_checked = csr_matrix(X_checked)
            if self.mst_method == "famst":
                # FAMST's approximate-neighbor backends operate on dense
                # features. Keep accepting sparse caller input, but convert it
                # here so selecting the approximate method is sufficient.
                X_checked = X_checked.toarray()
        # Cosine distance depends only on row direction. Normalize with a
        # scale-first algorithm so very large finite inputs cannot overflow
        # scikit-learn's dot products during distance calculation or PCA init.
        metric_features = (
            _cosine_safe_normalize(X_checked)
            if options["metric"] == "cosine"
            else X_checked
        )
        n_samples = X_checked.shape[0]
        if n_samples < 2:
            raise ValueError("X must contain at least two samples to construct an MST.")
        if options["include_knn_attraction"] and n_samples <= max(
            options["mst_neighbors"], 2 * options["n_msts"]
        ) + 1:
            raise ValueError(
                "include_knn_attraction requires enough samples for FAMST "
                "approximate-neighbor search."
            )
        self._activate_progress_reporting()
        graph_started = time.perf_counter()
        candidate_neighbor_edges: list[np.ndarray] = []
        gravity_neighbor_graphs: list[tuple[np.ndarray, np.ndarray]] = []
        candidate_edges_callback = (
            candidate_neighbor_edges.append
            if options["include_knn_attraction"]
            else None
        )

        refinements_need_graph = options["gravity_enabled"]
        feature_graph_width = options["gravity_neighbors"]
        reuse_graph_for_refinements = (
            self.mst_method == "famst"
            and options["metric"] == "euclidean"
            and not issparse(X_checked)
        )

        def capture_gravity_neighbor_graph(
            indices: np.ndarray, distances: np.ndarray
        ) -> None:
            gravity_neighbor_graphs.append((indices, distances))

        edges, tree_ranks = build_mst_edges(
            metric_features,
            n_msts=options["n_msts"],
            random_state=self.random_state,
            method=self.mst_method,
            neighbors=options["mst_neighbors"],
            inter_component_edges=options["inter_component_edges"],
            max_neighbors=options["max_neighbors"],
            representatives_per_component=options["representatives_per_component"],
            neighbor_device=options["resolved_device"],
            neighbor_backend=options["mst_neighbor_backend"],
            progress_callback=self._active_mst_progress_callback,
            return_ranks=True,
            metric=options["metric"],
            candidate_edges_callback=candidate_edges_callback,
            neighbor_graph_callback=capture_gravity_neighbor_graph
            if reuse_graph_for_refinements else None,
            neighbor_graph_neighbors=feature_graph_width
            if reuse_graph_for_refinements else None,
        )
        tree_ranks = np.asarray(tree_ranks, dtype=np.int32)
        if options["include_knn_attraction"]:
            if not candidate_neighbor_edges:
                raise RuntimeError(
                    "FAMST did not expose candidate neighbors for attraction; "
                    "try a larger input or disable include_knn_attraction."
                )
            edges, tree_ranks = _append_candidate_attraction_edges(
                edges,
                tree_ranks,
                candidate_neighbor_edges[-1],
                n_samples,
            )
        self.graph_construction_time_ = time.perf_counter() - graph_started
        if self._progress_reporter is not None:
            self._progress_reporter.phase_finished(
                "MST graph construction", self.graph_construction_time_
            )
        self._fit_edges(
            n_samples=n_samples,
            n_components=options["n_components"],
            edges=edges,
            tree_ranks=tree_ranks,
            fit_started=fit_started,
            resolved_device=options["resolved_device"],
            schedule_by_rank=self.mst_method != "prim",
            X_checked=X_checked,
            X_original=X,
            initialization_features=metric_features,
            initial_coordinates=self.init_coordinates,
            **{
                name: value
                for name, value in options.items()
                if name not in {
                    "n_components", "n_msts", "resolved_device",
                    "mst_neighbor_backend", "include_knn_attraction",
                    "gravity_neighbors", "gravity_resolution",
                    "gravity_strength", "gravity_min_dist", "gravity_core_quantile",
                    "gravity_mass_power",
                    "gravity_pack_regions", "gravity_pack_radius_quantile",
                    "gravity_pack_gap_fraction",
                    "gravity_enabled", "mean_shift_neighbors",
                    "mean_shift_bandwidth_factor", "mean_shift_step_size",
                    "mean_shift_iterations",
                }
            },
        )
        self.embedding_before_gravity_ = self.embedding_.copy()
        self.imste_fit_time_ = self.fit_time_
        feature_neighbor_graph = gravity_neighbor_graphs[-1] if gravity_neighbor_graphs else None
        if feature_neighbor_graph is not None:
            self.feature_neighbor_graph_ = feature_neighbor_graph
            self.feature_neighbor_graph_reused_ = True
        else:
            self.feature_neighbor_graph_reused_ = False
        if refinements_need_graph and feature_neighbor_graph is None:
            feature_neighbor_graph = self._feature_neighbor_graph(
                metric_features,
                feature_graph_width,
                options["metric"],
            )
            self.feature_neighbor_graph_ = feature_neighbor_graph
        if options["gravity_enabled"] and n_samples >= 3:
            if issparse(X_checked):
                raise ValueError("gravity refinement currently requires dense features.")
            from ._gravity import density_gravity_layout

            gravity_started = time.perf_counter()
            gravity_embedding, gravity_diagnostics = density_gravity_layout(
                X_checked,
                self.embedding_before_gravity_[:, :2],
                neighbors=min(options["gravity_neighbors"], n_samples - 1),
                resolution=options["gravity_resolution"],
                random_state=42 if self.random_state is None else self.random_state,
                core_quantile=options["gravity_core_quantile"],
                strength=options["gravity_strength"],
                min_dist=0.0,
                mass_power=options["gravity_mass_power"],
                pack_regions=options["gravity_pack_regions"],
                pack_radius_quantile=options["gravity_pack_radius_quantile"],
                pack_gap_fraction=options["gravity_pack_gap_fraction"],
                feature_neighbor_graph=feature_neighbor_graph,
                feature_neighbor_metric=(
                    "euclidean" if feature_neighbor_graph is not None else None
                ),
            )
            self.embedding_ = self.embedding_before_gravity_.copy()
            self.embedding_[:, :2] = gravity_embedding
            self.gravity_membership_ = gravity_diagnostics["membership"]
            self.gravity_point_mass_ = gravity_diagnostics["point_mass"]
            self.gravity_region_count_ = int(gravity_diagnostics["region_count"])
            self.gravity_regions_packed_ = bool(gravity_diagnostics["regions_packed"])
            self.gravity_min_dist_used_ = float(gravity_diagnostics["min_dist_threshold"])
            self.gravity_min_dist_reference_ = float(gravity_diagnostics["min_dist_reference"])
            self.gravity_min_dist_pairs_adjusted_ = int(
                gravity_diagnostics["min_dist_pairs_adjusted"]
            )
            self.gravity_graph_construction_time_ = float(
                gravity_diagnostics["graph_seconds"]
            )
            self.gravity_neighbor_search_time_ = float(
                gravity_diagnostics["neighbor_search_seconds"]
            )
            self.gravity_neighbor_graph_reused_ = bool(
                gravity_diagnostics["neighbor_graph_reused"]
            )
            self.gravity_force_time_ = float(gravity_diagnostics["force_seconds"])
            self.gravity_time_ = time.perf_counter() - gravity_started
        else:
            self.gravity_region_count_ = 0
            self.gravity_regions_packed_ = False
            self.gravity_min_dist_used_ = 0.0
            self.gravity_min_dist_reference_ = 0.0
            self.gravity_min_dist_pairs_adjusted_ = 0
            self.gravity_graph_construction_time_ = 0.0
            self.gravity_neighbor_search_time_ = 0.0
            self.gravity_neighbor_graph_reused_ = False
            self.gravity_force_time_ = 0.0
            self.gravity_time_ = 0.0
            for name in ("gravity_membership_", "gravity_point_mass_"):
                if hasattr(self, name):
                    delattr(self, name)
            if hasattr(self, "gravity_skipped_reason_"):
                del self.gravity_skipped_reason_
        self._apply_mean_shift(options)
        self._apply_gravity_min_dist_floor(options, feature_neighbor_graph)
        self.fit_time_ = time.perf_counter() - fit_started
        if self._progress_reporter is not None:
            self._progress_reporter.finish()
        return self

    def preprocessor(self, preprocessor: str | object, **options: object):
        """Attach an input transformer for raw text, images, graphs, or features.

        ``preprocessor`` may be ``'text'`` or ``'image'`` for the optional
        EmbeddingGemma 2 model, ``'graph'`` for NSPPK, or any scikit-learn-
        compatible transformer. The transformer is fitted before IMSTE when
        :meth:`fit` or :meth:`fit_transform` is called. This method configures
        and returns this estimator for chaining.

        Text and image presets accept ``model_name`` (default
        ``'google/embeddinggemma-2'``), ``batch_size`` (default 32), and
        ``device`` (default ``None``). Graph preset options are passed to
        ``NSPPK``. Custom transformers do not accept additional options.

        Parameters
        ----------
        preprocessor : {'text', 'image', 'graph'} or transformer
            Modality preset or transformer implementing ``fit`` and
            ``transform``.
        **options
            Preset-specific options.

        Returns
        -------
        self
            The configured IMSTE estimator.
        """
        from ._preprocessing import make_preprocessor

        self.input_preprocessor = make_preprocessor(
            preprocessor, options=options
        )
        return self

    def tune(
        self,
        X: object,
        y: object,
        *,
        search_space: Mapping[str, object] | None = None,
        k: int = 5,
        n_samples: int = 40,
        n_initial_points: int = 20,
        coverage_weight: float = 0.4,
        trial_callback: Callable[[dict[str, object]], None] | None = None,
        verbose: bool = False,
    ):
        """Tune settings for k-NN accuracy and spatial coverage of a 2D layout.

        Each search trial fits IMSTE once on all rows in ``X``; labels are
        used only to score that transductive embedding with shuffled,
        stratified five-fold cross-validation. Numeric ``(low, high)`` tuples
        define inclusive bounds, with integer or continuous dimensions inferred
        from the estimator parameter's declared default type. Parameters with
        a ``None`` default use their current value or the tuple endpoint types.
        Lists define explicit values
        to try, including categorical choices; tuples are reserved for numeric
        bounds. If ``search_space`` is omitted, the search varies ``n_msts``
        from 3 to 30, ``learning_rate`` from 0.01 to 0.1, ``negative_ratio``
        from 1 to 10, ``gravity_strength`` from 0.0 to 1.0, and
        ``mean_shift_iterations`` from 0 to 5. The first
        ``min(n_initial_points, n_samples)`` trials use
        Latin hypercube sampling; remaining trials use Gaussian-process Bayesian
        optimization.

        Trials maximize ``(1 - coverage_weight) * accuracy +
        coverage_weight * spatial_coverage``. Spatial coverage is normalized
        to [0, 1] from a robust, quantile-scaled 2D occupancy entropy score.
        The default ``coverage_weight=0.4`` gives accuracy 60% of the objective;
        set it to zero to tune accuracy alone.

        ``trial_callback``, when supplied, is called after each completed trial
        with its accuracy, coverage and objective scores, the best-so-far
        metrics and parameters, and a copy of that trial's embedding. This is
        useful for displaying live progress in a notebook.

        ``verbose`` controls progress output from the optimizer and trial/final
        estimator fits. It defaults to ``False``; ``trial_callback`` remains
        independent and is silent when omitted.

        The best configuration is then fitted on all rows and stored on this
        estimator. The returned scikit-optimize ``OptimizeResult`` contains the
        separate trial metrics, the selected parameters, and final embedding.
        ``best_score`` remains the selected trial's accuracy;
        ``best_objective`` is the value optimized by Bayesian search.
        ``n_components`` is fixed to 2 during tuning.

        Parameters
        ----------
        X, y : array-like
            Feature rows and corresponding single-output class labels.
        search_space : mapping
            Estimator parameter names mapped to numeric ``(low, high)`` tuple
            bounds or lists of explicit values to try. ``None`` uses defaults for
            ``n_msts`` (3–30), ``learning_rate`` (0.01–0.1), ``negative_ratio``
            (1–10), ``gravity_strength`` (0.0–1.0), and
            ``mean_shift_iterations`` (0–5).
        k : int, default=5
            Number of neighbors in the k-NN scoring classifier.
        n_samples : int, default=40
            Total number of parameter configurations to evaluate, including
            initial Latin hypercube trials.
        n_initial_points : int, default=20
            Maximum number of initial Latin hypercube trials.
        coverage_weight : float, default=0.4
            Weight of spatial coverage in the combined objective, from zero to
            one. Accuracy receives the remaining weight.
        trial_callback : callable or None, default=None
            Function receiving a trial-result dictionary after each evaluation.
        verbose : bool, default=False
            Whether to show optimizer and estimator fit progress.

        Returns
        -------
        scikit-optimize OptimizeResult
            Search history with accuracy, coverage and combined objective
            scores, plus the best fitted embedding. The estimator exposes
            ``best_params_``, ``best_score_``, ``best_coverage_``,
            ``best_objective_``, and ``tune_result_``.
        """
        from ._tuning import tune_estimator

        return tune_estimator(
            self,
            X,
            y,
            search_space=search_space,
            k=k,
            n_samples=n_samples,
            n_initial_points=n_initial_points,
            coverage_weight=coverage_weight,
            trial_callback=trial_callback,
            verbose=verbose,
        )

    def fit_transform(
        self,
        X: object,
        y: object = None,
        *,
        return_intermediate: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Fit and return final coordinates, optionally with the base layout.

        When ``return_intermediate=True``, return a pair containing the
        optimized IMSTE coordinates before refinement and the final coordinates
        after gravity and mean shift. Disabled refinements leave the corresponding
        stage unchanged.
        """
        if not isinstance(return_intermediate, (bool, np.bool_)):
            raise ValueError("return_intermediate must be a boolean.")
        self.fit(X, y=y)
        if return_intermediate:
            return self.embedding_before_gravity_.copy(), self.embedding_.copy()
        return self.embedding_.copy()
