"""Sparse-graph fitting and feature-transform methods for IMSTE."""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, issparse
from sklearn.utils.validation import check_array, check_is_fitted

from .mst import iterated_graph_edges


class _EstimatorGraphMethods:

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
        for name in ("feature_neighbor_graph_", "feature_neighbor_graph_reused_"):
            if hasattr(self, name):
                delattr(self, name)
        if options["include_knn_attraction"]:
            raise ValueError(
                "include_knn_attraction is available for feature-based fit(X), "
                "not fit_graph()."
            )
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
        if options["gravity_enabled"] and initialization_features is not None:
            if issparse(initialization_features):
                raise ValueError("gravity refinement currently requires dense features.")
        self._activate_progress_reporting()
        graph_started = time.perf_counter()
        edges, tree_ranks = iterated_graph_edges(
            distance_graph,
            options["n_msts"],
            progress_callback=self._active_mst_progress_callback,
            return_ranks=True,
        )
        self.graph_construction_time_ = time.perf_counter() - graph_started
        if self._progress_reporter is not None:
            self._progress_reporter.phase_finished(
                "MST graph construction", self.graph_construction_time_
            )
        tree_ranks = np.asarray(tree_ranks, dtype=np.int32)
        self._fit_edges(
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
                if name not in {
                    "n_components", "n_msts", "resolved_device",
                    "mst_neighbor_backend",
                }
            },
        )
        self.embedding_before_gravity_ = self.embedding_.copy()
        self.imste_fit_time_ = self.fit_time_
        n_samples = distance_graph.shape[0]
        feature_graph_width = options["gravity_neighbors"] if options["gravity_enabled"] else 1
        feature_neighbor_graph = None
        refinements_need_graph = options["gravity_enabled"]
        if refinements_need_graph and initialization_features is not None:
            if issparse(initialization_features) and options["gravity_enabled"]:
                raise ValueError("gravity refinement currently requires dense features.")
            feature_neighbor_graph = self._feature_neighbor_graph(
                initialization_features,
                feature_graph_width,
                options["metric"],
            )
            self.feature_neighbor_graph_ = feature_neighbor_graph
            self.feature_neighbor_graph_reused_ = False
        if options["gravity_enabled"] and initialization_features is not None and n_samples >= 3:
            from ._gravity import density_gravity_layout

            gravity_started = time.perf_counter()
            gravity_embedding, diagnostics = density_gravity_layout(
                initialization_features,
                self.embedding_before_gravity_[:, :2],
                neighbors=min(options["gravity_neighbors"], n_samples - 1),
                resolution=options["gravity_resolution"],
                random_state=42 if self.random_state is None else self.random_state,
                core_quantile=options["gravity_core_quantile"],
                strength=options["gravity_strength"],
                mass_power=options["gravity_mass_power"],
                pack_regions=options["gravity_pack_regions"],
                pack_radius_quantile=options["gravity_pack_radius_quantile"],
                pack_gap_fraction=options["gravity_pack_gap_fraction"],
                feature_neighbor_graph=feature_neighbor_graph,
                feature_neighbor_metric=options["metric"],
            )
            self.embedding_ = self.embedding_before_gravity_.copy()
            self.embedding_[:, :2] = gravity_embedding
            self.gravity_membership_ = diagnostics["membership"]
            self.gravity_point_mass_ = diagnostics["point_mass"]
            self.gravity_region_count_ = int(diagnostics["region_count"])
            self.gravity_regions_packed_ = bool(diagnostics["regions_packed"])
            self.gravity_graph_construction_time_ = float(diagnostics["graph_seconds"])
            self.gravity_neighbor_search_time_ = float(diagnostics["neighbor_search_seconds"])
            self.gravity_neighbor_graph_reused_ = bool(diagnostics["neighbor_graph_reused"])
            self.gravity_force_time_ = float(diagnostics["force_seconds"])
            self.gravity_time_ = time.perf_counter() - gravity_started
            if hasattr(self, "gravity_skipped_reason_"):
                del self.gravity_skipped_reason_
        else:
            self.gravity_region_count_ = 0
            self.gravity_regions_packed_ = False
            self.gravity_graph_construction_time_ = 0.0
            self.gravity_neighbor_search_time_ = 0.0
            self.gravity_neighbor_graph_reused_ = False
            self.gravity_force_time_ = 0.0
            self.gravity_time_ = 0.0
            if options["gravity_enabled"]:
                self.gravity_skipped_reason_ = (
                    "fit_graph() needs init_features to build the gravity graph."
                )
            elif hasattr(self, "gravity_skipped_reason_"):
                del self.gravity_skipped_reason_
            for name in ("gravity_membership_", "gravity_point_mass_"):
                if hasattr(self, name):
                    delattr(self, name)
        self._apply_mean_shift(options)
        self.fit_time_ = time.perf_counter() - fit_started
        if self._progress_reporter is not None:
            self._progress_reporter.finish()
        return self

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
