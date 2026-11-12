import numpy as np
import pytest
import torch
from sklearn.base import clone
from scipy.spatial.distance import cdist
from scipy import sparse

from imste import IteratedMinimumSpanningTreeEmbedder
from imste.inductive import InductiveIMSTE
from imste._estimator import (
    _append_candidate_attraction_edges,
    _iterated_mst_edges,
    _sample_negative_targets,
    _scale_initialization_coordinates,
    _standardized_pca_initialization,
)


def small_data():
    return np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]],
        dtype=float,
    )


def test_sklearn_api_and_training_transform():
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=1, n_epochs=3, batch_size=4, random_state=7
    )
    assert clone(estimator).get_params() == estimator.get_params()

    X = small_data()
    embedding = estimator.fit_transform(X)
    assert embedding.shape == (len(X), 2)
    assert np.isfinite(embedding).all()
    np.testing.assert_array_equal(estimator.transform(X), embedding)
    assert estimator.n_features_in_ == X.shape[1]
    assert estimator.device_ == "cpu"
    assert estimator.mst_representatives_per_component == 10


def test_fit_timing_diagnostics_are_finite_and_bounded():
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=1, n_epochs=0, random_state=5, multilevel_init=True
    ).fit(small_data())

    timings = np.array(
        [
            estimator.graph_construction_time_,
            estimator.multilevel_initialization_time_,
            estimator.final_embedding_optimization_time_,
            estimator.embedding_optimization_time_,
            estimator.fit_time_,
        ]
    )
    assert np.isfinite(timings).all()
    assert np.all(timings >= 0)
    assert estimator.graph_construction_time_ <= estimator.fit_time_
    assert (
        estimator.multilevel_initialization_time_
        <= estimator.embedding_optimization_time_
    )
    assert (
        estimator.final_embedding_optimization_time_
        <= estimator.embedding_optimization_time_
    )
    assert estimator.embedding_optimization_time_ <= estimator.fit_time_
    assert (
        estimator.graph_construction_time_ + estimator.embedding_optimization_time_
        <= estimator.fit_time_
    )
    assert (
        estimator.multilevel_initialization_time_
        + estimator.final_embedding_optimization_time_
        <= estimator.embedding_optimization_time_
    )


def test_verbose_fit_reports_each_macro_phase_when_it_finishes(capsys):
    IteratedMinimumSpanningTreeEmbedder(
        n_msts=1,
        n_epochs=0,
        random_state=5,
        multilevel_init=True,
        verbose=True,
    ).fit(small_data())

    output = capsys.readouterr().out
    assert "MST graph construction complete:" in output
    assert "Multilevel initialization complete:" in output
    assert "Final embedding optimization complete:" in output
    assert "Total measured:" in output


def test_verbose_multilevel_progress_uses_global_level_bars(capsys):
    IteratedMinimumSpanningTreeEmbedder(
        n_msts=1,
        n_epochs=0,
        multilevel_init=True,
        multilevel_base_size=2,
        multilevel_epochs=1,
        mst_method="prim",
        random_state=5,
        verbose=True,
    ).fit(small_data())

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "Multilevel coarsening" in output
    assert "Multilevel refinement epochs" in output
    assert "refine level 1 epochs" not in output


def test_candidate_neighbor_attractions_append_only_non_tree_edges():
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    ranks = np.array([1, 1, 1], dtype=np.int32)
    candidates = np.array([[0, 1], [0, 2], [2, 3], [3, 4]], dtype=np.int32)

    combined_edges, combined_ranks = _append_candidate_attraction_edges(
        edges, ranks, candidates, n_samples=5
    )

    assert combined_edges.shape == (5, 2)
    np.testing.assert_array_equal(combined_edges[-2:], [[0, 2], [3, 4]])
    np.testing.assert_array_equal(combined_ranks, [1, 1, 1, 1, 1])


def test_include_knn_attraction_adds_candidate_edges_to_fit_graph(monkeypatch):
    import imste._estimator as estimator_module

    base_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
    candidate_edges = np.array([[0, 1], [0, 2], [1, 3], [3, 4]], dtype=np.int32)

    def fake_build_mst_edges(X, *, candidate_edges_callback=None, **kwargs):
        if candidate_edges_callback is not None:
            candidate_edges_callback(candidate_edges)
        return base_edges, np.ones(len(base_edges), dtype=np.int32)

    monkeypatch.setattr(estimator_module, "build_mst_edges", fake_build_mst_edges)
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=1,
        n_epochs=0,
        mst_method="famst",
        include_knn_attraction=True,
        pca_init=False,
        multilevel_init=False,
        random_state=5,
    ).fit(np.random.RandomState(5).normal(size=(20, 2)))

    fitted_edges = {tuple(edge) for edge in estimator.graph_edges_.tolist()}
    assert fitted_edges == {(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 3)}
    assert estimator.graph_edges_.shape[0] == estimator.graph_ranks_.size


def test_knn_attraction_rejects_unsupported_graph_modes():
    with pytest.raises(ValueError, match="requires mst_method='famst'"):
        IteratedMinimumSpanningTreeEmbedder(
            mst_method="prim", include_knn_attraction=True
        ).fit(small_data())
    with pytest.raises(ValueError, match="requires metric='euclidean'"):
        IteratedMinimumSpanningTreeEmbedder(
            mst_method="famst", metric="cosine", include_knn_attraction=True
        ).fit(small_data())


def test_progress_callback_reports_completed_epochs_in_order():
    events = []
    IteratedMinimumSpanningTreeEmbedder(
        n_msts=1,
        n_epochs=3,
        random_state=5,
        mst_method="prim",
        negative_ratio=0,
        progress_callback=lambda completed, total: events.append((completed, total)),
    ).fit(small_data())
    assert events == [(1, 3), (2, 3), (3, 3)]


def test_seed_reproduces_embedding():
    X = small_data()
    params = dict(n_msts=1, n_epochs=4, batch_size=3, random_state=13)
    first = IteratedMinimumSpanningTreeEmbedder(**params).fit_transform(X)
    second = IteratedMinimumSpanningTreeEmbedder(**params).fit_transform(X)
    np.testing.assert_array_equal(first, second)


def test_default_pca_initialization_globally_scales_standardized_dense_pca():
    X = np.array([[0.0, 3.0, 4.0], [1.0, 1.0, 2.0], [2.0, 0.0, 1.0], [4.0, 2.0, 0.0]])
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=1, n_epochs=0, mst_method="prim", random_state=12
    ).fit(X)

    from sklearn.decomposition import PCA

    standardized = (X - X.mean(axis=0)) / X.std(axis=0)
    pca_scores = PCA(n_components=2, svd_solver="full").fit_transform(standardized)
    expected = _scale_initialization_coordinates(pca_scores, scaling="global")
    assert estimator.pca_init_scaling == "global"
    np.testing.assert_allclose(estimator.embedding_, expected, atol=1e-6)
    np.testing.assert_array_equal(
        estimator.embedding_,
        IteratedMinimumSpanningTreeEmbedder(
            n_msts=1, n_epochs=0, mst_method="prim", random_state=12
        )
        .fit(X)
        .embedding_,
    )


def test_pca_init_scaling_can_preserve_raw_pca_magnitudes():
    X = small_data()
    pca_scores = _standardized_pca_initialization(X, 2, random_state=6)
    raw_pca = IteratedMinimumSpanningTreeEmbedder(
        n_msts=1,
        n_epochs=0,
        mst_method="prim",
        random_state=6,
        pca_init_scaling="none",
    ).fit(X)
    random_init = IteratedMinimumSpanningTreeEmbedder(
        n_msts=1,
        n_epochs=0,
        mst_method="prim",
        random_state=6,
        pca_init=False,
    ).fit(X)

    np.testing.assert_allclose(raw_pca.embedding_, pca_scores, atol=1e-6)
    expected_rng = np.random.RandomState(6)
    expected_rng.randint(0, np.iinfo(np.int32).max)
    expected_random = (expected_rng.standard_normal((len(X), 2)) * 1e-3).astype(
        np.float32
    )
    np.testing.assert_array_equal(random_init.embedding_, expected_random)


@pytest.mark.parametrize("value", ["small", "minmax", ""])
def test_pca_init_scaling_rejects_unsupported_modes(value):
    with pytest.raises(ValueError, match="pca_init_scaling"):
        IteratedMinimumSpanningTreeEmbedder(pca_init_scaling=value).fit(small_data())


def test_pca_initialization_keeps_csr_input_sparse(monkeypatch):
    X = np.random.RandomState(8).normal(size=(9, 5))
    X[np.abs(X) < 0.8] = 0.0
    X_csr = sparse.csr_matrix(X)

    def fail_densify(*args, **kwargs):
        raise AssertionError("PCA initialization must not densify CSR input")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", fail_densify)
    sparse_scores = _standardized_pca_initialization(X_csr, 3, random_state=17)
    dense_scores = _standardized_pca_initialization(X, 3, random_state=17)

    assert np.isfinite(sparse_scores).all()
    np.testing.assert_allclose(np.abs(sparse_scores), np.abs(dense_scores), atol=1e-6)


def test_pca_initialization_zero_fills_extra_dimensions_and_constant_features():
    X = np.array([[0.0, 2.0, 5.0], [1.0, 4.0, 5.0], [2.0, 6.0, 5.0]])
    scores = _standardized_pca_initialization(X, 5, random_state=0)

    assert scores.shape == (3, 5)
    assert np.isfinite(scores).all()
    np.testing.assert_array_equal(scores[:, 3:], 0.0)


def test_pca_init_false_preserves_seeded_random_initialization():
    seed = 23
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=1,
        n_epochs=0,
        mst_method="prim",
        random_state=seed,
        pca_init=False,
    ).fit(small_data())
    rng = np.random.RandomState(seed)
    rng.randint(0, np.iinfo(np.int32).max)
    expected = (rng.standard_normal((len(small_data()), 2)) * 1e-3).astype(np.float32)

    np.testing.assert_array_equal(estimator.embedding_, expected)


def test_multilevel_initialization_returns_finite_embedding():
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=1,
        n_epochs=2,
        multilevel_init=True,
        multilevel_base_size=2,
        multilevel_epochs=2,
        mst_method="prim",
        negative_ratio=0,
        random_state=11,
    ).fit(small_data())
    assert estimator.embedding_.shape == (len(small_data()), 2)
    assert np.isfinite(estimator.embedding_).all()


def test_explicit_coordinates_take_precedence_over_multilevel(monkeypatch):
    coordinates = np.arange(len(small_data()) * 2, dtype=float).reshape(-1, 2)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("explicit coordinates must skip multilevel initialization")

    monkeypatch.setattr(
        IteratedMinimumSpanningTreeEmbedder,
        "_multilevel_initialization",
        fail_if_called,
    )
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=1,
        n_epochs=0,
        multilevel_init=True,
        init_coordinates=coordinates,
        mst_method="prim",
    ).fit(small_data())
    np.testing.assert_array_equal(estimator.embedding_, coordinates.astype(np.float32))


def test_fit_graph_uses_random_initialization_with_default_pca_option():
    graph = sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
    seed = 31
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=1, n_epochs=0, random_state=seed
    ).fit_graph(graph)
    rng = np.random.RandomState(seed)
    rng.randint(0, np.iinfo(np.int32).max)
    expected = (rng.standard_normal((2, 2)) * 1e-3).astype(np.float32)

    assert estimator.pca_init is True
    assert clone(estimator).pca_init is True
    np.testing.assert_array_equal(estimator.embedding_, expected)


def test_pca_init_requires_boolean():
    with pytest.raises(ValueError, match="pca_init must be a boolean"):
        IteratedMinimumSpanningTreeEmbedder(pca_init="yes").fit(small_data())


def test_transform_rejects_new_or_reordered_rows():
    X = small_data()
    estimator = IteratedMinimumSpanningTreeEmbedder(n_msts=1, n_epochs=0).fit(X)
    with pytest.raises(ValueError, match="original training rows"):
        estimator.transform(X[::-1])
    with pytest.raises(ValueError, match="original training rows"):
        estimator.transform(np.vstack([X, [3.0, 3.0]]))


def test_inductive_projection_maps_training_and_unseen_rows():
    X = small_data()
    estimator = InductiveIMSTE(
        embedder=IteratedMinimumSpanningTreeEmbedder(
            n_msts=1, n_epochs=0, random_state=11
        ),
        n_layers=1,
        layer_size=8,
        dropout=0.0,
        epochs=4,
        min_epochs=2,
        patience=2,
        batch_size=2,
    ).fit(X)

    training_projection = estimator.transform(X)
    novel_projection = estimator.transform([[3.0, 3.0]])
    assert training_projection.shape == (len(X), 2)
    assert novel_projection.shape == (1, 2)
    assert np.isfinite(training_projection).all()
    assert np.isfinite(novel_projection).all()
    assert estimator.reference_embedding_.shape == (len(X), 2)
    assert 2 <= estimator.epochs_trained_ <= 4
    assert np.isfinite(estimator.best_validation_loss_)
    assert np.isfinite(estimator.training_loss_)


def test_dataframe_transform_rejects_reordered_feature_names():
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame(small_data(), columns=["left", "right"])
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=1, n_epochs=0, random_state=3
    ).fit(X)

    with pytest.raises(ValueError, match="same feature names in the same order"):
        estimator.transform(X[["right", "left"]])


def test_dense_negative_sampling_falls_back_to_exact_complement():
    n_samples = 64
    packed_adjacency = np.full((n_samples, n_samples // 8), 0xFF, dtype=np.uint8)
    # Node 0 has exactly one non-neighbor. Rejection sampling alone would need
    # many draws; the bounded sampler should return that complement directly.
    packed_adjacency[0, 63 // 8] &= np.uint8(0xFF ^ (1 << (63 & 7)))
    sampled = _sample_negative_targets(
        np.random.RandomState(17),
        packed_adjacency,
        np.zeros(32, dtype=np.intp),
        n_samples,
    )
    np.testing.assert_array_equal(sampled, np.full(32, 63))


def test_inductive_mlp_stops_after_patience_once_minimum_epochs_are_met(monkeypatch):
    class ConstantMLP(torch.nn.Module):
        def __init__(self, n_features, n_components, n_layers, layer_size, dropout):
            super().__init__()
            self.output = torch.nn.Parameter(torch.ones(n_components))

        def forward(self, inputs):
            return (self.output * 0).expand(inputs.shape[0], -1)

    monkeypatch.setattr("imste.inductive._MLPProjector", ConstantMLP)
    estimator = InductiveIMSTE(
        embedder=IteratedMinimumSpanningTreeEmbedder(
            n_msts=1, n_epochs=0, random_state=19
        ),
        n_layers=1,
        layer_size=4,
        dropout=0.0,
        epochs=8,
        min_epochs=2,
        patience=2,
    ).fit(small_data())

    assert estimator.epochs_trained_ == 3


def test_duplicate_samples_keep_zero_distance_mst_edges():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=1, n_epochs=2, random_state=2
    ).fit(X)
    assert estimator.graph_edges_.shape == (len(X) - 1, 2)
    assert np.isfinite(estimator.embedding_).all()


def test_fit_graph_uses_only_minimum_distance_spanning_edges():
    graph = sparse.csr_matrix(
        np.array(
            [
                [0.0, 1.0, 9.0, 0.0],
                [1.0, 0.0, 2.0, 0.0],
                [9.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=1, n_epochs=0, random_state=2
    ).fit_graph(graph)

    assert estimator.graph_edges_.tolist() == [[0, 1], [1, 2]]
    assert estimator.graph_ranks_.tolist() == [1, 1]
    assert estimator.embedding_.shape == (4, 2)
    assert estimator.graph_input_ is True
    assert np.isfinite(estimator.embedding_).all()


def test_fit_graph_reuses_edges_and_preserves_disconnected_components():
    graph = sparse.csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=2, n_epochs=0, random_state=2
    ).fit_graph(graph)

    assert estimator.graph_edges_.shape == (6, 2)
    assert estimator.graph_ranks_.tolist() == [1, 1, 1, 2, 2, 2]
    assert np.all(
        estimator.graph_edges_[:, 0] // 3 == estimator.graph_edges_[:, 1] // 3
    )
    assert estimator.embedding_.shape == (6, 2)


def test_fit_graph_uses_minimum_reciprocal_distance_and_one_way_edges():
    graph = sparse.coo_matrix(
        (
            np.array([4.0, 1.0, 2.0, 1.5]),
            (np.array([0, 1, 1, 0]), np.array([1, 0, 2, 2])),
        ),
        shape=(3, 3),
    )
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=1, n_epochs=0, random_state=2
    ).fit_graph(graph)

    assert estimator.graph_edges_.tolist() == [[0, 1], [0, 2]]


def test_fit_graph_keeps_explicit_zero_distance_edges():
    graph = sparse.coo_matrix(
        (
            np.array([0.0, 0.0, 2.0, 2.0]),
            (np.array([0, 1, 1, 2]), np.array([1, 0, 2, 1])),
        ),
        shape=(3, 3),
    )
    estimator = IteratedMinimumSpanningTreeEmbedder(n_msts=1, n_epochs=0).fit_graph(
        graph
    )

    assert estimator.graph_edges_.tolist() == [[0, 1], [1, 2]]


@pytest.mark.parametrize(
    ("graph", "message"),
    [
        (np.eye(3), "SciPy sparse"),
        (sparse.csr_matrix((2, 3)), "square"),
        (sparse.csr_matrix([[0.0, -1.0], [-1.0, 0.0]]), "non-negative"),
        (sparse.csr_matrix([[0.0, np.nan], [np.nan, 0.0]]), "finite"),
        (sparse.csr_matrix((3, 3)), "off-diagonal edge"),
    ],
)
def test_fit_graph_rejects_invalid_graphs(graph, message):
    with pytest.raises(ValueError, match=message):
        IteratedMinimumSpanningTreeEmbedder(n_epochs=0).fit_graph(graph)


def test_famst_reuses_a_bottleneck_edge_when_more_trees_are_requested():
    X = np.array([[0.0], [1.0]])
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=2, n_epochs=0, random_state=3
    ).fit(X)

    assert estimator.graph_edges_.tolist() == [[0, 1], [0, 1]]
    assert estimator.graph_ranks_.tolist() == [1, 2]


def test_no_eligible_negative_pairs_is_supported():
    X = np.array([[0.0], [1.0]])
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=1, n_epochs=2, random_state=3
    ).fit(X)
    assert estimator.embedding_.shape == (2, 2)
    assert np.isfinite(estimator.embedding_).all()


def test_mps_device_requires_available_backend(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS-capable PyTorch installation"):
        IteratedMinimumSpanningTreeEmbedder(
            n_msts=1, n_epochs=0, random_state=5, device="mps"
        ).fit(small_data())


def test_rinnd_neighbor_backend_does_not_override_optimizer_device(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    estimator = IteratedMinimumSpanningTreeEmbedder(
        device="mps", mst_neighbor_backend="rinnd"
    )
    options = estimator._validated_fit_options()
    assert options["resolved_device"] == "mps"
    assert options["mst_neighbor_backend"] == "rinnd"


def test_explicit_rinnd_backend_rejects_cosine_metric():
    estimator = IteratedMinimumSpanningTreeEmbedder(
        mst_method="famst", mst_neighbor_backend="rinnd", metric="cosine"
    )
    with pytest.raises(ValueError, match="requires dense Euclidean FAMST input"):
        estimator._validated_fit_options()


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("n_msts", 0),
        ("mst_representatives_per_component", 0),
        ("mst_representatives_per_component", -1),
        ("mst_representatives_per_component", 1.5),
        ("mst_representatives_per_component", True),
        ("n_epochs", -1),
        ("batch_size", 0),
        ("negative_ratio", -1),
        ("learning_rate", 0),
        ("lambda_rep", -1),
        ("epsilon", 0),
    ],
)
def test_invalid_parameters_raise_value_error(parameter, value):
    estimator = IteratedMinimumSpanningTreeEmbedder(n_msts=1, n_epochs=0)
    setattr(estimator, parameter, value)
    with pytest.raises(ValueError):
        estimator.fit(small_data())


def test_invalid_data_raise_clear_error():
    with pytest.raises(ValueError, match="at least two samples"):
        IteratedMinimumSpanningTreeEmbedder(n_msts=1, n_epochs=0).fit([[1.0, 2.0]])
    with pytest.raises(ValueError):
        IteratedMinimumSpanningTreeEmbedder(n_msts=1, n_epochs=0).fit([[0.0], [np.nan]])


def test_exact_graph_uses_euclidean_distances():
    X = small_data()
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=2, n_epochs=0, random_state=19
    ).fit(X)
    expected, weights = _iterated_mst_edges(cdist(X, X, metric="euclidean"), 2)
    np.testing.assert_array_equal(estimator.graph_edges_, expected)
    np.testing.assert_array_equal(estimator.graph_weights_, weights)
    assert set(estimator.graph_weights_) == {1.0, 0.5}


def test_prim_ignores_famst_representative_count():
    X = small_data()
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=1,
        n_epochs=0,
        mst_method="prim",
        mst_representatives_per_component=1,
    ).fit(X)
    expected, _ = _iterated_mst_edges(cdist(X, X, metric="euclidean"), 1)

    np.testing.assert_array_equal(estimator.graph_edges_, expected)


def test_representative_count_is_passed_to_mst_builder(monkeypatch):
    captured = {}

    def fake_build_mst_edges(X, **kwargs):
        captured.update(kwargs)
        return (
            np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.intp),
            np.ones(4, dtype=np.float64),
        )

    monkeypatch.setattr("imste._estimator.build_mst_edges", fake_build_mst_edges)
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=1,
        n_epochs=0,
        mst_representatives_per_component=3,
    ).fit(small_data())

    assert estimator.mst_representatives_per_component == 3
    assert captured["representatives_per_component"] == 3


def test_graph_ranks_preserve_inverse_rank_weights():
    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=2, n_epochs=0, mst_method="prim", random_state=3
    ).fit(small_data())
    assert set(estimator.graph_ranks_) == {1, 2}
    np.testing.assert_array_equal(
        estimator.graph_weights_, 1.0 / estimator.graph_ranks_
    )
