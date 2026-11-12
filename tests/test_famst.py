import numpy as np
import pytest
import sys
import types
from scipy.sparse import csr_matrix

from imste.mst.famst import (
    _connect_components,
    _kruskal_ranked_trees,
    _ann_edges,
    _representative_component_pairs,
    _sample_component_representatives,
    iterated_famst_edges,
)


def _separated_components():
    X = np.array(
        [[0.0, 0.0], [10.0, 0.0], [1.0, 0.0], [11.0, 0.0], [20.0, 0.0], [21.0, 0.0]]
    )
    edges = {(0, 1): 10.0, (2, 3): 10.0, (4, 5): 1.0}
    groups = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    return X, edges, groups


def test_component_representatives_are_bounded_and_seeded():
    groups = [np.array([0, 1, 2]), np.arange(3, 23), np.array([23, 24])]
    points, components = _sample_component_representatives(
        groups, 5, np.random.RandomState(41)
    )
    repeated_points, repeated_components = _sample_component_representatives(
        groups, 5, np.random.RandomState(41)
    )

    np.testing.assert_array_equal(points, repeated_points)
    np.testing.assert_array_equal(components, repeated_components)
    np.testing.assert_array_equal(np.bincount(components), [3, 5, 2])
    assert set(points[components == 0]) == {0, 1, 2}
    assert set(points[components == 2]) == {23, 24}


def test_representative_mst_returns_unique_cross_component_pairs():
    X, _, groups = _separated_components()
    pairs = _representative_component_pairs(
        X, groups, 10, np.random.RandomState(7)
    )

    # The representative MST crosses components 0-1 twice, then 1-2.
    assert pairs == [(0, 1), (1, 2)]


def test_bridge_search_runs_once_for_each_representative_mst_pair():
    X, edges, _ = _separated_components()

    class RecordingRng:
        def __init__(self):
            self.rng = np.random.RandomState(5)
            self.choice_calls = 0

        def choice(self, values, size=None):
            self.choice_calls += 1 if size is None else size
            return self.rng.choice(values, size=size)

    rng = RecordingRng()
    bridges, labels = _connect_components(
        X, edges, n_candidates=1, rng=rng, representatives_per_component=2
    )

    bridge_pairs = {
        tuple(sorted((int(labels[left]), int(labels[right]))))
        for left, right, _, _ in bridges
    }
    assert bridge_pairs == {(0, 1), (1, 2)}
    assert len(bridges) == 2
    # One left and one right endpoint draw for each unique component pair.
    assert rng.choice_calls == 4


def test_famst_still_constructs_requested_tree(monkeypatch):
    X, edges, _ = _separated_components()
    monkeypatch.setattr(
        "imste.mst.famst._ann_edges",
        lambda X, neighbors, random_state: dict(edges),
    )

    result_edges, weights = iterated_famst_edges(
        X,
        n_msts=1,
        random_state=9,
        neighbors=1,
        inter_component_edges=1,
        max_neighbors=4,
        representatives_per_component=2,
        neighbor_backend="pynndescent",
    )

    assert result_edges.shape == (len(X) - 1, 2)
    np.testing.assert_array_equal(weights, np.ones(len(X) - 1))


def test_famst_jumps_to_neighbor_cap_after_first_failed_graph(monkeypatch):
    X, edges, _ = _separated_components()
    requested_neighbors = []
    extraction_calls = []

    def fake_ann(X, neighbors, random_state):
        requested_neighbors.append(neighbors)
        return dict(edges)

    def fake_kruskal(n_samples, candidate_edges, n_msts, **kwargs):
        extraction_calls.append(1)
        if len(extraction_calls) == 1:
            return (
                None,
                np.empty((0, 2), dtype=np.intp),
                np.empty(0, dtype=np.int32),
                0,
            )
        edges = np.zeros((n_samples - 1, 2), dtype=np.intp)
        ranks = np.ones(n_samples - 1, dtype=np.int32)
        return (edges, ranks), edges, ranks, 1

    monkeypatch.setattr("imste.mst.famst._ann_edges", fake_ann)
    monkeypatch.setattr(
        "imste.mst.famst._kruskal_ranked_trees", fake_kruskal
    )
    iterated_famst_edges(
        X,
        n_msts=1,
        random_state=9,
        neighbors=1,
        inter_component_edges=1,
        max_neighbors=4,
        representatives_per_component=2,
        neighbor_backend="pynndescent",
    )

    assert requested_neighbors == [2, 4]


def test_compiled_kruskal_returns_edge_disjoint_trees_and_integer_ranks():
    candidates = {
        (0, 1): 1.0,
        (1, 2): 1.0,
        (2, 3): 1.0,
        (0, 3): 1.0,
        (0, 2): 2.0,
        (1, 3): 2.0,
    }
    edges, ranks = _kruskal_ranked_trees(4, candidates, 2)
    assert edges.shape == (2 * (4 - 1), 2)
    np.testing.assert_array_equal(ranks, [1, 1, 1, 2, 2, 2])
    first = {tuple(sorted(map(int, edge))) for edge in edges[ranks == 1]}
    second = {tuple(sorted(map(int, edge))) for edge in edges[ranks == 2]}
    assert len(first) == len(second) == 3
    assert first.isdisjoint(second)


def test_famst_can_return_ranks_without_changing_weight_default(monkeypatch):
    X, edges, _ = _separated_components()
    monkeypatch.setattr(
        "imste.mst.famst._ann_edges",
        lambda X, neighbors, random_state: dict(edges),
    )
    kwargs = dict(
        X=X,
        n_msts=1,
        random_state=9,
        neighbors=1,
        inter_component_edges=1,
        max_neighbors=4,
        representatives_per_component=2,
        neighbor_backend="pynndescent",
    )
    edge_weights, weights = iterated_famst_edges(**kwargs)
    edge_ranks, ranks = iterated_famst_edges(**kwargs, return_ranks=True)
    np.testing.assert_array_equal(edge_weights, edge_ranks)
    np.testing.assert_array_equal(ranks, np.ones(len(ranks), dtype=np.int32))
    np.testing.assert_array_equal(weights, 1.0 / ranks)


def test_ann_edges_reuse_minimum_finite_returned_distance(monkeypatch):
    class FakeNNDescent:
        def __init__(self, *args, **kwargs):
            self.neighbor_graph = (
                np.array([[1, 2, -1], [0, 2, 2], [1, 0, 1]]),
                np.array([[3.0, 4.0, np.inf], [2.0, 1.5, np.nan], [2.5, 4.0, 3.5]]),
            )

    monkeypatch.setitem(
        sys.modules, "pynndescent", types.SimpleNamespace(NNDescent=FakeNNDescent)
    )
    edges = _ann_edges(np.zeros((3, 2)), neighbors=2, random_state=1)
    assert edges == {(0, 1): 2.0, (0, 2): 4.0, (1, 2): 1.5}


def test_rinnd_graph_uses_float32_and_nonself_neighbors(monkeypatch):
    captured = {}

    class FakeRINND:
        def __init__(self, data, **kwargs):
            captured["data"] = data
            captured["kwargs"] = kwargs
            self.neighbor_graph = (
                np.array([[1, 2], [0, 2], [1, 0]]),
                np.array([[1.0, 2.0], [1.0, 1.5], [1.5, 2.0]]),
            )

    from imste.mst import famst

    monkeypatch.setattr(famst, "_rinnd_class", lambda: FakeRINND)
    edges = famst._ann_edges_rinnd(np.zeros((3, 2), dtype=np.float64), 2, 11)

    assert captured["data"].dtype == np.float32
    assert captured["data"].flags.c_contiguous
    assert captured["kwargs"]["n_neighbors"] == 2
    assert captured["kwargs"]["metric"] == "euclidean"
    assert edges == {(0, 1): 1.0, (0, 2): 2.0, (1, 2): 1.5}


def test_explicit_rinnd_requires_compatible_install(monkeypatch):
    from imste.mst import famst

    monkeypatch.setattr(famst, "_rinnd_class", lambda: None)
    with pytest.raises(ImportError, match="imste\\[rinnd\\]"):
        famst._ann_edges_rinnd(np.zeros((5, 2)), 2, 1)


def test_explicit_rinnd_rejects_sparse_input():
    with pytest.raises(ValueError, match="requires dense input"):
        iterated_famst_edges(
            csr_matrix(np.zeros((6, 2))),
            n_msts=1,
            random_state=2,
            neighbors=2,
            inter_component_edges=1,
            max_neighbors=5,
            neighbor_backend="rinnd",
        )


def test_auto_prefers_rinnd_and_explicit_backend_dispatch(monkeypatch):
    X = np.zeros((6, 2), dtype=np.float64)
    complete_edges = {
        (left, right): float(right - left)
        for left in range(len(X))
        for right in range(left + 1, len(X))
    }
    calls = []
    monkeypatch.setattr("imste.mst.famst._rinnd_class", lambda: object)
    monkeypatch.setattr(
        "imste.mst.famst._ann_edges_rinnd",
        lambda X, neighbors, random_state, **kwargs: calls.append("rinnd") or complete_edges,
    )
    monkeypatch.setattr(
        "imste.mst.famst._ann_edges",
        lambda X, neighbors, random_state: calls.append("pynndescent") or complete_edges,
    )
    monkeypatch.setattr(
        "imste.mst.famst._ann_edges_mps",
        lambda X, neighbors, progress_callback=None: calls.append("mps_exact") or complete_edges,
    )

    kwargs = dict(
        X=X,
        n_msts=1,
        random_state=3,
        neighbors=2,
        inter_component_edges=1,
        max_neighbors=5,
        representatives_per_component=2,
    )
    for backend, expected in (
        ("auto", "rinnd"),
        ("pynndescent", "pynndescent"),
        ("mps_exact", "mps_exact"),
    ):
        iterated_famst_edges(**kwargs, neighbor_backend=backend)
        assert calls[-1] == expected


def test_auto_falls_back_to_device_backend_without_rinnd(monkeypatch):
    X = np.zeros((6, 2), dtype=np.float64)
    complete_edges = {
        (left, right): float(right - left)
        for left in range(len(X))
        for right in range(left + 1, len(X))
    }
    calls = []
    monkeypatch.setattr("imste.mst.famst._rinnd_class", lambda: None)
    monkeypatch.setattr(
        "imste.mst.famst._ann_edges",
        lambda X, neighbors, random_state: calls.append("pynndescent") or complete_edges,
    )
    iterated_famst_edges(
        X,
        n_msts=1,
        random_state=3,
        neighbors=2,
        inter_component_edges=1,
        max_neighbors=5,
        neighbor_backend="auto",
        neighbor_device="cpu",
    )
    assert calls == ["pynndescent"]


def test_auto_keeps_mps_search_when_rinnd_is_unavailable(monkeypatch):
    X = np.zeros((6, 2), dtype=np.float64)
    complete_edges = {
        (left, right): float(right - left)
        for left in range(len(X))
        for right in range(left + 1, len(X))
    }
    calls = []
    monkeypatch.setattr("imste.mst.famst._rinnd_class", lambda: None)
    monkeypatch.setattr(
        "imste.mst.famst._ann_edges_mps",
        lambda X, neighbors, progress_callback=None: calls.append("mps_exact") or complete_edges,
    )
    iterated_famst_edges(
        X,
        n_msts=1,
        random_state=3,
        neighbors=2,
        inter_component_edges=1,
        max_neighbors=5,
        neighbor_backend="auto",
        neighbor_device="mps",
    )
    assert calls == ["mps_exact"]


def test_auto_uses_pynndescent_for_sparse_input(monkeypatch):
    X = csr_matrix(np.zeros((6, 2), dtype=np.float64))
    complete_edges = {
        (left, right): float(right - left)
        for left in range(X.shape[0])
        for right in range(left + 1, X.shape[0])
    }
    calls = []
    monkeypatch.setattr("imste.mst.famst._rinnd_class", lambda: object)
    monkeypatch.setattr(
        "imste.mst.famst._ann_edges",
        lambda X, neighbors, random_state: calls.append("pynndescent") or complete_edges,
    )
    iterated_famst_edges(
        X,
        n_msts=1,
        random_state=3,
        neighbors=2,
        inter_component_edges=1,
        max_neighbors=5,
        neighbor_backend="auto",
    )
    assert calls == ["pynndescent"]
