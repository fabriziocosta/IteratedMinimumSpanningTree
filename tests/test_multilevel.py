import numpy as np

from imste._multilevel import (
    build_multilevel_hierarchy,
    coarsen_graph,
    expand_coordinates,
    propagate_labels,
    select_seeds,
)


def test_seed_selection_and_synchronous_tie_break_are_deterministic():
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    weights = np.ones(len(edges))
    first = select_seeds(6, edges, weights, target=3)
    np.testing.assert_array_equal(first, select_seeds(6, edges, weights, target=3))
    labels = propagate_labels(6, edges, weights, first)
    np.testing.assert_array_equal(labels, propagate_labels(6, edges, weights, first))
    assert labels.min() == 0 and labels.max() == 2

    weighted_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    weighted_scores = np.array([10.0, 1.0, 1.0, 10.0])
    np.testing.assert_array_equal(
        select_seeds(5, weighted_edges, weighted_scores, target=2), [1, 3]
    )
    tie_labels = propagate_labels(
        3, np.array([[0, 2], [1, 2]]), np.ones(2), np.array([0, 1])
    )
    np.testing.assert_array_equal(tie_labels, [0, 1, 0])


def test_components_stay_separate_and_quotient_weights_sum_crossings():
    # Two weighted triangles. Independent seeds are 0, 3; all crossing edges
    # between their groups should be summed exactly.
    edges = np.array([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]])
    weights = np.array([5.0, 1.0, 2.0, 5.0, 1.0, 2.0])
    labels, quotient_edges, quotient_weights = coarsen_graph(6, edges, weights, 4)
    assert len(np.unique(labels)) == 2
    assert labels[0] != labels[3]
    assert quotient_edges.shape == (0, 2)
    assert quotient_weights.size == 0

    # A connected chain has a single quotient edge with all crossing scores
    # combined, and ratio controls the requested group count.
    chain = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]])
    chain_weights = np.arange(1.0, 9.0)
    labels, quotient_edges, quotient_weights = coarsen_graph(9, chain, chain_weights, 3)
    assert len(np.unique(labels)) == 3
    expected = {}
    for (left, right), weight in zip(chain, chain_weights):
        a, b = sorted((int(labels[left]), int(labels[right])))
        if a != b:
            expected[(a, b)] = expected.get((a, b), 0.0) + weight
    actual = {
        tuple(edge): weight for edge, weight in zip(quotient_edges, quotient_weights)
    }
    assert actual == expected


def test_hierarchy_stops_when_coarsening_cannot_reduce():
    # Three isolated nodes cannot be reduced while preserving components.
    levels = build_multilevel_hierarchy(
        3, np.empty((0, 2), dtype=int), np.empty(0, dtype=int), base_size=1
    )
    assert levels == []


def test_expansion_uses_weighted_external_group_average_and_seeded_jitter():
    labels = np.array([0, 0, 1, 1, 2])
    edges = np.array([[0, 2], [1, 2], [2, 4]])
    weights = np.array([1.0, 3.0, 2.0])
    coarse = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    first = expand_coordinates(labels, edges, weights, coarse, random_state=8)
    second = expand_coordinates(labels, edges, weights, coarse, random_state=8)
    np.testing.assert_array_equal(first, second)
    # Group 0 vertices are placed at group 1; node 2 averages groups 0 and 2
    # with total weights 4 and 2, respectively.
    np.testing.assert_allclose(first[:2], np.broadcast_to(coarse[1], (2, 2)))
    np.testing.assert_allclose(first[2], (4 * coarse[0] + 2 * coarse[2]) / 6)
    # Node 4 has an external neighbor and group 2, while node 3 has none.
    np.testing.assert_allclose(first[4], coarse[1])
    assert not np.array_equal(first[3], coarse[1])
