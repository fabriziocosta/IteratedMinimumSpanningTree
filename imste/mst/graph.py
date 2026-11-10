"""Iterated minimum spanning forests over a supplied sparse distance graph."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def iterated_graph_edges(
    distance_graph: sparse.spmatrix,
    n_msts: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
    *,
    return_ranks: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ranked spanning forests using only explicitly stored graph edges.

    Distances determine edge ordering. Each iteration first consumes unused
    edges in distance order, then reuses the shortest previously used edges
    needed to connect each original graph component. Reuse increments the
    edge rank, matching the penalty used by the feature-based IMSTE builders.

    Directed entries are treated as undirected candidates. If both directions
    exist, the smaller stored distance is used. Explicitly stored zero
    distances are retained; implicit sparse zeros are absent edges.
    """
    if not sparse.issparse(distance_graph):
        raise ValueError("distance_graph must be a SciPy sparse matrix.")
    graph = distance_graph.tocoo(copy=True)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("distance_graph must be square.")
    if graph.shape[0] < 2:
        raise ValueError("distance_graph must contain at least two samples.")
    n_msts = int(n_msts)
    if n_msts < 1:
        raise ValueError("n_msts must be at least 1.")

    values = np.asarray(graph.data, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Stored distance_graph edge values must be finite and non-negative.")

    # Collapse duplicate and reciprocal entries into one undirected edge with
    # the minimum observed distance. Ignore self loops for tree construction.
    distances: dict[tuple[int, int], float] = {}
    for row, col, distance in zip(graph.row, graph.col, values):
        left, right = sorted((int(row), int(col)))
        if left == right:
            continue
        edge = (left, right)
        distances[edge] = min(float(distance), distances.get(edge, np.inf))

    n_samples = graph.shape[0]
    if not distances:
        raise ValueError("distance_graph must contain at least one off-diagonal edge.")

    edges = np.asarray(list(distances), dtype=np.intp).reshape(-1, 2)
    costs = np.asarray([distances[tuple(edge)] for edge in edges], dtype=np.float64)
    order = np.lexsort((edges[:, 1], edges[:, 0], costs))
    edges = edges[order]
    costs = costs[order]

    # Connectivity is based on edge presence, including explicitly stored
    # zero-distance edges.
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    topology = sparse.csr_matrix(
        (np.ones(rows.size, dtype=np.uint8), (rows, cols)),
        shape=(n_samples, n_samples),
    )
    n_components, labels = connected_components(topology, directed=False)

    edge_ranks: dict[tuple[int, int], int] = {}
    result_edges: list[tuple[int, int]] = []
    result_ranks: list[int] = []

    for tree_index in range(1, n_msts + 1):
        parent = np.arange(n_samples, dtype=np.intp)

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = int(parent[node])
            return node

        selected = 0
        for reused_pass in (False, True):
            for edge in edges:
                left, right = int(edge[0]), int(edge[1])
                edge_key = (left, right)
                was_used = edge_key in edge_ranks
                if was_used != reused_pass or labels[left] != labels[right]:
                    continue
                left_root, right_root = find(left), find(right)
                if left_root == right_root:
                    continue
                parent[right_root] = left_root
                rank = edge_ranks[edge_key] + 1 if was_used else tree_index
                edge_ranks[edge_key] = rank
                result_edges.append((left, right))
                result_ranks.append(rank)
                selected += 1

        expected = n_samples - n_components
        if selected != expected:
            raise ValueError("Could not construct a spanning forest from distance_graph.")
        if progress_callback is not None:
            progress_callback(tree_index, n_msts, f"Graph forest ({n_components} components)")

    edge_array = np.asarray(result_edges, dtype=np.intp).reshape(-1, 2)
    rank_array = np.asarray(result_ranks, dtype=np.int32)
    if return_ranks:
        return edge_array, rank_array
    return edge_array, 1.0 / rank_array.astype(np.float64)
