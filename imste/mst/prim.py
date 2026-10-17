"""Exact dense Prim construction for iterated Euclidean spanning trees."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics.pairwise import euclidean_distances


def iterated_prim_edges(
    X: object,
    n_msts: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
    *,
    return_ranks: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact trees using dense Prim, reusing bottleneck edges as needed.

    Dense inputs use a retained all-pairs distance matrix. CSR inputs compute
    one distance row at a time to avoid quadratic distance-matrix storage.
    """
    if issparse(X):
        return _iterated_sparse_prim_edges(
            csr_matrix(X), n_msts, progress_callback, return_ranks=return_ranks
        )
    distances = cdist(X, X, metric="euclidean")
    if not np.isfinite(distances).all():
        raise ValueError("Pairwise distances overflowed; rescale X before fitting.")
    np.fill_diagonal(distances, 0.0)
    return iterated_prim_distance_edges(
        distances, n_msts, progress_callback=progress_callback,
        return_ranks=return_ranks,
    )


def _iterated_sparse_prim_edges(
    X,
    n_msts: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
    *,
    return_ranks: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Prim over CSR rows without forming an all-pairs matrix."""
    n_samples = X.shape[0]
    used_neighbors: list[set[int]] = [set() for _ in range(n_samples)]
    edge_last_rank: dict[tuple[int, int], int] = {}
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    ranks: list[int] = []

    def squared_distances_from(sample: int) -> np.ndarray:
        values = euclidean_distances(X[sample], X, squared=True).ravel()
        if not np.isfinite(values).all():
            raise ValueError("Pairwise distances overflowed; rescale X before fitting.")
        return values

    for rank in range(1, n_msts + 1):
        in_tree = np.zeros(n_samples, dtype=bool)
        best = np.full(n_samples, np.inf, dtype=np.float64)
        parent = np.full(n_samples, -1, dtype=np.intp)
        parent_reused = np.zeros(n_samples, dtype=bool)
        best[0] = 0.0

        for _ in range(n_samples):
            candidates = np.where(in_tree, np.inf, best)
            node = int(np.argmin(candidates))
            if not np.isfinite(candidates[node]):
                remaining = np.flatnonzero(~in_tree)
                best_distance = np.inf
                best_left = -1
                best_right = -1
                for left in np.flatnonzero(in_tree):
                    distances = squared_distances_from(int(left))[remaining]
                    offset = int(np.argmin(distances))
                    distance = float(distances[offset])
                    if distance < best_distance:
                        best_distance = distance
                        best_left = int(left)
                        best_right = int(remaining[offset])
                if best_left < 0 or best_right < 0:
                    raise ValueError(f"Could not connect all samples in tree {rank}.")
                node = best_right
                best[node] = best_distance
                parent[node] = best_left
                parent_reused[node] = True

            in_tree[node] = True
            if parent[node] != -1:
                other = int(parent[node])
                edge = (min(other, node), max(other, node))
                edge_rank = edge_last_rank[edge] + 1 if parent_reused[node] else rank
                edge_last_rank[edge] = edge_rank
                edges.append((other, node))
                weights.append(1.0 / edge_rank)
                ranks.append(edge_rank)
                used_neighbors[other].add(node)
                used_neighbors[node].add(other)

            remaining = ~in_tree
            connectable = remaining.copy()
            if used_neighbors[node]:
                connectable[np.fromiter(used_neighbors[node], dtype=np.intp)] = False
            distances = squared_distances_from(node)
            improve = connectable & (distances <= best)
            best[improve] = distances[improve]
            parent[improve] = node
            parent_reused[improve] = False

        if progress_callback is not None:
            progress_callback(rank, n_msts, "Prim")

    edge_array = np.asarray(edges, dtype=np.intp).reshape(-1, 2)
    rank_array = np.asarray(ranks, dtype=np.int32)
    if return_ranks:
        return edge_array, rank_array
    return edge_array, np.asarray(weights, dtype=np.float64)


def iterated_prim_distance_edges(
    distances: np.ndarray,
    n_msts: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
    *,
    return_ranks: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct ranked trees, penalizing an edge each time it is reused."""
    n_samples = distances.shape[0]

    # A packed bit matrix stores which edges remain available after earlier
    # trees are removed. This also preserves legitimate zero-distance edges.
    available = np.full(
        (n_samples, (n_samples + 7) // 8), 0xFF, dtype=np.uint8
    )
    sample_ids = np.arange(n_samples)
    available[sample_ids, sample_ids >> 3] &= np.bitwise_not(
        (1 << (sample_ids & 7)).astype(np.uint8)
    )
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    ranks: list[int] = []
    edge_last_rank: dict[tuple[int, int], int] = {}

    for rank in range(1, n_msts + 1):
        in_tree = np.zeros(n_samples, dtype=bool)
        best = np.full(n_samples, np.inf, dtype=np.float64)
        parent = np.full(n_samples, -1, dtype=np.intp)
        parent_reused = np.zeros(n_samples, dtype=bool)
        best[0] = 0.0

        for _ in range(n_samples):
            candidates = np.where(in_tree, np.inf, best)
            node = int(np.argmin(candidates))
            if not np.isfinite(candidates[node]):
                # The unused graph cannot cross this cut. Reuse its shortest
                # complete-graph edge and increment that edge's penalty rank.
                remaining = np.flatnonzero(~in_tree)
                best_distance = np.inf
                best_left = -1
                best_right = -1
                for left in np.flatnonzero(in_tree):
                    distances_to_remaining = distances[left, remaining]
                    offset = int(np.argmin(distances_to_remaining))
                    distance = float(distances_to_remaining[offset])
                    if distance < best_distance:
                        best_distance = distance
                        best_left = int(left)
                        best_right = int(remaining[offset])
                if best_left < 0 or best_right < 0:
                    raise ValueError(f"Could not connect all samples in tree {rank}.")
                node = best_right
                best[node] = best_distance
                parent[node] = best_left
                parent_reused[node] = True

            in_tree[node] = True
            if parent[node] != -1:
                other = int(parent[node])
                edge = (min(other, node), max(other, node))
                if parent_reused[node]:
                    edge_rank = edge_last_rank[edge] + 1
                else:
                    edge_rank = rank
                edge_last_rank[edge] = edge_rank
                edges.append((other, node))
                weights.append(1.0 / edge_rank)
                ranks.append(edge_rank)
                available[other, node >> 3] &= np.uint8(
                    0xFF ^ (1 << (node & 7))
                )
                available[node, other >> 3] &= np.uint8(
                    0xFF ^ (1 << (other & 7))
                )

            remaining = ~in_tree
            available_row = np.unpackbits(
                available[node], bitorder="little", count=n_samples
            ).astype(bool)
            connectable = remaining & available_row
            improve = connectable & (distances[node] <= best)
            best[improve] = distances[node, improve]
            parent[improve] = node
            parent_reused[improve] = False
        if progress_callback is not None:
            progress_callback(rank, n_msts, "Prim")

    edge_array = np.asarray(edges, dtype=np.intp).reshape(-1, 2)
    rank_array = np.asarray(ranks, dtype=np.int32)
    if return_ranks:
        return edge_array, rank_array
    return edge_array, np.asarray(weights, dtype=np.float64)
