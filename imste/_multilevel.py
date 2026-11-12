"""Rank-only graph coarsening and coarse-to-fine coordinate expansion."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def _canonical_edges(
    edges: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if edges.shape[0] != weights.size:
        raise ValueError("edges and weights must contain the same number of entries.")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("edge weights must be finite and positive.")
    if not edges.size:
        return edges, weights
    edges = np.sort(edges, axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, weights = edges[order], weights[order]
    starts = np.r_[True, np.any(edges[1:] != edges[:-1], axis=1)]
    positions = np.flatnonzero(starts)
    return edges[positions], np.add.reduceat(weights, positions)


def select_seeds(
    n_nodes: int, edges: np.ndarray, weights: np.ndarray, target: int
) -> np.ndarray:
    """Select deterministic high weighted-degree seeds, preferring an independent set."""
    edges, weights = _canonical_edges(edges, weights)
    degree = np.zeros(n_nodes, dtype=np.float64)
    if edges.size:
        np.add.at(degree, edges[:, 0], weights)
        np.add.at(degree, edges[:, 1], weights)
    order = np.lexsort((np.arange(n_nodes), -degree))
    adjacency = [set() for _ in range(n_nodes)]
    for left, right in edges:
        adjacency[left].add(int(right))
        adjacency[right].add(int(left))
    selected: list[int] = []
    selected_set: set[int] = set()
    for node in order:
        node = int(node)
        if all(neighbor not in selected_set for neighbor in adjacency[node]):
            selected.append(node)
            selected_set.add(node)
            if len(selected) == target:
                break
    if len(selected) < target:
        for node in order:
            node = int(node)
            if node not in selected_set:
                selected.append(node)
                selected_set.add(node)
                if len(selected) == target:
                    break
    return np.asarray(selected, dtype=np.int64)


def propagate_labels(
    n_nodes: int, edges: np.ndarray, weights: np.ndarray, seeds: np.ndarray
) -> np.ndarray:
    """Synchronously assign nodes to neighboring seed labels with deterministic ties."""
    edges, weights = _canonical_edges(edges, weights)
    seeds = np.asarray(seeds, dtype=np.int64)
    labels = np.full(n_nodes, -1, dtype=np.int64)
    labels[seeds] = np.arange(len(seeds), dtype=np.int64)
    if edges.size:
        adjacency = [[] for _ in range(n_nodes)]
        for (left, right), weight in zip(edges, weights):
            adjacency[left].append((int(right), float(weight)))
            adjacency[right].append((int(left), float(weight)))
        while np.any(labels < 0):
            next_labels = labels.copy()
            changed = False
            for node in np.flatnonzero(labels < 0):
                votes: dict[int, float] = {}
                for neighbor, weight in adjacency[node]:
                    label = int(labels[neighbor])
                    if label >= 0:
                        votes[label] = votes.get(label, 0.0) + weight
                if votes:
                    next_labels[node] = min(
                        votes, key=lambda label: (-votes[label], label)
                    )
                    changed = True
            labels = next_labels
            if not changed:
                raise ValueError("Every connected component must contain a seed.")
    return labels


def coarsen_graph(
    n_nodes: int, edges: np.ndarray, weights: np.ndarray, reduction_ratio: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fine-node labels and the weighted quotient graph."""
    edges, weights = _canonical_edges(edges, weights)
    if n_nodes <= 1:
        return np.arange(n_nodes), np.empty((0, 2), dtype=np.int64), np.empty(0)
    rows = np.r_[edges[:, 0], edges[:, 1]]
    cols = np.r_[edges[:, 1], edges[:, 0]]
    adjacency = csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n_nodes, n_nodes))
    n_components, components = connected_components(adjacency, directed=False)
    target = max(n_components, int(np.ceil(n_nodes / reduction_ratio)))
    seeds = select_seeds(n_nodes, edges, weights, target)
    # Guarantee each component has a seed, preserving component separation.
    seeded_components = set(components[seeds].tolist())
    missing = [
        component
        for component in range(n_components)
        if component not in seeded_components
    ]
    if missing:
        degree = np.zeros(n_nodes)
        if edges.size:
            np.add.at(degree, edges[:, 0], weights)
            np.add.at(degree, edges[:, 1], weights)
        for component in missing:
            candidates = np.flatnonzero(components == component)
            seeds = np.r_[
                seeds, candidates[np.lexsort((candidates, -degree[candidates]))[0]]
            ]
    seeds = np.sort(seeds)
    labels = propagate_labels(n_nodes, edges, weights, seeds)
    # Label ids follow sorted seeds, so all ties are stable across runs.
    if edges.size:
        left, right = labels[edges[:, 0]], labels[edges[:, 1]]
        crossing = left != right
        quotient_edges, quotient_weights = _canonical_edges(
            np.column_stack((left[crossing], right[crossing])), weights[crossing]
        )
    else:
        quotient_edges = np.empty((0, 2), dtype=np.int64)
        quotient_weights = np.empty(0, dtype=np.float64)
    return labels, quotient_edges, quotient_weights


def build_multilevel_hierarchy(
    n_nodes: int,
    edges: np.ndarray,
    ranks: np.ndarray,
    *,
    reduction_ratio: int = 4,
    base_size: int = 1024,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build successive levels, each mapping previous nodes to its quotient."""
    current_edges, current_weights = _canonical_edges(
        edges, 1.0 / np.asarray(ranks, dtype=np.float64)
    )
    current_nodes = n_nodes
    levels = []
    while current_nodes > base_size:
        labels, next_edges, next_weights = coarsen_graph(
            current_nodes, current_edges, current_weights, reduction_ratio
        )
        next_nodes = int(labels.max()) + 1 if labels.size else 0
        if next_nodes >= current_nodes:
            break
        levels.append((labels, current_edges, current_weights))
        current_nodes, current_edges, current_weights = (
            next_nodes,
            next_edges,
            next_weights,
        )
    return levels


def expand_coordinates(
    labels: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray,
    coarse_coordinates: np.ndarray,
    *,
    random_state: int | None = 42,
) -> np.ndarray:
    """Expand group coordinates by weighted neighbor averages plus seeded jitter."""
    labels = np.asarray(labels, dtype=np.int64)
    coarse = np.asarray(coarse_coordinates, dtype=np.float64)
    n_components = coarse.shape[1]
    expanded = coarse[labels].copy()
    sums = np.zeros_like(expanded)
    totals = np.zeros(labels.size, dtype=np.float64)
    for (left, right), weight in zip(np.asarray(edges), np.asarray(weights)):
        left, right = int(left), int(right)
        gl, gr = labels[left], labels[right]
        if gl == gr:
            continue
        sums[left] += weight * coarse[gr]
        totals[left] += weight
        sums[right] += weight * coarse[gl]
        totals[right] += weight
    has_neighbors = totals > 0
    expanded[has_neighbors] = sums[has_neighbors] / totals[has_neighbors, None]
    if np.any(~has_neighbors):
        scale = float(np.sqrt(np.mean(np.var(coarse, axis=0)))) if coarse.size else 0.0
        jitter_scale = max(scale * 1e-3, 1e-6)
        rng = np.random.RandomState(random_state)
        expanded[~has_neighbors] += rng.normal(
            scale=jitter_scale, size=(np.count_nonzero(~has_neighbors), n_components)
        )
    return expanded.astype(np.float32)
