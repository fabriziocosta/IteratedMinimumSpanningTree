"""Sparse FAMST-style approximate Euclidean MST construction.

This follows the FAMST paper's ANN graph, component-connection, local
inter-component refinement, then Kruskal pipeline. Unlike the paper's single
tree output, this implementation extracts edge-disjoint trees from the sparse
candidate graph and increases the ANN neighborhood size when necessary.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numba import njit


ProgressCallback = Callable[[int, int, str], None]


@njit(cache=True)
def _component_roots(edges: np.ndarray, n_samples: int) -> np.ndarray:
    parent = np.arange(n_samples, dtype=np.int64)
    sizes = np.ones(n_samples, dtype=np.int64)
    for edge_index in range(len(edges)):
        left = edges[edge_index, 0]
        right = edges[edge_index, 1]
        left_root = left
        while parent[left_root] != left_root:
            left_root = parent[left_root]
        while parent[left] != left:
            next_left = parent[left]
            parent[left] = left_root
            left = next_left
        right_root = right
        while parent[right_root] != right_root:
            right_root = parent[right_root]
        while parent[right] != right:
            next_right = parent[right]
            parent[right] = right_root
            right = next_right
        if left_root == right_root:
            continue
        if sizes[left_root] < sizes[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        sizes[left_root] += sizes[right_root]
    for sample in range(n_samples):
        root = sample
        while parent[root] != root:
            root = parent[root]
        parent[sample] = root
    return parent


def _components(n_samples: int, edges: dict[tuple[int, int], float]):
    edge_array = np.asarray(list(edges), dtype=np.int64).reshape(-1, 2)
    roots = _component_roots(edge_array, n_samples)
    _, labels = np.unique(roots, return_inverse=True)
    group_lists: list[list[int]] = [[] for _ in range(int(labels.max()) + 1)]
    for point, label in enumerate(labels):
        group_lists[int(label)].append(point)
    groups = [np.asarray(group, dtype=np.intp) for group in group_lists]
    return labels, groups


def _distance(X: np.ndarray, left: int, right: int) -> float:
    delta = X[left] - X[right]
    value = float(np.sqrt(np.dot(delta, delta)))
    if not np.isfinite(value):
        raise ValueError("Pairwise distances overflowed; rescale X before fitting.")
    return value


@njit(cache=True)
def _undirected_csr(
    edges: np.ndarray, n_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    degree = np.zeros(n_samples, dtype=np.int64)
    for edge_index in range(len(edges)):
        degree[edges[edge_index, 0]] += 1
        degree[edges[edge_index, 1]] += 1
    indptr = np.empty(n_samples + 1, dtype=np.int64)
    indptr[0] = 0
    indptr[1:] = np.cumsum(degree)
    neighbors = np.empty(indptr[-1], dtype=np.int64)
    cursor = indptr[:-1].copy()
    for edge_index in range(len(edges)):
        left = edges[edge_index, 0]
        right = edges[edge_index, 1]
        neighbors[cursor[left]] = right
        cursor[left] += 1
        neighbors[cursor[right]] = left
        cursor[right] += 1
    return indptr, neighbors


@njit(cache=True)
def _squared_distance(X: np.ndarray, left: int, right: int) -> float:
    distance = 0.0
    for feature in range(X.shape[1]):
        delta = X[left, feature] - X[right, feature]
        distance += delta * delta
    return distance


@njit(cache=True)
def _refine_bridges(
    X: np.ndarray,
    labels: np.ndarray,
    indptr: np.ndarray,
    neighbors: np.ndarray,
    bridges: np.ndarray,
) -> np.ndarray:
    changed = True
    while changed:
        changed = False
        refined = np.empty_like(bridges)
        for bridge_index in range(len(bridges)):
            left = bridges[bridge_index, 0]
            right = bridges[bridge_index, 1]
            left_component = bridges[bridge_index, 2]
            right_component = bridges[bridge_index, 3]
            best_left = left
            best_right = right
            best_distance = _squared_distance(X, left, right)
            for offset in range(indptr[left], indptr[left + 1]):
                candidate_left = neighbors[offset]
                if labels[candidate_left] != left_component:
                    continue
                candidate_distance = _squared_distance(X, candidate_left, best_right)
                if candidate_distance < best_distance:
                    best_left = candidate_left
                    best_distance = candidate_distance
            for offset in range(indptr[right], indptr[right + 1]):
                candidate_right = neighbors[offset]
                if labels[candidate_right] != right_component:
                    continue
                candidate_distance = _squared_distance(X, best_left, candidate_right)
                if candidate_distance < best_distance:
                    best_right = candidate_right
                    best_distance = candidate_distance
            if best_left != left or best_right != right:
                changed = True
            refined[bridge_index, 0] = best_left
            refined[bridge_index, 1] = best_right
            refined[bridge_index, 2] = left_component
            refined[bridge_index, 3] = right_component
        bridges = refined
    return bridges


def _sample_component_representatives(
    groups: list[np.ndarray],
    max_per_component: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a bounded, reproducible set of points from each component."""
    representative_points: list[np.ndarray] = []
    representative_components: list[np.ndarray] = []
    for component, points in enumerate(groups):
        if len(points) > max_per_component:
            selected = np.sort(
                rng.choice(points, size=max_per_component, replace=False)
            )
        else:
            selected = points
        representative_points.append(selected)
        representative_components.append(
            np.full(len(selected), component, dtype=np.intp)
        )
    return (
        np.concatenate(representative_points),
        np.concatenate(representative_components),
    )


@njit(cache=True)
def _representative_prim(
    X: np.ndarray, points: np.ndarray, components: np.ndarray
) -> np.ndarray:
    n_representatives = len(points)
    selected = np.zeros(n_representatives, dtype=np.uint8)
    best_distances = np.full(n_representatives, np.inf, dtype=np.float64)
    parents = np.full(n_representatives, -1, dtype=np.int64)
    best_distances[0] = 0.0
    tree_edges = np.empty((max(0, n_representatives - 1), 2), dtype=np.int64)
    edge_count = 0
    for _ in range(n_representatives):
        current = -1
        current_distance = np.inf
        for candidate in range(n_representatives):
            if selected[candidate] == 0 and best_distances[candidate] < current_distance:
                current = candidate
                current_distance = best_distances[candidate]
        if current < 0 or not np.isfinite(current_distance):
            return tree_edges[:0]
        selected[current] = 1
        parent = parents[current]
        if parent >= 0:
            tree_edges[edge_count, 0] = components[current]
            tree_edges[edge_count, 1] = components[parent]
            edge_count += 1
        for candidate in range(n_representatives):
            if selected[candidate] != 0:
                continue
            distance_squared = 0.0
            for feature in range(X.shape[1]):
                delta = X[points[current], feature] - X[points[candidate], feature]
                distance_squared += delta * delta
            if distance_squared < best_distances[candidate]:
                best_distances[candidate] = distance_squared
                parents[candidate] = current
    return tree_edges[:edge_count]


def _representative_component_pairs(
    X: np.ndarray,
    groups: list[np.ndarray],
    max_per_component: int,
    rng: np.random.RandomState,
) -> list[tuple[int, int]]:
    """Return unique component pairs crossed by the representative MST.

    Prim's algorithm computes the exact Euclidean MST in quadratic time and
    linear auxiliary memory, without constructing a dense distance matrix.
    """
    points, components = _sample_component_representatives(
        groups, max_per_component, rng
    )
    n_representatives = len(points)
    component_pairs: set[tuple[int, int]] = set()
    tree_edges = _representative_prim(X, points, components)
    if n_representatives > 1 and len(tree_edges) != n_representatives - 1:
        raise ValueError("Could not construct an MST over component representatives.")
    for left_value, right_value in tree_edges:
        left, right = int(left_value), int(right_value)
        if left != right:
            component_pairs.add((min(left, right), max(left, right)))

    return sorted(component_pairs)


def _ann_edges(
    X: np.ndarray,
    neighbors: int,
    random_state: int | None,
) -> dict[tuple[int, int], float]:
    try:
        from pynndescent import NNDescent
    except ImportError as exc:
        raise ImportError(
            "mst_method='famst' requires the optional dependency. Install it with "
            "`python -m pip install 'imste[famst]'`."
        ) from exc

    index = NNDescent(
        X,
        n_neighbors=min(neighbors + 1, len(X)),
        metric="euclidean",
        random_state=random_state,
        n_jobs=-1,
    )
    indices, distances = index.neighbor_graph
    return _neighbor_edges_to_dict(indices, distances)


def _ann_edges_mps(X: np.ndarray, neighbors: int) -> dict[tuple[int, int], float]:
    """Build an exact k-neighbor candidate graph with chunked MPS distances."""
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS neighbor search requires an available Apple GPU.")

    device = torch.device("mps")
    features = torch.as_tensor(X, dtype=torch.float32, device=device)
    n_samples = len(X)
    norms = torch.sum(features * features, dim=1)
    chunk_size = min(512, n_samples)
    all_indices: list[np.ndarray] = []
    all_distances: list[np.ndarray] = []

    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        rows = torch.arange(start, stop, device=device)
        query = features[start:stop]
        distances_squared = (
            norms[start:stop, None]
            + norms[None, :]
            - 2.0 * (query @ features.T)
        ).clamp_min_(0.0)
        distances_squared[torch.arange(stop - start, device=device), rows] = torch.inf
        nearest_squared, nearest_indices = torch.topk(
            distances_squared,
            k=min(neighbors, n_samples - 1),
            dim=1,
            largest=False,
            sorted=False,
        )
        all_indices.append(nearest_indices.cpu().numpy())
        all_distances.append(torch.sqrt(nearest_squared).cpu().numpy())

    return _neighbor_edges_to_dict(
        np.concatenate(all_indices, axis=0),
        np.concatenate(all_distances, axis=0),
    )


def _neighbor_edges_to_dict(
    indices: np.ndarray, distances: np.ndarray
) -> dict[tuple[int, int], float]:
    """Make a deduplicated undirected weighted graph from a neighbor table."""
    n_samples = indices.shape[0]
    left = np.repeat(np.arange(n_samples, dtype=np.int32), indices.shape[1])
    right = indices.reshape(-1).astype(np.int32, copy=False)
    distance = distances.reshape(-1)
    valid = (right >= 0) & (right != left) & np.isfinite(distance)
    left = left[valid]
    right = right[valid]
    distance = distance[valid]
    lower = np.minimum(left, right)
    upper = np.maximum(left, right)
    keys = lower.astype(np.int64) * n_samples + upper
    if not keys.size:
        return {}
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    distance = distance[order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    unique_keys = keys[starts]
    minimum_distances = np.minimum.reduceat(distance, starts)
    return {
        (int(key // n_samples), int(key % n_samples)): float(value)
        for key, value in zip(unique_keys, minimum_distances)
    }


def _connect_components(
    X: np.ndarray,
    edges: dict[tuple[int, int], float],
    n_candidates: int,
    rng: np.random.RandomState,
    representatives_per_component: int = 10,
) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    """Add randomized inter-component candidates and refine locally.

    Each candidate stores (left, right, left_component, right_component).
    """
    labels, groups = _components(len(X), edges)
    if len(groups) <= 1:
        return [], labels

    bridge_rows: list[tuple[int, int, int, int]] = []
    component_pairs = _representative_component_pairs(
        X, groups, representatives_per_component, rng
    )
    for left_component, right_component in component_pairs:
        left_points = groups[left_component]
        right_points = groups[right_component]
        draw_count = n_candidates * n_candidates
        left_draws = np.asarray(rng.choice(left_points, size=draw_count), dtype=np.int64)
        right_draws = np.asarray(rng.choice(right_points, size=draw_count), dtype=np.int64)
        candidate_keys = left_draws * len(X) + right_draws
        _, first_draws = np.unique(candidate_keys, return_index=True)
        first_draws.sort()
        left_unique = left_draws[first_draws]
        right_unique = right_draws[first_draws]
        deltas = X[left_unique] - X[right_unique]
        distance_squared = np.einsum("ij,ij->i", deltas, deltas)
        best_indices = np.argsort(distance_squared, kind="stable")[:n_candidates]
        bridge_rows.extend(
            (int(left_unique[index]), int(right_unique[index]), left_component, right_component)
            for index in best_indices
        )

    bridges = np.asarray(bridge_rows, dtype=np.int64).reshape(-1, 4)
    if len(bridges):
        edge_pairs = np.asarray(list(edges), dtype=np.int64).reshape(-1, 2)
        indptr, neighbors = _undirected_csr(edge_pairs, len(X))
        bridges = _refine_bridges(X, labels, indptr, neighbors, bridges)

    bridge_result = []
    for left_value, right_value, left_component, right_component in bridges:
        left, right = int(left_value), int(right_value)
        edge = (left, right) if left < right else (right, left)
        edges.setdefault(edge, _distance(X, *edge))
        bridge_result.append((left, right, int(left_component), int(right_component)))
    return bridge_result, labels


@njit(cache=True)
def _find_root(parent: np.ndarray, item: int) -> int:
    root = item
    while parent[root] != root:
        root = parent[root]
    while parent[item] != item:
        next_item = parent[item]
        parent[item] = root
        item = next_item
    return root


@njit(cache=True)
def _union_roots(parent: np.ndarray, sizes: np.ndarray, left: int, right: int) -> bool:
    left_root = _find_root(parent, left)
    right_root = _find_root(parent, right)
    if left_root == right_root:
        return False
    if sizes[left_root] < sizes[right_root]:
        left_root, right_root = right_root, left_root
    parent[right_root] = left_root
    sizes[left_root] += sizes[right_root]
    return True


@njit(cache=True)
def _extract_ranked_msts(
    src: np.ndarray,
    dst: np.ndarray,
    order: np.ndarray,
    n_samples: int,
    n_msts: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    active = np.ones(len(src), dtype=np.uint8)
    result_edges = np.empty((n_msts * max(0, n_samples - 1), 2), dtype=np.int64)
    result_ranks = np.empty(n_msts * max(0, n_samples - 1), dtype=np.int32)
    output = 0
    completed = 0
    for rank in range(1, n_msts + 1):
        parent = np.arange(n_samples, dtype=np.int64)
        sizes = np.ones(n_samples, dtype=np.int64)
        count = 0
        for order_position in range(len(order)):
            edge_index = order[order_position]
            if active[edge_index] == 0:
                continue
            left = src[edge_index]
            right = dst[edge_index]
            if _union_roots(parent, sizes, left, right):
                result_edges[output, 0] = left
                result_edges[output, 1] = right
                result_ranks[output] = rank
                output += 1
                count += 1
                active[edge_index] = 0
                if count == n_samples - 1:
                    break
        if count != n_samples - 1:
            return result_edges[:output], result_ranks[:output], completed
        completed += 1
    return result_edges[:output], result_ranks[:output], completed


def _kruskal_edge_disjoint_trees(
    n_samples: int,
    candidate_edges: dict[tuple[int, int], float],
    n_msts: int,
    progress_callback: ProgressCallback | None = None,
    progress_detail: str = "FAMST",
) -> tuple[np.ndarray, np.ndarray] | None:
    if not candidate_edges:
        return None
    pairs = np.asarray(list(candidate_edges), dtype=np.int64)
    distances = np.fromiter(
        candidate_edges.values(), dtype=np.float64, count=len(candidate_edges)
    )
    finite = np.isfinite(distances)
    pairs = pairs[finite]
    distances = distances[finite]
    order = np.argsort(distances, kind="stable")
    result_edges, result_ranks, completed = _extract_ranked_msts(
        pairs[:, 0], pairs[:, 1], order.astype(np.int64), n_samples, n_msts
    )
    if completed != n_msts:
        return None
    for rank in range(1, n_msts + 1):
        if progress_callback is not None:
            progress_callback(rank, n_msts, progress_detail)
    return result_edges.astype(np.intp), result_ranks


def iterated_famst_edges(
    X: np.ndarray,
    n_msts: int,
    random_state: int | None,
    neighbors: int,
    inter_component_edges: int,
    max_neighbors: int | None,
    progress_callback: ProgressCallback | None = None,
    representatives_per_component: int = 10,
    *,
    neighbor_device: str = "cpu",
    return_ranks: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build edge-disjoint trees from an adaptive sparse neighbor graph."""
    n_samples = len(X)
    if n_samples <= 3:
        from .prim import iterated_prim_edges

        return iterated_prim_edges(
            X, n_msts, progress_callback=progress_callback,
            return_ranks=return_ranks,
        )

    rng = np.random.RandomState(random_state)
    configured_start_neighbors = min(neighbors, n_samples - 1)
    start_neighbors = min(
        max(configured_start_neighbors, 2 * n_msts), n_samples - 1
    )
    cap = min(
        n_samples - 1,
        max_neighbors
        if max_neighbors is not None
        else max(start_neighbors, 4 * start_neighbors, 4 * n_msts),
    )
    if cap < configured_start_neighbors:
        raise ValueError("mst_max_neighbors must be at least mst_neighbors.")
    start_neighbors = min(start_neighbors, cap)

    current_neighbors = start_neighbors
    while True:
        if current_neighbors >= n_samples - 1:
            from .prim import iterated_prim_edges

            return iterated_prim_edges(
                X, n_msts, progress_callback=progress_callback,
                return_ranks=return_ranks,
            )
        detail = f"FAMST ({current_neighbors} neighbors)"
        if progress_callback is not None:
            progress_callback(0, n_msts, detail)
        seed = int(rng.randint(0, np.iinfo(np.int32).max))
        if neighbor_device == "mps":
            candidate_edges = _ann_edges_mps(X, current_neighbors)
        else:
            candidate_edges = _ann_edges(X, current_neighbors, seed)
        _connect_components(
            X,
            candidate_edges,
            inter_component_edges,
            rng,
            representatives_per_component,
        )
        trees = _kruskal_edge_disjoint_trees(
            n_samples,
            candidate_edges,
            n_msts,
            progress_callback=progress_callback,
            progress_detail=detail,
        )
        if trees is not None:
            edges, ranks = trees
            if return_ranks:
                return edges, ranks
            return edges, 1.0 / ranks.astype(np.float64)
        if current_neighbors >= cap:
            raise ValueError(
                f"FAMST candidate graph could not supply {n_msts} edge-disjoint "
                f"spanning trees with up to {current_neighbors} neighbors. "
                "Increase mst_max_neighbors or reduce n_msts."
            )
        # A failed candidate graph means another full NNDescent build is
        # required. When the cap is close, skip intermediate rebuilds; for a
        # much larger cap, grow geometrically to avoid an oversized graph.
        if cap <= current_neighbors * 8:
            current_neighbors = cap
        else:
            current_neighbors = min(
                cap, max(current_neighbors + 1, current_neighbors * 2)
            )
