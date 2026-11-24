"""Gaussian-kernel mean-shift refinement for 2D embeddings."""

from __future__ import annotations

import numpy as np


def embedding_neighbor_indices(embedding: object, neighbors: int) -> np.ndarray:
    """Find fixed nearest-neighbor identities in an embedding's starting layout."""
    from sklearn.neighbors import NearestNeighbors

    coordinates = np.asarray(embedding, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 2:
        raise ValueError("embedding must have shape (n_samples >= 2, 2).")
    if not np.isfinite(coordinates).all():
        raise ValueError("embedding must contain only finite coordinates.")
    if isinstance(neighbors, (bool, np.bool_)) or not isinstance(
        neighbors, (int, np.integer)
    ) or not 1 <= int(neighbors) < len(coordinates):
        raise ValueError("neighbors must be in [1, n_samples).")

    k = int(neighbors)
    search = NearestNeighbors(
        n_neighbors=k, algorithm="kd_tree", n_jobs=-1
    )
    search.fit(coordinates)
    # Omitting X asks sklearn to query its indexed rows and exclude each
    # query point itself, avoiding a k+1 result and per-row cleanup.
    return search.kneighbors(return_distance=False)


def _mean_shift_iteration(
    coordinates: np.ndarray,
    graph_indices: np.ndarray,
    bandwidth: float,
    step_size: float,
) -> np.ndarray:
    """Compute one synchronous, allocation-light mean-shift iteration."""
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def update(current, neighbors, sigma, step):
        n_samples = current.shape[0]
        neighbor_count = neighbors.shape[1]
        output = np.empty_like(current)
        denominator = 2.0 * sigma * sigma
        for row in prange(n_samples):
            weight_sum = 0.0
            mean_x = 0.0
            mean_y = 0.0
            for column in range(neighbor_count):
                neighbor = neighbors[row, column]
                dx = current[neighbor, 0] - current[row, 0]
                dy = current[neighbor, 1] - current[row, 1]
                distance_squared = dx * dx + dy * dy
                weight = np.exp(-distance_squared / denominator)
                weight_sum += weight
                mean_x += current[neighbor, 0] * weight
                mean_y += current[neighbor, 1] * weight
            output[row, 0] = current[row, 0] + step * (
                mean_x / max(weight_sum, 1e-12) - current[row, 0]
            )
            output[row, 1] = current[row, 1] + step * (
                mean_y / max(weight_sum, 1e-12) - current[row, 1]
            )
        return output

    # float32 halves memory traffic and matches the public embedding dtype.
    result = update(
        np.ascontiguousarray(coordinates, dtype=np.float32),
        np.ascontiguousarray(graph_indices, dtype=np.int64),
        float(bandwidth),
        float(step_size),
    )
    result -= result.mean(axis=0, dtype=np.float64).astype(np.float32)
    return result


def gaussian_mean_shift_layout(
    embedding: object,
    *,
    neighbor_indices: object,
    neighbors: int = 50,
    bandwidth_factor: float = 1.0,
    step_size: float = 0.25,
    iterations: int = 0,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Shift coordinates using fixed 2D neighbors and refreshed 2D distances."""

    coordinates = np.asarray(embedding, dtype=np.float32).copy()
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 2:
        raise ValueError("embedding must have shape (n_samples >= 2, 2).")
    if not np.isfinite(coordinates).all():
        raise ValueError("embedding must contain only finite coordinates.")
    if isinstance(neighbors, (bool, np.bool_)) or not isinstance(
        neighbors, (int, np.integer)
    ) or not 1 <= int(neighbors) < len(coordinates):
        raise ValueError("neighbors must be in [1, n_samples).")
    if not np.isfinite(bandwidth_factor) or bandwidth_factor <= 0:
        raise ValueError("bandwidth_factor must be finite and positive.")
    if not np.isfinite(step_size) or not 0.0 <= step_size <= 1.0:
        raise ValueError("step_size must be in [0, 1].")
    if isinstance(iterations, (bool, np.bool_)) or not isinstance(
        iterations, (int, np.integer)
    ) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer.")

    graph_indices = np.asarray(neighbor_indices, dtype=np.int64)
    if graph_indices.ndim != 2 or graph_indices.shape[0] != len(coordinates):
        raise ValueError("neighbor_indices must have one row per sample.")
    if graph_indices.shape[1] < neighbors:
        raise ValueError("neighbor_indices is narrower than neighbors.")
    graph_indices = graph_indices[:, : int(neighbors)]
    if np.any(graph_indices < 0) or np.any(graph_indices >= len(coordinates)):
        raise ValueError("neighbor_indices contains invalid indices.")

    if iterations == 0 or step_size == 0:
        return coordinates.astype(np.float32), {"iterations": 0}

    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def kth_neighbor_distances(current, neighbors):
        output = np.empty(current.shape[0], dtype=np.float32)
        last_column = neighbors.shape[1] - 1
        for row in prange(current.shape[0]):
            neighbor = neighbors[row, last_column]
            dx = current[neighbor, 0] - current[row, 0]
            dy = current[neighbor, 1] - current[row, 1]
            output[row] = np.sqrt(dx * dx + dy * dy)
        return output

    for _ in range(int(iterations)):
        distances = kth_neighbor_distances(coordinates, graph_indices)
        bandwidth = max(
            float(np.median(distances)) * float(bandwidth_factor), 1e-12
        )
        coordinates = _mean_shift_iteration(
            coordinates, graph_indices, bandwidth, float(step_size)
        )

    return coordinates.astype(np.float32), {"iterations": int(iterations)}
