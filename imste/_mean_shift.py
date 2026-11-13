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
        n_neighbors=min(k + 1, len(coordinates)), algorithm="kd_tree", n_jobs=-1
    )
    search.fit(coordinates)
    indices = search.kneighbors(coordinates, return_distance=False)
    result = np.empty((len(coordinates), k), dtype=np.int64)
    for row, row_indices in enumerate(indices):
        result[row] = row_indices[row_indices != row][:k]
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

    coordinates = np.asarray(embedding, dtype=np.float64).copy()
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

    for _ in range(int(iterations)):
        offsets = coordinates[graph_indices] - coordinates[:, None, :]
        distances = np.linalg.norm(offsets, axis=2)
        bandwidth = max(
            float(np.median(distances[:, -1])) * float(bandwidth_factor), 1e-12
        )
        weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
        local_means = (coordinates[graph_indices] * weights[:, :, None]).sum(axis=1)
        local_means /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        coordinates += float(step_size) * (local_means - coordinates)
        coordinates -= coordinates.mean(axis=0)

    return coordinates.astype(np.float32), {"iterations": int(iterations)}
