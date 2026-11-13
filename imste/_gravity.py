"""Density-aware second-phase layout refinements for IMSTE embeddings."""

from __future__ import annotations

import time

import numpy as np


def apply_local_mexican_hat_force(
    embedding: object,
    point_mass: object,
    *,
    attraction_strength: float = 0.5,
    near_repulsion_strength: float = 2.5,
    attraction_range: float = 1.0,
    repulsion_range: float = 0.5,
    neighbors: int = 8,
    step_size: float = 0.05,
    iterations: int = 3,
    mass_power: float = 0.5,
    n_jobs: int = -1,
) -> np.ndarray:
    """Apply locally scaled Mexican-hat forces to 2D coordinates."""
    from sklearn.neighbors import NearestNeighbors

    coordinates = np.asarray(embedding, dtype=np.float64).copy()
    point_mass = np.asarray(point_mass, dtype=np.float64).reshape(-1)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 3:
        raise ValueError("embedding must have shape (n_samples >= 3, 2).")
    if point_mass.shape != (len(coordinates),):
        raise ValueError("point_mass must contain one value per embedded point.")
    if not np.isfinite(coordinates).all():
        raise ValueError("embedding must contain only finite coordinates.")
    if not np.isfinite(point_mass).all() or np.any(point_mass <= 0):
        raise ValueError("point_mass values must be finite and positive.")
    if not 1 <= int(neighbors) < len(coordinates):
        raise ValueError("neighbors must be in [1, n_samples).")
    if int(iterations) < 0:
        raise ValueError("iterations must be nonnegative.")
    if repulsion_range <= 0 or attraction_range <= repulsion_range:
        raise ValueError("attraction_range must exceed positive repulsion_range.")

    relative_mass = np.clip(point_mass / np.median(point_mass), 0.1, 10.0) ** mass_power
    for iteration in range(int(iterations)):
        index = NearestNeighbors(
            n_neighbors=int(neighbors) + 1,
            algorithm="kd_tree",
            n_jobs=n_jobs,
        ).fit(coordinates)
        distances, indices = index.kneighbors(coordinates, return_distance=True)
        keep = indices != np.arange(len(coordinates), dtype=np.int64)[:, None]
        distances = distances[keep].reshape(len(coordinates), int(neighbors))
        indices = indices[keep].reshape(len(coordinates), int(neighbors))
        local_scale = np.maximum(distances[:, -1], np.finfo(np.float64).eps)
        pair_scale = np.sqrt(local_scale[:, None] * local_scale[indices])
        sigma_attract = np.maximum(attraction_range * pair_scale, 1e-12)
        sigma_repel = np.maximum(repulsion_range * pair_scale, 1e-12)
        attraction = attraction_strength * np.exp(-0.5 * (distances / sigma_attract) ** 2)
        repulsion = near_repulsion_strength * np.exp(-0.5 * (distances / sigma_repel) ** 2)
        interaction = attraction - repulsion
        direction = (coordinates[indices] - coordinates[:, None, :]) / np.maximum(
            distances[:, :, None], 1e-12
        )
        coincident = distances <= 1e-12
        if np.any(coincident):
            source_rows = np.broadcast_to(
                np.arange(len(coordinates), dtype=np.int64)[:, None], indices.shape
            )
            lower = np.minimum(source_rows, indices)
            upper = np.maximum(source_rows, indices)
            pair_code = (lower * 73_856_093) ^ (upper * 19_349_663) ^ iteration
            angle = (pair_code % 1_000_003) * (2.0 * np.pi / 1_000_003)
            pair_direction = np.stack((np.cos(angle), np.sin(angle)), axis=2)
            pair_direction *= np.where(source_rows < indices, 1.0, -1.0)[:, :, None]
            direction[coincident] = pair_direction[coincident]
        neighbor_mass = relative_mass[indices]
        weighted_force = interaction[:, :, None] * direction * neighbor_mass[:, :, None]
        force = weighted_force.sum(axis=1) / np.maximum(
            (np.abs(interaction) * neighbor_mass).sum(axis=1, keepdims=True), 1e-12
        )
        movement = step_size * local_scale[:, None] * force
        movement_norm = np.linalg.norm(movement, axis=1)
        max_movement = step_size * local_scale
        shrink = np.minimum(1.0, max_movement / np.maximum(movement_norm, 1e-12))
        coordinates += movement * shrink[:, None]
        coordinates -= coordinates.mean(axis=0)
    return coordinates.astype(np.float32)


def pack_region_centers_on_ring(
    embedding: object,
    membership: object,
    *,
    radius_quantile: float = 0.9,
    gap_fraction: float = 0.25,
) -> np.ndarray:
    """Translate regions onto a compact ring while preserving within-region shape."""
    coordinates = np.asarray(embedding, dtype=np.float64)
    membership = np.asarray(membership, dtype=np.int64).reshape(-1)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("embedding must have shape (n_samples, 2).")
    if membership.shape != (len(coordinates),):
        raise ValueError("membership must contain one region id per embedded point.")
    if not np.isfinite(coordinates).all():
        raise ValueError("embedding must contain only finite coordinates.")
    if not np.isfinite(radius_quantile) or not 0.0 < radius_quantile < 1.0:
        raise ValueError("radius_quantile must be between 0 and 1, exclusive.")
    if not np.isfinite(gap_fraction) or gap_fraction < 0.0:
        raise ValueError("gap_fraction must be finite and nonnegative.")

    _, membership = np.unique(membership, return_inverse=True)
    region_count = int(membership.max()) + 1 if len(membership) else 0
    if region_count < 2:
        return coordinates.astype(np.float32)

    counts = np.bincount(membership, minlength=region_count)
    centers = np.zeros((region_count, 2), dtype=np.float64)
    np.add.at(centers, membership, coordinates)
    centers /= counts[:, None]
    radii = np.empty(region_count, dtype=np.float64)
    for region in range(region_count):
        offsets = coordinates[membership == region] - centers[region]
        radii[region] = np.quantile(np.linalg.norm(offsets, axis=1), radius_quantile)

    gap = gap_fraction * float(np.median(radii))
    ring_radius = float(np.sum(2.0 * radii + gap) / (2.0 * np.pi))
    global_center = np.average(centers, axis=0, weights=counts)
    angles = np.arctan2(
        centers[:, 1] - global_center[1], centers[:, 0] - global_center[0]
    )
    order = np.argsort(angles)
    targets = np.zeros_like(centers)
    for position, region in enumerate(order):
        angle = 2.0 * np.pi * position / region_count
        targets[region] = ring_radius * np.array([np.cos(angle), np.sin(angle)])
    packed = coordinates + targets[membership] - centers[membership]
    packed -= np.average(packed, axis=0)
    return packed.astype(np.float32)


def _compatible_neighbor_graph(
    graph: tuple[object, object] | None,
    *,
    n_samples: int,
    neighbors: int,
    metric: str | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Validate and normalize a reusable Euclidean feature-neighbor graph."""
    if graph is None or metric != "euclidean":
        return None
    try:
        raw_indices, raw_distances = graph
        indices = np.asarray(raw_indices)
        distances = np.asarray(raw_distances, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if (
        indices.ndim != 2
        or distances.shape != indices.shape
        or indices.shape[0] != n_samples
        or indices.shape[1] < neighbors
        or not np.issubdtype(indices.dtype, np.integer)
    ):
        return None

    indices = indices.astype(np.int64, copy=False)
    row_indices = np.arange(n_samples, dtype=np.int64)[:, None]
    valid = (
        (indices >= 0)
        & (indices < n_samples)
        & (indices != row_indices)
        & np.isfinite(distances)
        & (distances >= 0)
    )
    if np.any(valid.sum(axis=1) < neighbors):
        return None

    # Reject duplicate rows rather than silently assigning duplicate weight to
    # one point in the Leiden graph. ANN backends normally return unique rows.
    by_index = np.argsort(np.where(valid, indices, n_samples), axis=1, kind="stable")
    sorted_indices = np.take_along_axis(indices, by_index, axis=1)
    duplicates = sorted_indices[:, 1:] == sorted_indices[:, :-1]
    duplicates &= sorted_indices[:, 1:] < n_samples
    if np.any(duplicates):
        return None

    by_distance = np.argsort(
        np.where(valid, distances, np.inf), axis=1, kind="stable"
    )[:, :neighbors]
    selected_indices = np.take_along_axis(indices, by_distance, axis=1)
    selected_distances = np.take_along_axis(distances, by_distance, axis=1)
    return selected_indices, selected_distances.astype(np.float32, copy=False)


def density_gravity_layout(
    features: object,
    embedding: object,
    *,
    mode: str = "mexican_hat",
    neighbors: int = 15,
    resolution: float = 0.8,
    random_state: int = 42,
    core_quantile: float = 0.95,
    strength: float = 0.2,
    mass_power: float = 0.5,
    attraction_strength: float = 0.5,
    near_repulsion_strength: float = 2.5,
    force_neighbors: int = 8,
    attraction_range: float = 1.0,
    repulsion_range: float = 0.5,
    step_size: float = 0.05,
    iterations: int = 3,
    pack_regions: bool = False,
    pack_radius_quantile: float = 0.9,
    pack_gap_fraction: float = 0.25,
    feature_neighbor_graph: tuple[object, object] | None = None,
    feature_neighbor_metric: str | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build feature-space density regions and refine an IMSTE layout.

    A compatible Euclidean feature-neighbor graph is reused when supplied;
    otherwise RINND builds one. Leiden finds weighted density regions. Both
    modes pull points toward density-weighted region cores. ``density_core``
    applies only that regional pull; ``mexican_hat`` follows it with local
    attraction/repulsion to preserve within-region structure. No labels or
    notebook state are used.
    """
    from scipy.sparse import issparse

    if mode not in {"density_core", "mexican_hat"}:
        raise ValueError("mode must be 'density_core' or 'mexican_hat'.")
    if not isinstance(pack_regions, (bool, np.bool_)):
        raise ValueError("pack_regions must be a boolean.")
    if issparse(features):
        raise ValueError("gravity refinement currently requires dense features.")
    X = np.asarray(features, dtype=np.float32)
    coordinates = np.asarray(embedding, dtype=np.float64)
    if X.ndim != 2 or coordinates.shape != (len(X), 2) or len(X) < 3:
        raise ValueError("features and embedding must have matching rows and 2D coordinates.")
    if not np.isfinite(X).all() or not np.isfinite(coordinates).all():
        raise ValueError("features and embedding must contain only finite values.")
    if not 1 <= int(neighbors) < len(X):
        raise ValueError("neighbors must be in [1, n_samples).")

    try:
        import igraph as ig
        import leidenalg
    except ImportError as error:
        raise ImportError(
            "Gravity refinement requires igraph and leidenalg. Install them with "
            "`pip install 'igraph>=0.11' 'leidenalg>=0.10'`."
        ) from error

    graph_started = time.perf_counter()
    reusable_graph = _compatible_neighbor_graph(
        feature_neighbor_graph,
        n_samples=len(X),
        neighbors=int(neighbors),
        metric=feature_neighbor_metric,
    )
    neighbor_search_seconds = 0.0
    if reusable_graph is None:
        try:
            from rinnd import RINND
        except ImportError as error:
            raise ImportError(
                "Gravity refinement needs a compatible Euclidean feature-neighbor "
                "graph or RINND for a fresh search. Install it with "
                "`pip install 'rinnd>=0.3.1'`."
            ) from error
        import inspect

        search_options = {
            "metric": "euclidean",
            "n_neighbors": int(neighbors),
            "random_state": int(random_state),
            "n_trees": 8,
            "n_iters": max(5, int(round(np.log2(len(X))))),
            "max_candidates": max(20, min(60, int(round(1.5 * int(neighbors))))),
            "verbose": False,
        }
        try:
            if "n_jobs" in inspect.signature(RINND).parameters:
                search_options["n_jobs"] = -1
        except (TypeError, ValueError):
            pass
        search_started = time.perf_counter()
        search = RINND(np.ascontiguousarray(X), **search_options)
        indices, distances = search.neighbor_graph
        neighbor_search_seconds = time.perf_counter() - search_started
        indices = np.asarray(indices, dtype=np.int64)
        distances = np.asarray(distances, dtype=np.float32)
        neighbor_graph_reused = False
    else:
        indices, distances = reusable_graph
        neighbor_graph_reused = True
    radii = np.maximum(np.max(distances, axis=1), 1e-12)
    raw_mass = (np.median(radii) / radii) ** 2
    point_mass = np.clip(
        raw_mass, np.percentile(raw_mass, 1), np.percentile(raw_mass, 99)
    ).astype(np.float32)

    rows = np.repeat(np.arange(len(X), dtype=np.int64), indices.shape[1])
    columns = indices.reshape(-1)
    edge_distances = distances.reshape(-1)
    valid = (columns >= 0) & (columns != rows) & np.isfinite(edge_distances)
    rows, columns, edge_distances = rows[valid], columns[valid], edge_distances[valid]
    edge_weights = np.exp(
        -edge_distances / np.maximum(radii[rows], 1e-12)
    ).astype(np.float32)
    graph = ig.Graph(
        n=len(X), edges=zip(rows.tolist(), columns.tolist()), directed=False
    )
    graph.es["weight"] = edge_weights.tolist()
    graph.simplify(multiple=True, loops=True, combine_edges={"weight": "sum"})
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
        seed=int(random_state),
        n_iterations=2,
    )
    membership = np.asarray(partition.membership, dtype=np.int32)
    region_count = int(membership.max()) + 1
    region_mass = np.bincount(membership, weights=point_mass, minlength=region_count)
    graph_seconds = time.perf_counter() - graph_started

    order = np.argsort(membership, kind="stable")
    sorted_membership = membership[order]
    sorted_mass = point_mass[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_membership)) + 1, len(order)]
    thresholds = np.empty(region_count, dtype=np.float32)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        region = sorted_membership[start]
        thresholds[region] = np.quantile(sorted_mass[start:stop], core_quantile)
    core = point_mass >= thresholds[membership]
    core_mass = np.bincount(
        membership[core], weights=point_mass[core], minlength=region_count
    )
    centers = np.zeros((region_count, 2), dtype=np.float64)
    np.add.at(centers, membership[core], coordinates[core] * point_mass[core, None])
    centers /= core_mass[:, None]
    region_strength = np.clip(
        strength * (region_mass / np.median(region_mass)) ** mass_power,
        0.0,
        0.9,
    )
    # The feature-space regions must affect both modes. Mexican-hat then adds
    # local repulsion/attraction to retain structure around each pulled core.
    result = coordinates + region_strength[membership, None] * (
        centers[membership] - coordinates
    )

    force_started = time.perf_counter()
    if mode == "mexican_hat":
        result = apply_local_mexican_hat_force(
            result,
            point_mass,
            attraction_strength=attraction_strength,
            near_repulsion_strength=near_repulsion_strength,
            attraction_range=attraction_range,
            repulsion_range=repulsion_range,
            neighbors=force_neighbors,
            step_size=step_size,
            iterations=iterations,
            mass_power=mass_power,
        )
    if pack_regions:
        result = pack_region_centers_on_ring(
            result,
            membership,
            radius_quantile=pack_radius_quantile,
            gap_fraction=pack_gap_fraction,
        )
    result = np.asarray(result, dtype=np.float64)
    result -= result.mean(axis=0)
    result = result.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Gravity refinement produced non-finite coordinates.")
    force_seconds = time.perf_counter() - force_started
    return result, {
        "membership": membership,
        "point_mass": point_mass,
        "region_count": region_count,
        "graph_seconds": graph_seconds,
        "neighbor_search_seconds": neighbor_search_seconds,
        "neighbor_graph_reused": neighbor_graph_reused,
        "force_seconds": force_seconds,
        "regions_packed": bool(pack_regions and region_count > 1),
    }
