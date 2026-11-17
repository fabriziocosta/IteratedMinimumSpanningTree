"""Density-aware second-phase layout refinements for IMSTE embeddings."""

from __future__ import annotations

import time

import numpy as np


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


def _apply_local_distance_floor(
    embedding: np.ndarray,
    neighbor_indices: np.ndarray,
    min_dist_fraction: float,
    *,
    reference_embedding: np.ndarray | None = None,
    iterations: int = 32,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Repel too-close pairs on the feature-neighbor graph after gravity.

    The distance floor is iteratively relaxed over the local graph. More sweeps
    help points satisfy several neighbor constraints at once, especially for
    larger requested floors.
    """
    coordinates = np.asarray(embedding, dtype=np.float64)
    reference = (
        coordinates
        if reference_embedding is None
        else np.asarray(reference_embedding, dtype=np.float64)
    )
    if reference.shape != coordinates.shape:
        raise ValueError("reference_embedding must match embedding shape.")
    n_samples = len(coordinates)
    rows = np.repeat(np.arange(n_samples, dtype=np.int64), neighbor_indices.shape[1])
    columns = np.asarray(neighbor_indices, dtype=np.int64).reshape(-1)
    valid = (columns >= 0) & (columns < n_samples) & (columns != rows)
    rows, columns = rows[valid], columns[valid]
    if not len(rows):
        return coordinates, {
            "min_dist_reference": 0.0,
            "min_dist_threshold": 0.0,
            "min_dist_pairs_adjusted": 0,
        }

    base_distance = np.linalg.norm(reference[rows] - reference[columns], axis=1)
    nearest_distance = np.full(n_samples, np.inf, dtype=np.float64)
    np.minimum.at(nearest_distance, rows, base_distance)
    positive_nearest = nearest_distance[
        np.isfinite(nearest_distance) & (nearest_distance > 0)
    ]
    if not len(positive_nearest):
        return coordinates, {
            "min_dist_reference": 0.0,
            "min_dist_threshold": 0.0,
            "min_dist_pairs_adjusted": 0,
        }
    reference_distance = float(np.median(positive_nearest))
    threshold = float(min_dist_fraction * reference_distance)

    pairs = np.unique(np.sort(np.column_stack((rows, columns)), axis=1), axis=0)
    left, right = pairs[:, 0], pairs[:, 1]
    result = coordinates.copy()
    ever_adjusted = np.zeros(len(pairs), dtype=bool)
    for _ in range(iterations):
        delta = result[left] - result[right]
        distance = np.linalg.norm(delta, axis=1)
        shortfall = np.maximum(threshold - distance, 0.0)
        active = shortfall > 0
        if not np.any(active):
            break
        ever_adjusted |= active

        direction = np.zeros_like(delta)
        nonzero = distance > 1e-12
        direction[nonzero] = delta[nonzero] / distance[nonzero, None]
        if np.any(~nonzero & active):
            coincident = ~nonzero & active
            angles = np.mod(
                left[coincident] * 12.9898 + right[coincident] * 78.233,
                2 * np.pi,
            )
            direction[coincident, 0] = np.cos(angles)
            direction[coincident, 1] = np.sin(angles)

        correction = 0.5 * shortfall[:, None] * direction
        displacement = np.zeros_like(result)
        degree = np.zeros(n_samples, dtype=np.float64)
        np.add.at(displacement, left[active], correction[active])
        np.add.at(displacement, right[active], -correction[active])
        np.add.at(degree, left[active], 1.0)
        np.add.at(degree, right[active], 1.0)
        has_correction = degree > 0
        result[has_correction] += (
            displacement[has_correction] / degree[has_correction, None]
        )

    return result, {
        "min_dist_reference": reference_distance,
        "min_dist_threshold": threshold,
        "min_dist_pairs_adjusted": int(ever_adjusted.sum()),
    }


def density_gravity_layout(
    features: object,
    embedding: object,
    *,
    neighbors: int = 15,
    resolution: float = 1.0,
    random_state: int = 42,
    core_quantile: float = 0.95,
    strength: float = 0.2,
    min_dist: float = 0.0,
    mass_power: float = 0.5,
    pack_regions: bool = False,
    pack_radius_quantile: float = 0.9,
    pack_gap_fraction: float = 0.25,
    feature_neighbor_graph: tuple[object, object] | None = None,
    feature_neighbor_metric: str | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build feature-space density regions and refine an IMSTE layout.

    A compatible Euclidean feature-neighbor graph is reused when supplied;
    otherwise RINND builds one. Leiden finds weighted density regions, then
    points move toward density-weighted region cores. An optional local
    distance floor then separates feature-neighbor pairs whose 2D distances
    fall below ``min_dist`` times their median pre-gravity nearest-neighbor
    distance. No labels or notebook state are used.
    """
    from scipy.sparse import issparse

    if not isinstance(pack_regions, (bool, np.bool_)):
        raise ValueError("pack_regions must be a boolean.")
    if not np.isfinite(min_dist) or min_dist < 0.0:
        raise ValueError("min_dist must be finite and nonnegative.")
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
    if float(strength) == 0.0 and float(min_dist) == 0.0:
        return coordinates.astype(np.float32), {
            "membership": np.zeros(len(X), dtype=np.int32),
            "point_mass": np.ones(len(X), dtype=np.float32),
            "region_count": 0,
            "graph_seconds": 0.0,
            "neighbor_search_seconds": 0.0,
            "neighbor_graph_reused": feature_neighbor_graph is not None,
            "force_seconds": 0.0,
            "regions_packed": False,
            "min_dist_reference": 0.0,
            "min_dist_threshold": 0.0,
            "min_dist_pairs_adjusted": 0,
        }

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
    result = coordinates + region_strength[membership, None] * (
        centers[membership] - coordinates
    )

    force_started = time.perf_counter()
    if pack_regions:
        result = pack_region_centers_on_ring(
            result,
            membership,
            radius_quantile=pack_radius_quantile,
            gap_fraction=pack_gap_fraction,
        )
    result, min_dist_diagnostics = _apply_local_distance_floor(
        result, indices, float(min_dist), reference_embedding=coordinates
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
        **min_dist_diagnostics,
    }
