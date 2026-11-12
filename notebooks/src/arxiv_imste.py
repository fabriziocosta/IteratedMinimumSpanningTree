"""Notebook helpers for caching the arXiv projection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def apply_local_mexican_hat_force(
    embedding: object,
    point_mass: object,
    *,
    attraction_strength: float = 0.5,
    near_repulsion_strength: float = 1.5,
    attraction_range: float = 1.0,
    repulsion_range: float = 0.2,
    neighbors: int = 8,
    step_size: float = 0.05,
    iterations: int = 3,
    mass_power: float = 0.5,
    n_jobs: int = -1,
) -> np.ndarray:
    """Apply a density-weighted Mexican-hat interaction to local point pairs.

    Each iteration finds nearby pairs in the current 2D layout. The broader
    Gaussian attracts at middle distances; a narrower, stronger Gaussian
    repels only near-collisions. Both interaction ranges scale with each
    point's local ``neighbors``-radius, so the repulsive force stays local.
    Feature-space density mass weights the pull from neighboring points.
    """
    from sklearn.neighbors import NearestNeighbors

    coordinates = np.asarray(embedding, dtype=np.float64)
    point_mass = np.asarray(point_mass, dtype=np.float64).reshape(-1)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 3:
        raise ValueError("embedding must have shape (n_samples >= 3, 2).")
    if point_mass.shape != (len(coordinates),):
        raise ValueError("point_mass must contain one value per embedded point.")
    for name, value in {
        "attraction_strength": attraction_strength,
        "near_repulsion_strength": near_repulsion_strength,
        "attraction_range": attraction_range,
        "repulsion_range": repulsion_range,
        "step_size": step_size,
        "mass_power": mass_power,
    }.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if repulsion_range <= 0 or attraction_range <= repulsion_range:
        raise ValueError("attraction_range must exceed positive repulsion_range.")
    if isinstance(neighbors, bool) or int(neighbors) != neighbors or not 1 <= neighbors < len(coordinates):
        raise ValueError("neighbors must be an integer in [1, n_samples).")
    if isinstance(iterations, bool) or int(iterations) != iterations or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer.")
    if not np.isfinite(point_mass).all() or np.any(point_mass <= 0):
        raise ValueError("point_mass values must be finite and positive.")

    relative_mass = np.clip(point_mass / np.median(point_mass), 0.1, 10.0) ** mass_power
    for iteration in range(int(iterations)):
        index = NearestNeighbors(
            n_neighbors=int(neighbors) + 1, algorithm="kd_tree", n_jobs=n_jobs
        ).fit(coordinates)
        distances, indices = index.kneighbors(coordinates, return_distance=True)
        # Remove the query row itself explicitly; duplicate coordinates can
        # make it appear after another zero-distance neighbor.
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

def fit_density_gravity_layout(
    X: object,
    labels: object,
    embedding: object,
    *,
    cache_dir: str | Path,
    projection_id: str,
    dataset_id: str,
    settings: dict[str, Any],
    verbose: bool = True,
) -> dict[str, Any]:
    """Build or load an unsupervised density-gravity second-phase layout.

    Labels are used only for the optional reported silhouette. RINND and
    Leiden build regions from feature-space neighbors; density-core attraction
    and the local Mexican-hat force then modify the IMSTE coordinates.
    """
    import hashlib
    import time

    import numpy as np

    if not settings.get("enabled", True):
        return {
            "embedding": None,
            "seconds": 0.0,
            "region_count": 0,
            "silhouette": float("nan"),
            "cache_path": None,
            "cache_hit": False,
        }
    if settings.get("metric", "euclidean") != "euclidean":
        raise ValueError("The gravity phase currently requires metric='euclidean'.")
    mode = settings.get("mode", "density_core")
    if mode not in {"density_core", "mexican_hat"}:
        raise ValueError("Gravity mode must be 'density_core' or 'mexican_hat'.")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "algorithm_version": "local-mexican-hat-v3",
        "projection_id": str(projection_id),
        "dataset_id": str(dataset_id),
        **settings,
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    result_path = cache_path / f"arxiv_gravity_{cache_key}.npz"
    if result_path.is_file():
        with np.load(result_path, allow_pickle=False) as saved:
            return {
                "embedding": saved["embedding"],
                "seconds": float(saved["seconds"]),
                "region_count": int(saved["region_count"]),
                "silhouette": float(saved["silhouette"]),
                "cache_path": result_path,
                "cache_hit": True,
                "graph_seconds": float(saved["graph_seconds"]) if "graph_seconds" in saved.files else float(saved["seconds"]),
                "layout_seconds": float(saved["layout_seconds"]) if "layout_seconds" in saved.files else 0.0,
                "graph_cache_hit": bool(saved["graph_cache_hit"]) if "graph_cache_hit" in saved.files else True,
            }

    from sklearn.metrics import silhouette_score

    graph_settings = {
        "dataset_id": str(dataset_id),
        "neighbors": int(settings["neighbors"]),
        "resolution": float(settings["resolution"]),
        "random_state": int(settings["random_state"]),
        "metric": str(settings["metric"]),
    }
    graph_key = hashlib.sha256(
        json.dumps(graph_settings, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    graph_path = cache_path / f"arxiv_gravity_graph_{graph_key}.npz"
    graph_cache_hit = graph_path.is_file()
    graph_seconds = 0.0
    if graph_cache_hit:
        with np.load(graph_path, allow_pickle=False) as saved:
            membership = saved["membership"]
            node_mass = saved["node_mass"]
            region_mass = saved["region_mass"]
            region_count = int(saved["region_count"])
        print(f"Reusing cached gravity graph: {graph_path.name}")
    else:
        try:
            from rinnd import RINND
            import igraph as ig
            import leidenalg
        except ImportError as error:
            raise ImportError(
                "Gravity needs RINND, igraph, and leidenalg. Install RINND with "
                "`pip install 'imste[rinnd]'` and graph dependencies with "
                "`pip install igraph leidenalg`."
            ) from error

        graph_started = time.perf_counter()
        X_array = np.asarray(X)
        print(f"Building gravity graph for {len(X_array):,} rows…", flush=True)
        gravity_index = RINND(
            np.ascontiguousarray(X_array, dtype=np.float32),
            metric="euclidean",
            n_neighbors=int(settings["neighbors"]),
            random_state=int(settings["random_state"]),
            n_trees=8,
            n_iters=max(5, int(round(np.log2(len(X_array))))),
            max_candidates=max(20, min(60, int(round(1.5 * int(settings["neighbors"]))))),
            n_jobs=-1,
            verbose=bool(verbose),
        )
        gravity_indices, gravity_distances = gravity_index.neighbor_graph
        gravity_indices = np.asarray(gravity_indices, dtype=np.int64)
        gravity_distances = np.asarray(gravity_distances, dtype=np.float32)
        gravity_radii = np.max(gravity_distances, axis=1)
        raw_mass = (
            np.median(gravity_radii) / np.maximum(gravity_radii, 1e-12)
        ) ** 2
        node_mass = np.clip(
            raw_mass, np.percentile(raw_mass, 1), np.percentile(raw_mass, 99)
        ).astype(np.float32)

        gravity_rows = np.repeat(
            np.arange(len(X_array), dtype=np.int64), gravity_indices.shape[1]
        )
        gravity_cols = gravity_indices.reshape(-1)
        edge_distances = gravity_distances.reshape(-1)
        valid = (
            (gravity_cols >= 0)
            & (gravity_cols != gravity_rows)
            & np.isfinite(edge_distances)
        )
        gravity_rows = gravity_rows[valid]
        gravity_cols = gravity_cols[valid]
        edge_distances = edge_distances[valid]
        edge_weights = np.exp(
            -edge_distances / np.maximum(gravity_radii[gravity_rows], 1e-12)
        ).astype(np.float32)
        gravity_graph = ig.Graph(
            n=len(X_array), edges=zip(gravity_rows.tolist(), gravity_cols.tolist()),
            directed=False,
        )
        gravity_graph.es["weight"] = edge_weights.tolist()
        gravity_graph.simplify(
            multiple=True, loops=True, combine_edges={"weight": "sum"}
        )
        partition = leidenalg.find_partition(
            gravity_graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=float(settings["resolution"]),
            seed=int(settings["random_state"]),
            n_iterations=2,
        )
        membership = np.asarray(partition.membership, dtype=np.int32)
        region_count = int(membership.max()) + 1
        region_mass = np.bincount(
            membership, weights=node_mass, minlength=region_count
        )
        graph_seconds = time.perf_counter() - graph_started
        graph_tmp = graph_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            graph_tmp,
            membership=membership,
            node_mass=node_mass,
            region_mass=region_mass,
            region_count=np.asarray(region_count),
            graph_seconds=np.asarray(graph_seconds),
            graph_settings_json=np.asarray(json.dumps(graph_settings, sort_keys=True)),
        )
        graph_tmp.replace(graph_path)
        print(f"Built RINND/Leiden graph: {region_count:,} regions in {graph_seconds:.1f}s")

    order = np.argsort(membership, kind="stable")
    sorted_membership = membership[order]
    sorted_mass = node_mass[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_membership)) + 1, len(order)]
    thresholds = np.empty(region_count, dtype=np.float32)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        region = sorted_membership[start]
        thresholds[region] = np.quantile(
            sorted_mass[start:stop], float(settings["core_quantile"])
        )
    core = node_mass >= thresholds[membership]
    core_mass = np.bincount(
        membership[core], weights=node_mass[core], minlength=region_count
    )
    coordinates = np.asarray(embedding, dtype=np.float64)
    centers = np.zeros((region_count, coordinates.shape[1]), dtype=np.float64)
    np.add.at(
        centers,
        membership[core],
        coordinates[core] * node_mass[core, None],
    )
    centers /= core_mass[:, None]

    layout_started = time.perf_counter()
    if mode == "density_core":
        region_strength = np.clip(
            float(settings["strength"])
            * (region_mass / np.median(region_mass)) ** float(settings["mass_power"]),
            0.0,
            0.9,
        )
        result = coordinates + region_strength[membership, None] * (
            centers[membership] - coordinates
        )
    else:
        region_strength = np.clip(
            float(settings["strength"])
            * (region_mass / np.median(region_mass)) ** float(settings["mass_power"]),
            0.0,
            0.9,
        )
        result = coordinates + region_strength[membership, None] * (
            centers[membership] - coordinates
        )
        result = apply_local_mexican_hat_force(
            result,
            node_mass,
            attraction_strength=float(settings["attraction_strength"]),
            near_repulsion_strength=float(settings["near_repulsion_strength"]),
            attraction_range=float(settings["attraction_range"]),
            repulsion_range=float(settings["repulsion_range"]),
            neighbors=int(settings["force_neighbors"]),
            step_size=float(settings["step_size"]),
            iterations=int(settings["iterations"]),
            mass_power=float(settings["mass_power"]),
        )
    result -= result.mean(axis=0)
    result = result.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Gravity layout contains non-finite coordinates.")
    # Silhouette is invariant to uniform scaling. Keep pairwise dot products
    # well within range when an upstream embedding has unusually large values.
    silhouette_scale = max(float(np.max(np.abs(result))), 1.0)
    silhouette = float(
        silhouette_score(
            result[:, :2] / silhouette_scale,
            np.asarray(labels).astype(str),
            sample_size=min(5_000, len(labels)),
            random_state=2027,
        )
    )
    layout_seconds = time.perf_counter() - layout_started
    seconds = graph_seconds + layout_seconds
    temp_path = result_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temp_path,
        embedding=result,
        seconds=np.asarray(seconds),
        region_count=np.asarray(region_count),
        silhouette=np.asarray(silhouette),
        graph_seconds=np.asarray(graph_seconds),
        layout_seconds=np.asarray(layout_seconds),
        graph_cache_hit=np.asarray(graph_cache_hit),
        settings_json=np.asarray(json.dumps(settings, sort_keys=True)),
        projection_id=np.asarray(str(projection_id)),
    )
    temp_path.replace(result_path)
    print(
        f"Gravity mode {mode}: {region_count:,} regions; "
        f"graph {graph_seconds:.1f}s, force {layout_seconds:.1f}s"
    )
    print(f"Domain-label silhouette (sample ≤5,000): {silhouette:.4f}")
    print("Labels were used only for silhouette evaluation, not graph construction or forces.")
    return {
        "embedding": result,
        "seconds": seconds,
        "region_count": region_count,
        "silhouette": silhouette,
        "cache_path": result_path,
        "cache_hit": False,
        "graph_seconds": graph_seconds,
        "layout_seconds": layout_seconds,
        "graph_cache_hit": graph_cache_hit,
    }


def select_arxiv_experiment_subset(
    X,
    labels,
    *,
    dataset_id: str,
    mode: str = "balanced",
    sample_size: int = 20_000,
    random_state: int = 42,
):
    """Select a reproducible full, random, or class-balanced experiment set.

    Balanced sampling is without replacement and gives each class an equal
    number of unique rows (quotas can differ by one when the requested size is
    not divisible by the number of classes). The returned data identifier
    includes the selected source-row indices, so embedding caches cannot mix
    different subsets.
    """
    if mode not in {"full", "random", "balanced"}:
        raise ValueError("mode must be 'full', 'random', or 'balanced'.")
    if X.shape[0] != len(labels):
        raise ValueError("X and labels must contain the same number of rows.")
    if mode == "full":
        return X, labels, str(dataset_id)
    if isinstance(sample_size, bool) or not isinstance(sample_size, (int, np.integer)):
        raise ValueError("sample_size must be a positive integer.")
    if not 1 <= sample_size <= X.shape[0]:
        raise ValueError("sample_size must be between 1 and the number of rows.")

    rng = np.random.RandomState(random_state)
    label_values = np.asarray(labels)
    if mode == "random":
        indices = rng.choice(X.shape[0], size=int(sample_size), replace=False)
    else:
        classes, counts = np.unique(label_values, return_counts=True)
        max_balanced_size = int(len(classes) * counts.min())
        if sample_size > max_balanced_size:
            raise ValueError(
                f"A unique balanced sample can contain at most {max_balanced_size:,} "
                f"rows for {len(classes)} classes; requested {sample_size:,}. "
                "Reduce sample_size or use mode='random'."
            )
        base, remainder = divmod(int(sample_size), len(classes))
        quotas = np.full(len(classes), base, dtype=np.int64)
        # Give any remainder to classes with the most spare examples.
        remainder_order = np.argsort(-counts, kind="stable")
        quotas[remainder_order[:remainder]] += 1
        selected = []
        for class_value, quota in zip(classes, quotas):
            class_rows = np.flatnonzero(label_values == class_value)
            selected.append(rng.choice(class_rows, size=int(quota), replace=False))
        indices = np.concatenate(selected)
        rng.shuffle(indices)

    if hasattr(X, "iloc"):
        X_subset = X.iloc[indices].reset_index(drop=True)
    else:
        X_subset = X[indices]
    if hasattr(labels, "iloc"):
        labels_subset = labels.iloc[indices].reset_index(drop=True)
    else:
        labels_subset = labels[indices]

    indices_hash = hashlib.sha256(
        np.ascontiguousarray(indices, dtype=np.int64).view(np.uint8)
    ).hexdigest()
    identity = {
        "dataset_id": str(dataset_id),
        "mode": mode,
        "sample_size": int(sample_size),
        "random_state": int(random_state),
        "indices_sha256": indices_hash,
    }
    sample_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    return X_subset, labels_subset, sample_id


def save_arxiv_projection_cache(
    path: str | Path,
    *,
    projection_id: str,
    projection_settings: dict[str, Any],
    imste_settings_metadata: dict[str, Any],
    verbose: bool,
    umap_embedding: object,
    imste_embedding: object,
    labels: object,
    umap_seconds: float,
    imste_seconds: float,
    dataset_id: str,
) -> Path:
    """Atomically save both layouts and the fit metadata to the projection cache."""
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temp_path,
        projection_id=np.asarray(projection_id),
        umap=np.asarray(umap_embedding, dtype=np.float32),
        imste_multilevel=np.asarray(imste_embedding, dtype=np.float32),
        labels=np.asarray(labels, dtype="U"),
        umap_seconds=np.asarray(umap_seconds),
        imste_seconds=np.asarray(imste_seconds),
        dataset_id=np.asarray(dataset_id),
        settings_json=np.asarray(json.dumps(
            {
                "projection": projection_settings,
                "imste": imste_settings_metadata,
                "verbose": verbose,
            },
            sort_keys=True,
        )),
    )
    temp_path.replace(cache_path)
    return cache_path


def prepare_arxiv_projection_cache(
    path: str | Path,
    *,
    dataset_id: str,
    umap_settings: dict[str, Any],
    imste_settings: dict[str, Any],
) -> dict[str, Any]:
    """Build a stable cache identity and load matching projection arrays."""
    import hashlib

    metric = imste_settings["metric"]
    random_state = imste_settings["random_state"]
    imste_settings = {"mst_neighbor_backend": "auto", **imste_settings}
    umap_key = [
        umap_settings["n_neighbors"],
        umap_settings["n_epochs"],
        umap_settings["repulsion_strength"],
        umap_settings["min_dist"],
        umap_settings["metric"],
        umap_settings["nearest_neighbor_backend"],
        umap_settings["random_state"],
    ]
    projection_settings = {
        "data_id": dataset_id,
        "metric": metric,
        "umap": umap_key,
        "imste": [
            imste_settings["n_msts"],
            imste_settings["n_epochs"],
            imste_settings["negative_ratio"],
            imste_settings["lambda_rep"],
            imste_settings["mst_method"],
            imste_settings["mst_neighbors"],
            imste_settings["mst_neighbor_backend"],
            imste_settings["device"],
            metric,
            random_state,
        ],
        "multilevel": [
            imste_settings["multilevel_reduction_ratio"],
            imste_settings["multilevel_base_size"],
            imste_settings["multilevel_epochs"],
        ],
    }

    init_coordinates_id = None
    init_coordinates = imste_settings["init_coordinates"]
    if init_coordinates is not None:
        init_array = np.ascontiguousarray(init_coordinates)
        init_coordinates_id = {
            "shape": list(init_array.shape),
            "dtype": str(init_array.dtype),
            "sha256": hashlib.sha256(init_array.view(np.uint8)).hexdigest(),
        }

    imste_settings_metadata = {
        **{
            key: value
            for key, value in imste_settings.items()
            if key not in {"init_coordinates", "progress_callback", "mst_progress_callback"}
        },
        "init_coordinates": init_coordinates_id,
        "progress_callback": None,
        "mst_progress_callback": None,
    }
    new_parameter_defaults = {
        "n_components": 2,
        "batch_size": 32768,
        "learning_rate": 0.05,
        "epsilon": 1e-4,
        "logistic_margin": 1.0,
        "logistic_temperature": 0.5,
        "mst_inter_component_edges": 5,
        "mst_max_neighbors": None,
        "mst_representatives_per_component": 10,
        "mst_neighbor_backend": "auto",
        "pca_init": True,
        "pca_init_scaling": "global",
        "init_coordinates": None,
        "multilevel_init": True,
    }
    overrides = {}
    for key, default in new_parameter_defaults.items():
        value = imste_settings[key]
        if key == "init_coordinates":
            value = init_coordinates_id
        if value != default:
            overrides[key] = value
    if overrides:
        projection_settings["imste_overrides"] = overrides

    projection_id = hashlib.sha256(
        json.dumps(projection_settings, sort_keys=True).encode()
    ).hexdigest()[:16]
    cache_path = Path(path)
    cached = None
    umap_cached = None
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as saved:
            saved_settings = json.loads(str(saved["settings_json"].item()))
            saved_projection = saved_settings.get("projection", {})
            if (
                saved_projection.get("data_id") == dataset_id
                and saved_projection.get("umap") == umap_key
            ):
                umap_cached = {
                    "umap": saved["umap"],
                    "umap_seconds": float(saved["umap_seconds"]),
                }
            if str(saved["projection_id"].item()) == projection_id:
                cached = {
                    "umap": saved["umap"],
                    "imste_multilevel": saved["imste_multilevel"],
                    "umap_seconds": float(saved["umap_seconds"]),
                    "imste_seconds": float(saved["imste_seconds"]),
                }

    return {
        "projection_id": projection_id,
        "projection_settings": projection_settings,
        "imste_settings_metadata": imste_settings_metadata,
        "cached": cached,
        "umap_cached": umap_cached,
    }
