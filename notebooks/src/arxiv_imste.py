"""Notebook helpers for caching the arXiv projection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


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
