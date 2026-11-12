"""Notebook helpers for caching the arXiv projection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


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
