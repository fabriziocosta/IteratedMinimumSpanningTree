"""Load datasets and visualize their high-dimensional IMSTE embeddings."""

from __future__ import annotations

import numbers
import sys
import time
from pathlib import Path
from io import BytesIO
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from imste import IteratedMinimumSpanningTreeEmbedder


def _stratified_sample(
    X: object,
    labels: object,
    *,
    max_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X)
    labels = np.asarray(labels)
    if len(X) > max_samples:
        indices, _ = train_test_split(
            np.arange(len(X)),
            train_size=max_samples,
            random_state=random_state,
            stratify=labels,
        )
        X = X[indices]
        labels = labels[indices]
    return X, labels


def _load_openml(name: str) -> tuple[np.ndarray, np.ndarray]:
    dataset = fetch_openml(
        name,
        version=1,
        as_frame=False,
        parser="liac-arff",
        cache=False,
    )
    labels = np.unique(dataset.target, return_inverse=True)[1]
    return np.asarray(dataset.data), labels


def _load_kmnist() -> tuple[np.ndarray, np.ndarray]:
    """Download Kuzushiji-MNIST archives fresh without writing a local cache."""
    base_url = "https://codh.rois.ac.jp/kmnist/dataset/kmnist"

    def load_archive(filename: str) -> np.ndarray:
        with urlopen(f"{base_url}/{filename}") as response:
            archive_bytes = BytesIO(response.read())
        with np.load(archive_bytes) as archive:
            return np.asarray(archive["arr_0"])

    images = np.concatenate(
        [load_archive("kmnist-train-imgs.npz"), load_archive("kmnist-test-imgs.npz")]
    )
    labels = np.concatenate(
        [load_archive("kmnist-train-labels.npz"), load_archive("kmnist-test-labels.npz")]
    )
    return images.reshape(len(images), -1), labels


def _load_datasets(max_samples: int, random_state: int):
    loaders = {
        "MNIST (28×28)": lambda: _load_openml("mnist_784"),
        "Fashion-MNIST (28×28)": lambda: _load_openml("Fashion-MNIST"),
        "Kuzushiji-MNIST (28×28)": _load_kmnist,
    }
    datasets = {}
    errors = {}
    for name, loader in loaders.items():
        print(f"Loading {name} ...", flush=True)
        try:
            raw_X, raw_labels = loader()
            X, labels = _stratified_sample(
                raw_X,
                raw_labels,
                max_samples=max_samples,
                random_state=random_state,
            )
            datasets[name] = (X, labels)
            print(
                f"  {len(X):,} samples × {X.shape[1]:,} features; "
                f"{len(np.unique(labels))} classes",
                flush=True,
            )
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {error}"
            print(f"  skipped: {errors[name]}", flush=True)
    if not datasets:
        raise RuntimeError("No datasets loaded successfully.")
    return datasets, errors


def _plot_dataset_examples(
    name: str,
    X: np.ndarray,
    labels: np.ndarray,
    *,
    n_rows: int,
    random_state: int,
) -> None:
    """Show a random image grid with one column for each class."""
    image_side = int(round(np.sqrt(X.shape[1])))
    if image_side * image_side != X.shape[1]:
        raise ValueError(
            f"Cannot plot {name}: expected flattened square grayscale images, "
            f"got {X.shape[1]} features."
        )

    classes = np.unique(labels)
    rng = np.random.default_rng(random_state)
    fig, axes = plt.subplots(
        n_rows,
        len(classes),
        figsize=(max(6, 0.725 * len(classes)), max(1.1, 0.825 * n_rows)),
        squeeze=False,
    )
    for column, class_label in enumerate(classes):
        class_indices = np.flatnonzero(labels == class_label)
        selected = rng.choice(
            class_indices,
            size=min(n_rows, len(class_indices)),
            replace=False,
        )
        for row, sample_index in enumerate(selected):
            pixels = np.asarray(X[sample_index], dtype=np.float32).reshape(
                image_side, image_side
            )
            pixels = np.clip(pixels / 255.0, 0.0, 1.0)
            ax = axes[row, column]
            ax.imshow(pixels, cmap="gray_r", vmin=0.0, vmax=1.0)
            ax.set_axis_off()
            if row == 0:
                ax.set_title(str(class_label))
    fig.suptitle(f"{name}: sample images by class")
    fig.tight_layout()
    display(fig)
    plt.close(fig)


def _default_device() -> str:
    return "auto"


def run_high_dim_mst_gallery(
    *,
    max_samples: int = 1000,
    n_components: int = 2,
    n_msts: int = 4,
    n_epochs: int = 200,
    batch_size: int = 32768,
    learning_rate: float = 0.05,
    mst_method: str = "famst",
    mst_neighbor_backend: str = "auto",
    mst_neighbors: int = 15,
    mst_inter_component_edges: int = 5,
    metric: str = "euclidean",
    include_knn_attraction: bool = False,
    negative_ratio: int = 1,
    mst_max_neighbors: int | None = 15,
    mst_representatives_per_component: int = 10,
    lambda_rep: float = 0.5,
    epsilon: float = 1e-4,
    logistic_margin: float = 1.0,
    logistic_temperature: float = 0.5,
    pca_init: bool = True,
    pca_init_scaling: str = "global",
    sample_plot_rows: int = 2,
    plot_sample_images: bool = True,
    random_state: int = 42,
    device: str | None = None,
    verbose: bool = False,
    plot_embeddings: bool = True,
    multilevel_init: bool = True,
    multilevel_reduction_ratio: int = 4,
    multilevel_base_size: int = 1024,
    multilevel_epochs: int = 10,
    gravity_mode: str | None = "mexican_hat",
    gravity_neighbors: int = 15,
    gravity_resolution: float = 0.8,
    gravity_strength: float = 0.2,
    gravity_core_quantile: float = 0.95,
    gravity_mass_power: float = 0.5,
    gravity_attraction_strength: float = 0.5,
    gravity_near_repulsion_strength: float = 2.5,
    gravity_force_neighbors: int = 8,
    gravity_attraction_range: float = 1.0,
    gravity_repulsion_range: float = 0.5,
    gravity_step_size: float = 0.05,
    gravity_iterations: int = 3,
    gravity_pack_regions: bool = False,
    gravity_pack_radius_quantile: float = 0.9,
    gravity_pack_gap_fraction: float = 0.25,
) -> dict[str, object]:
    """Fit and display 2D IMSTE embeddings for several image datasets.

    Dataset loading, normalization, and fitting happen here so the notebook
    can remain a thin launcher. Failed remote dataset loads are reported and
    skipped. Set ``plot_embeddings=False`` to leave embedding plots to the
    caller, for example when comparing multiple methods side by side. Set
    ``plot_sample_images=False`` to skip the per-class image grids.
    ``negative_ratio`` controls sampled
    non-neighbors per endpoint of a positive edge, and ``mst_max_neighbors``
    caps FAMST's approximate-neighbor search. ``metric`` selects Euclidean or
    cosine graph distances; cosine uses exact Prim construction. Multilevel initialization is
    enabled by default for this gallery and can be configured with the
    ``multilevel_*`` options.
    Gravity runs as a second phase of the estimator by default. Set
    ``gravity_mode=None`` to return the base IMSTE layout only.
    """
    if max_samples < 10:
        raise ValueError("max_samples must be at least 10.")
    if mst_method not in {"prim", "famst"}:
        raise ValueError("mst_method must be either 'prim' or 'famst'.")
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("metric must be either 'euclidean' or 'cosine'.")
    if (
        isinstance(sample_plot_rows, (bool, np.bool_))
        or not isinstance(sample_plot_rows, numbers.Integral)
        or sample_plot_rows < 1
    ):
        raise ValueError("sample_plot_rows must be a positive integer.")
    resolved_device = _default_device() if device is None else device
    datasets, load_errors = _load_datasets(max_samples, random_state)

    embeddings = {}
    base_embeddings = {}
    summaries = []
    for name, (X, labels) in datasets.items():
        if plot_sample_images:
            _plot_dataset_examples(
                name,
                X,
                labels,
                n_rows=int(sample_plot_rows),
                random_state=random_state,
            )
        print(
            f"Fitting {name} with {mst_method.upper()} ({metric}) on "
            f"{resolved_device} ...",
            flush=True,
        )
        pipeline = Pipeline(
            [
                ("normalize", StandardScaler()),
                (
                    "imste",
                    IteratedMinimumSpanningTreeEmbedder(
                        n_msts=n_msts,
                        n_components=n_components,
                        n_epochs=n_epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        mst_method=mst_method,
                        mst_neighbor_backend=mst_neighbor_backend,
                        mst_neighbors=mst_neighbors,
                        mst_inter_component_edges=mst_inter_component_edges,
                        mst_representatives_per_component=mst_representatives_per_component,
                        metric=metric,
                        include_knn_attraction=include_knn_attraction,
                        mst_max_neighbors=mst_max_neighbors,
                        negative_ratio=negative_ratio,
                        lambda_rep=lambda_rep,
                        epsilon=epsilon,
                        logistic_margin=logistic_margin,
                        logistic_temperature=logistic_temperature,
                        pca_init=pca_init,
                        pca_init_scaling=pca_init_scaling,
                        random_state=random_state,
                        device=resolved_device,
                        verbose=verbose,
                        multilevel_init=multilevel_init,
                        multilevel_reduction_ratio=multilevel_reduction_ratio,
                        multilevel_base_size=multilevel_base_size,
                        multilevel_epochs=multilevel_epochs,
                        gravity_mode=gravity_mode,
                        gravity_neighbors=gravity_neighbors,
                        gravity_resolution=gravity_resolution,
                        gravity_strength=gravity_strength,
                        gravity_core_quantile=gravity_core_quantile,
                        gravity_mass_power=gravity_mass_power,
                        gravity_attraction_strength=gravity_attraction_strength,
                        gravity_near_repulsion_strength=gravity_near_repulsion_strength,
                        gravity_force_neighbors=gravity_force_neighbors,
                        gravity_attraction_range=gravity_attraction_range,
                        gravity_repulsion_range=gravity_repulsion_range,
                        gravity_step_size=gravity_step_size,
                        gravity_iterations=gravity_iterations,
                        gravity_pack_regions=gravity_pack_regions,
                        gravity_pack_radius_quantile=gravity_pack_radius_quantile,
                        gravity_pack_gap_fraction=gravity_pack_gap_fraction,
                    ),
                ),
            ]
        )
        started = time.perf_counter()
        embedding = pipeline.fit_transform(X)
        elapsed = time.perf_counter() - started
        print(f"  pipeline completed in {elapsed:.2f} seconds", flush=True)
        estimator = pipeline.named_steps["imste"]
        base_embeddings[name] = estimator.embedding_before_gravity_.copy()
        mst_summary = f"{n_msts} MSTs"
        feature_count = X.shape[1]
        print(
            f"  standardized IMSTE input: {embedding.shape[0]:,} samples × "
            f"{feature_count:,} features",
            flush=True,
        )
        embeddings[name] = embedding
        summaries.append(
            {
                "dataset": name,
                "samples": len(X),
                "input_features": X.shape[1],
                "features": feature_count,
                "classes": len(np.unique(labels)),
                "n_msts": n_msts,
                "n_components": n_components,
                "mst_method": mst_method,
                "mst_neighbor_backend": mst_neighbor_backend,
                "mst_neighbors": mst_neighbors,
                "mst_inter_component_edges": mst_inter_component_edges,
                "metric": metric,
                "include_knn_attraction": include_knn_attraction,
                "mst_max_neighbors": mst_max_neighbors,
                "mst_representatives_per_component": mst_representatives_per_component,
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "negative_ratio": negative_ratio,
                "lambda_rep": lambda_rep,
                "epsilon": epsilon,
                "logistic_margin": logistic_margin,
                "logistic_temperature": logistic_temperature,
                "pca_init": pca_init,
                "pca_init_scaling": pca_init_scaling,
                "multilevel_init": multilevel_init,
                "multilevel_reduction_ratio": multilevel_reduction_ratio,
                "multilevel_base_size": multilevel_base_size,
                "multilevel_epochs": multilevel_epochs,
                "gravity_mode": gravity_mode,
                "gravity_neighbors": estimator.gravity_neighbors,
                "gravity_resolution": estimator.gravity_resolution,
                "gravity_strength": estimator.gravity_strength,
                "gravity_core_quantile": estimator.gravity_core_quantile,
                "gravity_mass_power": estimator.gravity_mass_power,
                "gravity_attraction_strength": estimator.gravity_attraction_strength,
                "gravity_near_repulsion_strength": estimator.gravity_near_repulsion_strength,
                "gravity_force_neighbors": estimator.gravity_force_neighbors,
                "gravity_attraction_range": estimator.gravity_attraction_range,
                "gravity_repulsion_range": estimator.gravity_repulsion_range,
                "gravity_step_size": estimator.gravity_step_size,
                "gravity_iterations": estimator.gravity_iterations,
                "gravity_pack_regions": estimator.gravity_pack_regions,
                "gravity_pack_radius_quantile": estimator.gravity_pack_radius_quantile,
                "gravity_pack_gap_fraction": estimator.gravity_pack_gap_fraction,
                "gravity_region_count": estimator.gravity_region_count_,
                "gravity_regions_packed": estimator.gravity_regions_packed_,
                "gravity_seconds": estimator.gravity_time_,
                "imste_seconds": estimator.imste_fit_time_,
                "unique_graph_edges": len(estimator.graph_edges_),
                "seconds": elapsed,
                "device": estimator.device_,
                "verbose": verbose,
            }
        )

        if plot_embeddings:
            fig, ax = plt.subplots(figsize=(7, 5))
            points = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=labels,
                cmap="tab20",
                s=12,
                alpha=0.8,
                linewidths=0,
            )
            graph_description = (
                "approximate FAMST graph"
                if mst_method == "famst" and metric == "euclidean"
                else "exact Prim graph"
            )
            ax.set_title(
                f"{name} — {len(X):,} samples × "
                f"{feature_count:,} features passed to IMSTE; "
                f"{graph_description} ({metric}); {mst_summary}; "
                f"{n_epochs} epochs; {elapsed:.2f} s"
            )
            ax.set_xlabel("Embedding dimension 1")
            ax.set_ylabel("Embedding dimension 2")
            fig.colorbar(points, ax=ax, label="Class label")
            fig.tight_layout()
            display(fig)
            plt.close(fig)

    return {
        "embeddings": embeddings,
        "base_embeddings": base_embeddings,
        "datasets": datasets,
        "summary": pd.DataFrame(summaries),
        "load_errors": load_errors,
    }
