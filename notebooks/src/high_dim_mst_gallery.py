"""Load datasets and visualize their high-dimensional IMSTE embeddings."""

from __future__ import annotations

import numbers
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.datasets import fetch_openml, get_data_home
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
        cache=True,
    )
    labels = np.unique(dataset.target, return_inverse=True)[1]
    return np.asarray(dataset.data), labels


def _load_kmnist() -> tuple[np.ndarray, np.ndarray]:
    """Load Kuzushiji-MNIST from its official NumPy archives and cache locally."""
    base_url = "https://codh.rois.ac.jp/kmnist/dataset/kmnist"
    cache_dir = Path(get_data_home()) / "kmnist"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def load_archive(filename: str) -> np.ndarray:
        path = cache_dir / filename
        if not path.exists():
            temporary_path = path.with_suffix(path.suffix + ".tmp")
            try:
                urlretrieve(f"{base_url}/{filename}", temporary_path)
                temporary_path.replace(path)
            finally:
                temporary_path.unlink(missing_ok=True)
        with np.load(path) as archive:
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
    n_msts: int = 4,
    n_epochs: int = 200,
    mst_method: str = "famst",
    negative_ratio: int = 1,
    mst_max_neighbors: int | None = 15,
    lambda_rep: float = 0.5,
    pca_init: bool = True,
    sample_plot_rows: int = 2,
    plot_sample_images: bool = True,
    random_state: int = 42,
    device: str | None = None,
    plot_embeddings: bool = True,
    multilevel_init: bool = True,
    multilevel_reduction_ratio: int = 4,
    multilevel_base_size: int = 1024,
    multilevel_epochs: int = 10,
) -> dict[str, object]:
    """Fit and display 2D IMSTE embeddings for several image datasets.

    Dataset loading, normalization, and fitting happen here so the notebook
    can remain a thin launcher. Failed remote dataset loads are reported and
    skipped. Set ``plot_embeddings=False`` to leave embedding plots to the
    caller, for example when comparing multiple methods side by side. Set
    ``plot_sample_images=False`` to skip the per-class image grids.
    ``negative_ratio`` controls sampled
    non-neighbors per endpoint of a positive edge, and ``mst_max_neighbors``
    caps FAMST's approximate-neighbor search. Multilevel initialization is
    enabled by default for this gallery and can be configured with the
    ``multilevel_*`` options.
    """
    if max_samples < 10:
        raise ValueError("max_samples must be at least 10.")
    if mst_method not in {"prim", "famst"}:
        raise ValueError("mst_method must be either 'prim' or 'famst'.")
    if (
        isinstance(sample_plot_rows, (bool, np.bool_))
        or not isinstance(sample_plot_rows, numbers.Integral)
        or sample_plot_rows < 1
    ):
        raise ValueError("sample_plot_rows must be a positive integer.")
    resolved_device = _default_device() if device is None else device
    datasets, load_errors = _load_datasets(max_samples, random_state)

    embeddings = {}
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
            f"Fitting {name} with {mst_method.upper()} on {resolved_device} ...",
            flush=True,
        )
        pipeline = Pipeline(
            [
                ("normalize", StandardScaler()),
                (
                    "imste",
                    IteratedMinimumSpanningTreeEmbedder(
                        n_msts=n_msts,
                        n_components=2,
                        n_epochs=n_epochs,
                        mst_method=mst_method,
                        mst_max_neighbors=mst_max_neighbors,
                        learning_rate=0.05,
                        negative_ratio=negative_ratio,
                        lambda_rep=lambda_rep,
                        pca_init=pca_init,
                        random_state=random_state,
                        device=resolved_device,
                        multilevel_init=multilevel_init,
                        multilevel_reduction_ratio=multilevel_reduction_ratio,
                        multilevel_base_size=multilevel_base_size,
                        multilevel_epochs=multilevel_epochs,
                    ),
                ),
            ]
        )
        started = time.perf_counter()
        embedding = pipeline.fit_transform(X)
        elapsed = time.perf_counter() - started
        print(f"  pipeline completed in {elapsed:.2f} seconds", flush=True)
        estimator = pipeline.named_steps["imste"]
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
                "mst_method": mst_method,
                "mst_max_neighbors": mst_max_neighbors,
                "n_epochs": n_epochs,
                "negative_ratio": negative_ratio,
                "lambda_rep": lambda_rep,
                "pca_init": pca_init,
                "multilevel_init": multilevel_init,
                "multilevel_reduction_ratio": multilevel_reduction_ratio,
                "multilevel_base_size": multilevel_base_size,
                "multilevel_epochs": multilevel_epochs,
                "unique_graph_edges": len(estimator.graph_edges_),
                "seconds": elapsed,
                "device": estimator.device_,
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
                if mst_method == "famst"
                else "exact Prim graph"
            )
            ax.set_title(
                f"{name} — {len(X):,} samples × "
                f"{feature_count:,} features passed to IMSTE; "
                f"{graph_description}; {mst_summary}; "
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
        "datasets": datasets,
        "summary": pd.DataFrame(summaries),
        "load_errors": load_errors,
    }
