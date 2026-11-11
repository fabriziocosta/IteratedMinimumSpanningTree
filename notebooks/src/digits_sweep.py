"""Reusable helpers for the MNIST parameter-sweep notebook."""

from __future__ import annotations

import inspect
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE, trustworthiness
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Support running from a source checkout without requiring an editable install.
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from imste import IteratedMinimumSpanningTreeEmbedder


def load_mnist_data(sample_size=2000, random_state=42):
    """Load MNIST and standardize features in a stratified subsample.

    OpenML caches the full 70,000-row ``mnist_784`` dataset locally after the
    first download. ``sample_size=None`` returns all rows.
    """
    if sample_size is not None:
        if isinstance(sample_size, (bool, np.bool_)) or not isinstance(
            sample_size, (int, np.integer)
        ) or sample_size < 10:
            raise ValueError("sample_size must be at least 10, or None for all MNIST rows.")

    mnist = fetch_openml(
        "mnist_784", version=1, as_frame=False, parser="liac-arff", cache=True
    )
    X = np.asarray(mnist.data, dtype=np.float32)
    labels = np.asarray(mnist.target, dtype=np.int64)

    if sample_size is not None:
        if sample_size > len(X):
            raise ValueError(
                f"sample_size={sample_size} exceeds the {len(X)} available MNIST rows."
            )
        if sample_size < len(X):
            selected, _ = train_test_split(
                np.arange(len(X)),
                train_size=sample_size,
                random_state=random_state,
                stratify=labels,
            )
            X = X[selected]
            labels = labels[selected]

    X = StandardScaler().fit_transform(X)
    return X, labels


def format_duration(seconds: float) -> str:
    """Format elapsed time using the largest useful units."""
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours} hr {minutes} min" if minutes else f"{hours} hr"
    if minutes:
        return f"{minutes} min {seconds} sec"
    return f"{seconds} sec"


def _fit_and_score(X, labels, parameter, value, base_params, n_neighbors):
    params = dict(base_params)
    params[parameter] = value
    estimator = IteratedMinimumSpanningTreeEmbedder(**params)
    started = time.perf_counter()
    print(f"Fitting {parameter}={value} ...", flush=True)
    embedding = estimator.fit_transform(X)
    elapsed = time.perf_counter() - started
    print(
        f"Finished {parameter}={value} in {format_duration(elapsed)} "
        f"on {estimator.device_}",
        flush=True,
    )

    score = trustworthiness(
        X, embedding, n_neighbors=min(n_neighbors, len(X) - 1)
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn_accuracy = cross_val_score(
        KNeighborsClassifier(n_neighbors=5), embedding, labels, cv=cv
    ).mean()
    return {
        parameter: value,
        "trustworthiness": score,
        "5-NN 5-fold CV accuracy": knn_accuracy,
        "seconds": elapsed,
        "elapsed": format_duration(elapsed),
        "device": estimator.device_,
        "embedding": embedding,
    }


def run_sweep(X, labels, parameter, values, base_params, n_neighbors=10, n_jobs=1):
    """Fit and plot one digits embedding per value of an estimator parameter.

    Independent parameter values can run in parallel with ``n_jobs``. Keep the
    worker count modest because each fit builds a dense pairwise distance matrix.
    Returns result dictionaries (including embeddings) and a summary DataFrame.
    """
    if parameter not in IteratedMinimumSpanningTreeEmbedder().get_params():
        raise ValueError(f"Unknown estimator parameter: {parameter!r}")
    if not values:
        raise ValueError("values must contain at least one candidate")

    arguments = (
        (X, labels, parameter, value, base_params, n_neighbors)
        for value in values
    )
    if n_jobs == 1:
        results = [_fit_and_score(*args) for args in arguments]
    else:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(_fit_and_score)(*args) for args in arguments
            )

    ncols = min(3, len(results))
    nrows = math.ceil(len(results) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False,
        constrained_layout=True,
    )

    for index, result in enumerate(results):
        embedding = result["embedding"]
        value = result[parameter]
        score = result["trustworthiness"]
        knn_accuracy = result["5-NN 5-fold CV accuracy"]
        ax = axes.flat[index]
        points = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10",
            vmin=-0.5, vmax=9.5, s=7, alpha=0.8, linewidths=0,
        )
        ax.set_title(
            f"{parameter}={value}\n"
            f"trustworthiness={score:.3f}\n"
            f"5-NN 5-fold CV accuracy={knn_accuracy:.3f}"
        )
        ax.set_xlabel("Embedding dimension 1")
        ax.set_ylabel("Embedding dimension 2")

    for ax in axes.flat[len(values):]:
        ax.set_visible(False)
    colorbar = fig.colorbar(points, ax=list(axes.flat[:len(values)]), ticks=range(10))
    colorbar.set_label("Digit label")
    plt.show()

    summary = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "embedding"}
         for row in results]
    )
    return results, summary


def run_umap_comparisons(
    X,
    labels,
    configurations,
    n_epochs=500,
    random_state=42,
    trustworthiness_neighbors=10,
):
    """Fit and plot several UMAP settings with the same evaluation scores."""
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError(
            'UMAP comparison requires the notebook extra: '
            'python -m pip install "imste[notebook]"'
        ) from exc

    if not configurations:
        raise ValueError("configurations must contain at least one setting")

    results = []
    for configuration in configurations:
        name = configuration.get("name", "UMAP")
        n_neighbors = configuration["n_neighbors"]
        min_dist = configuration["min_dist"]
        reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_epochs=n_epochs,
            metric="euclidean",
            random_state=random_state,
            n_jobs=1,
        )
        print(f"Fitting UMAP: {name} ...", flush=True)
        started = time.perf_counter()
        embedding = reducer.fit_transform(X)
        elapsed = time.perf_counter() - started

        score = trustworthiness(
            X, embedding, n_neighbors=min(trustworthiness_neighbors, len(X) - 1)
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        knn_accuracy = cross_val_score(
            KNeighborsClassifier(n_neighbors=5), embedding, labels, cv=cv
        ).mean()
        results.append({
            "configuration": name,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "n_epochs": n_epochs,
            "trustworthiness": score,
            "5-NN 5-fold CV accuracy": knn_accuracy,
            "seconds": elapsed,
            "elapsed": format_duration(elapsed),
            "embedding": embedding,
        })
        print(f"Finished UMAP: {name} in {format_duration(elapsed)}", flush=True)

    ncols = min(3, len(results))
    nrows = math.ceil(len(results) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False,
        constrained_layout=True,
    )
    for index, result in enumerate(results):
        embedding = result["embedding"]
        ax = axes.flat[index]
        points = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10",
            vmin=-0.5, vmax=9.5, s=7, alpha=0.8, linewidths=0,
        )
        ax.set_title(
            f"{result['configuration']}\n"
            f"n_neighbors={result['n_neighbors']}, min_dist={result['min_dist']}\n"
            f"trustworthiness={result['trustworthiness']:.3f}; "
            f"5-NN CV={result['5-NN 5-fold CV accuracy']:.3f}\n"
            f"runtime={result['elapsed']}"
        )
        ax.set_xlabel("Embedding dimension 1")
        ax.set_ylabel("Embedding dimension 2")

    for ax in axes.flat[len(results):]:
        ax.set_visible(False)
    colorbar = fig.colorbar(points, ax=list(axes.flat[:len(results)]), ticks=range(10))
    colorbar.set_label("Digit label")
    plt.show()

    summary = pd.DataFrame(
        [{key: value for key, value in result.items() if key != "embedding"}
         for result in results]
    )
    return results, summary


def run_tsne_comparisons(
    X,
    labels,
    configurations,
    n_iterations=1000,
    random_state=42,
    trustworthiness_neighbors=10,
):
    """Fit and plot several t-SNE perplexities with shared evaluation scores."""
    if not configurations:
        raise ValueError("configurations must contain at least one setting")

    iteration_parameter = (
        "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    )
    results = []
    for configuration in configurations:
        name = configuration.get("name", "t-SNE")
        perplexity = configuration["perplexity"]
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
            n_jobs=-1,
            **{iteration_parameter: n_iterations},
        )
        print(f"Fitting t-SNE: {name} ...", flush=True)
        started = time.perf_counter()
        embedding = reducer.fit_transform(X)
        elapsed = time.perf_counter() - started

        score = trustworthiness(
            X, embedding, n_neighbors=min(trustworthiness_neighbors, len(X) - 1)
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        knn_accuracy = cross_val_score(
            KNeighborsClassifier(n_neighbors=5), embedding, labels, cv=cv
        ).mean()
        results.append({
            "configuration": name,
            "perplexity": perplexity,
            "iterations": n_iterations,
            "trustworthiness": score,
            "5-NN 5-fold CV accuracy": knn_accuracy,
            "seconds": elapsed,
            "elapsed": format_duration(elapsed),
            "embedding": embedding,
        })
        print(f"Finished t-SNE: {name} in {format_duration(elapsed)}", flush=True)

    ncols = min(3, len(results))
    nrows = math.ceil(len(results) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False,
        constrained_layout=True,
    )
    for index, result in enumerate(results):
        embedding = result["embedding"]
        ax = axes.flat[index]
        points = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10",
            vmin=-0.5, vmax=9.5, s=7, alpha=0.8, linewidths=0,
        )
        ax.set_title(
            f"{result['configuration']}\n"
            f"perplexity={result['perplexity']}\n"
            f"trustworthiness={result['trustworthiness']:.3f}; "
            f"5-NN CV={result['5-NN 5-fold CV accuracy']:.3f}\n"
            f"runtime={result['elapsed']}"
        )
        ax.set_xlabel("Embedding dimension 1")
        ax.set_ylabel("Embedding dimension 2")

    for ax in axes.flat[len(results):]:
        ax.set_visible(False)
    colorbar = fig.colorbar(points, ax=list(axes.flat[:len(results)]), ticks=range(10))
    colorbar.set_label("Digit label")
    plt.show()

    summary = pd.DataFrame(
        [{key: value for key, value in result.items() if key != "embedding"}
         for result in results]
    )
    return results, summary
