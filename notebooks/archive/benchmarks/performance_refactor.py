"""Benchmark IMSTE stage timings and compare full-fit time with UMAP.

Example:
    python benchmarks/performance_refactor.py --sizes 2000 10000 --repeats 3

UMAP and trustworthiness metrics require the ``notebook`` optional dependencies.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imste import IteratedMinimumSpanningTreeEmbedder


def _neighbor_preservation(X: np.ndarray, embedding: np.ndarray, k: int = 10) -> float:
    high = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)
    low = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(embedding)
    high_indices = high.kneighbors(return_distance=False)[:, 1:]
    low_indices = low.kneighbors(return_distance=False)[:, 1:]
    overlap = [
        len(set(high_row).intersection(low_row)) / k
        for high_row, low_row in zip(high_indices, low_indices)
    ]
    return float(np.mean(overlap))


def _quality_sample(X: np.ndarray, embedding: np.ndarray, seed: int) -> float:
    rng = np.random.RandomState(seed)
    count = min(len(X), 2000)
    selected = np.sort(rng.choice(len(X), size=count, replace=False))
    return float(trustworthiness(X[selected], embedding[selected], n_neighbors=10))


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[2000, 10000, 50000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-msts", type=int, default=15)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--skip-umap", action="store_true")
    args = parser.parse_args()

    # Warm Numba and ANN compilation outside measurements.
    warmup, _ = make_blobs(n_samples=256, n_features=args.features, random_state=3)
    IteratedMinimumSpanningTreeEmbedder(
        n_msts=2, n_epochs=2, random_state=3, mst_method="famst"
    ).fit(warmup)

    if not args.skip_umap:
        try:
            import umap
        except ImportError as exc:
            raise SystemExit(
                "UMAP comparison needs `python -m pip install -e '.[notebook]'`."
            ) from exc
    else:
        umap = None

    print(
        "n_samples,method,graph_seconds,optimization_seconds,fit_seconds,"
        "trustworthiness,knn_preservation"
    )
    for size in args.sizes:
        X, _ = make_blobs(
            n_samples=size,
            n_features=args.features,
            centers=20,
            cluster_std=3.0,
            random_state=42,
        )
        imste_runs: list[tuple[float, float, float]] = []
        qualities: list[float] = []
        knn_scores: list[float] = []
        for seed in range(args.repeats):
            model = IteratedMinimumSpanningTreeEmbedder(
                n_msts=args.n_msts,
                n_epochs=args.epochs,
                random_state=seed,
                mst_method="famst",
                mst_neighbors=15 if size <= 2000 else 120,
                mst_max_neighbors=512,
                device="cpu",
            ).fit(X)
            imste_runs.append(
                (
                    model.graph_construction_time_,
                    model.embedding_optimization_time_,
                    model.fit_time_,
                )
            )
            qualities.append(_quality_sample(X, model.embedding_, seed))
            knn_scores.append(_neighbor_preservation(X, model.embedding_))
        print(
            f"{size},IMSTE,{_median([x[0] for x in imste_runs]):.4f},"
            f"{_median([x[1] for x in imste_runs]):.4f},"
            f"{_median([x[2] for x in imste_runs]):.4f},"
            f"{_median(qualities):.4f},{_median(knn_scores):.4f}"
        )

        if umap is not None:
            elapsed: list[float] = []
            qualities = []
            knn_scores = []
            for seed in range(args.repeats):
                model = umap.UMAP(
                    n_neighbors=15,
                    n_components=2,
                    n_epochs=args.epochs,
                    random_state=seed,
                    n_jobs=1,
                )
                started = time.perf_counter()
                embedding = model.fit_transform(X)
                elapsed.append(time.perf_counter() - started)
                qualities.append(_quality_sample(X, embedding, seed))
                knn_scores.append(_neighbor_preservation(X, embedding))
            print(
                f"{size},UMAP,,,{_median(elapsed):.4f},"
                f"{_median(qualities):.4f},{_median(knn_scores):.4f}"
            )


if __name__ == "__main__":
    main()
