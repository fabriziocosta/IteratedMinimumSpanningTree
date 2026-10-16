"""Numba kernels for deterministic rank-scheduled IMSTE SGD."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import numpy as np
from numba import njit


def rank_phases(tree_ranks: np.ndarray, random_state: int | None) -> np.ndarray:
    """Choose one reproducible epoch phase per tree rank."""
    n_ranks = int(np.max(tree_ranks)) if tree_ranks.size else 0
    phases = np.zeros(n_ranks + 1, dtype=np.int64)
    rng = np.random.RandomState(random_state)
    for rank in range(2, n_ranks + 1):
        phases[rank] = rng.randint(rank)
    return phases


def active_edges_for_epoch(
    tree_ranks: np.ndarray, epoch: int, phases: np.ndarray
) -> np.ndarray:
    """Return scheduled edge indices; exposed for exact scheduler tests."""
    if not tree_ranks.size:
        return np.empty(0, dtype=np.intp)
    return np.flatnonzero(epoch % tree_ranks == phases[tree_ranks])


def expected_active_edge_count(tree_ranks: np.ndarray) -> float:
    """Expected active edge count, used for the normalized rank-weighted mean."""
    return float(np.sum(1.0 / tree_ranks)) if tree_ranks.size else 1.0


@njit(cache=True)
def _attraction_factor(d2: float, epsilon: float) -> float:
    return 2.0 / (1.0 + d2 + epsilon)


@njit(cache=True)
def _repulsion_factor(d2: float, margin: float, temperature: float) -> float:
    z = (margin - d2) / temperature
    if z >= 0.0:
        sigmoid = 1.0 / (1.0 + np.exp(-z))
    else:
        exp_z = np.exp(z)
        sigmoid = exp_z / (1.0 + exp_z)
    return 2.0 * sigmoid / temperature


@njit(cache=True)
def _find_neighbor(indices: np.ndarray, start: int, stop: int, target: int) -> bool:
    left = start
    right = stop
    while left < right:
        middle = (left + right) // 2
        value = indices[middle]
        if value < target:
            left = middle + 1
        else:
            right = middle
    return left < stop and indices[left] == target


@njit(cache=True)
def _sample_target(
    source: int, n_samples: int, indptr: np.ndarray, indices: np.ndarray
) -> int:
    # Rejection is fast for sparse graphs. Reservoir fallback remains exact for
    # unusually dense neighborhoods without allocating a complement array.
    for _ in range(32):
        target = np.random.randint(0, n_samples)
        if target != source and not _find_neighbor(
            indices, indptr[source], indptr[source + 1], target
        ):
            return target
    selected = -1
    available = 0
    for target in range(n_samples):
        if target == source or _find_neighbor(
            indices, indptr[source], indptr[source + 1], target
        ):
            continue
        available += 1
        if np.random.randint(0, available) == 0:
            selected = target
    return selected


@njit(cache=True)
def _active_edges_for_rank_epoch(
    grouped_edges: np.ndarray,
    rank_offsets: np.ndarray,
    phases: np.ndarray,
    epoch: int,
    schedule_by_rank: bool,
    shuffle_seed: int,
) -> np.ndarray:
    """Gather scheduled edges by rank, avoiding a scan of all edges."""
    if not schedule_by_rank:
        active = np.arange(len(grouped_edges), dtype=np.int64)
    else:
        active_count = 0
        for rank in range(1, len(rank_offsets)):
            if epoch % rank == phases[rank]:
                active_count += rank_offsets[rank] - rank_offsets[rank - 1]
        active = np.empty(active_count, dtype=np.int64)
        cursor = 0
        for rank in range(1, len(rank_offsets)):
            if epoch % rank != phases[rank]:
                continue
            start = rank_offsets[rank - 1]
            stop = rank_offsets[rank]
            for position in range(start, stop):
                active[cursor] = grouped_edges[position]
                cursor += 1

    np.random.seed(shuffle_seed)
    for idx in range(len(active) - 1, 0, -1):
        swap = np.random.randint(0, idx + 1)
        value = active[idx]
        active[idx] = active[swap]
        active[swap] = value
    return active


@njit(cache=True)
def _sample_epoch_negatives(
    active: np.ndarray,
    edges: np.ndarray,
    degree: np.ndarray,
    indptr: np.ndarray,
    adjacency: np.ndarray,
    n_samples: int,
    negative_ratio: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Sample all epoch negatives independently of embedding coordinates."""
    negatives = np.full((len(active), 2, negative_ratio), -1, dtype=np.int64)
    if negative_ratio == 0:
        return negatives, 0
    np.random.seed(seed)
    eligible_endpoints = 0
    for item in range(len(active)):
        edge_index = active[item]
        left = edges[edge_index, 0]
        right = edges[edge_index, 1]
        for endpoint_index in range(2):
            source = left if endpoint_index == 0 else right
            if degree[source] >= n_samples - 1:
                continue
            eligible_endpoints += 1
            for negative_index in range(negative_ratio):
                negatives[item, endpoint_index, negative_index] = _sample_target(
                    source, n_samples, indptr, adjacency
                )
    return negatives, eligible_endpoints * negative_ratio


@njit(cache=True)
def _recenter(embedding: np.ndarray) -> None:
    n_samples, n_components = embedding.shape
    for component in range(n_components):
        mean = 0.0
        for sample in range(n_samples):
            mean += embedding[sample, component]
        mean /= n_samples
        for sample in range(n_samples):
            embedding[sample, component] -= mean


@njit(cache=True)
def _apply_epoch_updates(
    embedding: np.ndarray,
    edges: np.ndarray,
    tree_ranks: np.ndarray,
    active: np.ndarray,
    negatives: np.ndarray,
    epoch: int,
    n_epochs: int,
    expected_active_edges: float,
    schedule_by_rank: bool,
    negative_count: int,
    learning_rate: float,
    lambda_rep: float,
    epsilon: float,
    margin: float,
    temperature: float,
) -> None:
    n_samples, n_components = embedding.shape
    # The mean-loss normalization divides per-coordinate updates by the graph
    # population. Undo that dilution so the configured learning rate remains a
    # usable per-sample step size as datasets grow. The factor of five was
    # calibrated against the former Adam default on representative embeddings.
    alpha = learning_rate * (5.0 * n_samples) * (1.0 - epoch / max(1, n_epochs))
    positive_scale = (1.0 - lambda_rep) / max(1.0, expected_active_edges)
    negative_scale = lambda_rep / max(1, negative_count)

    for item in range(len(active)):
        edge_index = active[item]
        left = edges[edge_index, 0]
        right = edges[edge_index, 1]
        d2 = 0.0
        for component in range(n_components):
            delta = embedding[left, component] - embedding[right, component]
            d2 += delta * delta
        edge_weight = 1.0 / tree_ranks[edge_index] if not schedule_by_rank else 1.0
        attraction_factor = (
            alpha * positive_scale * edge_weight * _attraction_factor(d2, epsilon)
        )
        for component in range(n_components):
            delta = embedding[left, component] - embedding[right, component]
            update = attraction_factor * delta
            embedding[left, component] -= update
            embedding[right, component] += update

        for endpoint_index in range(2):
            source = left if endpoint_index == 0 else right
            for negative_index in range(negatives.shape[2]):
                target = negatives[item, endpoint_index, negative_index]
                if target < 0:
                    continue
                d2 = 0.0
                for component in range(n_components):
                    delta = embedding[source, component] - embedding[target, component]
                    d2 += delta * delta
                repulsion_factor = (
                    alpha * negative_scale * _repulsion_factor(d2, margin, temperature)
                )
                for component in range(n_components):
                    delta = embedding[source, component] - embedding[target, component]
                    update = repulsion_factor * delta
                    embedding[source, component] += update
                    embedding[target, component] -= update

    if n_samples:
        _recenter(embedding)


def optimize_embedding(
    embedding: np.ndarray,
    edges: np.ndarray,
    tree_ranks: np.ndarray,
    *,
    n_epochs: int,
    learning_rate: float,
    negative_ratio: int,
    lambda_rep: float,
    epsilon: float,
    margin: float,
    temperature: float,
    random_state: int | None,
    schedule_by_rank: bool = True,
    progress_callback=None,
    timing_callback: Callable[[float, float], None] | None = None,
) -> None:
    """Optimize in place using scheduled edges or explicit inverse-rank weights."""
    n_samples = embedding.shape[0]
    edge_pairs = np.asarray(edges, dtype=np.int64)
    ranks = np.asarray(tree_ranks, dtype=np.int64)
    phases = rank_phases(ranks, random_state)
    expected_active_edges = expected_active_edge_count(ranks)
    max_rank = int(np.max(ranks)) if ranks.size else 0
    rank_counts = np.bincount(ranks, minlength=max_rank + 1)
    rank_offsets = np.cumsum(rank_counts, dtype=np.int64)
    grouped_edges = np.argsort(ranks, kind="stable").astype(np.int64, copy=False)
    source = np.concatenate((edge_pairs[:, 0], edge_pairs[:, 1]))
    target = np.concatenate((edge_pairs[:, 1], edge_pairs[:, 0]))
    order = np.lexsort((target, source))
    source = source[order]
    target = target[order]
    counts = np.bincount(source, minlength=n_samples)
    indptr = np.empty(n_samples + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    adjacency = target.astype(np.int64, copy=False)
    degree = np.diff(indptr)
    seed_rng = np.random.RandomState(random_state)
    for epoch in range(n_epochs):
        shuffle_seed = int(seed_rng.randint(0, np.iinfo(np.int32).max))
        sampling_seed = int(seed_rng.randint(0, np.iinfo(np.int32).max))
        active = _active_edges_for_rank_epoch(
            grouped_edges,
            rank_offsets,
            phases,
            epoch,
            schedule_by_rank,
            shuffle_seed,
        )
        sampling_started = perf_counter() if timing_callback is not None else 0.0
        negatives, negative_count = _sample_epoch_negatives(
            active,
            edge_pairs,
            degree,
            indptr,
            adjacency,
            n_samples,
            negative_ratio,
            sampling_seed,
        )
        sampling_elapsed = (
            perf_counter() - sampling_started if timing_callback is not None else 0.0
        )
        update_started = perf_counter() if timing_callback is not None else 0.0
        _apply_epoch_updates(
            embedding,
            edge_pairs,
            ranks,
            active,
            negatives,
            epoch,
            n_epochs,
            expected_active_edges,
            schedule_by_rank,
            negative_count,
            learning_rate,
            lambda_rep,
            epsilon,
            margin,
            temperature,
        )
        if timing_callback is not None:
            timing_callback(sampling_elapsed, perf_counter() - update_started)
        if progress_callback is not None:
            progress_callback(epoch + 1, n_epochs)
