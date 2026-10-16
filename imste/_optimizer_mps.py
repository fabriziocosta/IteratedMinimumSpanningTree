"""Batched PyTorch MPS optimizer for Apple GPUs.

Each active-edge batch is applied simultaneously, with batches processed
sequentially. Negative targets are sampled on CPU using the same sparse-graph
rejection sampler as the Numba optimizer, then copied to MPS with the edges.
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import numpy as np
import torch

from ._optimizer import (
    _active_edges_for_rank_epoch,
    _sample_epoch_negatives,
    expected_active_edge_count,
    rank_phases,
)


def optimize_embedding_mps(
    embedding: np.ndarray,
    edges: np.ndarray,
    tree_ranks: np.ndarray,
    *,
    n_epochs: int,
    batch_size: int,
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
    """Optimize coordinates on MPS with batched, simultaneous edge updates."""
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "device='mps' requires an MPS-capable PyTorch installation on a Mac "
            "with a supported Apple GPU."
        )

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

    device = torch.device("mps")
    coordinates = torch.as_tensor(embedding, dtype=torch.float32, device=device)
    edge_tensor = torch.as_tensor(edge_pairs, dtype=torch.long, device=device)
    rank_tensor = torch.as_tensor(ranks, dtype=torch.float32, device=device)
    seed_rng = np.random.RandomState(random_state)
    sampling_seconds = 0.0
    optimizer_started = perf_counter()

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
        active_pairs_cpu = edge_pairs[active]
        if negative_ratio > 0 and lambda_rep > 0.0:
            eligible_endpoints = int(
                np.count_nonzero(degree[active_pairs_cpu[:, 0]] < n_samples - 1)
                + np.count_nonzero(degree[active_pairs_cpu[:, 1]] < n_samples - 1)
            )
            negative_count = eligible_endpoints * negative_ratio
        else:
            negative_count = 0
        batch_seed_rng = np.random.RandomState(sampling_seed)
        for batch_start in range(0, len(active), batch_size):
            active_batch = active[batch_start : batch_start + batch_size]
            sample_started = perf_counter()
            if negative_count > 0:
                negatives, _ = _sample_epoch_negatives(
                    active_batch,
                    edge_pairs,
                    degree,
                    indptr,
                    adjacency,
                    n_samples,
                    negative_ratio,
                    int(batch_seed_rng.randint(0, np.iinfo(np.int32).max)),
                )
            else:
                negatives = np.empty((len(active_batch), 2, 0), dtype=np.int64)
            sampling_seconds += perf_counter() - sample_started

            active_tensor = torch.as_tensor(
                active_batch, dtype=torch.long, device=device
            )
            negatives_tensor = torch.as_tensor(
                negatives, dtype=torch.long, device=device
            )
            active_pairs = edge_tensor.index_select(0, active_tensor)
            left = active_pairs[:, 0]
            right = active_pairs[:, 1]

            alpha = learning_rate * (5.0 * n_samples) * (
                1.0 - epoch / max(1, n_epochs)
            )
            positive_scale = (1.0 - lambda_rep) / max(1.0, expected_active_edges)
            negative_scale = lambda_rep / max(1, negative_count)
            updates = torch.zeros_like(coordinates)

            delta = coordinates.index_select(0, left) - coordinates.index_select(0, right)
            d2 = torch.sum(delta * delta, dim=1)
            if schedule_by_rank:
                edge_weight = torch.ones_like(d2)
            else:
                edge_weight = torch.reciprocal(
                    rank_tensor.index_select(0, active_tensor)
                )
            attraction_factor = (
                alpha * positive_scale * edge_weight * (2.0 / (1.0 + d2 + epsilon))
            )
            positive_updates = attraction_factor.unsqueeze(1) * delta
            updates.index_add_(0, left, -positive_updates)
            updates.index_add_(0, right, positive_updates)

            if negatives_tensor.shape[2] > 0:
                sources = active_pairs.unsqueeze(2).expand_as(negatives_tensor)
                valid = negatives_tensor >= 0
                negative_sources = sources[valid]
                negative_targets = negatives_tensor[valid]
                negative_delta = coordinates.index_select(
                    0, negative_sources
                ) - coordinates.index_select(0, negative_targets)
                negative_d2 = torch.sum(negative_delta * negative_delta, dim=1)
                z = (margin - negative_d2) / temperature
                repulsion_factor = (
                    alpha * negative_scale * (2.0 / temperature) * torch.sigmoid(z)
                )
                negative_updates = repulsion_factor.unsqueeze(1) * negative_delta
                updates.index_add_(0, negative_sources, negative_updates)
                updates.index_add_(0, negative_targets, -negative_updates)

            coordinates.add_(updates)

        coordinates.sub_(coordinates.mean(dim=0, keepdim=True))
        if progress_callback is not None:
            progress_callback(epoch + 1, n_epochs)

    torch.mps.synchronize()
    embedding[...] = coordinates.cpu().numpy()
    if timing_callback is not None:
        elapsed = perf_counter() - optimizer_started
        timing_callback(sampling_seconds, max(0.0, elapsed - sampling_seconds))
