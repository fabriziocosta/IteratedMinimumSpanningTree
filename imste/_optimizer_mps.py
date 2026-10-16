"""Batched PyTorch MPS optimizer for Apple GPUs.

Each active-edge batch is applied simultaneously, with batches processed
sequentially. Negative targets are sampled on MPS by rejecting graph neighbors;
rare dense-neighborhood fallbacks use an exact CPU complement sampler.
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import numpy as np
import torch

from ._optimizer import (
    _active_edges_for_rank_epoch,
    expected_active_edge_count,
    rank_phases,
)


def _sample_negative_targets_mps(
    active_pairs: torch.Tensor,
    degree: torch.Tensor,
    adjacency_keys: torch.Tensor,
    adjacency_keys_cpu: np.ndarray,
    n_samples: int,
    negative_ratio: int,
    seed: int,
) -> torch.Tensor:
    """Rejection-sample non-neighbors on MPS, with a rare exact CPU fallback."""
    n_edges = active_pairs.shape[0]
    if negative_ratio == 0:
        return torch.empty((n_edges, 2, 0), dtype=torch.long, device="mps")

    sources = (
        active_pairs.unsqueeze(2)
        .expand(n_edges, 2, negative_ratio)
        .reshape(-1)
    )
    eligible = degree.index_select(0, sources) < n_samples - 1
    generator = torch.Generator(device="mps")
    generator.manual_seed(seed)
    candidate_targets = torch.randint(
        n_samples,
        (sources.numel(), 4),
        dtype=torch.long,
        device="mps",
        generator=generator,
    )
    candidate_sources = sources.unsqueeze(1).expand_as(candidate_targets)
    keys = candidate_sources * n_samples + candidate_targets
    locations = torch.searchsorted(adjacency_keys, keys.reshape(-1))
    in_bounds = locations < adjacency_keys.numel()
    bounded_locations = torch.clamp(locations, max=adjacency_keys.numel() - 1)
    blocked = adjacency_keys.index_select(0, bounded_locations) == keys.reshape(-1)
    blocked &= in_bounds
    valid = (
        (candidate_targets != candidate_sources)
        & ~blocked.reshape_as(candidate_targets)
        & eligible.unsqueeze(1)
    )
    has_candidate = valid.any(dim=1)
    selected = valid.to(torch.int64).argmax(dim=1)
    targets = candidate_targets.gather(1, selected.unsqueeze(1)).squeeze(1)
    invalid = ~has_candidate & eligible
    failed_positions = torch.nonzero(invalid, as_tuple=False).flatten()
    if failed_positions.numel():
        failed_sources = sources.index_select(0, failed_positions).cpu().numpy()
        failed_positions_cpu = failed_positions.cpu().numpy()
        fallback_rng = np.random.RandomState(seed)
        fallback_targets = np.empty(len(failed_sources), dtype=np.int64)
        for fallback_index, source in enumerate(failed_sources):
            start = int(source) * n_samples
            first = np.searchsorted(adjacency_keys_cpu, start)
            last = np.searchsorted(adjacency_keys_cpu, start + n_samples)
            blocked = adjacency_keys_cpu[first:last] - start
            available = np.ones(n_samples, dtype=bool)
            available[int(source)] = False
            available[blocked] = False
            fallback_targets[fallback_index] = int(
                fallback_rng.choice(np.flatnonzero(available))
            )
        targets_cpu = targets.cpu().numpy()
        targets_cpu[failed_positions_cpu] = fallback_targets
        targets = torch.as_tensor(targets_cpu, dtype=torch.long, device="mps")

    targets = targets.masked_fill(~eligible, -1)
    return targets.reshape(n_edges, 2, negative_ratio)


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
    degree = counts.astype(np.int64, copy=False)
    adjacency_keys_cpu = source.astype(np.int64, copy=False) * n_samples + target

    device = torch.device("mps")
    coordinates = torch.as_tensor(embedding, dtype=torch.float32, device=device)
    edge_tensor = torch.as_tensor(edge_pairs, dtype=torch.long, device=device)
    rank_tensor = torch.as_tensor(ranks, dtype=torch.float32, device=device)
    degree_tensor = torch.as_tensor(degree, dtype=torch.long, device=device)
    adjacency_keys = torch.as_tensor(
        adjacency_keys_cpu, dtype=torch.long, device=device
    )
    seed_rng = np.random.RandomState(random_state)
    sampling_events = []
    update_events = []
    use_device_events = timing_callback is not None and hasattr(torch.mps, "Event")
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
            active_tensor = torch.as_tensor(
                active_batch, dtype=torch.long, device=device
            )
            active_pairs = edge_tensor.index_select(0, active_tensor)
            sampling_start = sampling_end = update_end = None
            if use_device_events:
                sampling_start = torch.mps.Event(enable_timing=True)
                sampling_end = torch.mps.Event(enable_timing=True)
                update_end = torch.mps.Event(enable_timing=True)
                sampling_start.record()
            negatives_tensor = _sample_negative_targets_mps(
                active_pairs,
                degree_tensor,
                adjacency_keys,
                adjacency_keys_cpu,
                n_samples,
                negative_ratio if lambda_rep > 0.0 else 0,
                int(batch_seed_rng.randint(0, np.iinfo(np.int32).max)),
            )
            if sampling_end is not None:
                sampling_end.record()
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
            if update_end is not None:
                update_end.record()
                sampling_events.append((sampling_start, sampling_end))
                update_events.append((sampling_end, update_end))

        coordinates.sub_(coordinates.mean(dim=0, keepdim=True))
        if progress_callback is not None:
            progress_callback(epoch + 1, n_epochs)

    torch.mps.synchronize()
    embedding[...] = coordinates.cpu().numpy()
    if timing_callback is not None:
        if use_device_events:
            try:
                sampling_seconds = sum(
                    start.elapsed_time(end) for start, end in sampling_events
                ) / 1000.0
                update_seconds = sum(
                    start.elapsed_time(end) for start, end in update_events
                ) / 1000.0
            except RuntimeError:
                # Some PyTorch MPS builds can return out-of-order timestamps
                # for recorded events. Timing is diagnostic only, so preserve
                # the completed fit and fall back to the synchronized wall time.
                sampling_seconds = 0.0
                update_seconds = perf_counter() - optimizer_started
        else:
            sampling_seconds = 0.0
            update_seconds = perf_counter() - optimizer_started
        timing_callback(sampling_seconds, update_seconds)
