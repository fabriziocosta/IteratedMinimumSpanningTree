"""Pluggable minimum-spanning-tree construction backends."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .prim import iterated_prim_edges
from .graph import iterated_graph_edges


def build_mst_edges(
    X: np.ndarray,
    *,
    n_msts: int,
    random_state: int | None,
    method: str,
    neighbors: int,
    inter_component_edges: int,
    max_neighbors: int | None,
    representatives_per_component: int = 10,
    neighbor_device: str = "cpu",
    neighbor_backend: str = "auto",
    progress_callback: Callable[[int, int, str], None] | None = None,
    return_ranks: bool = False,
    metric: str = "euclidean",
    candidate_edges_callback: Callable[[np.ndarray], None] | None = None,
    neighbor_graph_callback: Callable[[np.ndarray, np.ndarray], None] | None = None,
    neighbor_graph_neighbors: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch MST construction to the selected backend."""
    if method == "prim":
        return iterated_prim_edges(
            X, n_msts, progress_callback=progress_callback,
            return_ranks=return_ranks,
            metric=metric,
        )
    if method == "famst":
        from .famst import iterated_famst_edges

        return iterated_famst_edges(
            X,
            n_msts,
            random_state,
            neighbors,
            inter_component_edges,
            max_neighbors,
            progress_callback,
            representatives_per_component,
            neighbor_device=neighbor_device,
            neighbor_backend=neighbor_backend,
            return_ranks=return_ranks,
            metric=metric,
            candidate_edges_callback=candidate_edges_callback,
            neighbor_graph_callback=neighbor_graph_callback,
            neighbor_graph_neighbors=neighbor_graph_neighbors,
        )
    raise ValueError("mst_method must be either 'prim' or 'famst'.")


__all__ = ["build_mst_edges", "iterated_prim_edges", "iterated_graph_edges"]
