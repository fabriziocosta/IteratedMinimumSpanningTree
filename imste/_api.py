"""Convenience functions for direct IMSTE fits."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import spmatrix

from ._estimator import IteratedMinimumSpanningTreeEmbedder


def fit_graph_transform(
    distance_graph: spmatrix,
    *,
    init_features: object | None = None,
    **estimator_params: Any,
) -> np.ndarray:
    """Fit IMSTE from a sparse distance graph and return its embedding.

    ``init_features`` optionally supplies one feature row per graph node for
    PCA initialization. Remaining keyword arguments configure the estimator.
    """
    estimator = IteratedMinimumSpanningTreeEmbedder(**estimator_params)
    return estimator.fit_graph_transform(
        distance_graph,
        init_features=init_features,
    )
