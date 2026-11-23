"""Convenience functions for direct IMSTE fits."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import spmatrix

from ._estimator import IteratedMinimumSpanningTreeEmbedder


def fit_graph_transform(
    distance_graph: spmatrix,
    *,
    y: object | None = None,
    X: object | None = None,
    init_features: object | None = None,
    **estimator_params: Any,
) -> np.ndarray:
    """Fit IMSTE from a sparse distance graph and return its embedding.

    If ``X`` is supplied, IMSTE first embeds those features, then uses that
    layout to initialize optimization over ``distance_graph``. The graph
    determines the final optimization objective. Positive ``label_strength``
    uses categorical labels in ``y``; positive ``target_strength`` uses
    continuous numeric targets.
    Alternatively, ``init_features`` supplies features for standardized-PCA
    initialization. These two initialization options cannot be combined.
    Remaining keyword arguments configure the estimator.
    """
    estimator = IteratedMinimumSpanningTreeEmbedder(**estimator_params)
    return estimator.fit_graph_transform(
        distance_graph,
        y=y,
        X=X,
        init_features=init_features,
    )
