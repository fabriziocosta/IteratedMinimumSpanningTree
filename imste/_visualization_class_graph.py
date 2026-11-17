"""Reusable class-mixture plots for labeled low-dimensional embeddings."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from .class_graph import _edge_component_overlap, _fit_class_mixture_nodes
from .mst.prim import iterated_prim_edges

NODE_EDGE_WIDTH = 1.1


def _overlap_to_alpha(score: float, faint: float, opaque: float) -> float:
    """Map chance-level symmetric cross-assignment (0.5) to full opacity."""
    confusion = np.clip(float(score) / 0.5, 0.0, 1.0)
    return float(faint + confusion * (opaque - faint))


def plot_class_mixture_mst(
    X: np.ndarray,
    labels: object,
    embeddings: Mapping[str, object],
    *,
    title: str | None = None,
    n_msts: int = 2,
    max_components_per_class: int = 3,
    min_component_size: int = 20,
    point_alpha: float = 0.4,
    min_edge_alpha: float = 0.2,
    max_edge_alpha: float = 0.9,
    random_state: int = 42,
    class_colors: Mapping[object, object] | None = None,
):
    """Plot class-specific GMM centers and iterated-MST edges over embeddings.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Samples aligned with labels and embeddings. Used to validate row count;
        all overlap scores are computed on each supplied 2D embedding.
    labels : array-like of shape (n_samples,)
        Class label for each sample. Labels are used to fit one GMM per class
        and are never used to fit the supplied embeddings.
    embeddings : mapping of str to array-like or fitted IMSTE estimator
        One or more 2D embeddings aligned row-for-row with ``X`` and ``labels``.
        Each embedding gets its own mixture centers and iterated MST graph.

    Other parameters control the number of iterated trees and maximum GMM
    components per class. Mixtures use full covariance matrices. Their means
    become graph nodes. Same-class edges are omitted; remaining edges connect
    different classes. For each edge, a pairwise posterior compares the two
    endpoint Gaussian densities with equal priors on each component's assigned
    samples in 2D. The cross-assignment probabilities are averaged across both
    directions. Edge width matches node outlines, and opacity increases with
    this symmetric overlap score. ``point_alpha`` controls the scatter-point
    opacity independently of edge opacity. ``class_colors`` can provide a
    label-to-color mapping to keep class colors consistent across plots.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of matplotlib.axes.Axes
        One plot axes per embedding; the figure also includes a shared class
        colorbar.
    edge_scores : pandas.DataFrame
        Per-edge classes, tree rank, both directional cross-assignment
        probabilities, their symmetric mean, and edge alpha.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    if X.ndim != 2 or len(X) != len(labels):
        raise ValueError("X must be 2D and have one row per label.")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values.")
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array.")
    if not embeddings:
        raise ValueError("Provide at least one named embedding.")
    if n_msts < 1 or max_components_per_class < 1:
        raise ValueError("n_msts and max_components_per_class must be positive.")
    if min_component_size < 2:
        raise ValueError("min_component_size must be at least 2.")
    if not (0.0 <= point_alpha <= 1.0):
        raise ValueError("point_alpha must be between 0 and 1.")
    if not (0.0 <= min_edge_alpha <= max_edge_alpha <= 1.0):
        raise ValueError("Edge alpha limits must satisfy 0 <= min <= max <= 1.")

    coordinates_by_method = {}
    for method, embedding in embeddings.items():
        coordinates = np.asarray(
            embedding.embedding_ if hasattr(embedding, "embedding_") else embedding,
            dtype=np.float64,
        )
        if coordinates.shape != (len(labels), 2) or not np.isfinite(coordinates).all():
            raise ValueError(f"Embedding {method!r} must be a finite (n_samples, 2) array.")
        coordinates_by_method[str(method)] = coordinates

    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError("At least two classes are required for a class graph.")
    class_to_code = {label: code for code, label in enumerate(classes)}
    cmap_name = "tab20" if len(classes) > 10 else "tab10"
    if class_colors is None:
        color_list = plt.get_cmap(cmap_name).colors[:len(classes)]
    else:
        missing = [label for label in classes if label not in class_colors]
        if missing:
            raise ValueError(f"class_colors is missing labels: {missing!r}")
        color_list = [class_colors[label] for label in classes]
    cmap = ListedColormap(color_list)
    norm = BoundaryNorm(np.arange(-0.5, len(classes) + 0.5), len(classes))

    mixtures = {}
    for method, coordinates in coordinates_by_method.items():
        nodes, _, _ = _fit_class_mixture_nodes(
            coordinates,
            labels,
            max_components=max_components_per_class,
            min_component_size=min_component_size,
            random_state=random_state,
        )
        mixtures[method] = nodes

    method_names = list(coordinates_by_method)
    fig, axes_with_cbar = plt.subplots(
        1,
        len(method_names) + 1,
        figsize=(7.5 * len(method_names) + 0.8, 6.8),
        gridspec_kw={"width_ratios": [1] * len(method_names) + [0.04]},
        squeeze=False,
    )
    plot_axes = axes_with_cbar.ravel()[:-1]
    colorbar_axis = axes_with_cbar.ravel()[-1]
    edge_rows: list[dict[str, object]] = []
    point_codes = np.asarray([class_to_code[label] for label in labels])
    class_scatter = None

    for method, ax in zip(method_names, plot_axes):
        embedding = coordinates_by_method[method]
        nodes = mixtures[method]
        node_positions = np.vstack([node["position"] for node in nodes])
        edges, ranks = iterated_prim_edges(
            node_positions, n_msts=n_msts, return_ranks=True
        )
        class_scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=point_codes,
            cmap=cmap,
            norm=norm,
            s=8,
            alpha=point_alpha,
            linewidths=0,
            zorder=1,
        )

        for (left, right), rank in zip(edges, ranks):
            left, right, rank = int(left), int(right), int(rank)
            node_a, node_b = nodes[left], nodes[right]
            if node_a["class_label"] == node_b["class_label"]:
                continue
            overlap_a_to_b, overlap_b_to_a, overlap = _edge_component_overlap(
                nodes, left, right
            )
            alpha = _overlap_to_alpha(overlap, min_edge_alpha, max_edge_alpha)
            curvature = 0.055 * (rank - (n_msts + 1) / 2)
            ax.add_patch(
                FancyArrowPatch(
                    node_a["position"],
                    node_b["position"],
                    arrowstyle="-",
                    connectionstyle=f"arc3,rad={curvature}",
                    linewidth=NODE_EDGE_WIDTH,
                    color="black",
                    alpha=alpha,
                    zorder=3,
                )
            )
            edge_rows.append(
                {
                    "embedding": method,
                    "class_a": node_a["class_label"],
                    "component_a": int(node_a["component_id"]),
                    "class_b": node_b["class_label"],
                    "component_b": int(node_b["component_id"]),
                    "mst_rank": rank,
                    "component_a_posterior_on_b": overlap_a_to_b,
                    "component_b_posterior_on_a": overlap_b_to_a,
                    "symmetric_component_overlap": overlap,
                    "edge_alpha": alpha,
                }
            )

        sizes = 76 + 100 * np.asarray([node["mixture_weight"] for node in nodes])
        node_classes = np.asarray([class_to_code[node["class_label"]] for node in nodes])
        ax.scatter(
            node_positions[:, 0],
            node_positions[:, 1],
            c=node_classes,
            cmap=cmap,
            norm=norm,
            s=sizes,
            edgecolors="black",
            linewidths=NODE_EDGE_WIDTH,
            zorder=4,
        )
        ax.set_title(f"{method}: class GMM centers + {n_msts} MSTs")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")

    fig.colorbar(
        class_scatter,
        cax=colorbar_axis,
        ticks=np.arange(len(classes)),
        label="Class",
    )
    alpha_handles = [
        Line2D(
            [0], [0], color="black", linewidth=NODE_EDGE_WIDTH,
            alpha=_overlap_to_alpha(score, min_edge_alpha, max_edge_alpha),
            label=f"{score:.1f}",
        )
        for score in (0.0, 0.25, 0.5)
    ]
    fig.legend(
        handles=alpha_handles,
        title="Symmetric cross-component probability (edge opacity)",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(title or "Class-mixture MST graph over embeddings", y=0.97)
    fig.subplots_adjust(left=0.03, right=0.96, bottom=0.19, top=0.89, wspace=0.13)
    return fig, plot_axes, pd.DataFrame(edge_rows)
