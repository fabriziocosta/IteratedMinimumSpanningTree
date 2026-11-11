"""Reusable Gaussian-mixture MST graph plots for low-dimensional embeddings."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from scipy.special import expit
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from imste.mst.prim import iterated_prim_edges

NODE_EDGE_WIDTH = 1.1


def _fit_class_mixture_nodes(
    embedding: np.ndarray,
    labels: np.ndarray,
    *,
    max_components: int,
    min_component_size: int,
    random_state: int,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    """Fit full-covariance GMMs per class and return component nodes and labels."""
    classes = np.unique(labels)
    nodes: list[dict[str, object]] = []
    sample_node_ids = np.full(len(labels), -1, dtype=np.intp)
    class_codes = np.empty(len(labels), dtype=np.intp)

    for class_code, class_label in enumerate(classes):
        sample_indices = np.flatnonzero(labels == class_label)
        class_codes[sample_indices] = class_code
        n_components = min(
            max_components,
            max(1, len(sample_indices) // min_component_size),
        )
        while True:
            mixture = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                reg_covar=1e-5,
                n_init=3,
                random_state=random_state + class_code,
            ).fit(embedding[sample_indices])
            component_ids = mixture.predict(embedding[sample_indices])
            counts = np.bincount(component_ids, minlength=n_components)
            if n_components == 1 or counts.min() >= min_component_size:
                break
            n_components -= 1

        for component_id in range(n_components):
            members = sample_indices[component_ids == component_id]
            node_id = len(nodes)
            sample_node_ids[members] = node_id
            nodes.append(
                {
                    "class_label": class_label,
                    "class_code": class_code,
                    "component_id": component_id,
                    "position": mixture.means_[component_id].copy(),
                    "covariance": mixture.covariances_[component_id].copy(),
                    "mixture_weight": float(mixture.weights_[component_id]),
                    "sample_indices": members,
                    "sample_coordinates": embedding[members].copy(),
                    "n_samples": len(members),
                }
            )

    if np.any(sample_node_ids < 0):
        raise RuntimeError("Some samples were not assigned to a class-mixture component.")
    return nodes, sample_node_ids, class_codes


def _edge_component_overlap(
    nodes: list[dict[str, object]],
    left: int,
    right: int,
) -> tuple[float, float, float]:
    """Return symmetric cross-component posterior overlap in 2D."""
    node_a, node_b = nodes[left], nodes[right]
    if node_a["class_label"] == node_b["class_label"]:
        raise ValueError("Same-class edges should be filtered before scoring.")
    indices_a = np.asarray(node_a["sample_indices"], dtype=np.intp)
    indices_b = np.asarray(node_b["sample_indices"], dtype=np.intp)
    covariance_a = np.asarray(node_a["covariance"])
    covariance_b = np.asarray(node_b["covariance"])
    log_density_a_on_b = multivariate_normal.logpdf(
        np.asarray(node_b["sample_coordinates"]),
        mean=np.asarray(node_a["position"]),
        cov=covariance_a,
        allow_singular=True,
    )
    log_density_b_on_b = multivariate_normal.logpdf(
        np.asarray(node_b["sample_coordinates"]),
        mean=np.asarray(node_b["position"]),
        cov=covariance_b,
        allow_singular=True,
    )
    log_density_a_on_a = multivariate_normal.logpdf(
        np.asarray(node_a["sample_coordinates"]),
        mean=np.asarray(node_a["position"]),
        cov=covariance_a,
        allow_singular=True,
    )
    log_density_b_on_a = multivariate_normal.logpdf(
        np.asarray(node_a["sample_coordinates"]),
        mean=np.asarray(node_b["position"]),
        cov=covariance_b,
        allow_singular=True,
    )
    # Equal priors make this an endpoint distinguishability score, independent
    # of the relative number of samples in each component.
    prob_a_to_b = float(np.mean(expit(log_density_a_on_b - log_density_b_on_b)))
    prob_b_to_a = float(np.mean(expit(log_density_b_on_a - log_density_a_on_a)))
    return prob_a_to_b, prob_b_to_a, (prob_a_to_b + prob_b_to_a) / 2.0


def _overlap_to_alpha(score: float, faint: float, opaque: float) -> float:
    """Map chance-level symmetric cross-assignment (0.5) to full opacity."""
    confusion = np.clip(float(score) / 0.5, 0.0, 1.0)
    return float(faint + confusion * (opaque - faint))


def compare_class_graph_relations(
    edge_scores: pd.DataFrame,
    *,
    method_a: str = "IMSTE",
    method_b: str = "UMAP",
    dataset: object | None = None,
) -> pd.DataFrame:
    """Rank class-pair MST relations unique to either embedding.

    Component-level edges are collapsed to unordered class pairs because the
    GMM components are fit independently in each embedding. A relation is
    considered present when at least one cross-class component edge connects
    that class pair. Duplicate appearances across iterated MSTs count once.
    ``edge_error`` is the mean chance-normalized symmetric cross-assignment
    probability across the distinct component edges for that class pair; the
    table also reports the maximum edge error and component-level details.

    Rows are sorted by descending edge error separately for relations unique
    to ``method_a`` and ``method_b``. A chance-level cross-assignment
    probability of 0.5 maps to edge error 1.0; perfect separation maps to 0.
    """
    required = {
        "embedding", "class_a", "component_a", "class_b", "component_b",
        "symmetric_component_overlap",
    }
    missing = required.difference(edge_scores.columns)
    if missing:
        raise ValueError(f"edge_scores is missing columns: {sorted(missing)}")
    if method_a == method_b:
        raise ValueError("method_a and method_b must be different embedding names.")
    frame = edge_scores.copy()
    if dataset is not None:
        if "dataset" not in frame:
            raise ValueError("dataset was specified but edge_scores has no dataset column.")
        frame = frame.loc[frame["dataset"] == dataset]
    frame = frame.loc[frame["embedding"].isin([method_a, method_b])]

    def label_key(value: object) -> tuple[str, str]:
        return type(value).__name__, str(value)

    # method -> unordered label pair -> component edge -> edge error
    relations: dict[str, dict[tuple[object, object], dict[tuple[object, object], float]]] = {
        method_a: {}, method_b: {}
    }
    for row in frame.to_dict("records"):
        left, right = row["class_a"], row["class_b"]
        if label_key(left) <= label_key(right):
            class_pair = (left, right)
            component_pair = (row["component_a"], row["component_b"])
        else:
            class_pair = (right, left)
            component_pair = (row["component_b"], row["component_a"])
        error = float(np.clip(row["symmetric_component_overlap"] / 0.5, 0.0, 1.0))
        method = row["embedding"]
        relation_edges = relations[method].setdefault(class_pair, {})
        # The same component edge can appear in multiple iterated MSTs.
        relation_edges[component_pair] = max(error, relation_edges.get(component_pair, 0.0))

    rows: list[dict[str, object]] = []
    for source, target in ((method_a, method_b), (method_b, method_a)):
        unique_pairs = set(relations[source]).difference(relations[target])
        ranked = []
        for class_pair in unique_pairs:
            component_errors = relations[source][class_pair]
            errors = list(component_errors.values())
            ranked.append(
                {
                    "class_a": class_pair[0],
                    "class_b": class_pair[1],
                    "present_in": source,
                    "missing_from": target,
                    "edge_error": float(np.mean(errors)),
                    "max_component_edge_error": float(np.max(errors)),
                    "n_component_edges": len(errors),
                    "component_edges": tuple(
                        (component_a, component_b, component_errors[(component_a, component_b)])
                        for component_a, component_b in sorted(
                            component_errors, key=lambda pair: (str(pair[0]), str(pair[1]))
                        )
                    ),
                }
            )
        ranked.sort(key=lambda row: (-row["edge_error"], -row["max_component_edge_error"],
                                     label_key(row["class_a"]), label_key(row["class_b"])))
        for rank, row in enumerate(ranked, start=1):
            rows.append({"rank": rank, **row})

    return pd.DataFrame(
        rows,
        columns=[
            "rank", "class_a", "class_b", "present_in", "missing_from",
            "edge_error", "max_component_edge_error", "n_component_edges",
            "component_edges",
        ],
    )


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
