"""Analysis helpers for class-mixture minimum-spanning-tree relations."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

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
    """Backward-compatible import for the plotting API in ``imste.visualization``."""
    from .visualization import plot_class_mixture_mst as plotter

    return plotter(
        X,
        labels,
        embeddings,
        title=title,
        n_msts=n_msts,
        max_components_per_class=max_components_per_class,
        min_component_size=min_component_size,
        point_alpha=point_alpha,
        min_edge_alpha=min_edge_alpha,
        max_edge_alpha=max_edge_alpha,
        random_state=random_state,
        class_colors=class_colors,
    )
