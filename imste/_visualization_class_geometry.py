"""Reusable 2D renderers for IMSTE embeddings.

Matplotlib, pandas, SciPy, scikit-learn, Datashader, and ipywidgets are optional
visualization dependencies. Import this module only in visualization workflows.
"""

from __future__ import annotations

from html import escape as html_escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch, PathPatch
from matplotlib.path import Path as MplPath
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from ._visualization_core import (
    _as_observations,
    _embedding_array,
    _plot_embedding_frame,
    cross_validate_embedding_knn,
)
from ._visualization_interactive import _add_compound_path_union_outline


def _fit_component_ellipses(
    points: np.ndarray,
    *,
    pixel_area: float,
    n_std: float,
    max_ellipses: int,
    min_precision: float,
    min_recall: float,
    random_state: int,
):
    """Fit one to ``max_ellipses`` ellipses, splitting when one overcovers.

    Precision estimates the fraction of the total fitted ellipse area occupied
    by the component. Recall estimates the fraction of component pixels inside
    their assigned ellipse. The smallest split meeting both thresholds wins;
    otherwise the best precision/recall balance is used.
    """
    sample_size = min(len(points), 10_000)
    sample_indices = np.linspace(0, len(points) - 1, sample_size, dtype=int)
    sample = points[sample_indices]
    candidates = []

    for n_ellipses in range(1, min(max_ellipses, len(sample) // 3) + 1):
        if n_ellipses == 1:
            sample_labels = np.zeros(len(sample), dtype=np.intp)
            centers = sample.mean(axis=0, keepdims=True)
        else:
            model = KMeans(
                n_clusters=n_ellipses,
                n_init=10,
                random_state=random_state,
            ).fit(sample)
            sample_labels = model.labels_
            centers = model.cluster_centers_

        ellipses = []
        enclosed_count = 0
        total_ellipse_area = 0.0
        valid = True
        for cluster_id, center in enumerate(centers):
            cluster = sample[sample_labels == cluster_id]
            if len(cluster) < 3:
                valid = False
                break
            covariance = np.cov(cluster, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues[order], 0.0)
            if eigenvalues[1] <= np.finfo(float).eps:
                valid = False
                break
            eigenvectors = eigenvectors[:, order]
            inverse_covariance = (
                eigenvectors / eigenvalues
            ) @ eigenvectors.T
            delta = cluster - center
            mahalanobis_squared = np.einsum(
                "ij,jk,ik->i", delta, inverse_covariance, delta
            )
            enclosed_count += int((mahalanobis_squared <= n_std**2).sum())
            total_ellipse_area += (
                np.pi * n_std**2 * np.sqrt(eigenvalues[0] * eigenvalues[1])
            )
            angle = np.degrees(
                np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
            )
            ellipses.append(
                (
                    center,
                    2 * n_std * np.sqrt(eigenvalues[0]),
                    2 * n_std * np.sqrt(eigenvalues[1]),
                    angle,
                )
            )
        if not valid:
            continue

        precision = min(1.0, (len(points) * pixel_area) / total_ellipse_area)
        recall = enclosed_count / len(sample)
        score = 2 * precision * recall / max(precision + recall, 1e-12)
        candidates.append((n_ellipses, precision, recall, score, ellipses))
        if precision >= min_precision and recall >= min_recall:
            return ellipses, n_ellipses, precision, recall

    if not candidates:
        return [], 0, 0.0, 0.0
    best = max(candidates, key=lambda candidate: candidate[3] - 0.02 * (candidate[0] - 1))
    return best[4], best[0], best[1], best[2]


def _render_class_neighborhood_geometry(
    embedding: np.ndarray,
    obs: pd.DataFrame,
    *,
    k: int = 5,
    color_key: str = "celltype",
    cmap: str = "tab20",
    alpha_range: tuple[float, float] = (0.2, 0.9),
    cv_neighbors: int = 5,
    cv_splits: int = 5,
    cv_random_state: int = 42,
    distance_std_threshold: float | None = 1.0,
    ax=None,
    ellipse_ax=None,
    ellipse_overlay_ax=None,
    ellipse_n_std: float = 2.0,
    ellipse_max_per_patch: int = 3,
    ellipse_min_precision: float = 0.65,
    ellipse_min_recall: float = 0.8,
    legend_position: str | None = None,
    draw_union: bool = True,
):
    """Plot the union of same-class local convex hulls in the 2D embedding.

    For each point, its neighborhood contains the point itself and its ``k``
    nearest other points from the same class. The convex hulls of those
    neighborhoods are combined into one filled path per class, so overlaps
    within a class receive that class's alpha only once. A stratified
    cross-validated kNN classifier runs on the 2D coordinates first; incorrectly
    classified and CV-ineligible points are omitted. Then, by default, points
    whose mean distance to their same-class ``k`` nearest neighbors exceeds
    that class's mean plus one standard deviation are also removed. Set
    ``distance_std_threshold=None`` to disable the distance filter. Class patch
    fill alpha is scaled linearly by the number of retained cells between
    ``alpha_range[0]`` and ``alpha_range[1]``.
    Neighborhoods without three non-collinear coordinates are skipped.
    Ellipses are split into up to ``ellipse_max_per_patch`` clusters when the
    estimated area precision or pixel coverage falls below its threshold.
    """
    if (
        not isinstance(k, (int, np.integer))
        or isinstance(k, bool)
        or k < 1
    ):
        raise ValueError("k must be a positive integer.")
    if (
        len(alpha_range) != 2
        or not np.isfinite(alpha_range).all()
        or not 0 <= alpha_range[0] <= alpha_range[1] <= 1
    ):
        raise ValueError("alpha_range must be a (min, max) pair in [0, 1].")
    if distance_std_threshold is not None and (
        not np.isfinite(distance_std_threshold) or distance_std_threshold < 0
    ):
        raise ValueError("distance_std_threshold must be non-negative or None.")
    if not np.isfinite(ellipse_n_std) or ellipse_n_std <= 0:
        raise ValueError("ellipse_n_std must be positive and finite.")
    if (
        not isinstance(ellipse_max_per_patch, (int, np.integer))
        or isinstance(ellipse_max_per_patch, bool)
        or ellipse_max_per_patch < 1
    ):
        raise ValueError("ellipse_max_per_patch must be a positive integer.")
    if not 0 <= ellipse_min_precision <= 1 or not 0 <= ellipse_min_recall <= 1:
        raise ValueError("ellipse precision and recall thresholds must be in [0, 1].")
    if color_key not in obs:
        raise KeyError(f"Color column {color_key!r} is not present in obs.")

    coordinates = _embedding_array(embedding)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("embedding must contain at least two coordinates per row.")
    coordinates = np.ascontiguousarray(coordinates[:, :2])
    if not np.isfinite(coordinates).all():
        raise ValueError("embedding coordinates must be finite.")
    if len(obs) != len(coordinates):
        raise ValueError("obs rows must match the embedding rows.")

    labels = pd.Categorical(obs[color_key])
    cv_results = cross_validate_embedding_knn(
        coordinates,
        obs,
        label_key=color_key,
        n_neighbors=cv_neighbors,
        n_splits=cv_splits,
        random_state=cv_random_state,
    )
    cv_correct_mask = cv_results["correct_mask"]
    category_colors = plt.get_cmap(cmap, max(len(labels.categories), 1))
    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure
    ellipse_axes = [
        ellipse_axis
        for ellipse_axis in (ellipse_ax, ellipse_overlay_ax)
        if ellipse_axis is not None
    ]
    if any(ellipse_axis.figure is not fig for ellipse_axis in ellipse_axes):
        raise ValueError("ax and ellipse axes must belong to the same figure.")
    if legend_position is None:
        legend_position = "right" if created_figure else "inside"
    if legend_position not in {"inside", "right", "bottom"}:
        raise ValueError("legend_position must be 'inside', 'right', or 'bottom'.")
    class_patches = []
    filtered_plot_mask = np.zeros(len(coordinates), dtype=bool)
    filtered_outliers = 0

    for category_index, category in enumerate(labels.categories):
        class_indices = np.flatnonzero(
            (labels.codes == category_index) & cv_correct_mask
        )
        if len(class_indices) < 2:
            continue

        class_coordinates = coordinates[class_indices]
        n_same_class_neighbors = min(k, len(class_indices) - 1)
        distance_search = NearestNeighbors(n_neighbors=n_same_class_neighbors + 1)
        distance_search.fit(class_coordinates)
        neighbor_distances, neighbor_indices = distance_search.kneighbors(
            class_coordinates, return_distance=True
        )
        mean_neighbor_distances = np.empty(len(class_indices), dtype=np.float64)
        for local_index, (distances, neighbors) in enumerate(
            zip(neighbor_distances, neighbor_indices)
        ):
            other_mask = neighbors != local_index
            mean_neighbor_distances[local_index] = distances[other_mask][
                :n_same_class_neighbors
            ].mean()

        if distance_std_threshold is None:
            inlier_mask = np.ones(len(class_indices), dtype=bool)
        else:
            cutoff = mean_neighbor_distances.mean() + (
                distance_std_threshold * mean_neighbor_distances.std()
            )
            inlier_mask = mean_neighbor_distances <= cutoff
        filtered_outliers += int((~inlier_mask).sum())
        class_indices = class_indices[inlier_mask]
        class_coordinates = class_coordinates[inlier_mask]
        if len(class_indices) < 3:
            continue
        filtered_plot_mask[class_indices] = True

        n_neighbors = min(k + 1, len(class_indices))
        neighbors = NearestNeighbors(n_neighbors=n_neighbors)
        neighbors.fit(class_coordinates)
        neighborhood_indices = neighbors.kneighbors(
            class_coordinates, return_distance=False
        )

        path_vertices = []
        path_codes = []
        seen_neighborhoods = set()
        for neighborhood in neighborhood_indices:
            neighborhood_key = tuple(sorted(map(int, neighborhood)))
            if neighborhood_key in seen_neighborhoods:
                continue
            seen_neighborhoods.add(neighborhood_key)
            neighborhood_coordinates = class_coordinates[neighborhood]
            try:
                hull = ConvexHull(neighborhood_coordinates)
            except QhullError:
                continue

            polygon = neighborhood_coordinates[hull.vertices]
            polygon = np.vstack((polygon, polygon[0]))
            codes = np.full(len(polygon), MplPath.LINETO, dtype=np.uint8)
            codes[0] = MplPath.MOVETO
            codes[-1] = MplPath.CLOSEPOLY
            path_vertices.append(polygon)
            path_codes.append(codes)

        color = category_colors(category_index)
        if path_vertices:
            compound_path = MplPath(
                np.concatenate(path_vertices), np.concatenate(path_codes)
            )
            class_patches.append(
                (category_index, category, color, compound_path, len(class_indices))
            )

    legend_handles = []
    ellipse_legend_handles = []
    n_ellipses = 0
    n_split_patches = 0
    split_patches_by_class = {}
    class_counts = [record[4] for record in class_patches]
    if class_counts:
        min_count, max_count = min(class_counts), max(class_counts)
        for _, category, color, compound_path, n_cells in class_patches:
            if min_count == max_count:
                patch_alpha = (alpha_range[0] + alpha_range[1]) / 2
            else:
                patch_alpha = alpha_range[0] + (
                    (n_cells - min_count) / (max_count - min_count)
                ) * (alpha_range[1] - alpha_range[0])
            border_alpha = min(patch_alpha + 0.1, 0.95)
            component_labels, n_components, x_grid, y_grid = (
                _add_compound_path_union_outline(
                    ax,
                    compound_path,
                    color,
                    linewidth=1.0,
                    alpha=border_alpha,
                    draw_outline=draw_union,
                )
            )
            if draw_union:
                ax.add_patch(
                    PathPatch(
                        compound_path,
                        facecolor=(*color[:3], patch_alpha),
                        edgecolor="none",
                    )
                )
                legend_handles.append(
                    Patch(
                        facecolor=(*color[:3], patch_alpha),
                        edgecolor=(*color[:3], border_alpha),
                        linewidth=1.0,
                        label=str(category),
                    )
                )
            if ellipse_axes:
                class_drew_ellipse = False
                for component_id in range(1, n_components + 1):
                    rows, columns = np.nonzero(component_labels == component_id)
                    if len(rows) < 3:
                        continue
                    component_points = np.column_stack(
                        (x_grid[columns], y_grid[rows])
                    )
                    pixel_area = (
                        (x_grid[1] - x_grid[0]) * (y_grid[1] - y_grid[0])
                    )
                    fitted, split_count, _, _ = _fit_component_ellipses(
                        component_points,
                        pixel_area=pixel_area,
                        n_std=ellipse_n_std,
                        max_ellipses=ellipse_max_per_patch,
                        min_precision=ellipse_min_precision,
                        min_recall=ellipse_min_recall,
                        random_state=cv_random_state,
                    )
                    if split_count > 1:
                        n_split_patches += 1
                        split_patches_by_class[category] = (
                            split_patches_by_class.get(category, 0) + 1
                        )
                    if split_count > 1:
                        ellipse_paths = []
                        for center, width, height, angle in fitted:
                            ellipse = Ellipse(
                                center, width=width, height=height, angle=angle
                            )
                            ellipse_paths.append(
                                ellipse.get_path().transformed(ellipse.get_transform())
                            )
                        ellipse_union_path = MplPath.make_compound_path(*ellipse_paths)
                        for target_ax in ellipse_axes:
                            target_ax.add_patch(
                                PathPatch(
                                    ellipse_union_path,
                                    facecolor=(*color[:3], 0.12),
                                    edgecolor="none",
                                    zorder=1,
                                )
                            )
                            _add_compound_path_union_outline(
                                target_ax,
                                ellipse_union_path,
                                color,
                                linewidth=1.2,
                                alpha=0.9,
                            )
                    else:
                        for center, width, height, angle in fitted:
                            for target_ax in ellipse_axes:
                                target_ax.add_patch(
                                    Ellipse(
                                        center,
                                        width=width,
                                        height=height,
                                        angle=angle,
                                        facecolor=(*color[:3], 0.12),
                                        edgecolor=(*color[:3], 0.9),
                                        linewidth=1.2,
                                        zorder=1,
                                    )
                                )
                    if fitted:
                        class_drew_ellipse = True
                    n_ellipses += len(fitted)
                if class_drew_ellipse:
                    ellipse_legend_handles.append(
                        Patch(
                            facecolor=(*color[:3], 0.12),
                            edgecolor=(*color[:3], 0.9),
                            linewidth=1.2,
                            label=str(category),
                        )
                    )

    if draw_union and filtered_plot_mask.any():
        ax.update_datalim(coordinates[filtered_plot_mask])
    ax.autoscale_view()
    ax.margins(0.03)
    ax.set_aspect("equal", adjustable="datalim")
    if distance_std_threshold is None:
        filter_description = "distance filter off"
    else:
        filter_description = (
            f"{filtered_outliers} cells filtered at class mean + "
            f"{distance_std_threshold:g} SD"
        )
    if draw_union:
        ax.set_title(
            f"Unions of {k}-nearest-neighbor hulls by {color_key}\n"
            f"{filter_description}; 2D CV accuracy "
            f"{cv_results['accuracy']:.3f} (balanced "
            f"{cv_results['balanced_accuracy']:.3f})"
        )
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    if legend_handles:
        if legend_position == "right":
            legend_options = {
                "bbox_to_anchor": (1.02, 1), "loc": "upper left", "frameon": False
            }
        elif legend_position == "bottom":
            legend_options = {
                "bbox_to_anchor": (0.5, -0.16),
                "loc": "upper center",
                "ncol": min(5, len(legend_handles)),
                "frameon": False,
            }
        else:
            legend_options = {
                "loc": "upper right", "frameon": True, "framealpha": 0.8
            }
        ax.legend(
            handles=legend_handles,
            title=color_key,
            **legend_options,
        )
    if ellipse_axes:
        split_summary = ", ".join(
            f"{category}: {count}"
            for category, count in split_patches_by_class.items()
        ) or "none"
        for target_ax in ellipse_axes:
            target_ax.set_title(
                f"{ellipse_n_std:g}σ ellipses: {n_ellipses} total, "
                f"{n_split_patches} patches split; classes: {split_summary}"
            )
            target_ax.set_xlabel("Embedding 1")
            target_ax.set_ylabel("Embedding 2")
            target_ax.set_aspect("equal", adjustable="datalim")
        if ellipse_ax is not None and ellipse_legend_handles:
            if legend_position == "bottom":
                ellipse_legend_options = {
                    "bbox_to_anchor": (0.5, -0.16),
                    "loc": "upper center",
                    "ncol": min(5, len(ellipse_legend_handles)),
                    "frameon": False,
                }
            elif legend_position == "right":
                ellipse_legend_options = {
                    "bbox_to_anchor": (1.02, 1),
                    "loc": "upper left",
                    "frameon": False,
                }
            else:
                ellipse_legend_options = {
                    "loc": "upper right",
                    "frameon": True,
                    "framealpha": 0.8,
                }
            ellipse_ax.legend(
                handles=ellipse_legend_handles,
                title=color_key,
                **ellipse_legend_options,
            )
    if created_figure:
        fig.tight_layout()
    return fig, ax


def plot_knn_union(
    embedding: np.ndarray,
    labels,
    *,
    label_key: str | None = None,
    k: int = 5,
    cmap: str = "tab20",
    alpha_range: tuple[float, float] = (0.2, 0.9),
    cv_neighbors: int = 5,
    cv_splits: int = 5,
    random_state: int = 42,
    distance_std_threshold: float | None = 1.0,
    ax=None,
    legend_position: str | None = None,
):
    """Render the cross-validated union of same-label kNN neighborhood hulls.

    ``embedding`` may be an array or fitted IMSTE estimator. ``labels`` may be
    a one-dimensional sequence/Series or a metadata DataFrame. For a DataFrame,
    set ``label_key`` to select its label column.
    """
    obs, label_key = _as_observations(labels, label_key)
    return _render_class_neighborhood_geometry(
        embedding,
        obs,
        k=k,
        color_key=label_key,
        cmap=cmap,
        alpha_range=alpha_range,
        cv_neighbors=cv_neighbors,
        cv_splits=cv_splits,
        cv_random_state=random_state,
        distance_std_threshold=distance_std_threshold,
        ax=ax,
        legend_position=legend_position,
        draw_union=True,
    )


def plot_ellipses(
    embedding: np.ndarray,
    labels,
    *,
    label_key: str | None = None,
    k: int = 5,
    cmap: str = "tab20",
    cv_neighbors: int = 5,
    cv_splits: int = 5,
    random_state: int = 42,
    distance_std_threshold: float | None = 1.0,
    n_std: float = 2.0,
    max_ellipses_per_region: int = 3,
    min_precision: float = 0.65,
    min_recall: float = 0.8,
    point_size: float = 7,
    point_alpha: float = 0.5,
    ax=None,
    legend_position: str | None = None,
):
    """Render ellipses for array or fitted IMSTE coordinates and aligned labels."""
    obs, label_key = _as_observations(labels, label_key)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure
    _plot_embedding_frame(
        embedding,
        obs,
        color_keys=(label_key,),
        point_size=point_size,
        point_alpha=point_alpha,
        ax=ax,
        legend_position=legend_position,
    )
    return _render_class_neighborhood_geometry(
        embedding,
        obs,
        k=k,
        color_key=label_key,
        cmap=cmap,
        cv_neighbors=cv_neighbors,
        cv_splits=cv_splits,
        cv_random_state=random_state,
        distance_std_threshold=distance_std_threshold,
        ax=ax,
        ellipse_ax=ax,
        ellipse_n_std=n_std,
        ellipse_max_per_patch=max_ellipses_per_region,
        ellipse_min_precision=min_precision,
        ellipse_min_recall=min_recall,
        legend_position=legend_position,
        draw_union=False,
    )
