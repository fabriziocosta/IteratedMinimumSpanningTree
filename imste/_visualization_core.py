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

def _embedding_array(embedding) -> np.ndarray:
    """Resolve coordinates from an array or fitted IMSTE estimator."""
    if hasattr(embedding, "embedding_"):
        embedding = embedding.embedding_
    elif hasattr(embedding, "reference_embedding_"):
        embedding = embedding.reference_embedding_
    coordinates = np.asarray(embedding, dtype=np.float64)
    return coordinates


def _plot_embedding_frame(
    embedding: np.ndarray,
    obs: pd.DataFrame,
    *,
    color_keys: tuple[str, ...] = ("tech", "celltype"),
    point_size: float = 7,
    point_alpha: float = 0.8,
    ax=None,
    legend_position: str | None = None,
):
    """Plot the embedding colored by available metadata columns."""
    embedding = _embedding_array(embedding)
    if len(obs) != len(embedding):
        raise ValueError("obs rows must match the embedding rows.")
    color_keys = [key for key in color_keys if key in obs]
    if not color_keys:
        raise ValueError("None of the requested color columns are present in obs.")
    if ax is None:
        fig, axes = plt.subplots(
            1,
            len(color_keys),
            figsize=(1.3 * point_size * len(color_keys), point_size),
            squeeze=False,
        )
        created_figure = True
    else:
        axes = np.atleast_1d(ax).ravel()
        if len(axes) != len(color_keys):
            raise ValueError("Provide one axes object for each color key.")
        fig = axes[0].figure
        axes = axes.reshape(1, -1)
        created_figure = False
    if legend_position is None:
        legend_position = "right" if created_figure else "inside"
    if legend_position not in {"inside", "right", "bottom"}:
        raise ValueError("legend_position must be 'inside', 'right', or 'bottom'.")

    for ax, key in zip(axes.ravel(), color_keys):
        labels = pd.Categorical(obs[key])
        palette = plt.get_cmap("tab20", max(len(labels.categories), 1))
        point_colors = palette(
            np.clip(labels.codes, 0, max(len(labels.categories) - 1, 0))
        )
        point_colors[labels.codes < 0] = (0.5, 0.5, 0.5, 1.0)
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=point_colors,
            s=point_size,
            alpha=point_alpha,
            linewidths=0,
            zorder=3,
        )
        ax.set_title(f"IMSTE colored by {key}")
        ax.set_xlabel("IMSTE 1")
        ax.set_ylabel("IMSTE 2")
        ax.set_aspect("equal", adjustable="box")
        handles = [
            Line2D(
                [], [], marker="o", linestyle="none", color=palette(index),
                markersize=6, label=str(value)
            )
            for index, value in enumerate(labels.categories)
        ]
        if np.any(labels.codes < 0):
            handles.append(
                Line2D(
                    [], [], marker="o", linestyle="none", color="0.5",
                    markersize=6, label="(missing)"
                )
            )
        if legend_position == "right":
            legend_options = {
                "bbox_to_anchor": (1.02, 1), "loc": "upper left", "frameon": False
            }
        elif legend_position == "bottom":
            legend_options = {
                "bbox_to_anchor": (0.5, -0.16),
                "loc": "upper center",
                "ncol": min(7, max(1, len(labels.categories))),
                "frameon": False,
            }
        else:
            legend_options = {
                "loc": "upper right", "frameon": True, "framealpha": 0.8
            }
        ax.legend(
            handles,
            [handle.get_label() for handle in handles],
            title=key,
            **legend_options,
        )
    if created_figure:
        fig.tight_layout()
    return fig, axes


def _as_observations(labels, label_key: str | None = None):
    """Normalize a label vector or metadata frame to a one-column frame."""
    if isinstance(labels, pd.DataFrame):
        if label_key is None:
            if labels.shape[1] != 1:
                raise ValueError("Set label_key when labels is a multi-column DataFrame.")
            label_key = str(labels.columns[0])
        if label_key not in labels:
            raise KeyError(f"Label column {label_key!r} is not present in labels.")
        return labels, label_key
    if isinstance(labels, pd.Series):
        label_key = label_key or str(labels.name or "label")
        return labels.to_frame(name=label_key), label_key
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("labels must be a one-dimensional array or Series.")
    label_key = label_key or "label"
    return pd.DataFrame({label_key: values}), label_key


def plot_embedding(
    embedding: np.ndarray,
    labels,
    *,
    label_key: str | None = None,
    color_keys: tuple[str, ...] | None = None,
    point_size: float = 7,
    point_alpha: float = 0.8,
    ax=None,
    legend_position: str | None = None,
):
    """Plot embedding points colored by one or more metadata columns or labels."""
    if isinstance(labels, pd.DataFrame) and label_key is None and color_keys:
        obs = labels
    else:
        obs, label_key = _as_observations(labels, label_key)
    if color_keys is None:
        color_keys = (label_key,)
    return _plot_embedding_frame(
        embedding,
        obs,
        color_keys=color_keys,
        point_size=point_size,
        point_alpha=point_alpha,
        ax=ax,
        legend_position=legend_position,
    )


def render_datashader_embedding(
    embedding,
    labels=None,
    *,
    label_key: str | None = None,
    color_key: dict[str, str] | None = None,
    width: int = 1000,
    height: int = 1000,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    preserve_aspect: bool = False,
    how: str = "eq_hist",
    min_alpha: int = 40,
    background: str = "white",
    max_spread_px: int = 0,
    spread_px: int = 0,
    spread_threshold: float = 0.5,
    spread_shape: str = "circle",
    draw_border: bool = True,
):
    """Rasterize a 2D embedding with Datashader.

    ``embedding`` may be an ``(n, 2)`` array or a fitted IMSTE estimator.
    ``labels`` may be a one-dimensional vector, Series, or DataFrame column;
    categorical labels are shaded separately using ``color_key``. If labels
    are omitted, the function shades point density. The returned Datashader
    image can be displayed directly in a notebook or converted with
    ``image.to_pil()`` for Matplotlib.

    Datashader is an optional dependency installed by ``imste[notebook]``.
    ``x_range`` and ``y_range`` allow rendering a zoomed view without dropping
    points from the source embedding. Set ``spread_px`` above zero to enlarge
    every rendered mark by a fixed number of pixels on each side. Alternatively,
    set ``max_spread_px`` above zero for density-aware spreading, which stops
    once the requested local pixel density is reached. Only one spread mode may
    be enabled at a time. Set ``preserve_aspect`` to use the same pixel scale
    for both coordinate axes; this may add empty canvas margins to avoid
    stretching the embedding. Set ``draw_border=False`` to omit the thin gray
    border around the returned image.
    """
    coordinates = _embedding_array(embedding)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("embedding must be a two-dimensional coordinate array.")
    if len(coordinates) < 1 or not np.isfinite(coordinates).all():
        raise ValueError("embedding must contain finite coordinates.")
    for name, value in (("width", width), ("height", height)):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")
    if (
        isinstance(min_alpha, (bool, np.bool_))
        or not isinstance(min_alpha, (int, np.integer))
        or not 0 <= min_alpha <= 255
    ):
        raise ValueError("min_alpha must be an integer between 0 and 255.")
    if (
        isinstance(max_spread_px, (bool, np.bool_))
        or not isinstance(max_spread_px, (int, np.integer))
        or max_spread_px < 0
    ):
        raise ValueError("max_spread_px must be a non-negative integer.")
    if (
        isinstance(spread_px, (bool, np.bool_))
        or not isinstance(spread_px, (int, np.integer))
        or spread_px < 0
    ):
        raise ValueError("spread_px must be a non-negative integer.")
    if max_spread_px > 0 and spread_px > 0:
        raise ValueError("Set either spread_px or max_spread_px, not both.")
    if (
        not np.isfinite(spread_threshold)
        or not 0.0 <= spread_threshold <= 1.0
    ):
        raise ValueError("spread_threshold must be between 0 and 1.")
    for name, value in (("x_range", x_range), ("y_range", y_range)):
        if value is None:
            continue
        if (
            len(value) != 2
            or not np.isfinite(value).all()
            or value[0] >= value[1]
        ):
            raise ValueError(f"{name} must contain two finite increasing values.")

    if not isinstance(preserve_aspect, (bool, np.bool_)):
        raise ValueError("preserve_aspect must be a boolean.")
    if not isinstance(draw_border, (bool, np.bool_)):
        raise ValueError("draw_border must be a boolean.")
    if preserve_aspect:
        x_limits = np.asarray(
            x_range if x_range is not None else (coordinates[:, 0].min(), coordinates[:, 0].max()),
            dtype=np.float64,
        )
        y_limits = np.asarray(
            y_range if y_range is not None else (coordinates[:, 1].min(), coordinates[:, 1].max()),
            dtype=np.float64,
        )
        pixel_scale = max(
            (x_limits[1] - x_limits[0]) / int(width),
            (y_limits[1] - y_limits[0]) / int(height),
            np.finfo(np.float64).eps,
        )
        x_center = float(x_limits.mean())
        y_center = float(y_limits.mean())
        x_half_span = pixel_scale * int(width) / 2
        y_half_span = pixel_scale * int(height) / 2
        x_range = (x_center - x_half_span, x_center + x_half_span)
        y_range = (y_center - y_half_span, y_center + y_half_span)

    try:
        import datashader as ds
        import datashader.transfer_functions as tf
    except ImportError as error:
        raise ImportError(
            "Datashader is required for this renderer. Install it with "
            "`pip install 'imste[notebook]'` or `pip install datashader`."
        ) from error

    frame = pd.DataFrame({"x": coordinates[:, 0], "y": coordinates[:, 1]})
    canvas = ds.Canvas(
        plot_width=int(width),
        plot_height=int(height),
        x_range=x_range,
        y_range=y_range,
    )
    if labels is None:
        aggregate = canvas.points(frame, "x", "y", ds.count())
        image = tf.shade(aggregate, how=how, min_alpha=int(min_alpha))
    else:
        observations, label_key = _as_observations(labels, label_key)
        if len(observations) != len(coordinates):
            raise ValueError("labels rows must match the embedding rows.")
        values = observations[label_key].astype("object")
        values = values.where(pd.notna(values), "(missing)").astype(str)
        frame[label_key] = pd.Categorical(values)
        aggregate = canvas.points(frame, "x", "y", ds.count_cat(label_key))
        categories = frame[label_key].cat.categories
        palette_name = "tab10" if len(categories) <= 10 else "tab20"
        palette = plt.get_cmap(palette_name, max(len(categories), 1))
        normalized_color_key = {
            str(category): to_hex(palette(index))
            for index, category in enumerate(categories)
        }
        if color_key is not None:
            normalized_color_key.update(
                {str(key): color for key, color in color_key.items()}
            )
        # Datashader's categorical compositor multiplies a (height, width,
        # categories) buffer by an RGB matrix. Shade one category at a time and
        # composite the RGBA layers instead; this avoids unstable BLAS
        # matrix-multiply warnings on some platforms and keeps category colors.
        category_images = [
            tf.shade(
                aggregate.sel({label_key: category}),
                cmap=normalized_color_key[str(category)],
                how=how,
                min_alpha=int(min_alpha),
            )
            for category in categories
        ]
        image = tf.stack(*category_images, how="over")
    if spread_px > 0:
        image = tf.spread(image, px=int(spread_px), shape=spread_shape)
    elif max_spread_px > 0:
        image = tf.dynspread(
            image,
            threshold=spread_threshold,
            max_px=int(max_spread_px),
            shape=spread_shape,
        )
    image = tf.set_background(image, background)
    if draw_border:
        bordered_data = image.data.copy()
        gray = np.uint32(0xFF808080)
        # Use two source pixels so the border stays even when a large render
        # is reduced to fit a notebook panel.
        bordered_data[:2, :] = gray
        bordered_data[-2:, :] = gray
        bordered_data[:, :2] = gray
        bordered_data[:, -2:] = gray
        image = tf.Image(
            bordered_data,
            coords=image.coords,
            dims=image.dims,
            attrs=image.attrs,
        )
    return image


def plot_density_parent_tree(
    embedding: np.ndarray,
    labels,
    *,
    label_key: str | None = None,
    point_size: float = 8,
    point_alpha: float = 0.0,
    edge_alpha: float = 0.5,
    cross_label_edge_alpha: float = 0.1,
    edge_alpha_exponent: float = 1.2,
    linewidth: float = 0.7,
    k: int = 40,
    edge_length_std_threshold: float = 0.85,
    ax=None,
    legend_position: str = "bottom",
):
    """Plot a KDE density-parent graph over a labeled 2D embedding.

    Each point connects to its ``k`` nearest points with higher KDE density,
    or all such points when fewer than ``k`` exist. Equal-density points are
    ordered by row index so every non-root point has a parent. Parent links
    fade with rank, and unusually long links can be filtered by their length.
    """
    coordinates = _embedding_array(embedding)
    obs, label_key = _as_observations(labels, label_key)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("embedding must be a two-dimensional coordinate array.")
    if len(obs) != len(coordinates):
        raise ValueError("obs rows must match the embedding rows.")
    if label_key not in obs:
        raise KeyError(f"Label column {label_key!r} is not present in obs.")
    if len(coordinates) < 2 or not np.isfinite(coordinates).all():
        raise ValueError("embedding must contain at least two finite coordinates.")
    if legend_position not in {"inside", "right", "bottom"}:
        raise ValueError("legend_position must be 'inside', 'right', or 'bottom'.")
    if not np.isfinite(edge_length_std_threshold) or edge_length_std_threshold < 0:
        raise ValueError("edge_length_std_threshold must be finite and non-negative.")
    if not np.isfinite(edge_alpha_exponent) or edge_alpha_exponent < 0:
        raise ValueError("edge_alpha_exponent must be finite and non-negative.")
    if not np.isfinite(edge_alpha) or not 0.0 <= edge_alpha <= 1.0:
        raise ValueError("edge_alpha must be between 0 and 1.")
    if not np.isfinite(cross_label_edge_alpha) or not 0.0 <= cross_label_edge_alpha <= 1.0:
        raise ValueError("cross_label_edge_alpha must be between 0 and 1.")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer.")
    k = min(int(k), len(coordinates) - 1)

    density = gaussian_kde(coordinates.T)(coordinates.T)
    density_order = np.lexsort((np.arange(len(coordinates)), density))
    density_rank = np.empty(len(coordinates), dtype=np.intp)
    density_rank[density_order] = np.arange(len(coordinates))

    parents = np.full((len(coordinates), k), -1, dtype=np.intp)
    unresolved = density_order[:-1].copy()
    search_tree = cKDTree(coordinates)
    neighbor_count = min(max(2, k + 1), len(coordinates))
    while unresolved.size:
        _, neighbors = search_tree.query(
            coordinates[unresolved], k=neighbor_count, workers=-1
        )
        if neighbor_count == 1:
            neighbors = neighbors[:, None]
        higher_density = density_rank[neighbors] > density_rank[unresolved, None]
        enough_parents = higher_density.sum(axis=1) >= k
        all_neighbors_seen = neighbor_count == len(coordinates)
        resolved_mask = enough_parents | all_neighbors_seen
        if np.any(resolved_mask):
            resolved_neighbors = neighbors[resolved_mask]
            resolved_higher_density = higher_density[resolved_mask]
            for row_index, point_index in enumerate(unresolved[resolved_mask]):
                higher_neighbors = resolved_neighbors[row_index][
                    resolved_higher_density[row_index]
                ]
                parents[point_index, :min(k, len(higher_neighbors))] = higher_neighbors[:k]
        unresolved = unresolved[~resolved_mask]
        if unresolved.size:
            neighbor_count = min(2 * neighbor_count, len(coordinates))

    categories = pd.Categorical(obs[label_key])
    palette = plt.get_cmap("tab20", max(len(categories.categories), 1))
    point_colors = palette(
        np.clip(categories.codes, 0, max(len(categories.categories) - 1, 0))
    )
    point_colors[categories.codes < 0] = (0.5, 0.5, 0.5, 1.0)
    edge_mask = parents >= 0
    child_grid, edge_rank_grid = np.indices(parents.shape)
    child_indices = child_grid[edge_mask]
    parent_indices = parents[edge_mask]
    edge_ranks = edge_rank_grid[edge_mask] + 1
    edge_lengths = np.linalg.norm(
        coordinates[child_indices] - coordinates[parent_indices], axis=1
    )
    mean_edge_length = float(edge_lengths.mean())
    std_edge_length = float(edge_lengths.std())
    edge_cutoff = mean_edge_length + edge_length_std_threshold * std_edge_length
    keep_edge = edge_lengths <= edge_cutoff
    children, parent_indices = child_indices[keep_edge], parent_indices[keep_edge]
    segments = np.stack((coordinates[children], coordinates[parent_indices]), axis=1)
    edge_colors = (point_colors[children] + point_colors[parent_indices]) / 2
    edge_ranks = edge_ranks[keep_edge]
    cross_label = categories.codes[children] != categories.codes[parent_indices]
    base_alpha = np.where(cross_label, cross_label_edge_alpha, edge_alpha)
    edge_colors[:, 3] *= base_alpha / edge_ranks**edge_alpha_exponent

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    ax.add_collection(LineCollection(segments, colors=edge_colors, linewidths=linewidth, zorder=1))
    ax.scatter(
        coordinates[:, 0], coordinates[:, 1], c=point_colors,
        s=point_size, alpha=point_alpha, linewidths=0, zorder=2,
    )
    ax.set_title(
        f"KDE density-parent graph (k={k}); mean edge {mean_edge_length:.3g}; "
        f"{int((~keep_edge).sum())} edges omitted (>{edge_length_std_threshold:g} SD)"
    )
    ax.set_xlabel("IMSTE 1")
    ax.set_ylabel("IMSTE 2")
    ax.set_aspect("equal", adjustable="box")
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=palette(index),
               markersize=6, label=str(value))
        for index, value in enumerate(categories.categories)
    ]
    if np.any(categories.codes < 0):
        handles.append(Line2D([], [], marker="o", linestyle="none", color="0.5",
                              markersize=6, label="(missing)"))
    if legend_position == "right":
        legend_options = {"bbox_to_anchor": (1.02, 1), "loc": "upper left", "frameon": False}
    elif legend_position == "bottom":
        legend_options = {
            "bbox_to_anchor": (0.5, -0.16), "loc": "upper center",
            "ncol": min(7, max(1, len(categories.categories))), "frameon": False,
        }
    else:
        legend_options = {"loc": "upper right", "frameon": True, "framealpha": 0.8}
    ax.legend(handles, [handle.get_label() for handle in handles], title=label_key, **legend_options)
    return fig, ax


def cross_validate_embedding_knn(
    embedding: np.ndarray,
    obs: pd.DataFrame,
    *,
    label_key: str = "celltype",
    n_neighbors: int = 5,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Cross-validate a kNN classifier using only the 2D embedding.

    Rows with missing labels are ignored. The split count is reduced when a
    class has fewer examples than requested, while classes with fewer than two
    examples raise an error because stratified cross-validation is impossible.
    """
    if (
        not isinstance(n_neighbors, (int, np.integer))
        or isinstance(n_neighbors, bool)
        or n_neighbors < 1
    ):
        raise ValueError("n_neighbors must be a positive integer.")
    if (
        not isinstance(n_splits, (int, np.integer))
        or isinstance(n_splits, bool)
        or n_splits < 2
    ):
        raise ValueError("n_splits must be an integer of at least 2.")
    if label_key not in obs:
        raise KeyError(f"Label column {label_key!r} is not present in obs.")

    coordinates = _embedding_array(embedding)
    if len(obs) != len(coordinates):
        raise ValueError("obs rows must match the embedding rows.")

    labels = pd.Categorical(obs[label_key]).remove_unused_categories()
    original_codes = np.asarray(labels.codes)
    original_classes = np.asarray(labels.categories)
    if len(original_classes) == 0:
        raise ValueError("At least one labeled class is required for CV.")

    original_counts = np.bincount(
        original_codes[original_codes >= 0], minlength=len(original_classes)
    )
    eligible_classes = np.flatnonzero(original_counts >= 2)
    if len(eligible_classes) < 2:
        raise ValueError("At least two classes need two or more instances for CV.")
    class_map = np.full(len(original_classes), -1, dtype=np.intp)
    class_map[eligible_classes] = np.arange(len(eligible_classes))
    valid = (original_codes >= 0) & (class_map[np.maximum(original_codes, 0)] >= 0)
    X = coordinates[valid]
    y = class_map[original_codes[valid]]
    classes = original_classes[eligible_classes]
    counts = np.bincount(y, minlength=len(classes))
    min_class_count = int(counts.min())
    effective_splits = min(int(n_splits), min_class_count)
    min_train_size = len(y) - int(np.ceil(len(y) / effective_splits))
    if n_neighbors > min_train_size:
        raise ValueError(
            f"n_neighbors={n_neighbors} exceeds the smallest CV training fold "
            f"({min_train_size} instances)."
        )

    cv = StratifiedKFold(
        n_splits=effective_splits, shuffle=True, random_state=random_state
    )
    predictions = cross_val_predict(
        KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=1),
        X,
        y,
        cv=cv,
        n_jobs=1,
    )
    matrix = confusion_matrix(y, predictions, labels=np.arange(len(classes)))
    correct_mask = np.zeros(len(labels), dtype=bool)
    correct_mask[valid] = y == predictions
    return {
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "n_neighbors": int(n_neighbors),
        "n_splits": effective_splits,
        "classes": classes,
        "y_true": y,
        "y_pred": predictions,
        "valid_mask": valid,
        "correct_mask": correct_mask,
        "n_excluded": int(len(labels) - valid.sum()),
        "confusion_matrix": matrix,
    }
