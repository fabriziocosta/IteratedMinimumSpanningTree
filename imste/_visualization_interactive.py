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

from ._visualization_core import _as_observations, _embedding_array


def interactive_force(
    embedding: np.ndarray,
    labels,
    *,
    label_key: str | None = None,
    initial_k: int = 10,
    initial_iterations: int = 20,
    initial_distance_fraction: float = 0.25,
    max_k: int = 50,
    max_iterations: int = 200,
    point_size: float = 6,
    label_colors: dict | None = None,
):
    """Show an interactive label-guided adjustment of a 2D embedding.

    For each point, the viewer separately finds its ``k`` nearest same-label
    points and its ``k`` nearest different-label points in the original
    embedding. During each iteration it moves toward the same-label neighbors
    and repels from different-label neighbors only when they are closer than a
    small separation margin. Balance 0 applies only different-label repulsion;
    balance 1 applies only the same-label springs. The movement slider
    interpolates between the original and adjusted coordinates. Neighbor
    identities remain fixed during the adjustment. Each same-label neighbor
    pair uses a symmetric spring whose rest length is the selected fraction of
    its original distance.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as error:
        raise ImportError(
            "Install ipywidgets to use the interactive embedding viewer."
        ) from error

    obs, label_key = _as_observations(labels, label_key)

    if label_key not in obs:
        raise KeyError(f"Label column {label_key!r} is not present in obs.")
    coordinates = _embedding_array(embedding)
    if len(obs) != len(coordinates):
        raise ValueError("obs rows must match the embedding rows.")
    n_samples = len(coordinates)
    if n_samples < 2:
        raise ValueError("At least two points are required for nearest neighbors.")
    if not isinstance(max_k, (int, np.integer)) or isinstance(max_k, bool) or max_k < 1:
        raise ValueError("max_k must be a positive integer.")
    max_k = min(int(max_k), n_samples - 1)
    if not isinstance(max_iterations, (int, np.integer)) or isinstance(max_iterations, bool) or max_iterations < 0:
        raise ValueError("max_iterations must be a non-negative integer.")

    labels = pd.Categorical(obs[label_key])
    label_codes = labels.codes
    same_neighbors = np.zeros((n_samples, max_k), dtype=np.intp)
    different_neighbors = np.zeros((n_samples, max_k), dtype=np.intp)
    same_neighbor_mask = np.zeros((n_samples, max_k), dtype=bool)
    different_neighbor_mask = np.zeros((n_samples, max_k), dtype=bool)
    for label_code in np.unique(label_codes):
        class_rows = np.flatnonzero(label_codes == label_code)
        other_rows = np.flatnonzero(label_codes != label_code)

        n_same = min(max_k, max(0, len(class_rows) - 1))
        if n_same:
            same_model = NearestNeighbors(n_neighbors=n_same).fit(
                coordinates[class_rows]
            )
            # kneighbors() with X omitted excludes each fitted row itself.
            same_neighbors[class_rows, :n_same] = class_rows[
                same_model.kneighbors(return_distance=False)
            ]
            same_neighbor_mask[class_rows, :n_same] = True

        n_different = min(max_k, len(other_rows))
        if n_different:
            different_model = NearestNeighbors(n_neighbors=n_different).fit(
                coordinates[other_rows]
            )
            different_neighbors[class_rows, :n_different] = other_rows[
                different_model.kneighbors(
                    coordinates[class_rows], return_distance=False
                )
            ]
            different_neighbor_mask[class_rows, :n_different] = True
    # Scale updates to the visible embedding, not to the often tiny nearest-
    # neighbor spacing in dense single-cell plots.
    global_scale = float(np.sqrt(np.mean(np.var(coordinates, axis=0))))
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = max(float(np.ptp(coordinates, axis=0).mean()), 1e-8)
    step_limit = 0.025 * global_scale
    repulsion_radius = 0.08 * global_scale
    coordinate_min = coordinates.min(axis=0)
    coordinate_max = coordinates.max(axis=0)
    coordinate_span = coordinate_max - coordinate_min
    viewport_padding = np.maximum(coordinate_span * 0.05, 0.08 * global_scale)
    viewport_min = coordinate_min - viewport_padding
    viewport_max = coordinate_max + viewport_padding

    default_k = min(max(1, int(initial_k)), max_k)
    default_iterations = min(max(0, int(initial_iterations)), int(max_iterations))
    k_control = widgets.IntSlider(
        value=default_k,
        min=1,
        max=max_k,
        step=1,
        description="",
        continuous_update=False,
        layout=widgets.Layout(width="480px"),
    )
    iteration_control = widgets.IntSlider(
        value=default_iterations,
        min=0,
        max=int(max_iterations),
        step=1,
        description="",
        continuous_update=False,
        layout=widgets.Layout(width="480px"),
    )
    balance_control = widgets.FloatSlider(
        value=0.5,
        min=0.0,
        max=1.0,
        step=0.05,
        description="",
        readout_format=".2f",
        continuous_update=False,
        layout=widgets.Layout(width="480px"),
    )
    distance_control = widgets.FloatSlider(
        value=float(np.clip(initial_distance_fraction, 0.05, 1.0)),
        min=0.05,
        max=1.0,
        step=0.05,
        description="",
        readout_format=".2f",
        continuous_update=False,
        layout=widgets.Layout(width="480px"),
    )
    movement_control = widgets.FloatSlider(
        value=1.0,
        min=0.0,
        max=1.0,
        step=0.05,
        description="",
        readout_format=".2f",
        continuous_update=False,
        layout=widgets.Layout(width="480px"),
    )
    status = widgets.HTML(
        value='<span style="color:#666">Ready</span>',
        layout=widgets.Layout(margin="4px 0 0 420px"),
    )

    cache: dict[tuple[int, int, float, float], np.ndarray] = {}
    pair_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def same_label_pairs(k: int):
        if int(k) in pair_cache:
            return pair_cache[int(k)]
        mask = same_neighbor_mask[:, : int(k)]
        rows = np.broadcast_to(np.arange(n_samples)[:, None], mask.shape)[mask]
        columns = same_neighbors[:, : int(k)][mask]
        keys = np.unique(
            np.minimum(rows, columns).astype(np.int64) * n_samples
            + np.maximum(rows, columns)
        )
        left = (keys // n_samples).astype(np.intp)
        right = (keys % n_samples).astype(np.intp)
        initial_distances = np.linalg.norm(
            coordinates[right] - coordinates[left], axis=1
        )
        pair_cache.clear()
        pair_cache[int(k)] = (left, right, initial_distances)
        return pair_cache[int(k)]

    def adjusted_coordinates(
        k: int,
        n_iterations: int,
        balance: float,
        distance_fraction: float,
    ) -> np.ndarray:
        balance = float(balance)
        distance_fraction = float(distance_fraction)
        key = (int(k), int(n_iterations), balance, distance_fraction)
        if key in cache:
            return cache[key]
        pair_left, pair_right, initial_pair_distances = same_label_pairs(k)
        rest_lengths = np.maximum(
            initial_pair_distances * distance_fraction,
            0.005 * global_scale,
        )
        different_indices = different_neighbors[:, : int(k)]
        different_mask = different_neighbor_mask[:, : int(k)]
        current = coordinates.copy()

        for _ in range(int(n_iterations)):
            pair_delta = current[pair_right] - current[pair_left]
            pair_distances = np.linalg.norm(pair_delta, axis=1)
            pair_direction = pair_delta / np.maximum(
                pair_distances[:, None], np.finfo(np.float64).eps
            )
            coincident = pair_distances <= np.finfo(np.float64).eps
            if coincident.any():
                # Give exact overlaps a stable direction so their nonzero rest
                # length can separate them instead of leaving a zero force.
                angles = (
                    (pair_left[coincident].astype(np.int64) * 73856093)
                    ^ (pair_right[coincident].astype(np.int64) * 19349663)
                ) % 1_000_003
                angles = angles * (2.0 * np.pi / 1_000_003)
                pair_direction[coincident] = np.column_stack(
                    (np.cos(angles), np.sin(angles))
                )
            spring_magnitude = pair_distances - rest_lengths
            spring_force = spring_magnitude[:, None] * pair_direction
            same_label_force = np.column_stack(
                (
                    np.bincount(pair_left, weights=spring_force[:, 0], minlength=n_samples)
                    - np.bincount(pair_right, weights=spring_force[:, 0], minlength=n_samples),
                    np.bincount(pair_left, weights=spring_force[:, 1], minlength=n_samples)
                    - np.bincount(pair_right, weights=spring_force[:, 1], minlength=n_samples),
                )
            )
            nearby_different = current[different_indices]
            away_vectors = current[:, None, :] - nearby_different
            neighbor_distances = np.linalg.norm(away_vectors, axis=2)
            close_different = different_mask & (neighbor_distances < repulsion_radius)
            shortfall = np.maximum(repulsion_radius - neighbor_distances, 0.0)
            unit_away = away_vectors / np.maximum(
                neighbor_distances[:, :, None], np.finfo(np.float64).eps
            )
            repulsion = unit_away * shortfall[:, :, None] * 0.5
            repulsion[~close_different] = 0.0
            repulsion_count = close_different.sum(axis=1, keepdims=True)
            mean_repulsion = np.divide(
                repulsion.sum(axis=1),
                repulsion_count,
                out=np.zeros_like(current),
                where=repulsion_count != 0,
            )
            force = (
                balance * 3.0 * same_label_force
                + (1.0 - balance) * mean_repulsion * 0.5
            )
            force_size = np.linalg.norm(force, axis=1)
            scale = np.minimum(
                1.0,
                step_limit / np.maximum(force_size, np.finfo(np.float64).tiny),
            )
            current += force * scale[:, None]

        cache.clear()
        cache[key] = current
        return current

    def render_update(
        k: int,
        iterations: int,
        balance: float,
        distance_fraction: float,
        movement: float,
    ):
        adjusted = adjusted_coordinates(k, iterations, balance, distance_fraction)
        shown = coordinates + float(movement) * (adjusted - coordinates)
        palette = plt.get_cmap("tab20", max(len(labels.categories), 1))
        category_colors = [
            (label_colors or {}).get(str(label), palette(index))
            for index, label in enumerate(labels.categories)
        ]
        point_colors = np.asarray([
            category_colors[code] if code >= 0 else (0.5, 0.5, 0.5, 1.0)
            for code in label_codes
        ])
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(
            shown[:, 0],
            shown[:, 1],
            c=point_colors,
            s=point_size,
            alpha=0.65,
            linewidths=0,
        )
        ax.set_xlabel("IMSTE 1")
        ax.set_ylabel("IMSTE 2")
        ax.set_xlim(viewport_min[0], viewport_max[0])
        ax.set_ylim(viewport_min[1], viewport_max[1])
        ax.set_aspect("equal", adjustable="box")
        legend_handles = [
            Patch(facecolor=category_colors[index], edgecolor="none", label=str(label))
            for index, label in enumerate(labels.categories)
        ]
        if np.any(label_codes < 0):
            legend_handles.append(
                Patch(facecolor="0.5", edgecolor="none", label="(missing)")
            )
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                title=label_key,
                bbox_to_anchor=(0.5, -0.12),
                loc="upper center",
                ncol=min(6, len(legend_handles)),
                frameon=False,
            )
        fig.tight_layout()
        display(fig)
        plt.close(fig)

    def update(
        k: int,
        iterations: int,
        balance: float,
        distance_fraction: float,
        movement: float,
    ):
        status.value = (
            '<span style="color:#9a6700;font-weight:600">'
            '⏳ Computing…</span>'
        )
        try:
            render_update(k, iterations, balance, distance_fraction, movement)
        except Exception as error:
            status.value = (
                '<span style="color:#b42318;font-weight:600">'
                f'Failed: {html_escape(str(error))}</span>'
            )
            raise
        status.value = '<span style="color:#188038;font-weight:600">✓ Done</span>'

    output = widgets.interactive_output(
        update,
        {
            "k": k_control,
            "iterations": iteration_control,
            "balance": balance_control,
            "distance_fraction": distance_control,
            "movement": movement_control,
        },
    )
    label_width = "420px"
    controls = widgets.VBox(
        [
            widgets.HBox(
                [
                    widgets.HTML(
                        value=(
                            '<div style="text-align:right;padding-right:12px;'
                            f'white-space:nowrap">{html_escape(label)}</div>'
                        ),
                        layout=widgets.Layout(width=label_width),
                    ),
                    control,
                ],
                layout=widgets.Layout(align_items="center"),
            )
            for label, control in (
                ("k per group:", k_control),
                ("Iterations:", iteration_control),
                ("Balance (repel → same-label spring):", balance_control),
                ("Rest distance (fraction of original):", distance_control),
                ("Movement:", movement_control),
            )
        ] + [status],
        layout=widgets.Layout(align_items="flex-start"),
    )
    viewer = widgets.VBox([controls, output])
    return viewer


def _add_compound_path_union_outline(
    ax,
    path,
    color,
    *,
    linewidth: float = 1.0,
    alpha: float = 0.95,
    draw_outline: bool = True,
    zorder: float = 2,
):
    """Draw only the exterior outline of a compound polygon path.

    Matplotlib strokes each subpath independently. Rasterizing the filled
    compound path and contouring its mask gives the outline of the union, with
    overlaps between neighborhood hulls left unoutlined.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    width, height = 1600, 1400
    xmin, ymin, xmax, ymax = path.get_extents().extents
    xpad = max((xmax - xmin) * 0.01, np.finfo(float).eps)
    ypad = max((ymax - ymin) * 0.01, np.finfo(float).eps)
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = ymin - ypad, ymax + ypad

    raster_figure = Figure(figsize=(8, 7), dpi=200)
    raster_canvas = FigureCanvasAgg(raster_figure)
    raster_ax = raster_figure.add_axes((0, 0, 1, 1))
    raster_ax.set_xlim(xmin, xmax)
    raster_ax.set_ylim(ymin, ymax)
    raster_ax.set_axis_off()
    raster_ax.add_patch(
        PathPatch(path, facecolor="black", edgecolor="none", antialiased=True)
    )
    raster_canvas.draw()
    pixels = np.asarray(raster_canvas.buffer_rgba())
    union_mask = (pixels[:, :, 0] < 128)[::-1, :]
    component_labels, n_components = ndimage.label(
        union_mask, structure=np.ones((3, 3), dtype=bool)
    )

    x_grid = np.linspace(xmin, xmax, width)
    y_grid = np.linspace(ymin, ymax, height)
    if draw_outline:
        ax.contour(
            x_grid,
            y_grid,
            union_mask.astype(np.float32),
            levels=(0.5,),
            colors=[(*color[:3], alpha)],
            linewidths=linewidth,
            zorder=zorder,
        )
    return component_labels, n_components, x_grid, y_grid
