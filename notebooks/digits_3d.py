"""Interactive 3D IMSTE visualization for the MNIST notebook."""

from __future__ import annotations

import time
import traceback

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from IPython.display import clear_output, display
from sklearn.manifold import trustworthiness
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

try:  # Notebook execution puts this directory directly on sys.path.
    from digits_sweep import format_duration, load_mnist_data
except ModuleNotFoundError:  # Also support importing as notebooks.digits_3d.
    from .digits_sweep import format_duration, load_mnist_data
from imste import IteratedMinimumSpanningTreeEmbedder


def _available_devices() -> list[str]:
    devices = ["auto", "cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def fit_and_plot_mst_3d(
    X,
    labels,
    n_msts=15,
    n_epochs=200,
    batch_size=32768,
    learning_rate=0.05,
    negative_ratio=5,
    lambda_rep=0.5,
    epsilon=1e-4,
    random_state=42,
    device="auto",
    mst_method="famst",
    trustworthiness_neighbors=10,
):
    """Fit one 3D IMSTE embedding and return its scores and Plotly figure."""
    X = np.asarray(X)
    labels = np.asarray(labels)
    if X.ndim != 2 or len(X) != len(labels):
        raise ValueError("X must be 2D and have one label per row")
    if len(X) < 25:
        raise ValueError("At least 25 samples are required for 5-fold 5-NN scoring")

    estimator = IteratedMinimumSpanningTreeEmbedder(
        n_msts=n_msts,
        n_components=3,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        negative_ratio=negative_ratio,
        lambda_rep=lambda_rep,
        epsilon=epsilon,
        random_state=random_state,
        device=device,
        mst_method=mst_method,
    )
    print(
        f"Fitting 3D IMSTE embedding with n_msts={n_msts}, "
        "rank-scheduled edge updates ...",
        flush=True,
    )
    started = time.perf_counter()
    embedding = estimator.fit_transform(X)
    elapsed = time.perf_counter() - started

    n_neighbors = min(trustworthiness_neighbors, (len(X) - 1) // 2)
    score = trustworthiness(X, embedding, n_neighbors=n_neighbors)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    knn_accuracy = cross_val_score(
        KNeighborsClassifier(n_neighbors=5), embedding, labels, cv=cv
    ).mean()
    elapsed_text = format_duration(elapsed)
    print(
        f"Finished 3D IMSTE embedding in {elapsed_text} on {estimator.device_}",
        flush=True,
    )

    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=embedding[:, 0],
                y=embedding[:, 1],
                z=embedding[:, 2],
                mode="markers",
                marker={
                    "size": 3,
                    "opacity": 0.8,
                    "color": labels.astype(int),
                    "colorscale": "Turbo",
                    "cmin": 0,
                    "cmax": 9,
                    "colorbar": {
                        "title": "Digit",
                        "tickmode": "array",
                        "tickvals": list(range(10)),
                    },
                },
                customdata=labels,
                hovertemplate="Digit %{customdata}<extra></extra>",
                showlegend=False,
            )
        ]
    )
    figure.update_layout(
        title=(
            f"3D IMSTE ({mst_method.upper()}, n_msts={n_msts})<br>"
            f"trustworthiness={score:.3f}; 5-NN CV={knn_accuracy:.3f}; "
            f"runtime={elapsed_text} ({estimator.device_})"
        ),
        scene={
            "xaxis_title": "Dimension 1",
            "yaxis_title": "Dimension 2",
            "zaxis_title": "Dimension 3",
            "aspectmode": "cube",
        },
        height=750,
        width=1000,
        margin={"l": 0, "r": 0, "t": 110, "b": 0},
        legend={
            "title": {"text": "Digit"},
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.08,
        },
    )
    summary = pd.DataFrame([{
        "n_msts": n_msts,
        "mst_method": mst_method,
        "n_components": 3,
        "trustworthiness": score,
        "5-NN 5-fold CV accuracy": knn_accuracy,
        "seconds": elapsed,
        "elapsed": elapsed_text,
        "device": estimator.device_,
    }])
    return embedding, summary, figure


def _load_balanced_pool(max_instances, random_state):
    X_pool, labels_pool = load_mnist_data(
        sample_size=max_instances,
        random_state=random_state,
    )
    rng = np.random.RandomState(random_state)
    class_indices = [
        rng.permutation(np.flatnonzero(labels_pool == label))
        for label in np.unique(labels_pool)
    ]
    per_class = min(len(indices) for indices in class_indices)
    balanced_order = np.stack(
        [indices[:per_class] for indices in class_indices], axis=1
    ).reshape(-1)
    return X_pool[balanced_order], labels_pool[balanced_order]


def display_interactive_mst_3d(
    max_instances=4000,
    initial_sample_count=1000,
    random_state=42,
):
    """Display sliders for a 3D IMSTE embedding and fit on button click."""
    if max_instances < 500:
        raise ValueError("max_instances must be at least 500.")
    if initial_sample_count < 500:
        raise ValueError("initial_sample_count must be at least 500.")
    initial_sample_count = min(initial_sample_count, max_instances)

    status = widgets.HTML(value="Loading stratified MNIST sample pool...")
    display(status)
    X_pool, labels_pool = _load_balanced_pool(max_instances, random_state)
    max_instances = len(X_pool)
    initial_sample_count = min(initial_sample_count, max_instances)
    minimum_sample_count = min(500, max_instances)
    status.value = (
        f"Balanced pool available: {max_instances:,} samples. "
        f"Default selection: {initial_sample_count:,} samples, 15 MSTs with approximate FAMST, "
        "200 epochs. Click Fit 3D embedding to run."
    )

    sample_count = widgets.IntSlider(
        value=initial_sample_count,
        min=minimum_sample_count,
        max=max_instances,
        step=250,
        description="Instances",
        continuous_update=False,
        style={"description_width": "initial"},
    )
    n_msts = widgets.IntSlider(
        value=15, min=1, max=64, step=1,
        description="MSTs", continuous_update=False,
        style={"description_width": "initial"},
    )
    batch_size = widgets.IntSlider(
        value=32768, min=4096, max=131072, step=4096,
        description="Batch size (MPS)", continuous_update=False,
        style={"description_width": "initial"},
    )
    mst_method = widgets.Dropdown(
        options=[("FAMST (approximate)", "famst"), ("Prim (exact)", "prim")],
        value="famst",
        description="MST builder",
        style={"description_width": "initial"},
    )
    n_epochs = widgets.IntSlider(
        value=200, min=100, max=2000, step=100,
        description="Epochs", continuous_update=False,
        style={"description_width": "initial"},
    )
    learning_rate = widgets.FloatSlider(
        value=0.05, min=0.01, max=0.10, step=0.01,
        readout_format=".2f", description="Learning rate",
        continuous_update=False,
        style={"description_width": "initial"},
    )
    negative_ratio = widgets.IntSlider(
        value=5, min=0, max=10, step=1,
        description="Negative ratio", continuous_update=False,
        style={"description_width": "initial"},
    )
    lambda_rep = widgets.FloatSlider(
        value=0.5, min=0.0, max=1.0, step=0.05,
        readout_format=".2f", description="Repulsion share (λ)",
        continuous_update=False,
        style={"description_width": "initial"},
    )
    epsilon = widgets.FloatLogSlider(
        value=1e-4, base=10, min=-8, max=-2, step=0.1,
        description="Epsilon", continuous_update=False,
        style={"description_width": "initial"},
    )
    random_state_slider = widgets.IntSlider(
        value=random_state, min=0, max=1000, step=1,
        description="Random state", continuous_update=False,
        style={"description_width": "initial"},
    )
    trustworthiness_neighbors = widgets.IntSlider(
        value=10, min=5, max=50, step=5,
        description="Trustworthiness k", continuous_update=False,
        style={"description_width": "initial"},
    )
    device_default = "auto"
    device = widgets.Dropdown(
        options=_available_devices(),
        value=device_default,
        description="Device",
        style={"description_width": "initial"},
    )
    batch_size.layout.display = "none"

    def update_batch_size_visibility(change) -> None:
        batch_size.layout.display = "" if change["new"] == "mps" else "none"

    fit_button = widgets.Button(
        description="Fit 3D embedding",
        button_style="primary",
        icon="play",
    )
    reset_button = widgets.Button(
        description="Reset sliders",
        tooltip="Restore default values",
        icon="undo",
    )
    output = widgets.Output()

    def reset_sliders(_=None):
        sample_count.value = min(1000, max_instances)
        n_msts.value = 15
        batch_size.value = 32768
        mst_method.value = "famst"
        n_epochs.value = 200
        learning_rate.value = 0.05
        negative_ratio.value = 5
        lambda_rep.value = 0.5
        epsilon.value = 1e-4
        random_state_slider.value = 42
        trustworthiness_neighbors.value = 10
        device.value = device_default

    def fit_and_display(_=None):
        fit_button.disabled = True
        status.value = "Fitting the 3D embedding with the selected settings..."
        with output:
            clear_output(wait=True)
            try:
                n_samples = sample_count.value
                X = X_pool[:n_samples]
                labels = labels_pool[:n_samples]
                embedding, summary, figure = fit_and_plot_mst_3d(
                    X,
                    labels,
                    n_msts=n_msts.value,
                    batch_size=batch_size.value,
                    mst_method=mst_method.value,
                    n_epochs=n_epochs.value,
                    learning_rate=learning_rate.value,
                    negative_ratio=negative_ratio.value,
                    lambda_rep=lambda_rep.value,
                    epsilon=epsilon.value,
                    random_state=random_state_slider.value,
                    device=device.value,
                    trustworthiness_neighbors=trustworthiness_neighbors.value,
                )
                display(figure)
                display(summary)
                status.value = "Embedding ready. Adjust the sliders and fit again."
            except Exception:
                status.value = "The fit failed; see the error details below."
                traceback.print_exc()
            finally:
                fit_button.disabled = False

    device.observe(update_batch_size_visibility, names="value")
    fit_button.on_click(fit_and_display)
    reset_button.on_click(reset_sliders)
    controls = widgets.VBox(
        [
            widgets.HBox([sample_count, n_msts]),
            widgets.HBox([batch_size]),
            widgets.HBox([mst_method]),
            widgets.HBox([n_epochs, learning_rate]),
            widgets.HBox([negative_ratio, lambda_rep, epsilon]),
            widgets.HBox([random_state_slider, trustworthiness_neighbors, device]),
        ]
    )
    display(controls, widgets.HBox([fit_button, reset_button]), output)
    return {
        "controls": controls,
        "fit_button": fit_button,
        "reset_button": reset_button,
        "output": output,
        "status": status,
        "sliders": {
            "sample_count": sample_count,
            "n_msts": n_msts,
            "batch_size": batch_size,
            "mst_method": mst_method,
            "n_epochs": n_epochs,
            "learning_rate": learning_rate,
            "negative_ratio": negative_ratio,
            "lambda_rep": lambda_rep,
            "epsilon": epsilon,
            "random_state": random_state_slider,
            "trustworthiness_neighbors": trustworthiness_neighbors,
            "device": device,
        },
    }
