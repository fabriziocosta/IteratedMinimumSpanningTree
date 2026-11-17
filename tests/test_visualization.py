import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from imste.class_graph import plot_class_mixture_mst as legacy_plot_class_mixture_mst
from imste.visualization import plot_class_mixture_mst, plot_embedding


def test_plot_embedding_accepts_series_and_returns_figure_and_axes():
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    labels = pd.Series(["a", "a", "b", "b"], name="class")

    figure, axes = plot_embedding(coordinates, labels)

    assert figure is axes[0, 0].figure
    assert axes.shape == (1, 1)
    assert axes[0, 0].get_title() == "IMSTE colored by class"
    plt.close(figure)


def test_plot_embedding_rejects_label_count_mismatch():
    with pytest.raises(ValueError, match="obs rows must match"):
        plot_embedding(np.zeros((3, 2)), ["a", "b"])


def test_class_mixture_plot_is_in_visualization_api_and_legacy_import_still_works(
    monkeypatch,
):
    expected = object()
    monkeypatch.setattr("imste.visualization.plot_class_mixture_mst", lambda *a, **k: expected)

    result = legacy_plot_class_mixture_mst(
        np.zeros((2, 1)), np.array(["a", "b"]), {"IMSTE": np.zeros((2, 2))}
    )

    assert result is expected
    assert callable(plot_class_mixture_mst)


def test_class_mixture_plot_returns_axes_and_edge_scores():
    rng = np.random.default_rng(2)
    labels = np.repeat(["left", "right"], 24)
    embedding = np.vstack(
        [rng.normal(loc=(-2.0, 0.0), scale=0.25, size=(24, 2)),
         rng.normal(loc=(2.0, 0.0), scale=0.25, size=(24, 2))]
    )

    figure, axes, edge_scores = plot_class_mixture_mst(
        embedding,
        labels,
        {"IMSTE": embedding, "UMAP": embedding + np.array([0.1, -0.1])},
        n_msts=1,
        max_components_per_class=1,
        min_component_size=5,
        random_state=0,
    )

    assert len(axes) == 2
    assert {"embedding", "class_a", "class_b", "symmetric_component_overlap"}.issubset(
        edge_scores.columns
    )
    assert set(edge_scores["embedding"]) == {"IMSTE", "UMAP"}
    assert len(edge_scores) == 2
    plt.close(figure)
