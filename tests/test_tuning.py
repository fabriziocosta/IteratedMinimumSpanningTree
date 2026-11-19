from types import SimpleNamespace

import numpy as np
import pytest

from imste import IteratedMinimumSpanningTreeEmbedder
from imste import _tuning


class _Dimension:
    def __init__(self, low_or_categories, high=None, *, name=None):
        self.name = name
        if high is None:
            self.categories = list(low_or_categories)
        else:
            self.low = low_or_categories
            self.high = high


class _IntegerDimension(_Dimension):
    pass


class _RealDimension(_Dimension):
    pass


class _CategoricalDimension(_Dimension):
    pass


def _fake_skopt(monkeypatch):
    calls = {}

    def gp_minimize(func, dimensions, **kwargs):
        calls.update(kwargs)
        calls["dimensions"] = dimensions
        points = []
        scores = []
        for index in range(kwargs["n_calls"]):
            point = []
            for dimension in dimensions:
                if hasattr(dimension, "categories"):
                    point.append(dimension.categories[index % len(dimension.categories)])
                elif isinstance(dimension.low, int):
                    point.append(dimension.low + index % (dimension.high - dimension.low + 1))
                else:
                    point.append((dimension.low + dimension.high) / 2)
            points.append(point)
            scores.append(func(point))
        best_index = int(np.argmin(scores))
        return SimpleNamespace(
            x=points[best_index],
            fun=scores[best_index],
            x_iters=points,
            func_vals=np.asarray(scores),
        )

    monkeypatch.setattr(
        _tuning,
        "_load_skopt",
        lambda: (gp_minimize, _IntegerDimension, _RealDimension, _CategoricalDimension),
    )
    return calls


def test_tune_runs_lhs_budget_scores_and_refits_best(monkeypatch):
    calls = _fake_skopt(monkeypatch)
    fitted = []

    def fake_fit_transform(self, X, y=None):
        embedding = np.asarray(X, dtype=float)[:, :2].copy()
        self.embedding_ = embedding
        return embedding

    def fake_fit(self, X, y=None):
        self.embedding_ = np.asarray(X, dtype=float)[:, :2].copy()
        fitted.append(self.get_params())
        return self

    monkeypatch.setattr(IteratedMinimumSpanningTreeEmbedder, "fit_transform", fake_fit_transform)
    monkeypatch.setattr(IteratedMinimumSpanningTreeEmbedder, "fit", fake_fit)

    X = np.array(
        [[-2.0, i] for i in range(5)] + [[2.0, i] for i in range(5)],
        dtype=float,
    )
    y = np.array(["left"] * 5 + ["right"] * 5)
    mapper = IteratedMinimumSpanningTreeEmbedder(random_state=7)
    intermediate = []
    result = mapper.tune(
        X,
        y,
        search_space={
            "n_msts": (1, 3),
            "mst_max_neighbors": (40, 80),
            "learning_rate": (0.01, 0.1),
            "include_knn_attraction": [False, True],
        },
        n_samples=3,
        n_initial_points=20,
        trial_callback=intermediate.append,
    )

    assert calls["n_calls"] == 3
    assert calls["n_initial_points"] == 3
    assert calls["initial_point_generator"] == "lhs"
    assert calls["random_state"] == 7
    assert [type(d).__name__ for d in calls["dimensions"]] == [
        "_IntegerDimension", "_IntegerDimension", "_RealDimension",
        "_CategoricalDimension",
    ]
    assert result.best_score == pytest.approx(1.0)
    assert result.cv_scores.shape == (3,)
    assert result.best_embedding.shape == (len(X), 2)
    assert mapper.best_params_ == result.best_params
    assert mapper.best_score_ == result.best_score
    assert mapper.n_components == 2
    assert mapper.embedding_.shape == (len(X), 2)
    assert fitted[-1]["n_msts"] == mapper.n_msts
    assert len(intermediate) == 3
    assert intermediate[-1]["embedding"].shape == (len(X), 2)
    assert "best_params" in intermediate[-1]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"k": 0}, "k must be a positive integer"),
        ({"n_samples": 0}, "n_samples must be a positive integer"),
    ],
)
def test_tune_rejects_invalid_budget_or_k(monkeypatch, kwargs, message):
    _fake_skopt(monkeypatch)
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.array([0] * 5 + [1] * 5)
    with pytest.raises(ValueError, match=message):
        IteratedMinimumSpanningTreeEmbedder().tune(
            X, y, search_space={"n_msts": (1, 2)}, **kwargs
        )


def test_tune_requires_five_members_of_each_class(monkeypatch):
    _fake_skopt(monkeypatch)
    X = np.arange(18, dtype=float).reshape(9, 2)
    y = np.array([0] * 4 + [1] * 5)
    with pytest.raises(ValueError, match="at least 5 samples in every class"):
        IteratedMinimumSpanningTreeEmbedder().tune(
            X, y, search_space={"n_msts": (1, 2)}
        )


def test_tune_rejects_searching_embedding_dimensions(monkeypatch):
    _fake_skopt(monkeypatch)
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.array([0] * 5 + [1] * 5)
    with pytest.raises(ValueError, match="n_components is fixed at 2"):
        IteratedMinimumSpanningTreeEmbedder().tune(
            X, y, search_space={"n_components": (2, 3)}
        )
