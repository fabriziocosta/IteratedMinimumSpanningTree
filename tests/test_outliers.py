import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from imste import OutlierDetectionResult, detect_outliers
from imste import _outliers


def _exact_embedding_neighbors(coordinates, k, *, random_state):
    search = NearestNeighbors(n_neighbors=k + 1).fit(coordinates)
    distances, indices = search.kneighbors(coordinates)
    out_distances = np.empty((len(coordinates), k))
    out_indices = np.empty((len(coordinates), k), dtype=int)
    for row in range(len(coordinates)):
        keep = indices[row] != row
        out_distances[row] = distances[row, keep][:k]
        out_indices[row] = indices[row, keep][:k]
    return out_distances, out_indices


def test_fast_embedding_neighbors_prefers_rinnd_and_returns_nonself_rows(monkeypatch):
    import imste.mst.famst as famst

    captured = {}

    class FakeRINND:
        def __init__(self, data, **kwargs):
            captured["data"] = data
            captured["kwargs"] = kwargs
            self.neighbor_graph = (
                np.array([[1, 2], [0, 2], [1, 0]], dtype=np.int32),
                np.array([[0.1, 2.0], [0.1, 1.9], [1.9, 2.0]], dtype=np.float32),
            )

    monkeypatch.setattr(famst, "_rinnd_class", lambda: FakeRINND)
    coordinates = np.arange(6, dtype=np.float64).reshape(3, 2)
    distances, indices = _outliers._fast_embedding_neighbors(
        coordinates, 2, random_state=7
    )

    assert captured["data"].dtype == np.float32
    assert captured["data"].flags.c_contiguous
    assert captured["kwargs"]["n_neighbors"] == 2
    assert captured["kwargs"]["random_state"] == 7
    np.testing.assert_array_equal(indices, [[1, 2], [0, 2], [1, 0]])
    np.testing.assert_allclose(distances, [[0.1, 2.0], [0.1, 1.9], [1.9, 2.0]])


def test_lof_ranks_sparse_local_density_and_identifies_zero_reciprocity(monkeypatch):
    monkeypatch.setattr(_outliers, "_fast_embedding_neighbors", _exact_embedding_neighbors)
    embedding = np.array([[0.0], [0.1], [0.2], [0.3], [10.0]])

    scores, mutual_counts, ranked, outliers = _outliers._score_embedding(embedding, k=2)

    assert scores[4] > scores[3] == scores[0]
    np.testing.assert_array_equal(mutual_counts, [1, 2, 2, 1, 0])
    assert ranked[0] == 4
    np.testing.assert_array_equal(outliers, [4])


@pytest.mark.parametrize(
    "embedding, k, message",
    [
        (np.zeros((1, 2)), 1, "at least two rows"),
        (np.array([[0.0], [1.0]]), 0, "positive integer"),
        (np.array([[0.0], [1.0]]), True, "positive integer"),
        (np.array([[0.0], [1.0]]), 2, "smaller than the number of samples"),
        (np.array([[0.0], [np.inf]]), 1, "finite values"),
    ],
)
def test_score_embedding_validates_input_and_k(monkeypatch, embedding, k, message):
    monkeypatch.setattr(_outliers, "_fast_embedding_neighbors", _exact_embedding_neighbors)
    with pytest.raises(ValueError, match=message):
        _outliers._score_embedding(embedding, k)


def test_detect_outliers_uses_ten_dimensions_by_default(monkeypatch):
    import imste._estimator as estimator_module

    fitted = {}
    expected_embedding = np.array([[0.0], [0.1], [0.2], [0.3], [10.0]])

    def fake_fit(self, X):
        fitted["n_components"] = self.n_components
        self.embedding_ = expected_embedding.copy()
        return self

    monkeypatch.setattr(
        estimator_module.IteratedMinimumSpanningTreeEmbedder,
        "fit",
        fake_fit,
    )
    monkeypatch.setattr(_outliers, "_fast_embedding_neighbors", _exact_embedding_neighbors)

    result = detect_outliers(np.arange(10).reshape(5, 2), k=2)

    assert isinstance(result, OutlierDetectionResult)
    assert fitted["n_components"] == 10
    np.testing.assert_array_equal(result.embedding, expected_embedding)
    assert result.estimator.n_components == 10
    assert result.lof_scores.shape == (5,)
    assert result.mutual_neighbor_counts.shape == (5,)
    assert result.ranked_indices[0] == 4
    np.testing.assert_array_equal(result.outlier_indices, [4])


def test_detect_outliers_allows_embedding_dimension_override(monkeypatch):
    import imste._estimator as estimator_module

    seen = {}

    def fake_fit(self, X):
        seen["n_components"] = self.n_components
        self.embedding_ = np.array([[0.0], [1.0], [2.0]])
        return self

    monkeypatch.setattr(
        estimator_module.IteratedMinimumSpanningTreeEmbedder,
        "fit",
        fake_fit,
    )
    monkeypatch.setattr(_outliers, "_fast_embedding_neighbors", _exact_embedding_neighbors)
    detect_outliers(
        np.arange(6).reshape(3, 2),
        k=1,
        estimator_params={"n_components": 3},
    )
    assert seen["n_components"] == 3


def test_estimator_outliers_uses_instance_and_returns_it(monkeypatch):
    import imste._estimator as estimator_module

    embedding = np.array([[0.0], [0.1], [0.2], [0.3], [10.0]])
    def fake_fit_transform(self, X):
        self.embedding_ = embedding.copy()
        return embedding.copy()

    monkeypatch.setattr(
        estimator_module.IteratedMinimumSpanningTreeEmbedder,
        "fit_transform",
        fake_fit_transform,
    )
    monkeypatch.setattr(_outliers, "_fast_embedding_neighbors", _exact_embedding_neighbors)
    mapper = estimator_module.IteratedMinimumSpanningTreeEmbedder(n_components=5)

    mapper.fit_transform(np.arange(10).reshape(5, 2))
    result = mapper.outliers(k=2)

    assert result.estimator is mapper
    assert mapper.n_components == 5
    np.testing.assert_array_equal(result.embedding, embedding)
    np.testing.assert_array_equal(result.outlier_indices, [4])


def test_estimator_outliers_requires_fitted_embedding():
    from imste import IteratedMinimumSpanningTreeEmbedder

    with pytest.raises(ValueError, match="Fit the estimator before calling outliers"):
        IteratedMinimumSpanningTreeEmbedder().outliers(k=1)


def test_estimator_outliers_averages_ranks_across_dimensions(monkeypatch):
    import imste._estimator as estimator_module

    input_rows = np.arange(8, dtype=float).reshape(4, 2)
    embeddings = {
        dimension: np.full((4, 1), dimension, dtype=float)
        for dimension in (5, 10, 30)
    }
    rankings = {
        5: np.array([0, 1, 2, 3]),
        10: np.array([0, 2, 1, 3]),
        30: np.array([0, 1, 3, 2]),
    }
    dimension_outliers = {
        5: np.array([0, 3]),
        10: np.array([0]),
        30: np.array([0, 2]),
    }

    def fake_fit(self, X, y=None):
        self.embedding_ = embeddings[self.n_components].copy()
        self.X_fit_ = np.asarray(X).copy()
        return self

    def fake_score(embedding, k, *, random_state):
        dimension = int(embedding[0, 0])
        ranked = rankings[dimension]
        scores = np.empty(len(ranked), dtype=float)
        scores[ranked] = np.arange(len(ranked), 0, -1)
        mutual = np.ones(len(ranked), dtype=np.int64)
        mutual[dimension_outliers[dimension]] = 0
        return scores, mutual, ranked.copy(), dimension_outliers[dimension].copy()

    monkeypatch.setattr(estimator_module.IteratedMinimumSpanningTreeEmbedder, "fit", fake_fit)
    monkeypatch.setattr(_outliers, "_score_embedding", fake_score)
    mapper = estimator_module.IteratedMinimumSpanningTreeEmbedder(n_components=2)
    mapper.X_fit_ = input_rows.copy()
    mapper.embedding_ = np.zeros((4, 2))
    original_embedding = mapper.embedding_.copy()

    result = mapper.outliers(k=1, n_components=[5, 10, 30])

    assert result.components == (5, 10, 30)
    np.testing.assert_allclose(result.average_ranks, [1.0, 7 / 3, 3.0, 11 / 3])
    np.testing.assert_array_equal(result.ranked_indices, [0, 1, 2, 3])
    np.testing.assert_array_equal(result.outlier_frequency, [3, 0, 1, 1])
    np.testing.assert_array_equal(result.outlier_indices, [0])
    np.testing.assert_array_equal(mapper.embedding_, original_embedding)
    assert mapper.n_components == 2


@pytest.mark.parametrize("dimensions", [[], [2, 2], [2, 0], [2, True], "2,3"])
def test_estimator_outliers_validates_component_list(dimensions):
    from imste import IteratedMinimumSpanningTreeEmbedder

    mapper = IteratedMinimumSpanningTreeEmbedder(n_components=2)
    mapper.embedding_ = np.zeros((4, 2))
    with pytest.raises(ValueError, match="n_components"):
        mapper.outliers(k=1, n_components=dimensions)


def test_detect_outliers_rejects_non_mapping_params():
    with pytest.raises(ValueError, match="mapping or None"):
        detect_outliers(np.arange(8).reshape(4, 2), estimator_params=[("n_epochs", 0)])
