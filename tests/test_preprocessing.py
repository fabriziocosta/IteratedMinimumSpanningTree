from types import ModuleType

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer

from imste import IteratedMinimumSpanningTreeEmbedder
from imste._preprocessing import _EmbeddingGemmaTransformer, make_preprocessor


def _mapper():
    return IteratedMinimumSpanningTreeEmbedder(
        n_msts=1,
        n_epochs=0,
        device="cpu",
        mst_method="prim",
        pca_init=False,
        multilevel_init=False,
        gravity_strength=0,
        mean_shift_iterations=0,
    )


def test_custom_transformer_preprocesses_raw_text_and_clones():
    texts = [
        "red apple fruit",
        "red berry fruit",
        "blue sky cloud",
        "blue ocean water",
        "green leaf plant",
        "green grass plant",
    ]
    mapper = _mapper().preprocessor(TfidfVectorizer())

    assert isinstance(clone(mapper).input_preprocessor, TfidfVectorizer)
    embedding = mapper.fit_transform(texts)

    assert embedding.shape == (len(texts), 2)
    assert np.isfinite(embedding).all()
    assert mapper.transform(texts).shape == embedding.shape
    with pytest.raises(ValueError, match="original training rows"):
        mapper.transform(["new unseen document", *texts[1:]])


@pytest.mark.parametrize(
    "modality, inputs",
    [
        ("text", ["first text", "second text"]),
        ("image", [np.zeros((4, 4), dtype=np.uint8), np.ones((4, 4), dtype=np.uint8)]),
    ],
)
def test_embeddinggemma_adapter_encodes_batches(monkeypatch, modality, inputs):
    received = {}

    class FakeModel:
        def __init__(self, model_name, device=None):
            received["model_name"] = model_name
            received["device"] = device

        def get_sentence_embedding_dimension(self):
            return 3

        def encode(self, values, **kwargs):
            received["values"] = values
            received["kwargs"] = kwargs
            return np.arange(len(values) * 3, dtype=np.float32).reshape(len(values), 3)

    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeModel
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", fake_module)

    transformer = _EmbeddingGemmaTransformer(modality, batch_size=7, device="cpu")
    vectors = transformer.fit_transform(inputs)

    assert vectors.shape == (2, 3)
    assert received["model_name"] == "google/embeddinggemma-2"
    assert received["device"] == "cpu"
    assert received["kwargs"]["batch_size"] == 7
    assert received["kwargs"]["show_progress_bar"] is False


def test_preprocessor_presets_and_validation(monkeypatch):
    class FakeNSPPK(BaseEstimator, TransformerMixin):
        def __init__(self, radius=1):
            self.radius = radius

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return np.zeros((len(X), 2))

    fake_module = ModuleType("nsppk")
    fake_module.NSPPK = FakeNSPPK
    monkeypatch.setitem(__import__("sys").modules, "nsppk", fake_module)

    graph_preprocessor = make_preprocessor("graph", options={"radius": 2})
    assert isinstance(graph_preprocessor, FakeNSPPK)
    assert graph_preprocessor.radius == 2
    assert make_preprocessor("text").model_name == "google/embeddinggemma-2"
    assert make_preprocessor("image").modality == "image"

    with pytest.raises(ValueError, match="must be 'text'"):
        make_preprocessor("audio")
    with pytest.raises(TypeError, match="sklearn-compatible transformer"):
        make_preprocessor(object())
