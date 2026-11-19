"""Input transformers for raw-data IMSTE workflows."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class _EmbeddingGemmaTransformer(BaseEstimator, TransformerMixin):
    """Sentence Transformers adapter for text or image inputs."""

    def __init__(
        self,
        modality: str,
        model_name: str = "google/embeddinggemma-2",
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.modality = modality
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

    def fit(self, X: object, y: object = None) -> "_EmbeddingGemmaTransformer":
        del y
        if self.modality not in {"text", "image"}:
            raise ValueError("modality must be either 'text' or 'image'.")
        if isinstance(self.batch_size, (bool, np.bool_)) or not isinstance(
            self.batch_size, (int, np.integer)
        ) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "The text and image preprocessors require Sentence Transformers. "
                "Install them with `python -m pip install 'imste[preprocessing]'`."
            ) from exc

        self.model_ = SentenceTransformer(self.model_name, device=self.device)
        get_dimension = getattr(self.model_, "get_embedding_dimension", None)
        if not callable(get_dimension):
            # Keep compatibility with older Sentence Transformers releases.
            get_dimension = self.model_.get_sentence_embedding_dimension
        self.n_features_out_ = get_dimension()
        return self

    def transform(self, X: object) -> np.ndarray:
        check_is_fitted(self, attributes=["model_"])
        inputs = _as_modality_samples(X, self.modality)
        vectors = self.model_.encode(
            inputs,
            batch_size=int(self.batch_size),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def _as_modality_samples(X: object, modality: str) -> list[object]:
    """Normalize single inputs and batches to the sequence expected by encode."""
    if modality == "text":
        if isinstance(X, str):
            return [X]
        if isinstance(X, np.ndarray):
            if X.ndim != 1:
                raise ValueError(
                    "Text input must be a string or one-dimensional sequence of strings."
                )
            values = X.tolist()
        else:
            try:
                values = list(X)  # type: ignore[arg-type]
            except TypeError as exc:
                raise ValueError(
                    "Text input must be a string or one-dimensional sequence of strings."
                ) from exc
        if not all(isinstance(value, str) for value in values):
            raise ValueError("Every text input must be a string.")
        return values

    if modality == "image":
        if isinstance(X, (str, Path)):
            return [X]
        if isinstance(X, np.ndarray):
            if X.ndim in {2, 3}:
                return [X]
            if X.ndim == 4:
                return list(X)
            raise ValueError(
                "Image input must be one image, a sequence of images, or a 4D image batch."
            )
        try:
            values = list(X)  # type: ignore[arg-type]
        except TypeError:
            return [X]
        return values

    try:
        return list(X)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("Graph input must be a sequence of graph objects.") from exc


def make_preprocessor(
    preprocessor: str | object,
    *,
    options: Mapping[str, Any] | None = None,
) -> object:
    """Resolve a modality preset or validate a supplied sklearn transformer."""
    options = {} if options is None else dict(options)
    if isinstance(preprocessor, str):
        if preprocessor in {"text", "image"}:
            unexpected = set(options) - {"model_name", "batch_size", "device"}
            if unexpected:
                raise TypeError(
                    "Text/image preprocessor options must be among "
                    "'model_name', 'batch_size', and 'device'; got "
                    f"{sorted(unexpected)}."
                )
            return _EmbeddingGemmaTransformer(modality=preprocessor, **options)
        if preprocessor == "graph":
            try:
                from nsppk import NSPPK
            except ImportError as exc:
                raise ImportError(
                    "The graph preprocessor requires NSPPK. Install it with "
                    "`python -m pip install git+https://github.com/Xfcosta/NSPPK.git`."
                ) from exc
            return NSPPK(**options)
        raise ValueError(
            "preprocessor must be 'text', 'image', 'graph', or a transformer."
        )

    if options:
        raise TypeError("Options can only be passed with a modality preset string.")
    if not callable(getattr(preprocessor, "fit", None)) or not callable(
        getattr(preprocessor, "transform", None)
    ):
        raise TypeError(
            "preprocessor must be a modality name or sklearn-compatible transformer."
        )
    return preprocessor


def fit_transform_input(
    transformer: object,
    X: object,
    y: object = None,
) -> object:
    """Fit one transformer and return its feature representation."""
    fit_transform = getattr(transformer, "fit_transform", None)
    if callable(fit_transform):
        return fit_transform(X, y) if y is not None else fit_transform(X)
    fit = getattr(transformer, "fit")
    fit(X, y) if y is not None else fit(X)
    return transformer.transform(X)
