"""Out-of-sample projection for IMSTE embeddings."""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted
from scipy.sparse import issparse

from ._estimator import (
    IteratedMinimumSpanningTreeEmbedder,
    _finite_nonnegative,
    _positive_integer,
)


class _MLPProjector(nn.Module):
    """Map standardized features to standardized IMSTE coordinates."""

    def __init__(
        self,
        n_features: int,
        n_components: int,
        n_layers: int,
        layer_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_size = n_features
        for _ in range(n_layers):
            layers.extend(
                (nn.Linear(input_size, layer_size), nn.ReLU(), nn.Dropout(dropout))
            )
            input_size = layer_size
        layers.append(nn.Linear(input_size, n_components))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def _scaled_feature_rows(X, indices, mean, scale) -> np.ndarray:
    """Densify only the requested CSR batch for the dense projection MLP."""
    rows = X[indices]
    if issparse(rows):
        rows = rows.toarray()
    return np.asarray((rows - mean) / scale, dtype=np.float32)


class InductiveIMSTE(TransformerMixin, BaseEstimator):
    """Fit IMSTE coordinates and learn an MLP mapping for any input rows.

    Unlike :class:`imste.IteratedMinimumSpanningTreeEmbedder`, this estimator's
    ``transform`` always uses the learned MLP, including on its training rows.
    The exact coordinates optimized by IMSTE are available as
    ``reference_embedding_``.

    Parameters
    ----------
    embedder : IteratedMinimumSpanningTreeEmbedder or None, default=None
        Transductive IMSTE estimator used to create reference coordinates. A
        fresh estimator with its default settings is used when omitted.
    n_layers : int, default=6
        Number of hidden layers in the projection MLP.
    layer_size : int, default=128
        Number of units in each hidden layer.
    dropout : float, default=0.1
        Dropout probability after each hidden layer; must be in ``[0, 1)``.
    epochs : int, default=200
        Maximum MLP training epochs.
    min_epochs : int, default=100
        Minimum epochs before early stopping can occur.
    patience : int, default=20
        Stop after this many epochs without validation improvement.
    batch_size : int, default=256
        Number of rows per MLP update and prediction batch.
    learning_rate : float, default=0.001
        Adam learning rate for the MLP.
    """

    def __init__(
        self,
        embedder: IteratedMinimumSpanningTreeEmbedder | None = None,
        n_layers: int = 6,
        layer_size: int = 128,
        dropout: float = 0.1,
        epochs: int = 200,
        min_epochs: int = 100,
        patience: int = 20,
        batch_size: int = 256,
        learning_rate: float = 0.001,
    ) -> None:
        self.embedder = embedder
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.dropout = dropout
        self.epochs = epochs
        self.min_epochs = min_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def fit(self, X: object, y: object = None) -> "InductiveIMSTE":
        """Fit the reference embedding and its MLP projection."""
        del y
        fit_started = time.perf_counter()
        n_layers = _positive_integer("n_layers", self.n_layers)
        layer_size = _positive_integer("layer_size", self.layer_size)
        dropout = _finite_nonnegative("dropout", self.dropout)
        if dropout >= 1:
            raise ValueError("dropout must be a finite number in [0, 1).")
        epochs = _positive_integer("epochs", self.epochs)
        min_epochs = _positive_integer("min_epochs", self.min_epochs)
        patience = _positive_integer("patience", self.patience)
        batch_size = _positive_integer("batch_size", self.batch_size)
        learning_rate = _finite_nonnegative(
            "learning_rate", self.learning_rate, strictly_positive=True
        )
        if min_epochs > epochs:
            raise ValueError("min_epochs must not exceed epochs.")

        if self.embedder is None:
            self.embedder_ = IteratedMinimumSpanningTreeEmbedder()
        elif isinstance(self.embedder, IteratedMinimumSpanningTreeEmbedder):
            self.embedder_ = clone(self.embedder)
        else:
            raise ValueError(
                "embedder must be an IteratedMinimumSpanningTreeEmbedder or None."
            )
        self.embedder_.fit(X)
        self.reference_embedding_ = self.embedder_.embedding_.copy()
        self.n_features_in_ = self.embedder_.n_features_in_
        if hasattr(self.embedder_, "feature_names_in_"):
            self.feature_names_in_ = self.embedder_.feature_names_in_.copy()
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_

        X_checked = self.embedder_.X_fit_
        n_samples = X_checked.shape[0]
        validation_size = max(1, int(np.ceil(0.1 * n_samples)))
        if validation_size >= n_samples:
            raise ValueError("MLP projection requires at least two training rows.")

        # Reuse the graph estimator's seed so a single random_state controls the
        # full inductive fit.
        rng = np.random.RandomState(self.embedder_.random_state)
        split_order = rng.permutation(n_samples)
        validation_indices = split_order[:validation_size]
        training_indices = split_order[validation_size:]

        X_training = X_checked[training_indices]
        target_training = self.reference_embedding_[training_indices]
        if issparse(X_training):
            self.X_mean_ = np.asarray(X_training.mean(axis=0)).ravel()
            second_moment = np.asarray(
                X_training.multiply(X_training).mean(axis=0)
            ).ravel()
            self.X_scale_ = np.sqrt(
                np.maximum(second_moment - self.X_mean_**2, 0.0)
            )
        else:
            self.X_mean_ = X_training.mean(axis=0)
            self.X_scale_ = X_training.std(axis=0)
        self.X_scale_[self.X_scale_ == 0.0] = 1.0
        self.target_mean_ = target_training.mean(axis=0)
        self.target_scale_ = target_training.std(axis=0)
        self.target_scale_[self.target_scale_ == 0.0] = 1.0

        targets_scaled = np.asarray(
            (self.reference_embedding_ - self.target_mean_) / self.target_scale_,
            dtype=np.float32,
        )
        device = torch.device(self.embedder_.device_)
        sparse_input = issparse(X_checked)
        if sparse_input:
            X_tensor = None
        else:
            X_scaled = np.asarray(
                (X_checked - self.X_mean_) / self.X_scale_, dtype=np.float32
            )
            X_tensor = torch.as_tensor(X_scaled, device=device)
        targets_tensor = torch.as_tensor(targets_scaled, device=device)
        seed = int(rng.randint(0, np.iinfo(np.int32).max))

        projection_started = time.perf_counter()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.projector_ = _MLPProjector(
                X_checked.shape[1],
                self.reference_embedding_.shape[1],
                n_layers,
                layer_size,
                dropout,
            ).to(device)
            optimizer = torch.optim.Adam(self.projector_.parameters(), lr=learning_rate)
            best_validation_loss = float("inf")
            best_state = None
            epochs_without_improvement = 0
            self.projector_.train()

            for epoch in range(epochs):
                order = rng.permutation(training_indices)
                for start in range(0, len(order), batch_size):
                    batch = order[start : start + batch_size]
                    target_batch = torch.as_tensor(
                        batch, dtype=torch.long, device=device
                    )
                    if sparse_input:
                        input_batch = torch.as_tensor(
                            _scaled_feature_rows(
                                X_checked, batch, self.X_mean_, self.X_scale_
                            ),
                            device=device,
                        )
                    else:
                        input_batch = X_tensor[target_batch]
                    prediction = self.projector_(input_batch)
                    loss = torch.mean(
                        (prediction - targets_tensor[target_batch]) ** 2
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                self.projector_.eval()
                with torch.no_grad():
                    if sparse_input:
                        validation_error = 0.0
                        validation_count = 0
                        for start in range(0, len(validation_indices), batch_size):
                            batch = validation_indices[start : start + batch_size]
                            target_batch = torch.as_tensor(
                                batch, dtype=torch.long, device=device
                            )
                            input_batch = torch.as_tensor(
                                _scaled_feature_rows(
                                    X_checked, batch, self.X_mean_, self.X_scale_
                                ),
                                device=device,
                            )
                            error = torch.sum(
                                (
                                    self.projector_(input_batch)
                                    - targets_tensor[target_batch]
                                )
                                ** 2
                            )
                            validation_error += float(error.cpu())
                            validation_count += (
                                len(batch) * self.reference_embedding_.shape[1]
                            )
                        validation_loss = validation_error / validation_count
                    else:
                        validation_tensor = torch.as_tensor(
                            validation_indices, dtype=torch.long, device=device
                        )
                        validation_prediction = self.projector_(
                            X_tensor[validation_tensor]
                        )
                        validation_loss = float(
                            torch.mean(
                                (
                                    validation_prediction
                                    - targets_tensor[validation_tensor]
                                )
                                ** 2
                            ).cpu()
                        )
                self.projector_.train()
                if not np.isfinite(validation_loss):
                    raise ValueError(
                        "MLP validation loss became non-finite; lower the learning "
                        "rate or rescale the input data."
                    )
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_state = {
                        key: value.detach().clone()
                        for key, value in self.projector_.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if (
                    epoch + 1 >= min_epochs
                    and epochs_without_improvement >= patience
                ):
                    break

            if best_state is None:
                raise ValueError("MLP training produced no finite validation checkpoint.")
            self.projector_.load_state_dict(best_state)
            self.projector_.eval()
            with torch.no_grad():
                if sparse_input:
                    training_error = 0.0
                    training_count = 0
                    for start in range(0, len(training_indices), batch_size):
                        batch = training_indices[start : start + batch_size]
                        target_batch = torch.as_tensor(
                            batch, dtype=torch.long, device=device
                        )
                        input_batch = torch.as_tensor(
                            _scaled_feature_rows(
                                X_checked, batch, self.X_mean_, self.X_scale_
                            ),
                            device=device,
                        )
                        error = torch.sum(
                            (
                                self.projector_(input_batch)
                                - targets_tensor[target_batch]
                            )
                            ** 2
                        )
                        training_error += float(error.cpu())
                        training_count += (
                            len(batch) * self.reference_embedding_.shape[1]
                        )
                    training_loss = torch.tensor(training_error / training_count)
                else:
                    training_tensor = torch.as_tensor(
                        training_indices, dtype=torch.long, device=device
                    )
                    training_prediction = self.projector_(X_tensor[training_tensor])
                    training_loss = torch.mean(
                        (
                            training_prediction
                            - targets_tensor[training_tensor]
                        )
                        ** 2
                    )

        if device.type == "mps":
            torch.mps.synchronize()
        self.projection_fit_time_ = time.perf_counter() - projection_started
        self.training_loss_ = float(training_loss.cpu())
        self.best_validation_loss_ = best_validation_loss
        self.epochs_trained_ = epoch + 1
        self.fit_time_ = time.perf_counter() - fit_started
        return self

    def transform(self, X: object) -> np.ndarray:
        """Project rows with the learned MLP, including the training rows."""
        check_is_fitted(self, attributes=["projector_", "reference_embedding_"])
        self.embedder_._check_feature_names(X)
        X_checked = check_array(
            X, dtype=np.float64, ensure_2d=True, accept_sparse="csr"
        )
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features, but this estimator was fitted "
                f"with {self.n_features_in_} features."
            )
        device = torch.device(self.embedder_.device_)
        predictions: list[np.ndarray] = []
        self.projector_.eval()
        if issparse(X_checked):
            X_scaled = None
        else:
            X_scaled = np.asarray(
                (X_checked - self.X_mean_) / self.X_scale_, dtype=np.float32
            )
        with torch.no_grad():
            for start in range(0, X_checked.shape[0], self.batch_size):
                stop = min(start + self.batch_size, X_checked.shape[0])
                if X_scaled is None:
                    batch_features = _scaled_feature_rows(
                        X_checked,
                        slice(start, stop),
                        self.X_mean_,
                        self.X_scale_,
                    )
                else:
                    batch_features = X_scaled[start:stop]
                batch = torch.as_tensor(batch_features, device=device)
                predictions.append(self.projector_(batch).cpu().numpy())
        standardized = np.concatenate(predictions, axis=0).astype(np.float64)
        return standardized * self.target_scale_ + self.target_mean_
