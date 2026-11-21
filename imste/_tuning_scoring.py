"""Scorer resolution helpers used by IMSTE parameter tuning."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    get_scorer,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def _roc_scorer(estimator, X, y) -> float:
    classes = np.asarray(estimator.classes_)
    probabilities = estimator.predict_proba(X)
    if len(classes) == 2:
        return float(roc_auc_score(np.asarray(y) == classes[1], probabilities[:, 1]))
    return float(
        roc_auc_score(
            y,
            probabilities,
            labels=classes,
            multi_class="ovr",
            average="macro",
        )
    )


def _average_precision_scorer(estimator, X, y) -> float:
    classes = np.asarray(estimator.classes_)
    probabilities = estimator.predict_proba(X)
    if len(classes) == 2:
        return float(
            average_precision_score(np.asarray(y) == classes[1], probabilities[:, 1])
        )
    targets = label_binarize(y, classes=classes)
    return float(average_precision_score(targets, probabilities, average="macro"))


def _f1_scorer(estimator, X, y) -> float:
    return float(
        f1_score(y, estimator.predict(X), average="macro", zero_division=0)
    )


def resolve_scorer(scoring: str | Callable) -> tuple[Callable, str]:
    """Resolve a sklearn scorer or a named shortcut to a scorer callable."""
    if callable(scoring):
        return scoring, getattr(scoring, "__name__", "custom scorer")
    if not isinstance(scoring, str):
        raise ValueError("scoring must be a scikit-learn scorer or a scorer name.")

    shortcut = scoring.strip().lower().replace("-", "_").replace(" ", "_")
    if shortcut in {"accuracy", "acc"}:
        return get_scorer("accuracy"), "accuracy"
    if shortcut in {"roc", "roc_auc"}:
        return _roc_scorer, "roc"
    if shortcut in {"apr", "average_precision", "average_precision_score"}:
        return _average_precision_scorer, "apr"
    if shortcut in {"f_score", "fscore", "f1", "f1_score"}:
        return _f1_scorer, "f-score"

    try:
        return get_scorer(scoring), scoring
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Unknown scoring shortcut or scikit-learn scorer: {scoring!r}. "
            "Use 'accuracy', 'roc', 'apr', 'f-score', a sklearn scorer name, "
            "or a scorer callable."
        ) from exc


def classifier_accuracy(estimator, X, y) -> float:
    """Return accuracy for a fitted sklearn classifier."""
    return float(accuracy_score(y, estimator.predict(X)))
