"""Bayesian parameter search for labeled IMSTE embeddings."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from inspect import Parameter, signature
from numbers import Integral, Real as NumericReal
from typing import TYPE_CHECKING, Callable

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_X_y

if TYPE_CHECKING:
    from ._estimator import IteratedMinimumSpanningTreeEmbedder


_DEFAULT_SEARCH_SPACE = {
    "n_msts": (3, 30),
    "learning_rate": (0.01, 0.1),
    "negative_ratio": (1, 10),
    "gravity_strength": (0.0, 1.0),
    "mean_shift_iterations": (0, 5),
}


def _load_skopt():
    try:
        from skopt import gp_minimize
        from skopt.space import Categorical, Integer, Real
    except ImportError as exc:
        raise ImportError(
            "IteratedMinimumSpanningTreeEmbedder.tune requires scikit-optimize. "
            "Install it with `python -m pip install 'imste[tuning]'`."
        ) from exc
    return gp_minimize, Integer, Real, Categorical


def _search_dimensions(estimator, search_space, Integer, RealDimension, Categorical):
    if not isinstance(search_space, Mapping) or not search_space:
        raise ValueError("search_space must be a non-empty mapping of parameter ranges.")

    estimator_parameters = estimator.get_params(deep=True)
    constructor_parameters = signature(type(estimator).__init__).parameters
    names = list(search_space)
    dimensions = []
    for name, specification in search_space.items():
        if name not in estimator_parameters:
            raise ValueError(f"Unknown estimator parameter in search_space: {name!r}.")
        if name == "n_components":
            raise ValueError("n_components is fixed at 2 during tuning and cannot be searched.")

        # Tuples denote numeric bounds; lists denote explicit choices.
        if isinstance(specification, tuple) and (
            len(specification) != 2
            or not all(
                isinstance(value, NumericReal) and not isinstance(value, bool)
                for value in specification
            )
        ):
            raise ValueError(
                f"Tuple for {name!r} must contain exactly two numeric bounds."
            )
        is_bounds = isinstance(specification, tuple)
        if is_bounds:
            low, high = specification
            if not np.isfinite([low, high]).all() or low >= high:
                raise ValueError(f"Bounds for {name!r} must be finite and increasing.")
            current = estimator_parameters[name]
            constructor_parameter = constructor_parameters.get(name)
            parameter_default = (
                Parameter.empty
                if constructor_parameter is None
                else constructor_parameter.default
            )
            type_reference = (
                parameter_default
                if parameter_default is not Parameter.empty and parameter_default is not None
                else current
            )
            if isinstance(type_reference, (bool, np.bool_)):
                raise ValueError(
                    f"Boolean parameter {name!r} needs an explicit categorical list."
                )
            if isinstance(type_reference, (Integral, np.integer)):
                if int(low) != low or int(high) != high:
                    raise ValueError(f"Integer parameter {name!r} requires integer bounds.")
                dimensions.append(Integer(int(low), int(high), name=name))
            elif isinstance(type_reference, (NumericReal, np.floating)):
                dimensions.append(RealDimension(float(low), float(high), name=name))
            elif type_reference is None and all(
                isinstance(value, Integral) for value in specification
            ):
                dimensions.append(Integer(int(low), int(high), name=name))
            elif type_reference is None and all(
                isinstance(value, NumericReal) for value in specification
            ):
                dimensions.append(RealDimension(float(low), float(high), name=name))
            else:
                raise ValueError(
                    f"Numeric bounds for {name!r} require an integer or float estimator parameter."
                )
        else:
            if not isinstance(specification, list) or len(specification) == 0:
                raise ValueError(
                    f"Range for {name!r} must be a numeric (low, high) tuple "
                    "or a non-empty list of categorical choices."
                )
            dimensions.append(Categorical(list(specification), name=name))
    return names, dimensions


def spatial_coverage_score(embedding: object, *, bins: int | None = None) -> float:
    """Return a robust, scale-independent score for 2D point-space coverage.

    Coordinates are scaled using their 1st and 99th percentiles, clipped to a
    unit square, and summarized by the entropy of a 2D occupancy histogram.
    Entropy is converted to effective occupied cells and normalized to
    ``[0, 1]``. The default grid resolution adapts to sample count.
    """
    coordinates = np.asarray(embedding, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 2:
        raise ValueError("embedding must have shape (n_samples >= 2, 2).")
    if not np.isfinite(coordinates).all():
        raise ValueError("embedding must contain only finite coordinates.")
    if bins is None:
        bins = int(np.clip(np.ceil(np.sqrt(len(coordinates) / 2)), 12, 48))
    if isinstance(bins, (bool, np.bool_)) or not isinstance(
        bins, (Integral, np.integer)
    ) or bins < 2:
        raise ValueError("bins must be an integer of at least 2.")

    lower, upper = np.quantile(coordinates, [0.01, 0.99], axis=0)
    span = upper - lower
    scaled = np.full_like(coordinates, 0.5)
    variable_axes = span > 0
    scaled[:, variable_axes] = (
        (coordinates[:, variable_axes] - lower[variable_axes])
        / span[variable_axes]
    )
    scaled = np.clip(scaled, 0.0, 1.0)

    occupancy, _, _ = np.histogram2d(
        scaled[:, 0], scaled[:, 1], bins=int(bins), range=((0.0, 1.0), (0.0, 1.0))
    )
    probabilities = occupancy.ravel()
    probabilities = probabilities[probabilities > 0] / len(coordinates)
    entropy = -np.sum(probabilities * np.log(probabilities))
    effective_cells = np.exp(entropy)
    total_cells = int(bins) ** 2
    return float(np.clip((effective_cells - 1.0) / (total_cells - 1.0), 0.0, 1.0))


def tune_estimator(
    estimator: "IteratedMinimumSpanningTreeEmbedder",
    X: object,
    y: object,
    *,
    search_space: Mapping[str, object] | None = None,
    k: int = 5,
    n_samples: int = 40,
    n_initial_points: int = 20,
    coverage_weight: float = 0.4,
    trial_callback: Callable[[dict[str, object]], None] | None = None,
    verbose: bool = False,
):
    """Search using CV accuracy combined with robust spatial coverage."""
    gp_minimize, Integer, Real, Categorical = _load_skopt()

    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (Integral, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer.")
    if (isinstance(n_samples, (bool, np.bool_))
            or not isinstance(n_samples, (Integral, np.integer)) or n_samples < 1):
        raise ValueError("n_samples must be a positive integer.")
    if (isinstance(n_initial_points, (bool, np.bool_))
            or not isinstance(n_initial_points, (Integral, np.integer))
            or n_initial_points < 1):
        raise ValueError("n_initial_points must be a positive integer.")
    if (isinstance(coverage_weight, (bool, np.bool_))
            or not isinstance(coverage_weight, NumericReal)
            or not np.isfinite(coverage_weight)
            or not 0.0 <= coverage_weight <= 1.0):
        raise ValueError("coverage_weight must be a finite number in [0, 1].")
    if trial_callback is not None and not callable(trial_callback):
        raise ValueError("trial_callback must be callable or None.")
    if not isinstance(verbose, (bool, np.bool_)):
        raise ValueError("verbose must be a boolean.")

    X_checked, y_checked = check_X_y(
        X, y, accept_sparse="csr", ensure_2d=True, dtype=None
    )
    classes, class_counts = np.unique(y_checked, return_counts=True)
    if len(classes) < 2:
        raise ValueError("y must contain at least two classes for classification.")
    if class_counts.min() < 5:
        raise ValueError(
            "5-fold stratified tuning requires at least 5 samples in every class."
        )
    smallest_training_fold = len(y_checked) - int(np.ceil(len(y_checked) / 5))
    if k > smallest_training_fold:
        raise ValueError(
            f"k={k} exceeds the smallest 5-fold training split ({smallest_training_fold})."
        )

    if search_space is None:
        search_space = _DEFAULT_SEARCH_SPACE
    search_space = dict(search_space)
    supervised_tuning = float(estimator.label_strength) > 0.0
    if "label_strength" in search_space:
        strength_values = search_space["label_strength"]
        if isinstance(strength_values, tuple):
            supervised_tuning |= float(strength_values[1]) > 0.0
        elif isinstance(strength_values, list):
            supervised_tuning |= any(float(value) > 0.0 for value in strength_values)
    names, dimensions = _search_dimensions(
        estimator, search_space, Integer, Real, Categorical
    )
    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=estimator.random_state
    )
    folds = list(cv.split(X_checked, y_checked))
    cv_scores: list[float] = []
    coverage_scores: list[float] = []
    objective_scores: list[float] = []
    best_objective_so_far = -np.inf
    best_accuracy_so_far = -np.inf
    best_coverage_so_far = -np.inf
    best_parameters_so_far: dict[str, object] | None = None

    def objective(values):
        nonlocal best_objective_so_far, best_accuracy_so_far
        nonlocal best_coverage_so_far, best_parameters_so_far
        parameters = dict(zip(names, values))
        candidate = clone(estimator)
        candidate.set_params(n_components=2, **parameters)
        candidate.verbose = bool(verbose)
        if supervised_tuning:
            fold_accuracies = []
            fold_coverages = []
            embedding = None
            for training_indices, validation_indices in folds:
                fold_labels = np.full(len(y_checked), None, dtype=object)
                fold_labels[training_indices] = y_checked[training_indices]
                fold_estimator = clone(candidate)
                fold_embedding = fold_estimator.fit_transform(
                    X_checked,
                    y=fold_labels if fold_estimator.label_strength > 0.0 else None,
                )
                if embedding is None:
                    embedding = fold_embedding
                classifier = KNeighborsClassifier(n_neighbors=int(k))
                classifier.fit(
                    fold_embedding[training_indices], y_checked[training_indices]
                )
                fold_accuracies.append(
                    float(
                        np.mean(
                            classifier.predict(fold_embedding[validation_indices])
                            == y_checked[validation_indices]
                        )
                    )
                )
                fold_coverages.append(spatial_coverage_score(fold_embedding))
            accuracy = float(np.mean(fold_accuracies))
            coverage = float(np.mean(fold_coverages))
        else:
            embedding = candidate.fit_transform(X_checked)
            accuracy = float(
                cross_val_score(
                    KNeighborsClassifier(n_neighbors=int(k)),
                    embedding,
                    y_checked,
                    cv=cv,
                    scoring="accuracy",
                ).mean()
            )
            coverage = spatial_coverage_score(embedding)
        objective_score = (
            (1.0 - float(coverage_weight)) * accuracy
            + float(coverage_weight) * coverage
        )
        cv_scores.append(accuracy)
        coverage_scores.append(coverage)
        objective_scores.append(objective_score)
        is_best = objective_score > best_objective_so_far
        if is_best:
            best_objective_so_far = objective_score
            best_accuracy_so_far = accuracy
            best_coverage_so_far = coverage
            best_parameters_so_far = parameters.copy()
        if trial_callback is not None:
            trial_callback(
                {
                    "trial": len(cv_scores),
                    "parameters": parameters.copy(),
                    "score": accuracy,
                    "accuracy": accuracy,
                    "coverage": coverage,
                    "objective": objective_score,
                    "best_score": float(best_accuracy_so_far),
                    "best_accuracy": float(best_accuracy_so_far),
                    "best_coverage": float(best_coverage_so_far),
                    "best_objective": float(best_objective_so_far),
                    "best_params": best_parameters_so_far.copy(),
                    "embedding": embedding.copy(),
                    "is_best": is_best,
                }
            )
        return -objective_score

    initial_count = min(int(n_initial_points), int(n_samples))
    with warnings.catch_warnings():
        if not verbose:
            # Repeated proposals are normal in a bounded Bayesian search. Keep
            # skopt's warning about replacing them with random points out of
            # otherwise quiet notebook runs.
            warnings.filterwarnings(
                "ignore",
                message=r"The objective has been evaluated at point .* before, using random point .*",
                category=UserWarning,
                module=r"skopt\.optimizer\.optimizer",
            )
        result = gp_minimize(
            objective,
            dimensions,
            n_calls=int(n_samples),
            n_initial_points=initial_count,
            initial_point_generator="lhs",
            random_state=estimator.random_state,
            verbose=bool(verbose),
        )

    best_index = int(np.argmin(result.func_vals))
    best_parameters = dict(zip(names, result.x))
    estimator.set_params(n_components=2, **best_parameters)
    original_verbose = estimator.verbose
    try:
        estimator.verbose = bool(verbose)
        estimator.fit(
            X_checked,
            y=y_checked if estimator.label_strength > 0.0 else None,
        )
    finally:
        estimator.verbose = original_verbose

    result.best_params = best_parameters
    result.best_score = float(cv_scores[best_index])
    result.best_accuracy = result.best_score
    result.best_coverage = float(coverage_scores[best_index])
    result.best_objective = float(objective_scores[best_index])
    result.coverage_weight = float(coverage_weight)
    result.search_space = search_space
    result.k = int(k)
    result.n_samples = int(n_samples)
    result.n_initial_points = initial_count
    result.cv_scores = np.asarray(cv_scores, dtype=float)
    result.coverage_scores = np.asarray(coverage_scores, dtype=float)
    result.objective_scores = np.asarray(objective_scores, dtype=float)
    result.best_embedding = estimator.embedding_.copy()
    estimator.best_params_ = best_parameters
    estimator.best_score_ = result.best_score
    estimator.best_accuracy_ = result.best_accuracy
    estimator.best_coverage_ = result.best_coverage
    estimator.best_objective_ = result.best_objective
    estimator.tune_result_ = result
    return result
