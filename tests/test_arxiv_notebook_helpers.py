import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "src"))
from arxiv_imste import select_arxiv_experiment_subset


def test_balanced_arxiv_subset_has_equal_unique_class_counts_and_stable_id():
    X = np.arange(120, dtype=np.float32).reshape(40, 3)
    labels = pd.Series(pd.Categorical(["major"] * 28 + ["minor"] * 12))

    first = select_arxiv_experiment_subset(
        X, labels, dataset_id="source", mode="balanced", sample_size=16,
        random_state=7,
    )
    second = select_arxiv_experiment_subset(
        X, labels, dataset_id="source", mode="balanced", sample_size=16,
        random_state=7,
    )

    X_subset, labels_subset, sample_id = first
    assert X_subset.shape == (16, 3)
    assert labels_subset.value_counts().to_dict() == {"major": 8, "minor": 8}
    assert sample_id == second[2]
    np.testing.assert_array_equal(X_subset, second[0])


def test_random_arxiv_subset_is_reproducible_and_cache_id_tracks_sample():
    X = np.arange(90, dtype=np.float32).reshape(30, 3)
    labels = np.asarray(["a"] * 20 + ["b"] * 10)

    first = select_arxiv_experiment_subset(
        X, labels, dataset_id="source", mode="random", sample_size=12,
        random_state=3,
    )
    same = select_arxiv_experiment_subset(
        X, labels, dataset_id="source", mode="random", sample_size=12,
        random_state=3,
    )
    changed = select_arxiv_experiment_subset(
        X, labels, dataset_id="source", mode="random", sample_size=12,
        random_state=4,
    )

    assert first[2] == same[2]
    assert first[2] != changed[2]
    np.testing.assert_array_equal(first[0], same[0])


def test_balanced_arxiv_subset_rejects_oversampling_minority_classes():
    X = np.zeros((11, 2))
    labels = np.asarray(["a"] * 8 + ["b"] * 3)

    with pytest.raises(ValueError, match="at most 6 rows"):
        select_arxiv_experiment_subset(
            X, labels, dataset_id="source", mode="balanced", sample_size=8,
        )
