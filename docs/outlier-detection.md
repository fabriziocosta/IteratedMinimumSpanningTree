# Outlier detection

`detect_outliers()` fits an IMSTE embedding with 10 dimensions by default, then
scores local density in that embedding. Its higher-dimensional default keeps
more neighborhood detail for anomaly scoring than the usual two-dimensional
visualization. The neighbor graph is built with RINND when available and
PyNNDescent otherwise; that same fast graph is used for both LOF scores and
reciprocal-neighbor counts.

## Fit and rank samples

```python
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

from imste import IteratedMinimumSpanningTreeEmbedder

digits = load_digits()
X = StandardScaler().fit_transform(digits.data)

mapper = IteratedMinimumSpanningTreeEmbedder(
    n_components=10,
    n_epochs=100,
    random_state=42,
)
embedding = mapper.fit_transform(X)
result = mapper.outliers(k=20, n_components=[5, 10, 30])
top_candidates = result.ranked_indices[:100]
detected_outliers = result.outlier_indices
```

`mapper.outliers()` scores the fitted embedding, so first fit the estimator
with `fit` or `fit_transform`. Set `n_components` and other IMSTE options in
the usual estimator constructor. `k` must be between 1 and `n_samples - 1`.
When `n_components` is a sequence, IMSTE refits cloned estimators for each
requested dimension, reusing the current fitted model when its dimension is in
the sequence. The original estimator remains unchanged. Multiple dimensions
require a feature-based fit; pass `y` if refitting uses positive
`label_strength`.
The standalone
`detect_outliers(X, k=20, estimator_params=None)` convenience function remains
available and defaults to `n_components=10`.

## Read the result

`result.lof_scores` contains one standard local outlier factor score per input
row. It compares a sample's local reachability density with the densities of
its neighbors; values above 1 indicate lower density than nearby samples, and
higher values rank as stronger outlier candidates. Scores are relative to this
dataset and are not calibrated probabilities.

`result.mutual_neighbor_counts` gives the number of the sample's `k` nearest
neighbors that also include it among their own `k` nearest neighbors for the
first evaluated dimension. `result.component_rankings` and
`result.outliers_by_component` retain the per-dimension results.
`result.average_ranks` gives each row's mean LOF rank across dimensions, with
rank 1 best; `result.ranked_indices` sorts by that average. The
`result.outlier_frequency` count shows in how many dimensions each row had zero
reciprocal neighbors. `result.outlier_indices` contains rows with zero
reciprocal neighbors in every evaluated dimension, sorted by average rank. All
indices refer to the input row order. The embedding and LOF scores describe the
first evaluated dimension.

The result also includes `result.embedding` and the fitted IMSTE
`result.estimator`. See the [MNIST outlier notebook](../notebooks/mnist_outlier_detection.ipynb)
for a balanced sample and a 10×10 image grid of the 100 highest-scoring rows.
