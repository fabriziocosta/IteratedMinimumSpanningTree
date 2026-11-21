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

from imste import detect_outliers

digits = load_digits()
X = StandardScaler().fit_transform(digits.data)

result = detect_outliers(
    X,
    k=20,
    estimator_params={"n_epochs": 100, "random_state": 42},
)
top_candidates = result.ranked_indices[:100]
detected_outliers = result.outlier_indices
```

`k` must be between 1 and `n_samples - 1`. Any estimator setting can be passed
through `estimator_params`; it can also override the default
`n_components=10`. Labels are not used by the detector.

## Read the result

`result.lof_scores` contains one standard local outlier factor score per input
row. It compares a sample's local reachability density with the densities of
its neighbors; values above 1 indicate lower density than nearby samples, and
higher values rank as stronger outlier candidates. Scores are relative to this
dataset and are not calibrated probabilities.

`result.mutual_neighbor_counts` gives the number of the sample's `k` nearest
neighbors that also include it among their own `k` nearest neighbors.
`result.outlier_indices` contains rows with zero reciprocal neighbors,
ordered by descending LOF score. `result.ranked_indices` ranks every row by LOF,
which is useful when a fixed number of top candidates is needed. These indices
refer to the input row order.

The result also includes `result.embedding` and the fitted IMSTE
`result.estimator`. See the [MNIST outlier notebook](../notebooks/mnist_outlier_detection.ipynb)
for a balanced sample and a 10×10 image grid of the 100 highest-scoring rows.
