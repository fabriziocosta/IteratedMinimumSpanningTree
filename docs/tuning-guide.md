# Tuning IMSTE layouts for class separation and space use

`IteratedMinimumSpanningTreeEmbedder.tune()` searches IMSTE parameters for a
two-dimensional layout that balances label predictability with use of the
plot area. It uses scikit-optimize's Gaussian-process search with Latin
hypercube initial samples.

Install the optional optimizer with:

```bash
python -m pip install "imste[tuning]"
```

The complete MNIST workflow, including live trial results, is in
[`notebooks/tune_mnist.ipynb`](../notebooks/tune_mnist.ipynb). Its reusable
loading and display helpers are in
[`notebooks/src/digits_sweep.py`](../notebooks/src/digits_sweep.py).

## Plan for the compute cost

The default search performs 40 full embedding fits and then one final fit of
the selected settings. Every trial embeds all rows passed to `tune()`; the
cross-validation scores labels in that embedding and does not refit IMSTE per
fold. Cost therefore grows with both the dataset and number of trials. For an
initial check, use a small representative dataset or a balanced subset and a
shorter budget such as `n_samples=4, n_initial_points=4`. Keep at least five
examples of each class in any subset because the default scoring uses 5-fold
stratified cross-validation. The two-dimensional
[getting-started example](getting-started.md) is a quick place to begin before
moving to MNIST.

## What each trial scores

With `label_strength=0`, each parameter proposal fits one unsupervised
embedding on all rows in `X`. A 5-fold stratified CV evaluates a k-NN
classifier using that embedding and `y`, with mean accuracy as the
class-separation score. This is transductive: held-out feature rows were
present when the embedding was fitted.

Set `label_strength` above zero to make tuning supervised. Each trial then
fits five embeddings over all feature rows, masking each validation fold's
labels from that fold's embedding fit. The k-NN classifier is trained on the
fold's training coordinates and scored on its validation coordinates. Coverage
is averaged over the five embeddings. This avoids using validation labels in
the supervised attraction weights, while the feature rows remain transductive.
If the best strength is positive, the final best-parameter refit uses all
supplied labels. Searching
`label_strength` over a range or list containing a positive value also enables
this masked-fold protocol for every trial, including trials where the proposed
strength is zero. Supervised tuning therefore costs about five embedding fits
per trial rather than one.

The second score measures robust spatial coverage without labels. The public
`imste.spatial_coverage_score(embedding)` helper:

1. Scales each coordinate between its 1st and 99th percentiles and clips values
   to a unit square, limiting the effect of outliers and global coordinate
   scale.
2. Counts points in a two-dimensional grid. By default, the number of bins per
   axis is `ceil(sqrt(n_samples / 2))`, clamped to the range 12–48.
3. Computes the entropy of the occupied-cell probabilities, converts it to an
   effective number of occupied cells, and normalizes the result to `[0, 1]`.

The search maximizes this weighted score:

```text
objective = (1 - coverage_weight) * accuracy + coverage_weight * coverage
```

The default `coverage_weight=0.4` gives accuracy 60% of the objective and
coverage 40%. Set it to zero for accuracy-only tuning. A larger weight favors
layouts that spread points more evenly across the plot, even when that costs
some accuracy. The objective and both component scores are reported so this
tradeoff remains visible.

## Search space and budget

Pass a mapping from estimator parameter names to numeric bounds or categorical
choices. By default, `tune()` searches the following parameters and ranges:

| Parameter | Default range | Dimension |
| --- | --- | --- |
| `n_msts` | `(3, 30)` | Integer |
| `learning_rate` | `(0.01, 0.1)` | Continuous |
| `negative_ratio` | `(1, 10)` | Integer |
| `gravity_strength` | `(0.0, 1.0)` | Continuous |
| `mean_shift_iterations` | `(0, 5)` | Integer |

To use these defaults, omit `search_space`. A two-number tuple specifies
inclusive numeric bounds. The estimator's declared default type determines
whether the search dimension is integer or continuous. Parameters with a
`None` default use their current value or, when that is also `None`, the tuple's
endpoint types. A list specifies the
exact values to try; use this for booleans, string-valued parameters, or a
fixed set of numeric candidates. Tuples are reserved for bounds, and
categorical choices must be written as lists. `n_components` is fixed at 2 and
cannot be searched.

```python
from imste import IteratedMinimumSpanningTreeEmbedder

mapper = IteratedMinimumSpanningTreeEmbedder(
    random_state=42,
)

search = mapper.tune(X, y)

print(search.best_accuracy, search.best_coverage, search.best_objective)
embedding = mapper.embedding_
```

`n_samples` is the total number of parameter evaluations, including the
initial Latin hypercube samples. Defaults are `n_samples=40`,
`n_initial_points=20`, `k=5`, and `coverage_weight=0.4`; when the budget is
smaller, all trials use the initial-point sampler. Each embedding fit uses the
estimator's `n_epochs` setting, which defaults to 200. The estimator's
`random_state` seeds the search and the shuffled CV folds.

Each configuration fits IMSTE on the full supplied data, followed by one final
fit of the selected configuration. Search cost therefore grows with the number
of trials and the cost of an IMSTE fit. Set `n_samples` and
`n_initial_points` explicitly when you need a smaller experiment.

## Results and live progress

The returned scikit-optimize result includes the search history and these
additional fields:

| Field | Meaning |
| --- | --- |
| `best_params` | Parameter values selected by the combined objective |
| `best_score` / `best_accuracy` | 5-fold accuracy for the selected configuration |
| `best_coverage` | Spatial coverage score for the selected configuration |
| `best_objective` | Positive weighted score optimized by the search |
| `cv_scores` | Accuracy for every trial |
| `coverage_scores` | Coverage for every trial |
| `objective_scores` | Positive weighted score for every trial |
| `best_embedding` | Final refitted embedding |

The estimator is refitted on all rows using `best_params`. Its corresponding
attributes are `best_params_`, `best_score_`, `best_accuracy_`,
`best_coverage_`, `best_objective_`, and `tune_result_`. The native
scikit-optimize `result.fun` remains negative because the optimizer minimizes;
use `best_objective` for the positive score.

`trial_callback` is called after each completed trial with a dictionary
containing:

- `trial`, `parameters`, and `embedding` for the current proposal
- `accuracy` (also available as `score`), `coverage`, and `objective`
- `best_accuracy` (also available as `best_score`), `best_coverage`,
  `best_objective`, and `best_params` for the best proposal so far
- `is_best`, which is true when the current trial improves the combined score

The callback can update a table or display the best embedding while the search
runs. For plots, use the public `imste.visualization.plot_embedding` helper
described in the [visualization guide](visualization-guide.md), as the MNIST
tuning notebook does. Its reusable
`make_live_tuning_display()` helper accepts `figsize=(width, height)` in inches
so the live image size can be set directly in the notebook call.

`tune(..., verbose=False)` suppresses optimizer and estimator fit progress by
default. It produces no progress output when `trial_callback=None`. Supplying a
`trial_callback` still invokes it after every trial, regardless of `verbose`;
set `verbose=True` to also show optimizer and fit progress.

## Data requirements

`X` and `y` must have the same number of rows, and `y` must contain at least two
classes with at least five examples in every class. `k` must fit within the
smallest 5-fold training split. The MNIST notebook lets you select MNIST,
Fashion-MNIST, or Kuzushiji-MNIST and defaults to a balanced sample of 2,000
images per class. It reads the repository cache when available and downloads
the selected dataset otherwise.
