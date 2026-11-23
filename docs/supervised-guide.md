# Supervised embeddings

IMSTE can use categorical labels to reweight the attraction of edges that are
already in its graph. This is an optional way to guide the layout when class
labels are available. With the default `label_strength=0`, fitting remains
unsupervised and labels do not affect the embedding.

## Fit with labels

Pass one label per row as `y` and choose a positive `label_strength`:

```python
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

from imste import IteratedMinimumSpanningTreeEmbedder

digits = load_digits()
X = StandardScaler().fit_transform(digits.data)

mapper = IteratedMinimumSpanningTreeEmbedder(
    label_strength=0.6,
    random_state=42,
)
embedding = mapper.fit_transform(X, y=digits.target)
```

For each graph edge, same-class endpoints receive the multiplier
`exp(label_strength / 2)` and different-class endpoints receive
`exp(-label_strength / 2)`. The adjusted edge weights are normalized before
optimization. Increasing `label_strength` increases the relative attraction
weight of same-class edges. It does not add graph edges or directly repel
different classes.

`y` must contain one scalar categorical label per input row when
`label_strength` is positive. `None` and NaN entries are treated as unknown;
edges involving an unknown label keep their ordinary inverse-rank weight.
Labels guide a transductive embedding of the rows passed to `fit` or
`fit_transform`; this option does not train a classifier or provide a mapping
for new rows. For new-row projection, see the [inductive guide](inductive-guide.md).

The same reweighting is available for a supplied graph:

```python
mapper = IteratedMinimumSpanningTreeEmbedder(label_strength=0.6)
embedding = mapper.fit_graph_transform(distance_graph, y=labels)
```

To search for a useful label strength alongside other estimator parameters,
see the [tuning guide](tuning-guide.md), which describes the masked-fold
protocol used for supervised tuning.

## Fit with a continuous target

For a numeric outcome, use `target_strength` and `target_bandwidth` instead of
`label_strength`. Targets are robustly scaled, then graph edges receive a
Gaussian similarity weight: similar target values strengthen their attraction
and distant values weaken it. Edges touching an unknown target keep their
ordinary weight. The target bandwidth is measured in robust target-scale units.

```python
import numpy as np
from imste import IteratedMinimumSpanningTreeEmbedder

target = np.log1p(retweet_counts)
mapper = IteratedMinimumSpanningTreeEmbedder(
    target_strength=1.0,
    target_bandwidth=1.0,
    random_state=42,
)
embedding = mapper.fit_transform(X, y=target)
```

The categorical and continuous supervision modes are mutually exclusive. A
positive `target_strength` requires a one-dimensional, non-constant numeric
target with at least two known values. `None` and NaN entries are treated as
unknown. To tune a target-shaped layout against held-out regression
performance, see the `task="regression"` section in the
[tuning guide](tuning-guide.md).
