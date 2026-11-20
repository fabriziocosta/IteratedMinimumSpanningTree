# Getting started

IMSTE is a scikit-learn transformer for learning a low-dimensional layout of a
fixed dataset. It builds a graph from ranked minimum spanning tree iterations
and optimizes one coordinate vector per input row. The default graph builder
uses a sparse approximate neighbor graph; exact Prim construction is also
available.

## Install

Install the CPU-capable core package from PyPI:

```bash
python -m pip install imste
```

Install optional features only when you need them:

```bash
python -m pip install "imste[gpu]"           # PyTorch optimizer and InductiveIMSTE
python -m pip install "imste[notebook]"       # plotting, widgets, and notebook tools
python -m pip install "imste[preprocessing]"  # text and image vectorization
python -m pip install "imste[tuning]"         # Bayesian parameter search
```

The graph preprocessing preset also requires `NSPPK`, which is installed from
its source repository; see the [preprocessing guide](preprocessing-guide.md).
From a source checkout, append the desired extra to `.` (for example,
`python -m pip install ".[gpu]"`).

## Embed a dataset

This example uses scikit-learn's built-in digits dataset and needs no download:

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

from imste import IteratedMinimumSpanningTreeEmbedder

digits = load_digits()
X = StandardScaler().fit_transform(digits.data)

mapper = IteratedMinimumSpanningTreeEmbedder(random_state=42)
embedding = mapper.fit_transform(X)

plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, s=10, cmap="tab10")
plt.show()
```

`fit_transform` returns an array with shape `(n_samples, n_components)`. The
labels in this example are used only to color points after fitting; IMSTE does
not use them to construct the layout. Scale features when they use substantially
different units or ranges.

## Choose the graph builder

The default `mst_method="famst"` uses approximate nearest neighbors and is
designed for sparse graph construction. Choose `mst_method="prim"` for exact
trees. Exact Prim can take quadratic time; for CSR input it computes distance
rows on demand rather than storing a dense all-pairs matrix.

```python
mapper = IteratedMinimumSpanningTreeEmbedder(
    n_msts=5,
    mst_method="famst",
    mst_neighbors=15,
    random_state=42,
)
embedding = mapper.fit_transform(X)
```

For an existing sparse distance graph, use `fit_graph()` or
`fit_graph_transform()`. The matrix must be square and contain finite,
non-negative stored distances. Implicit zeros mean absent edges; explicitly
stored zero-distance edges are retained. See the [estimator API](api) for
details.

## Training rows and new rows

`IteratedMinimumSpanningTreeEmbedder` is transductive: its `transform` method
returns fitted coordinates for the original rows in their original order. It
does not project new observations. Install the `gpu` extra to use
`InductiveIMSTE`, which fits an MLP to approximate a mapping from feature rows
to embedding coordinates.

## Next steps

* [Preprocess raw text, images, and graph inputs](preprocessing-guide.md)
* [Understand multilevel initialization](multilevel-initialization.md)
* [Tune layouts for class separation and coverage](tuning-guide.md)
* [Explore visualization helpers](visualization-guide.md)
* [Read the method overview](whitepaper.md)
