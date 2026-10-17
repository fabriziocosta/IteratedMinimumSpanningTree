# IMSTE

**Iterated Minimum Spanning Tree Embedding (IMSTE)** is a scikit-learn
transformer for learning low-dimensional coordinates from numeric data. It
builds a weighted graph from ranked minimum spanning trees, then
optimizes one coordinate vector per sample.

The package has two explicit estimators: `IteratedMinimumSpanningTreeEmbedder`
learns coordinates for the training rows, while `InductiveIMSTE` also learns
an MLP that maps new rows into that space.

## Install

From a source checkout:

```bash
python -m pip install .
```

For notebooks, install the optional plotting and widget dependencies:

```bash
python -m pip install ".[notebook]"
```

Both estimators accept dense arrays and SciPy CSR matrices. The graph builders
keep CSR inputs sparse. `InductiveIMSTE` densifies only each MLP mini-batch.
The default CPU FAMST graph builder uses PyNNDescent, which is installed with
the package. MPS uses exact chunked nearest-neighbor search for dense FAMST
inputs; CSR inputs use PyNNDescent on CPU for graph construction. To use only
the exact Prim builder, select `mst_method="prim"`.

## Quick start

IMSTE uses Euclidean distances to build its graph. Scale features first when
they have different ranges. This small example uses scikit-learn's built-in
digits dataset, so it does not need a download:

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

from imste import IteratedMinimumSpanningTreeEmbedder

digits = load_digits()
rng = np.random.default_rng(42)
rows = rng.choice(len(digits.data), size=500, replace=False)
X = StandardScaler().fit_transform(digits.data[rows])

mapper = IteratedMinimumSpanningTreeEmbedder(
    n_msts=5,
    n_epochs=300,
    random_state=42,
)
embedding = mapper.fit_transform(X)

print(embedding.shape)  # (500, 2)
```

For larger datasets, FAMST is the default sparse approximate builder. To use
the exact Prim builder instead, set `mst_method="prim"`:

```python
mapper = IteratedMinimumSpanningTreeEmbedder(
    n_msts=5,
    mst_method="famst",
    mst_neighbors=15,
    mst_inter_component_edges=5,
    random_state=42,
)
embedding = mapper.fit_transform(X)
```

FAMST starts with an approximate k-nearest-neighbor graph. It samples up to
`mst_representatives_per_component` points from each component, builds an exact
MST over those representatives, and searches bridge candidates only for the
unique component pairs crossed by that MST. The default representative cap is
10. It refines those connections and extracts trees from the sparse candidate
graph. FAMST can increase the neighbor count when needed, up to
`mst_max_neighbors` (by default, four times `mst_neighbors`). If earlier trees
leave a local bottleneck, later trees reuse only the edges needed to reconnect
the residual components. Every reuse increments that edge's rank, lowering its
attraction weight to `1 / rank`. The requested `n_msts` is therefore the number
of tree iterations, not a promise that every edge is unique across trees. The
result is approximate and is not guaranteed to match the trees from the
complete Euclidean graph. The sparse mode also samples negative pairs without
allocating the core estimator's quadratic adjacency bit matrix. The
implementation follows Almansoori and Telek's
[FAMST paper](https://arxiv.org/abs/2507.14261).

The labels are not passed to the estimator. Use them afterward to color or
score a visualization if needed.

## How it works

### Build a graph

The estimator computes pairwise Euclidean distances, constructs `n_msts`
spanning-tree iterations, and joins their ranked edges into one graph. An edge
used for the first time receives that tree's rank. When a bottleneck requires
reusing it, its rank increments by one, so the repeated edge receives less
weight. For the approximate FAMST graph, an edge of rank `r` is scheduled once
every `r` epochs on average. Exact Prim processes every edge each epoch and
scales its attraction by `1 / r`. Both modes use the same normalized
inverse-rank objective. The estimator exposes these integer ranks as
`graph_ranks_` and weights as `graph_weights_ = 1 / rank`.

### Optimize coordinates

The embedding objective uses squared Euclidean distances. For an embedded pair
`i, j`, let `d² = sum((y_i - y_j) ** 2)`. Graph edges use logarithmic
attraction, `log1p(d² + epsilon)`. Sampled non-edges use logistic repulsion,
`softplus((logistic_margin - d²) / logistic_temperature)`.

`lambda_rep` sets the repulsion share of the objective; attraction receives
`1 - lambda_rep`. Positive and negative updates use the corresponding mean
reductions; attraction uses `sum(1 / graph_ranks_)` as its normalizer. Each
positive edge contributes `negative_ratio`
sampled non-neighbors from each eligible endpoint. The CPU Numba optimizer uses
float32 coordinates, linear learning-rate decay, and recenters once per epoch.
It rescales the mean objective by sample count to retain a useful per-coordinate
SGD step size. The optional MPS backend applies each epoch's edge updates as a
GPU batch, so its update trajectory differs from the sequential CPU optimizer.
Fixed seeds reproduce CPU runs with the same implementation; MPS batch
accumulation can introduce small floating-point differences. CPU and MPS
coordinates need not match each other or the former Adam optimizer.

The default `mst_method="famst"` builds an approximate sparse graph. It can
increase the neighbor count as needed, subject to `mst_max_neighbors`, then
reuse bottleneck edges with an increasing rank penalty. Select
`mst_method="prim"` for exact trees. Dense input uses quadratic pairwise
memory; CSR input avoids the all-pairs distance matrix but still takes
quadratic time.

## Choose how to transform rows

### Transductive: training rows only

Use the core estimator when you want optimized coordinates for the fitted
rows. Its `transform` accepts only the exact training matrix in its original
row order; it raises an error for new or reordered rows.

```python
from imste import IteratedMinimumSpanningTreeEmbedder

mapper = IteratedMinimumSpanningTreeEmbedder(n_msts=10).fit(X_train)
Z_train = mapper.transform(X_train)
```

### Inductive: map new rows with an MLP

Import `InductiveIMSTE` from the explicit `imste.inductive` module. It fits a
transductive IMSTE embedding, then trains an MLP to predict those coordinates:

```python
from imste import IteratedMinimumSpanningTreeEmbedder
from imste.inductive import InductiveIMSTE

mapper = InductiveIMSTE(
    embedder=IteratedMinimumSpanningTreeEmbedder(
        n_msts=10,
        random_state=42,
    ),
    n_layers=6,
    layer_size=128,
    dropout=0.1,
    min_epochs=100,
    patience=20,
).fit(X_train)

Z_train = mapper.transform(X_train)  # MLP predictions for training rows
Z_new = mapper.transform(X_new)      # MLP predictions for new rows
Z_reference = mapper.reference_embedding_  # exact optimized IMSTE coordinates
```

`InductiveIMSTE.transform` always uses the MLP, including for training rows.
Its `reference_embedding_` contains the exact transductive coordinates that
serve as MLP targets. The MLP standardizes its inputs and targets, holds out
10% of the input rows for validation, and restores the checkpoint with the
best validation loss. The IMSTE coordinates use all input rows; the validation
split only controls MLP early stopping. Predictions approximate the reference
embedding and do not extend or re-optimize its graph.

## Render an embedding

Reusable Matplotlib renderers for the kNN hull union and fitted ellipses, plus
an interactive label-guided force viewer, live in `imste.visualization`. Install
the notebook extras first (`python -m pip install "imste[notebook]"`), then pass
an embedding and either a one-dimensional label vector or a metadata DataFrame.

```python
from imste.visualization import interactive_force, plot_ellipses, plot_knn_union
from IPython.display import display

fig, ax = plot_knn_union(
    mapper, labels, k=7, cv_neighbors=30, cv_splits=5,
    distance_std_threshold=0.75,
)
fig, ax = plot_ellipses(
    mapper, labels, k=7, n_std=2.0, max_ellipses_per_region=2,
)
viewer = interactive_force(
    mapper, labels, initial_k=10, initial_iterations=20,
)
display(viewer)
```

For a DataFrame, select its label column with `label_key="cell_type"`. Each
renderer returns its Matplotlib figure and axes, except `interactive_force`,
which returns an `ipywidgets` viewer. The force controls adjust same-label
springs, different-label repulsion, iteration count, and movement. Renderers
accept a fitted `IteratedMinimumSpanningTreeEmbedder` or `InductiveIMSTE`
directly; they use `embedding_` or the inductive model's `reference_embedding_`.
Raw coordinate arrays remain supported.

## Parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `n_msts` | `15` | Number of spanning-tree iterations; bottleneck edge reuse is penalized by incrementing its rank |
| `n_components` | `2` | Embedding dimensions |
| `n_epochs` | `200` | Coordinate-optimization epochs |
| `batch_size` | `32768` | Maximum active edges per MPS update batch; ignored by CPU optimizer |
| `learning_rate` | `0.05` | Coordinate optimizer learning rate |
| `negative_ratio` | `5` | Non-neighbors sampled per endpoint of each positive edge |
| `lambda_rep` | `0.5` | Repulsion share, from 0 to 1 |
| `logistic_margin` | `1.0` | Squared-distance margin for repulsion |
| `logistic_temperature` | `0.5` | Logistic repulsion softness; must be positive |
| `epsilon` | `1e-4` | Smoothing for the logarithmic attraction |
| `random_state` | `42` | Seed for initialization, shuffling, and sampling |
| `device` | `"auto"` | `"auto"` and `"cpu"` select CPU; `"mps"` selects Apple's PyTorch GPU backend |
| `mst_method` | `"famst"` | Sparse approximate `"famst"` or exact dense `"prim"` |
| `mst_neighbors` | `15` | Initial candidate neighbors for FAMST |
| `mst_inter_component_edges` | `5` | Candidate connections per FAMST component pair |
| `mst_representatives_per_component` | `10` | Maximum representatives per component for selecting FAMST bridge pairs |
| `mst_max_neighbors` | `None` | FAMST neighbor cap; defaults to four times `mst_neighbors` |

The selected device is available as `mapper.device_` after fitting.
`device="auto"` keeps graph construction and optimization on CPU. With
`device="mps"`, FAMST builds its exact chunked nearest-neighbor candidate graph
on the Apple GPU, then performs component connection and tree extraction on
CPU; the batched optimizer also runs on MPS. MPS requires
`torch.backends.mps.is_available()` to be true. For example:

```python
mapper = IteratedMinimumSpanningTreeEmbedder(
    n_msts=5,
    n_epochs=500,
    device="mps",
    random_state=42,
)
embedding = mapper.fit_transform(X)
```

MPS samples negative pairs and computes updates for each batch on the GPU, then
applies batches sequentially. A rare exact CPU fallback handles unusually dense
neighborhoods. MPS and CPU builds use different neighbor-search methods, so
their graph and embedding can differ. Larger batches can improve GPU utilization
and use more memory; changing the batch size also changes the MPS update
trajectory. CPU runs ignore this setting.

### MLP parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `n_layers` | `6` | Hidden layers in `InductiveIMSTE` |
| `layer_size` | `128` | Units in each hidden layer |
| `dropout` | `0.1` | Dropout probability |
| `epochs` | `200` | Maximum MLP training epochs |
| `min_epochs` | `100` | Minimum epochs before early stopping |
| `patience` | `20` | Epochs without validation improvement before stopping |
| `batch_size` | `256` | Rows per MLP update and prediction batch |
| `learning_rate` | `0.001` | MLP optimizer learning rate |

These are `InductiveIMSTE` parameters. Configure the graph and coordinate
optimization by passing an `IteratedMinimumSpanningTreeEmbedder` as `embedder`.

## Notebooks

- [Performance refactor benchmark](benchmarks/performance_refactor.py): report
  separate FAMST, optimizer, and total fit medians with neighborhood metrics,
  alongside a comparable UMAP run. Local reference results are in
  [benchmarks/RESULTS.md](benchmarks/RESULTS.md).
- [MST scaling benchmark](notebooks/mst_scaling_benchmark.ipynb): compare exact
  Prim with approximate FAMST graph construction over increasing dataset sizes.
- [Interactive 2D embedding](notebooks/digits_2d_interactive.ipynb): choose a
  dataset, select exact Prim or approximate FAMST, configure fit settings, and
  optionally compare MLP predictions with the optimized coordinates.
- [Interactive 3D embedding](notebooks/digits_3d_interactive.ipynb): select
  exact Prim or approximate FAMST and explore three-dimensional embeddings
  with Plotly.
- [Parameter sweep](notebooks/digits_parameter_sweep.ipynb): compare IMSTE
  settings and reference embeddings on MNIST.
- [High-dimensional dataset gallery](notebooks/high_dim_mst_gallery.ipynb):
  view embeddings across image datasets.

For the full method description and equations, see the
[IMSTE whitepaper](WHITEPAPER.md).
