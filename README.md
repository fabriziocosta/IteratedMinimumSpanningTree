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
The default FAMST graph builder prefers RINND for dense Euclidean inputs when
RINND is installed. Otherwise it uses PyNNDescent on CPU or exact chunked
nearest-neighbor search on MPS. Install the optional RINND backend with
`python -m pip install 'imste[rinnd]'`. The neighbor-search backend can be
selected independently from the optimizer's `device`; CSR inputs continue to
use PyNNDescent. To use only the exact Prim builder, select
`mst_method="prim"`.

For an existing sparse graph whose stored values are distances, use
`fit_graph()` instead of passing the matrix to `fit()` (which treats rows as
feature vectors). Supply a square SciPy sparse matrix with finite,
non-negative values. Each stored off-diagonal entry is an edge; implicit zeros
are absent edges, while explicitly stored zero-distance edges are retained.
Edges are treated as undirected, and reciprocal entries use the smaller
distance. Disconnected components remain separate spanning forests. Each
iteration prefers unused edges; when needed, it reuses short edges and
increments their rank. The optimizer uses the usual inverse-rank attraction
weights. Optionally pass `init_features` with one feature row per graph node to
seed the coordinates from standardized PCA scores. These features affect only
initialization; the supplied graph remains the optimizer's objective. Without
`init_features`, seeded random initialization is used. This path exposes the
result through `embedding_` rather than `transform()`.

## Quick start

IMSTE uses Euclidean distance by default and also supports cosine distance
with `metric="cosine"`. Scale features first when they have different ranges.
This small example uses scikit-learn's built-in
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
    metric="cosine",
    random_state=42,
)
embedding = mapper.fit_transform(X)

print(embedding.shape)  # (500, 2)
```

```python
from imste import IteratedMinimumSpanningTreeEmbedder

mapper = IteratedMinimumSpanningTreeEmbedder(n_msts=10, n_epochs=300)
mapper.fit_graph(distance_graph, init_features=X)  # PCA-seed from X
embedding = mapper.embedding_
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
complete graph for the selected metric. Cosine metric uses exact Prim even
when `mst_method="famst"`, since FAMST's refinement is Euclidean-specific.
The sparse mode also samples negative pairs without
allocating the core estimator's quadratic adjacency bit matrix. The
implementation follows Almansoori and Telek's
[FAMST paper](https://arxiv.org/abs/2507.14261).

The optimizer normally attracts along ranked MST edges only. For a denser
local objective, `include_knn_attraction=True` also attracts along FAMST's
approximate candidate-neighbor edges as rank-one positive edges. This option
is available for Euclidean `fit(X)` calls with `mst_method="famst"`; it
increases optimization graph size
and can change neighborhood fidelity. The
[balanced arXiv experiment notebook](notebooks/05_arxiv_balanced_imste_experiments.ipynb)
compares this option with PCA/multilevel and UMAP-warm-start initializations on
a deterministic, class-balanced subset.

The labels are not passed to the estimator. Use them afterward to color or
score a visualization if needed.

## How it works

### Build a graph

The estimator computes pairwise distances using `metric`, constructs `n_msts`
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

Reusable Matplotlib renderers for the KDE density-parent tree, kNN hull union,
and fitted ellipses, plus an interactive label-guided force viewer, live in
`imste.visualization`. Install the notebook extras first
(`python -m pip install "imste[notebook]"`), then pass
an embedding and either a one-dimensional label vector or a metadata DataFrame.
For dense embeddings, `render_datashader_embedding` rasterizes the point cloud
instead of drawing every point individually. It returns a Datashader image,
supports categorical labels and custom colors, and accepts coordinate ranges
for zoomed views. Set `spread_px` to enlarge all marks by a fixed pixel radius,
or use `max_spread_px` for density-aware spreading of sparse regions.

```python
from imste.visualization import (
    interactive_force,
    plot_density_parent_tree,
    plot_ellipses,
    plot_knn_union,
    render_datashader_embedding,
)
from IPython.display import display

fig, ax = plot_knn_union(
    mapper, labels, k=7, cv_neighbors=30, cv_splits=5,
    distance_std_threshold=0.75,
)
fig, ax = plot_ellipses(
    mapper, labels, k=7, n_std=2.0, max_ellipses_per_region=2,
)
fig, ax = plot_density_parent_tree(mapper, labels, label_key="cell_type")
viewer = interactive_force(
    mapper, labels, initial_k=10, initial_iterations=20,
)
display(viewer)
image = render_datashader_embedding(mapper, labels, width=1200, height=900)
display(image)
```

For a DataFrame, select its label column with `label_key="cell_type"`. Each
renderer returns its Matplotlib figure and axes, except `interactive_force`,
which returns an `ipywidgets` viewer. The force controls adjust same-label
springs, different-label repulsion, iteration count, and movement. Renderers
accept a fitted `IteratedMinimumSpanningTreeEmbedder` or `InductiveIMSTE`
directly; they use `embedding_` or the inductive model's `reference_embedding_`.
Raw coordinate arrays remain supported.

For class-level graph overlays across one or more embeddings, use
`imste.class_graph.plot_class_mixture_mst`. It fits full-covariance Gaussian
mixtures per class in each 2D embedding, connects their component centers with
iterated MSTs, and maps symmetric cross-component posterior probability to
edge opacity. For each edge, the two endpoint Gaussian densities form an
equal-prior pairwise posterior, evaluated on the samples assigned to the
opposite component in each direction. The directional probabilities are
averaged to score the edge; chance-level overlap (0.5) is fully opaque. It
returns the figure, plot axes, and a table of per-edge scores.

To compare the class-pair relations selected by two embedding graphs, use
`imste.class_graph.compare_class_graph_relations`. It collapses component
edges to unordered class pairs, removes duplicate edges across MST iterations,
and ranks relations present in only one graph by mean chance-normalized edge
error. The per-edge maximum and component pairs remain available for review.

## Parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `n_msts` | `15` | Number of spanning-tree iterations; bottleneck edge reuse is penalized by incrementing its rank |
| `n_components` | `2` | Embedding dimensions |
| `pca_init` | `True` | Initialize `fit(X)` with PCA on feature-wise standardized inputs; extra PCA dimensions are zero-filled |
| `pca_init_scaling` | `"global"` | Apply one uniform scale to fit all PCA coordinates into `[-1, 1]`, or use `"none"` to preserve raw magnitudes |
| `multilevel_init` | `False` | Opt into rank-only graph coarsening and coarse-to-fine initialization |
| `multilevel_reduction_ratio` | `4` | Approximate number of nodes represented by each node per coarsening level |
| `multilevel_base_size` | `1024` | Stop coarsening at or below this graph size |
| `multilevel_epochs` | `10` | Refinement epochs at each coarsened graph level |
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
| `metric` | `"euclidean"` | Feature distance for the graph: `"euclidean"` or `"cosine"` |
| `mst_method` | `"famst"` | Sparse approximate `"famst"` or exact dense `"prim"` |
| `mst_neighbor_backend` | `"auto"` | `"auto"`, `"rinnd"`, `"pynndescent"`, or `"mps_exact"` FAMST neighbor search |
| `mst_neighbors` | `15` | Initial candidate neighbors for FAMST |
| `include_knn_attraction` | `False` | Also attract along FAMST candidate-neighbor edges in Euclidean `fit(X)` |
| `gravity_mode` | `"mexican_hat"` | Optional density-aware second phase; set to `None` to disable |
| `gravity_neighbors` | `15` | Feature-space neighbors for the RINND/Leiden region graph |
| `gravity_resolution` | `0.8` | Leiden partition resolution |
| `gravity_strength` | `0.2` | Attraction toward each density-weighted region core |
| `gravity_core_quantile` | `0.95` | Within-region density quantile defining the core points |
| `gravity_mass_power` | `0.5` | Exponent controlling point and region mass effects |
| `gravity_attraction_strength` | `0.5` | Attraction strength for local Mexican-hat forces |
| `gravity_near_repulsion_strength` | `2.5` | Repulsion strength for close embedded neighbors |
| `gravity_force_neighbors` | `8` | Embedded neighbors used by the local force |
| `gravity_attraction_range` | `1.0` | Attraction radius relative to local embedded neighbor distance |
| `gravity_repulsion_range` | `0.5` | Repulsion radius relative to local embedded neighbor distance |
| `gravity_step_size` | `0.05` | Maximum force step as a fraction of local scale |
| `gravity_iterations` | `3` | Local-force refinement iterations |
| `gravity_pack_regions` | `False` | After gravity, translate Leiden regions onto a compact ring while preserving each region's internal shape |
| `gravity_pack_radius_quantile` | `0.9` | Quantile used to estimate region radii for ring packing |
| `gravity_pack_gap_fraction` | `0.25` | Gap between packed regions as a fraction of their median radius |
| `mst_inter_component_edges` | `5` | Candidate connections per FAMST component pair |
| `mst_representatives_per_component` | `10` | Maximum representatives per component for selecting FAMST bridge pairs |
| `mst_max_neighbors` | `None` | FAMST neighbor cap; defaults to four times `mst_neighbors` |

Fitted estimators expose `graph_construction_time_`,
`multilevel_initialization_time_`, `final_embedding_optimization_time_`,
`embedding_optimization_time_`, and `fit_time_` in seconds. With `verbose=True`,
the fit summary reports each macro phase separately. MPS negative sampling and
coordinate updates are timed together to avoid per-batch Metal event recording
on long fits.

When gravity follows a feature-based FAMST fit, it reuses the fitted Euclidean
neighbor search when that graph contains enough valid neighbors. The
`gravity_neighbor_graph_reused_` and `gravity_neighbor_search_time_` attributes
report reuse and any fresh-search time. Prim fits and graph-only fits use a
fresh search when gravity has features available.

By default, `fit(X)` starts coordinate optimization from PCA scores computed
after feature-wise standardization and uniformly scaled by the maximum absolute
score into `[-1, 1]`. This preserves the relative scale between PCA axes. Set
`pca_init_scaling="none"` to preserve raw PCA score magnitudes, or
`pca_init=False` to use the seeded random initialization instead.
`fit_graph()` uses the same PCA initialization when `init_features` is supplied;
otherwise it uses random initialization. Pass `init_coordinates` with shape
`(n_samples, n_components)` to start from explicit coordinates instead:

```python
mapper = IteratedMinimumSpanningTreeEmbedder(
    n_msts=5,
    init_coordinates=umap_coordinates,
)
embedding = mapper.fit_transform(X)
```
The coordinate rows must correspond to `X` in the same order. Explicit
coordinates take precedence over `pca_init` and are used at their supplied scale.

Gravity is an optional second phase for dense Euclidean inputs with at least
two embedding dimensions. Its Leiden regions can be translated onto a compact
ring with `gravity_pack_regions=True`. This rigid translation preserves
within-region distances, but changes distances between regions; use it when a
compact region view is more useful than preserving the fitted global layout.

Set `multilevel_init=True` to initialize from the fitted ranked graph instead.
The initializer scores graph edges by inverse rank, coarsens each connected
component independently, then lays out the smallest graph from PCA of pooled
group features when `pca_init=True` (the default). With `pca_init=False`, it
uses seeded random coordinates instead. The layout then expands and refines
the graph level by level. Coarse-level
quotient edges use summed inverse-rank weights and run on the estimator's
selected CPU or MPS optimizer. The final fit still uses the original graph and
its normal rank scheduling. Explicit `init_coordinates` skip this process; if
the graph cannot be reduced, the configured PCA or random initialization is
used.

The selected optimizer device is available as `mapper.device_` after fitting.
`device="auto"` keeps optimization on CPU. With `device="mps"`, the batched
optimizer runs on the Apple GPU. The independent
`mst_neighbor_backend="auto"` setting prefers RINND for dense Euclidean FAMST
inputs, while `"mps_exact"` selects exact chunked neighbor search on MPS. MPS
requires
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

The notebooks are ordered as a short progression from standard high-dimensional
examples to controlled and biological IMSTE–UMAP comparisons:

1. [00 — High-dimensional dataset gallery](notebooks/00_high_dim_mst_gallery.ipynb):
   run IMSTE on standard image datasets and inspect embedding quality and fit
   times.
2. [01 — Synthetic IMSTE and UMAP comparison](notebooks/01_imste_umap_synthetic_comparison.ipynb):
   compare global layout and local-neighbor behavior on controlled geometries.
3. [02 — Biological embeddings and trajectories](notebooks/02_biological_imste_umap_comparison.ipynb):
   evaluate annotated Paul15 and PBMC3k single-cell datasets, then compare
   GSE52529 myoblast embeddings by observed differentiation time and calculate
   diffusion pseudotime from the GEO FPKM matrix.
4. [Prepare the arXiv dataset](data/arxiv/prepare_arxiv_dataset.ipynb):
   download or reuse the Kaggle metadata, then create one reusable paper-vector
   cache. Increase its size by replacing the existing cache; smaller selections
   reuse a prefix without writing another dataset copy.
5. [03 — arXiv large-dataset comparison](notebooks/03_arxiv_large_dataset_comparison.ipynb):
   consume the prepared vectors, reuse or fit UMAP and IMSTE coordinates, and
   save a projection cache for interactive exploration.
6. [04 — Interactive embedding rendering](notebooks/04_interactive_embedding_rendering.ipynb):
   interactively render cached MNIST-family and arXiv embeddings with Datashader.

Earlier exploratory notebooks are in
[notebooks/archive](notebooks/archive/). This includes the [interactive 2D
embedding](notebooks/archive/digits_2d_interactive.ipynb), [interactive 3D
embedding](notebooks/archive/digits_3d_interactive.ipynb), [parameter
sweep](notebooks/archive/digits_parameter_sweep.ipynb), and [MST scaling
benchmark](notebooks/archive/mst_scaling_benchmark.ipynb).

Python helper modules used by the notebooks are in [notebooks/src](notebooks/src/).
For the full method description and equations, see the
[IMSTE whitepaper](WHITEPAPER.md).
