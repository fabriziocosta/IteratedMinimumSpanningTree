# IMSTE: Iterated Minimum Spanning Tree Embedding

## Abstract

Iterated Minimum Spanning Tree Embedding (IMSTE) is an unsupervised method for
placing a finite set of high-dimensional observations in a lower-dimensional
coordinate space. It builds a weighted graph from ranked spanning-tree
iterations. Earlier trees use shorter connections where possible; later trees
prefer unused edges and reuse bottleneck edges when needed to connect the
residual graph. A rank-based weight controls the influence of each edge. The
graph is then embedded by optimizing an attractive loss on graph edges
together with a sampled repulsive loss on non-edges.

The method is deliberately direct. It uses Euclidean distances, ranked
spanning-tree construction, and a small differentiable objective. The default
FAMST backend builds an approximate sparse candidate graph; exact Prim is also
available. The core estimator does not use class labels or define an
out-of-sample projection. An optional `InductiveIMSTE` wrapper learns an
approximate MLP projection. This paper describes the implementation in this
repository and its practical limits; it makes no claim that the method is a
drop-in replacement for established embedding methods.

## 1. Motivation and scope

An embedding aims to represent relationships among observations with fewer
coordinates than the original feature space. IMSTE represents those
relationships through a sequence of ranked spanning trees. The default FAMST
backend starts from an approximate nearest-neighbor graph, while exact Prim
considers the complete Euclidean graph. Later iterations prefer unused edges;
when a bottleneck requires reuse, the edge's rank increases and its attraction
weight decreases.

The core estimator is intended for exploratory visualization and geometric
inspection of a fixed dataset. It returns learned coordinates for training
rows, and its `transform` method accepts only those same rows in their original
order. `InductiveIMSTE` can approximate coordinates for new rows with a
separately trained MLP.

## 2. Notation

Let the input be a matrix

$$
X = [x_1,\ldots,x_n]^\top \in \mathbb{R}^{n\times p},
$$

In the matrix dimensions, n is the number of samples and p is the number of
features.

Each sample receives a coordinate in the embedding space, where the output
dimension is q:

$$
y_i \in \mathbb{R}^q.
$$

The number of trees is R.

## 3. Graph construction

The graph uses Euclidean distances between input rows. The default
`mst_method="famst"` builds an approximate sparse nearest-neighbor candidate
graph and adds component-connecting bridge candidates. The alternative
`mst_method="prim"` constructs exact trees over the complete Euclidean graph.
Dense Prim input stores all pairwise distances; CSR Prim input computes
distance rows on demand. An existing sparse distance graph can also be supplied
through `fit_graph()`.

$$
D_{ij}=\sqrt{\sum_{k=1}^{p}(x_{ik}-x_{jk})^2}.
$$

For each requested iteration, the method finds a spanning tree, preferring
available unused edges. If removing earlier trees leaves a bottleneck, the
builder reuses only the edges needed to connect residual components. A newly
selected edge receives the iteration number as its rank; each reuse increments
that edge's previous rank. Its inverse-rank weight is:

$$
w_{ij}=\frac{1}{r}.
$$

The edge weight depends on its tree rank, not on a separately applied distance
decay.

For connected feature-based input, when all requested iterations succeed, the
graph stores $R(n-1)$ edge occurrences. Reused bottleneck edges mean the number
of distinct undirected edges can be smaller. FAMST increases its neighbor count
and repairs local bottlenecks when possible; it raises an error if it cannot
construct the requested iterations within its configured neighbor cap. Exact
Prim can fall back to the shortest complete-graph edge crossing a disconnected
residual cut.
Its explicit edge-availability mask allows distinct samples at zero distance
to remain valid edges.

$$
|E|_{\mathrm{occurrences}}=R(n-1).
$$

## 4. Coordinate objective

Each sample receives a trainable coordinate stored as a contiguous float32
array. By default, `fit(X)` initializes it from PCA scores computed on
feature-wise standardized input, then applies one global scale so all scores
fit in `[-1, 1]`. Set `pca_init=False` to use small seeded Gaussian noise.
`fit_graph()` uses random initialization unless `init_features` are supplied.
A Numba-compiled SGD kernel updates coordinates directly on CPU; the optional
MPS backend uses PyTorch.

### 4.1 Attraction on graph edges

For each edge in the graph, the squared Euclidean distance between its
embedded coordinates is:

$$
d_{ij}^2=\sum_{k=1}^{q}(y_{ik}-y_{jk})^2.
$$

The estimator uses squared Euclidean distance, so $z_{ij}=d_{ij}^2$. The
logarithmic objective uses the pairwise edge probability:

$$
q_{ij}=\frac{1}{1+z_{ij}+\epsilon}.
$$

The attractive term is

$$
L_{\mathrm{attr}}=
\frac{1}{\sum_{(i,j)\in E}w_{ij}}\sum_{(i,j)\in E}
w_{ij}\bigl[-\log(q_{ij})\bigr]
=\frac{1}{\sum_{(i,j)\in E}w_{ij}}\sum_{(i,j)\in E}
w_{ij}\log(1+z_{ij}+\epsilon).
$$

Minimizing it brings graph-connected samples together. The MST-rank weight
applies to positive edges only.

For approximate FAMST graphs, the normalized rank-weighted objective can be
estimated by activating a rank $r$ edge once every $r$ epochs and using
rank-independent updates while it is active. Deterministic phases spread the
higher-rank updates across epochs. Exact Prim instead processes every edge in
every epoch and multiplies each attraction update by its explicit inverse-rank
weight $1/r_{ij}$. Both modes divide positive updates by
$\sum_{(i,j)\in E}1/r_{ij}$; negative updates are normalized by the eligible
sampled-pair count for that epoch.

This logarithmic objective is the only built-in attraction. It always acts on
squared Euclidean distances and uses the same edge weights and weight-sum
normalization described above.

### 4.2 Repulsion on sampled non-edges

For each active undirected positive edge, the implementation samples negative
targets for both endpoints. This makes sampling independent of the arbitrary
orientation used to store an edge. Targets are sampled uniformly from nodes
that are neither the source nor one of its graph neighbors. The
squared embedding distance for a sampled pair is:

$$
d_{uv}^2=\sum_{k=1}^{q}(y_{uk}-y_{vk})^2.
$$

For each sampled negative pair, the logistic margin loss is:

Here $z_{uv}=d_{uv}^2$ is the squared Euclidean embedding distance.

$$
L_{uv}^{-}=\operatorname{softplus}\left(\frac{m-z_{uv}}{\tau}\right),
$$

with `logistic_margin` $m$ and `logistic_temperature` $\tau$.
The negative loss is averaged over sampled pairs without MST-rank weights.
Nearby negative pairs incur a larger cost and are pushed apart. If a graph
has no eligible negative pairs, the repulsive term is zero for that step.

The combined normalized objective is

$$
L=(1-\lambda)L_{\mathrm{attr}}+\lambda L_{\mathrm{rep}},
$$

The parameter `lambda_rep` is the repulsion share $\lambda\in[0,1]$; the
attraction share is $1-\lambda$. The default $\lambda=0.5$ gives equal
weight to the two terms. The implementation normalizes the weighted
positive-edge losses by their graph-wide total weight; the negative term is
unchanged.

The estimator uses the logarithmic attraction and logistic repulsion described
above. It optimizes sample coordinates directly. `fit_graph()` accepts a
square sparse distance graph, treats stored off-diagonal entries as edges, and
preserves disconnected components as spanning forests. With `init_features`,
it can use standardized PCA scores to initialize coordinates; the supplied
graph remains the optimization objective.

## 5. Optimization procedure

For each epoch, the implementation selects edges whose rank schedule is due,
shuffles those edges, and processes each positive pair and its negative
samples sequentially. Linear learning-rate decay is applied across epochs.
The normalized mean gradient is rescaled by the sample count, with a
calibration factor chosen to match the former optimizer's default quality.
After each epoch it subtracts the coordinate mean, removing global translation
drift without changing pairwise differences. `device="auto"` and
`device="cpu"` select the sequential CPU Numba optimizer. `device="mps"`
selects a PyTorch MPS backend that computes updates for each active-edge batch
together and applies batches sequentially. This path has a different trajectory
from sequential CPU SGD. FAMST neighbor search is selected independently by
`mst_neighbor_backend`, so RINND can build the candidate graph while the
optimizer uses MPS. Component connection and tree extraction remain on CPU.
MPS samples negative pairs and updates coordinates on
the GPU, with a rare exact CPU fallback for unusually dense neighborhoods. MPS
batch size controls memory use and update grouping; CPU runs ignore it.

The estimator uses Euclidean distance for graph construction and squared
Euclidean distance in the embedding objective. Rank $r$ determines update
frequency; `graph_weights_` remains available as the corresponding $1/r$
diagnostic.

| Parameter | Default | Role |
| --- | ---: | --- |
| `n_msts` | 15 | Number of spanning-tree iterations; bottleneck edges can be reused with increased rank |
| `mst_method` | `famst` | Approximate sparse FAMST trees or exact Prim trees |
| `mst_neighbor_backend` | `auto` | FAMST neighbor search: prefer RINND when compatible, or select PyNNDescent / exact MPS search |
| `mst_neighbors` | 15 | Initial FAMST neighbor count; the builder targets at least `2 * n_msts` |
| `mst_inter_component_edges` | 5 | Bridge candidates retained per selected component pair |
| `mst_representatives_per_component` | 10 | Maximum component representatives used to select bridge pairs |
| `mst_max_neighbors` | `None` | FAMST neighbor cap; `None` permits growth up to four times the starting count |
| `logistic_margin` | 1.0 | Squared-distance margin for logistic repulsion |
| `logistic_temperature` | 0.5 | Softness of the logistic repulsion; must be positive |
| `n_components` | 2 | Number of output dimensions |
| `n_epochs` | 200 | Number of rank-scheduled optimization epochs |
| `batch_size` | 32768 | Maximum active edges per MPS batch; ignored by CPU optimizer |
| `learning_rate` | 0.05 | Base SGD learning rate, sample-scaled and linearly decayed |
| `negative_ratio` | 5 | Negative samples per endpoint of each positive edge |
| `lambda_rep` | 0.5 | Repulsion share; attraction uses `1 - lambda_rep` |
| `epsilon` | `1e-4` | Smoothing for the pairwise edge probability |
| `random_state` | 42 | Seed for initialization, shuffling, and sampling |
| `device` | `auto` | `auto`/`cpu` use Numba CPU; `mps` uses batched PyTorch MPS updates |
| `pca_init` | `True` | Use standardized PCA scores to initialize feature-based fits |
| `pca_init_scaling` | `global` | Scale PCA scores uniformly into `[-1, 1]`; `none` preserves their magnitudes |

With `mst_neighbor_backend="auto"`, dense Euclidean FAMST inputs use RINND when
available; otherwise graph construction retains its device-based backend
selection. RINND is optional (`pip install 'imste[rinnd]'`). The transductive
optimizer uses Numba on CPU or PyTorch on MPS, depending on `device`,
independently of the neighbor-search backend.

## 6. Practical use

Euclidean distance makes feature scale part of the model. Features with larger
numeric ranges can dominate graph construction, so inputs should be scaled
appropriately for the dataset. The gallery standardizes a stratified sample
before fitting. Labels in that gallery are used only to color plots; they are
not passed into the estimator's optimization.

```python
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

from imste import IteratedMinimumSpanningTreeEmbedder

digits = load_digits()
X = StandardScaler().fit_transform(digits.data)

embedding = IteratedMinimumSpanningTreeEmbedder(
    n_msts=10,
    n_components=2,
    random_state=42,
).fit_transform(X)
```

The returned array has shape `(n_samples, n_components)`. Coordinates are
centered after every epoch, but their orientation, reflection, and scale are
not otherwise fixed. Interpret the embedding through relative
relationships and visual patterns, rather than as a coordinate system with
absolute units.

## 7. Computational cost

Dense input to the exact `mst_method="prim"` backend constructs a pairwise
distance matrix requiring quadratic memory in the number of samples. CSR input
computes distance rows on demand and avoids storing that matrix:

$$
O(n^2)
$$

Computing all distances scales with the number of samples squared times the
number of features:

$$
O(n^2p)
$$

The dense repeated MST construction takes:

$$
O(Rn^2)
$$

When all requested iterations succeed, the graph stores this many edge
occurrences, including repeated bottleneck edges:

$$
|E|_{\mathrm{occurrences}}=R(n-1)
$$

The default `mst_method="famst"` backend builds a sparse nearest-neighbor
graph. `mst_neighbor_backend="auto"` prefers RINND for dense Euclidean input;
PyNNDescent remains available for CPU and sparse builds, and exact chunked MPS
search is selectable. It samples up to `mst_representatives_per_component`
points per ANN component, builds an exact MST over those representatives, and
uses its cross-component edges to select unique component pairs for bridge
search and refinement. It extracts ranked trees, reusing bottleneck edges when
needed. For a fixed neighbor count and a small number of ANN components, its
graph storage grows approximately linearly with the number of rows. Its result
is approximate and does not guarantee the same trees as the complete Euclidean
graph. PyNNDescent remains available as a fallback and explicit backend. Each
backend can increase the neighbor count, within `mst_max_neighbors`, when the candidate graph lacks
enough unused edges. This follows the FAMST approach by Almansoori and
Telek (2025).

Optimization cost grows with the number of epochs, graph edges, negative ratio,
and output dimension. Let m denote the negative ratio and q the output
dimension. A rough per-epoch bound is:

$$
O(|E|(1+m)q).
$$

Large datasets can therefore require substantial memory and runtime even
when the final embedding has only two dimensions. Sampling a subset before
fitting is often practical for visualization.

## 8. Reproducibility and estimator behavior

`random_state` controls the initial coordinates, edge order within epochs, and
negative sampling. It makes CPU runs repeatable under the same software and
hardware configuration. Different compute backends or software versions may
produce small numerical differences.

The transductive `IteratedMinimumSpanningTreeEmbedder` follows the scikit-learn
`fit`, `fit_transform`, and `transform` interface. Since all coordinates are
optimized jointly, `transform` returns stored coordinates only for the exact
training matrix in its original row order. Use `imste.inductive.InductiveIMSTE`
when new rows need projections. This separate estimator fits an MLP to predict
the reference coordinates; its default has six hidden layers of 128 units and
0.1 dropout. It uses a 10% validation split, at least 100 epochs, and patience
of 20. The MLP is an extension around the core IMSTE objective, not part of the
embedding algorithm described here.

## 9. Limitations and interpretation

- Poor feature scaling can cause some dimensions to dominate Euclidean distances
  and produce a poor graph.
- Dense exact Prim uses quadratic pairwise-distance memory; CSR exact Prim
  avoids storing that matrix but still has quadratic time in sample count.
- The FAMST backend is approximate; quality depends on ANN recall and the
  candidate graph may fail to support the requested number of ranked spanning
  tree iterations under its configured neighbor cap.
- The embedding is stochastic and non-convex. A fixed seed supports
  reproducibility but does not establish that a result is unique or globally
  optimal.
- The repulsive term samples graph non-neighbors rather than modeling all
  non-edge pairs exactly; the sampled negative losses are unweighted.
- Output distances and axis values are not calibrated quantities. Rotation,
  reflection, and overall scale do not carry intrinsic meaning.
- The core IMSTE objective does not itself define an out-of-sample projection.
  `InductiveIMSTE` approximates it with an MLP trained on fitted coordinates.
  Labels are not used during either fit.

## 10. Summary

IMSTE constructs a rank-weighted union of spanning-tree iterations, reusing
bottleneck edges when necessary, and learns low-dimensional coordinates by
balancing attraction along graph edges against sampled repulsion between graph
non-neighbors. The approach is compact and easy to inspect. Use exact Prim for
smaller datasets where exact trees matter, or FAMST when sparse approximate
trees are an acceptable tradeoff for larger datasets.
