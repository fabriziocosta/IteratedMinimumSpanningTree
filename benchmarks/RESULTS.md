# Performance refactor measurements

These measurements are a local CPU reference, not a hardware-independent
performance claim. Inputs are 32-feature, 20-center `make_blobs` datasets with
`cluster_std=3`, seed 42. Fits used 15 MSTs and 50 epochs. The 2,000-row
results use seeds 0 and 1; the 10,000-row results use seed 0. Numba and
PyNNDescent were warmed on a separate 256-row input. UMAP used 15
neighbors, 50 epochs, and matching seeds. Trustworthiness was measured on at
most 2,000 rows; kNN preservation is mean overlap of 10-neighbor sets.

| Samples | Version / method | Graph (s) | Optimizer (s) | Fit (s) | Trustworthiness | kNN preservation |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2,000 | Baseline Torch / FAMST | 2.477 | 0.825 | 3.302 | 0.9816 | 0.1323 |
| 2,000 | Numba SGD / FAMST | 0.992 | 0.157 | 1.150 | 0.9808 | 0.1336 |
| 2,000 | UMAP reference | — | — | 3.989 | 0.9822 | 0.1495 |
| 10,000 | Numba SGD / FAMST | 19.295 | 0.971 | 20.266 | 0.9807 | 0.0321 |
| 10,000 | UMAP reference | — | — | 3.811 | 0.9805 | 0.0342 |

The 2,000-row comparison used two warmed runs for baseline and refactor. The
10,000-row FAMST run needed a 120-neighbor starting graph and cap of 512 to
construct all requested trees; its figures are from one run. A 50,000-
row run was not practical in this pass because graph construction already
dominated the 10,000-row fit. The 10,000-row result shows that optimizer work is
small after the refactor, while adaptive ANN construction remains the main
performance target.

## M5 Pro FAMST retry experiment

On the workspace's Apple M5 Pro (arm64), a 10,000-row, 32-feature, 15-tree,
50-epoch fit with seed 0 was measured before and after changing FAMST's retry
policy. The old policy rebuilt the ANN graph at each doubled neighbor count;
the new policy jumps to the configured cap when it is within eight times the
current count, while retaining gradual growth when the cap is much larger.

| Retry policy | Graph (s) | Optimizer (s) | Fit (s) | Trustworthiness | kNN preservation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Double each retry | 20.217 | 0.676 | 20.894 | 0.9807 | 0.0321 |
| Skip intermediate retry | 14.474 | 0.693 | 15.167 | 0.9807 | 0.0321 |

These are single-run measurements. The 2,000-row workload with the new policy
measured 0.578 s graph construction, 0.102 s optimization, and 0.681 s total
fit time across two runs; its trustworthiness was 0.9808 and kNN preservation
was 0.1336. Small graphs retain geometric neighbor growth to avoid jumping to
an unnecessarily large candidate graph.

## M5 Pro 60,000-row MPS utilization experiment

MNIST `mnist_784` was loaded from the local OpenML cache, stratified to 60,000
rows, and standardized. The run used 15 MSTs, 500 epochs, FAMST with 120
neighbors (cap 512), seed 42, `device="mps"`, batch size 131,072, and negative
ratio 30. FAMST's exact chunked MPS neighbor search was enabled for this run.
The Mac's `ioreg` `Device Utilization %` value was sampled every 0.5 seconds and
split between graph construction and optimization using the epoch callback.

| Stage | Time (s) | `ioreg` samples | Mean GPU utilization | Peak |
| --- | ---: | ---: | ---: | ---: |
| FAMST graph build | 10.27 | 15 | 42.1% | 100% |
| MPS optimization | 42.96 | 78 | 90.4% | 98% |
| Whole fit sample mean | 53.31 | 93 | 82.6% | 100% |

The embedding was finite and the graph contained 899,985 edges. The ratio-30
setting is an explicit workload choice, not a default: it samples more negative
pairs and takes substantially longer than ratio 5. GPU utilization depends on
the selected negative ratio, batch size, data, and other system activity.

Reproduce with:

```sh
python benchmarks/performance_refactor.py --sizes 2000 10000 50000 --repeats 3 --epochs 50
```

The script prints graph-construction, optimizer, and total-fit timings along
with the two neighborhood metrics. It warms compilation outside measured
runs and includes UMAP when the notebook extras are installed.
