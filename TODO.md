# TODO

## Evaluate PCA initialization and scale calibration

- [ ] Add and evaluate PCA-based initialization for the embedding coordinates.
  - Keep the current small random initialization as a selectable baseline.
  - For dense inputs, use the leading principal component scores and centre them.
  - Preserve the PCA geometry with one global scale factor. Do not standardize each output axis separately unless whitening is an intentional experiment, since that changes the relative variance of the principal directions.
  - Define explicit behavior for CSR inputs. Avoid silently densifying a sparse matrix; either provide a sparse-compatible PCA path or report a clear unsupported-input error.

- [ ] Calibrate the PCA scale against the embedding loss.
  - The input-space Euclidean distances determine the ranked MST graph. The layout loss acts on squared distances between embedding coordinates.
  - The default repulsion uses margin `1.0` and temperature `0.5`; therefore, the PCA coordinates' raw scale affects how much repulsion is active at the start.
  - Use a deterministic sample of eligible non-edge pairs to estimate a typical initial embedding distance. Apply one scalar so this distance is near the repulsion margin, then test a small range of scale factors.
  - Record the initial positive-edge and sampled-negative distance distributions, so the chosen scale is interpretable.

- [ ] Compare initializations under controlled conditions.
  - Run random and PCA starts on the same datasets with the same graph, optimizer settings, and matched random seeds where applicable.
  - Repeat across several seeds and dataset types. Include the existing synthetic benchmark and at least one representative real dataset.
  - Record final objective components, trustworthiness, k-nearest-neighbour preservation, a global-structure or graph-connectivity measure, convergence behaviour, and runtime.
  - Include both initialization methods for UMAP as a control where practical, since UMAP supports PCA and random initializations.
  - Do not claim PCA initialization improves IMSTE unless repeated comparisons support that conclusion.

- [ ] Add tests and documentation for the initialization option.
  - Check centring, deterministic output, coordinate shape, global scaling, and failure messages for unsupported sparse inputs.
  - Document how initialization scale interacts with `logistic_margin` and `logistic_temperature`.
  - Keep the current random initialization as the default until the controlled comparison shows a reliable benefit.

### PCA versus classical MDS

For Euclidean input distances, classical MDS produces the same principal-coordinate geometry as PCA, up to rotation or reflection. Start by testing PCA rather than implementing a separate dense classical-MDS initialization. Revisit classical MDS if the project later supports non-Euclidean or precomputed dissimilarities.

## Scale IMSTE to large datasets

- [ ] Add a large-dataset benchmark that measures the stages separately.
  - Include representative dense and CSR inputs, with sample sizes that grow beyond the current small and medium examples.
  - Record neighbor-graph construction time, MST extraction and refinement time, coordinate-optimization time, peak memory, and total runtime.
  - Record graph and embedding quality alongside runtime, including neighbor preservation and a global-structure measure.
  - Compare FAMST with exact Prim only at sizes where the exact run is feasible; report when the exact baseline is omitted.
  - Reuse a constructed graph when comparing optimizer or initialization settings so each comparison isolates the changed stage.
  - Document the hardware, dataset dimensions, sparsity, and benchmark procedure.

- [ ] Investigate coarse-to-fine initialization for large graphs.
  - Prototype recursive graph coarsening, layout of the coarsened graph, then expansion and refinement across levels.
  - Preserve IMSTE's ranked multi-MST edge information during coarsening and expansion.
  - Compare with random and PCA initialization using the same graph, optimizer settings, seeds, runtime, and quality measures.
  - Keep the approach experimental unless results show a repeatable benefit.

## Reuse the feature-space neighbor graph for gravity

- [ ] Avoid rebuilding the RINND graph for gravity when a compatible feature-space neighbor graph is already available from IMSTE/FAMST.
  - Reuse its neighbor indices and distances to compute local radii, density weights, and the weighted Leiden graph.
  - Check that the graph uses Euclidean distances and includes enough neighbors for the gravity settings; retain a fresh-search fallback when it is missing or incompatible.
  - Compare graph-construction time and verify that reuse preserves the intended regions, density weights, and gravity output.

## Add a Datashader embedding visualization

- [ ] Add an optional Datashader renderer for large two-dimensional embeddings.
  - Aggregate points into screen pixels so dense regions remain interpretable without drawing every point as a separate marker.
  - Support categorical labels with a stable color mapping and a density shading mode.
  - Redraw the aggregation for the visible coordinate range when users zoom or pan, so the display reflects the points in that view.
  - Keep the existing Matplotlib renderers for ordinary-sized embeddings and make Datashader an optional visualization dependency.
  - Define behavior for missing labels, points outside the current view, and empty selections.
  - Add a gallery example comparing ordinary scatter rendering with pixel aggregation at increasing sample sizes, including zoomed views.
  - Document the renderer's limits and when to use it; retain access to individual point metadata for selection or inspection where the plotting frontend permits it.

