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
