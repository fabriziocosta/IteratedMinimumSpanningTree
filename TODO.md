# TODO



## Reuse the feature-space neighbor graph for gravity

- [x] Avoid rebuilding the feature-neighbor graph for gravity when a compatible graph is available from a feature-based FAMST fit.
  - [x] Reuse neighbor indices and distances to compute local radii, density weights, and the weighted Leiden graph.
  - [x] Check that the graph is Euclidean and wide enough; retain a fresh-search fallback when it is missing or incompatible.
  - [ ] Compare graph-construction time and verify that reuse preserves the intended regions, density weights, and gravity output.

## Try a mean-shift gravity variant

- [ ] Implement mean shift as a separate experimental gravity mode; keep the current gravity method as the baseline because it already works well.
  - Use the original-space approximate-neighbor graph and distances to define kernel-weighted neighborhoods, reusing the existing graph where compatible.
  - Move each point in 2D toward the weighted mean of its feature-space neighbors' current 2D coordinates.
  - Retain short-range repulsion based on 2D proximity to prevent projected points from collapsing or overlapping.
  - Reduce exposed attraction parameters where adaptive bandwidths and normalized mean-shift updates can determine them; keep only necessary repulsion controls.
  - Compare both methods on the same datasets and settings, measuring runtime, cluster separation, local-neighbor preservation, and stability.
  - Decide whether to switch the default only after the comparison shows a consistent improvement.

## Add a Datashader embedding visualization

- [x] Add an optional Datashader renderer for large two-dimensional embeddings.
  - [x] Aggregate points into screen pixels so dense regions remain interpretable without drawing every point as a separate marker.
  - [x] Support categorical labels with stable color mappings and a density shading mode.
  - [x] Accept explicit coordinate ranges so callers can render a zoomed view.
  - [x] Keep existing Matplotlib renderers and make Datashader an optional visualization dependency.
  - [x] Define behavior for missing labels, points outside the current view, and empty embeddings.
  - [x] Draw a thin gray image border by default, with a `draw_border` option to turn it off.
  - [x] Document installation, usage, coordinate ranges, and spreading controls.
- [ ] Refresh the raster automatically when a user zooms or pans; the current renderer accepts ranges, but the frontend must call it again.
- [ ] Add a gallery comparing ordinary scatter rendering with pixel aggregation at increasing sample sizes, including zoomed views.
- [ ] Document renderer limitations and connect individual point metadata to selection or inspection where the plotting frontend permits it.

