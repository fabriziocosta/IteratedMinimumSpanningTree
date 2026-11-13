# TODO



## Reuse the feature-space neighbor graph for gravity and mean shift

- [x] Capture the high-dimensional feature-space neighbor graph during feature-based FAMST construction and reuse it for gravity and mean shift.
- [x] Preserve the MST candidate graph by using the same captured graph, with a wider neighbor table for refinement controls.
- [x] Reuse neighbor indices and distances to compute gravity densities and the mean-shift kernel weights.
- [x] Apply gravity first, then mean shift over the current 2D coordinates of those same feature neighbors.
- [x] Allow zero gravity pull and zero mean-shift iterations to disable each refinement.
  - [ ] Compare graph-construction time and verify that reuse preserves the intended regions, density weights, and gravity output.

## Mean-shift refinement

- [x] Add mean shift as an optional estimator phase after gravity.
- [x] Expose the main controls and the sequential comparison in notebook 07.
- [ ] Compare base, gravity, and gravity-then-mean-shift layouts on the same datasets and settings.

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

