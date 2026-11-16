"""Public plotting helpers for IMSTE embeddings."""

from ._visualization_core import (
    _as_observations,
    _embedding_array,
    _plot_embedding_frame,
    cross_validate_embedding_knn,
    plot_density_parent_tree,
    plot_embedding,
    render_datashader_embedding,
)
from ._visualization_interactive import (
    _add_compound_path_union_outline,
    interactive_force,
)
from ._visualization_class_geometry import (
    _fit_component_ellipses,
    _render_class_neighborhood_geometry,
    plot_ellipses,
    plot_knn_union,
)

__all__ = [
    "cross_validate_embedding_knn",
    "interactive_force",
    "plot_density_parent_tree",
    "plot_ellipses",
    "plot_embedding",
    "plot_knn_union",
    "render_datashader_embedding",
]
