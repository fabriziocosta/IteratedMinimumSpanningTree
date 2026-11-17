"""Reusable Scanpy helpers for single-cell expression preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HVGSelection:
    """Gene ranking and selection metadata from excess-variance HVG selection."""

    ranked_genes: pd.DataFrame
    selected_gene_names: np.ndarray
    selection_mask: np.ndarray
    n_batches: int
    method: str


def sample_indices_by_group(
    group_values,
    *,
    eligible_mask=None,
    max_per_group: int | None = None,
    random_state: int = 42,
):
    """Choose reproducible row indices, with an optional cap within each group."""
    groups = np.asarray(group_values, dtype=str)
    if eligible_mask is None:
        eligible = np.ones(len(groups), dtype=bool)
    else:
        eligible = np.asarray(eligible_mask, dtype=bool)
        if eligible.shape != groups.shape:
            raise ValueError("eligible_mask must have one value per group label.")
    if max_per_group is not None and (
        not isinstance(max_per_group, (int, np.integer)) or max_per_group < 1
    ):
        raise ValueError("max_per_group must be a positive integer or None.")

    rng = np.random.default_rng(random_state)
    selected = []
    for group in pd.unique(groups[eligible]):
        candidates = np.flatnonzero(eligible & (groups == group))
        count = len(candidates) if max_per_group is None else min(len(candidates), max_per_group)
        selected.append(rng.choice(candidates, count, replace=False))
    if not selected:
        raise ValueError("No eligible cells remain for sampling.")
    return np.sort(np.concatenate(selected))


def filter_count_data(
    adata,
    *,
    min_genes_per_cell: int | None = None,
    min_cells_per_gene: int | None = None,
    count_layer: str | None = None,
):
    """Return a filtered AnnData copy while retaining filtered raw counts.

    Cell filtering runs before gene filtering. Genes with zero detections are
    always removed, even when ``min_cells_per_gene`` is ``None``.
    """
    from scipy import sparse

    for name, value in (
        ("min_genes_per_cell", min_genes_per_cell),
        ("min_cells_per_gene", min_cells_per_gene),
    ):
        if value is not None and (not isinstance(value, (int, np.integer)) or value < 1):
            raise ValueError(f"{name} must be a positive integer or None.")

    result = adata.copy()
    if count_layer is not None:
        if count_layer not in result.layers:
            raise KeyError(f"Count layer {count_layer!r} is missing; available layers: {list(result.layers.keys())}")
        result.X = result.layers[count_layer].copy()

    if sparse.issparse(result.X):
        result.X = result.X.tocsr()
        result.X.eliminate_zeros()
        detected_per_cell = np.asarray(result.X.getnnz(axis=1)).ravel()
        raw_total_per_cell = np.asarray(result.X.sum(axis=1)).ravel()
    else:
        result.X = np.asarray(result.X)
        detected_per_cell = np.count_nonzero(result.X, axis=1)
        raw_total_per_cell = np.asarray(result.X.sum(axis=1)).ravel()

    result.obs["n_genes_detected"] = detected_per_cell.astype(np.int32)
    result.obs["total_counts_raw"] = raw_total_per_cell
    if min_genes_per_cell is not None:
        keep_cells = detected_per_cell >= min_genes_per_cell
        result = result[keep_cells].copy()

    if sparse.issparse(result.X):
        detected_per_gene = np.asarray(result.X.getnnz(axis=0)).ravel()
    else:
        detected_per_gene = np.count_nonzero(result.X, axis=0)
    gene_threshold = max(1, min_cells_per_gene or 1)
    keep_genes = detected_per_gene >= gene_threshold
    if not keep_genes.any():
        raise ValueError("Cell and gene quality filters removed every gene.")
    result = result[:, keep_genes].copy()
    result.var["n_cells_detected"] = detected_per_gene[keep_genes].astype(np.int32)
    if result.n_obs == 0:
        raise ValueError("Cell quality filtering removed every cell.")

    result.layers["raw_counts"] = result.X.copy()
    return result


def normalize_log1p(adata, *, target_sum: float | None = 10_000):
    """Normalize counts per cell when requested, then apply Scanpy log1p in place."""
    import scanpy as sc

    if target_sum is not None:
        if target_sum <= 0:
            raise ValueError("target_sum must be positive or None.")
        sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def preprocess_count_data(
    adata,
    *,
    target_sum: float = 10_000,
    min_genes_per_cell: int | None = None,
    min_cells_per_gene: int | None = None,
    count_layer: str | None = None,
):
    """Filter raw counts, preserve them in a layer, and return normalized log1p data."""
    filtered = filter_count_data(
        adata,
        min_genes_per_cell=min_genes_per_cell,
        min_cells_per_gene=min_cells_per_gene,
        count_layer=count_layer,
    )
    return normalize_log1p(filtered, target_sum=target_sum)


def gene_mean_variance(matrix):
    """Return per-gene population mean and variance for dense or sparse data."""
    from scipy import sparse

    if sparse.issparse(matrix):
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        mean_square = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
    else:
        matrix = np.asarray(matrix)
        mean = matrix.mean(axis=0)
        mean_square = np.square(matrix).mean(axis=0)
    variance = np.maximum(mean_square - np.square(mean), 0.0)
    return mean.astype(np.float64), variance.astype(np.float64)


def fit_mean_variance_trend(mean, variance):
    """Estimate expected variance with a spline in log1p mean/log1p variance space."""
    from scipy.interpolate import UnivariateSpline

    x = np.log1p(np.asarray(mean, dtype=np.float64))
    y = np.log1p(np.asarray(variance, dtype=np.float64))
    unique_x, inverse = np.unique(x, return_inverse=True)
    median_y = np.array([np.median(y[inverse == i]) for i in range(len(unique_x))])
    if len(unique_x) == 1:
        return np.full_like(mean, np.expm1(median_y[0]), dtype=np.float64)
    degree = min(3, len(unique_x) - 1)
    trend = UnivariateSpline(unique_x, median_y, k=degree, s=float(len(unique_x)))
    return np.maximum(np.expm1(trend(x)), 0.0)


def select_hvgs_excess_variance(
    adata,
    *,
    n_top_genes: int = 2_000,
    batch_key: str | None = None,
    batch_aware: bool = False,
) -> HVGSelection:
    """Rank HVGs by excess variance, optionally selecting by batch consensus.

    The returned table is always sorted by global excess variance. In batch
    mode, selection ranks batches by top-N vote count, then mean within-batch
    excess variance, global excess variance, and gene name.
    """
    if not isinstance(n_top_genes, (int, np.integer)) or n_top_genes < 1:
        raise ValueError("n_top_genes must be a positive integer.")
    if batch_aware and batch_key is not None and batch_key not in adata.obs:
        raise KeyError(f"Batch column {batch_key!r} is missing from adata.obs.")

    mean_expression, observed_variance = gene_mean_variance(adata.X)
    expected_variance = fit_mean_variance_trend(mean_expression, observed_variance)
    global_excess = observed_variance - expected_variance
    if not np.isfinite(global_excess).all():
        raise ValueError("Mean–variance trend produced non-finite excess-variance scores.")

    n_genes = adata.n_vars
    gene_names = np.asarray(adata.var_names, dtype=str)
    batch_selected_count = np.zeros(n_genes, dtype=np.int32)
    n_batches = 0
    if batch_aware and batch_key is not None:
        labels = adata.obs[batch_key].astype(str).to_numpy()
        batches = pd.unique(labels)
        batch_score_sum = np.zeros(n_genes, dtype=np.float64)
        for batch in batches:
            indices = np.flatnonzero(labels == batch)
            if len(indices) < 2:
                raise ValueError(f"Batch {batch!r} has fewer than two cells after filtering.")
            batch_mean, batch_variance = gene_mean_variance(adata.X[indices])
            batch_excess = batch_variance - fit_mean_variance_trend(batch_mean, batch_variance)
            batch_score_sum += batch_excess
            batch_order = np.lexsort((gene_names, -batch_excess))
            batch_selected_count[batch_order[:min(n_top_genes, n_genes)]] += 1
        n_batches = len(batches)
        mean_batch_excess = batch_score_sum / n_batches
        selection_order = np.lexsort((
            gene_names,
            -global_excess,
            -mean_batch_excess,
            -batch_selected_count,
        ))
        method = "batch_consensus"
    else:
        mean_batch_excess = np.full(n_genes, np.nan, dtype=np.float64)
        selection_order = np.lexsort((gene_names, -global_excess))
        method = "global_excess_variance"

    selected_count = min(n_top_genes, n_genes)
    selection_mask = np.zeros(n_genes, dtype=bool)
    selection_mask[selection_order[:selected_count]] = True
    selection_rank = np.empty(n_genes, dtype=np.int32)
    selection_rank[selection_order] = np.arange(1, n_genes + 1)

    adata.var["mean_expression"] = mean_expression
    adata.var["observed_variance"] = observed_variance
    adata.var["expected_variance"] = expected_variance
    adata.var["excess_variance"] = global_excess
    adata.var["batch_selected_count"] = batch_selected_count
    adata.var["mean_batch_excess_variance"] = mean_batch_excess
    adata.var["highly_variable"] = selection_mask

    ranked = pd.DataFrame({
        "gene": gene_names,
        "mean_expression": mean_expression,
        "observed_variance": observed_variance,
        "expected_variance": expected_variance,
        "excess_variance": global_excess,
        "batch_selected_count": batch_selected_count,
        "mean_batch_excess_variance": mean_batch_excess,
        "highly_variable": selection_mask,
        "selection_rank": selection_rank,
    }).sort_values(["excess_variance", "gene"], ascending=[False, True], kind="stable")
    ranked = ranked.reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, n_genes + 1))
    selected_names = gene_names[selection_order[:selected_count]]
    return HVGSelection(ranked, selected_names, selection_mask, n_batches, method)


def select_hvgs_scanpy(adata, *, n_top_genes: int, flavor: str = "seurat"):
    """Return a copy containing Scanpy-selected HVGs without subsetting the input."""
    import scanpy as sc

    if n_top_genes < 1:
        raise ValueError("n_top_genes must be positive.")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=min(n_top_genes, adata.n_vars),
        flavor=flavor,
        subset=False,
    )
    return adata[:, adata.var["highly_variable"].to_numpy()].copy()


def scale_and_pca(
    adata,
    *,
    n_pcs: int,
    random_state: int = 42,
    zero_center: bool = True,
    max_value: float | None = None,
):
    """Scale an HVG-only AnnData object, run PCA, and return float32 coordinates."""
    import scanpy as sc

    if n_pcs < 1:
        raise ValueError("n_pcs must be positive.")
    n_components = min(n_pcs, adata.n_obs - 1, adata.n_vars - 1)
    if n_components < 1:
        raise ValueError("PCA requires at least two cells and two genes.")
    sc.pp.scale(adata, zero_center=zero_center, max_value=max_value)
    sc.tl.pca(
        adata,
        n_comps=n_components,
        zero_center=zero_center,
        random_state=random_state,
    )
    coordinates = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    if not np.isfinite(coordinates).all():
        raise ValueError("PCA produced non-finite coordinates.")
    return coordinates
