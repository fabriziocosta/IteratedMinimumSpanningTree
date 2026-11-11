"""Data loading, fitting, and relation analysis for biological IMSTE notebooks."""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd


def align_by_axis_flips(reference, candidate):
    """Apply the best signed axis permutation from the eight 2D symmetries."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError("reference and candidate must be aligned (n_samples, 2) arrays")
    if len(reference) == 0 or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("embeddings must be non-empty and finite")

    ref = reference - reference.mean(axis=0, keepdims=True)
    cand = candidate - candidate.mean(axis=0, keepdims=True)
    ref_scale = np.sqrt(np.mean(np.sum(ref**2, axis=1)))
    cand_scale = np.sqrt(np.mean(np.sum(cand**2, axis=1)))
    if ref_scale == 0 or cand_scale == 0:
        raise ValueError("embeddings must have non-zero spread")
    ref /= ref_scale
    cand /= cand_scale

    transforms = []
    for swap_axes in (False, True):
        permutation = np.array([1, 0]) if swap_axes else np.array([0, 1])
        for signs, flip in (
            (np.array([1.0, 1.0]), "none"),
            (np.array([-1.0, 1.0]), "flip X"),
            (np.array([1.0, -1.0]), "flip Y"),
            (np.array([-1.0, -1.0]), "flip X and Y"),
        ):
            name = f"swap X/Y + {flip}" if swap_axes and flip != "none" else (
                "swap X/Y" if swap_axes else flip
            )
            transforms.append((permutation, signs, name))
    scores = [
        float(np.mean(np.sum((ref - cand[:, permutation] * signs) ** 2, axis=1)))
        for permutation, signs, _ in transforms
    ]
    best = int(np.argmin(scores))
    permutation, signs, name = transforms[best]
    return candidate[:, permutation] * signs, {"transform": name, "normalized_mse": scores[best]}


def load_biological_datasets(*, random_state=42, n_hvg=2000, n_pcs=50):
    """Load full biological datasets and return scaled-HVG PCA coordinates."""
    import scanpy as sc
    from scipy import sparse

    if n_hvg < 1 or n_pcs < 1:
        raise ValueError("n_hvg and n_pcs must be positive integers.")

    def dense_float32(matrix):
        if sparse.issparse(matrix):
            matrix = matrix.toarray()
        return np.asarray(matrix, dtype=np.float32)

    paul = sc.datasets.paul15()
    paul_labels = paul.obs["paul15_clusters"].astype(str).to_numpy()
    paul_root = int(paul.uns["iroot"])
    stage_by_cluster = {
        **{cluster: "Progenitor" for cluster in ("7MEP", "9GMP", "10GMP")},
        **{cluster: "Differentiating lineage" for cluster in (
            "1Ery", "2Ery", "3Ery", "4Ery", "5Ery", "6Ery",
            "12Baso", "13Baso", "14Mo", "15Mo", "16Neu", "17Neu",
        )},
        **{cluster: "Specialized cell" for cluster in ("8Mk", "11DC", "18Eos", "19Lymph")},
    }
    unknown_clusters = set(paul_labels) - set(stage_by_cluster)
    if unknown_clusters:
        raise ValueError(f"No developmental-stage annotation for Paul15 clusters: {sorted(unknown_clusters)}")
    paul_stages = np.asarray([stage_by_cluster[label] for label in paul_labels])
    sc.pp.normalize_total(paul, target_sum=1e4)
    sc.pp.log1p(paul)
    sc.pp.highly_variable_genes(paul, n_top_genes=n_hvg, flavor="seurat", subset=True)
    sc.pp.scale(paul, max_value=10)
    sc.pp.pca(paul, n_comps=n_pcs, random_state=random_state)
    paul_X = dense_float32(paul.obsm["X_pca"])

    pbmc = sc.datasets.pbmc3k()
    pbmc_annotated = sc.datasets.pbmc3k_processed()
    sc.pp.filter_cells(pbmc, min_genes=200)
    sc.pp.filter_genes(pbmc, min_cells=3)
    common = pbmc.obs_names[pbmc.obs_names.isin(pbmc_annotated.obs_names)]
    pbmc = pbmc[common].copy()
    pbmc_labels = pbmc_annotated.obs.loc[common, "louvain"].astype(str).to_numpy()
    sc.pp.normalize_total(pbmc, target_sum=1e4)
    sc.pp.log1p(pbmc)
    sc.pp.highly_variable_genes(pbmc, n_top_genes=n_hvg, flavor="seurat", subset=True)
    sc.pp.scale(pbmc, max_value=10)
    sc.pp.pca(pbmc, n_comps=n_pcs, random_state=random_state)
    pbmc_X = dense_float32(pbmc.obsm["X_pca"])

    return {
        "Paul15 myeloid progenitors": {
            "X": paul_X,
            "labels": paul_labels,
            "stage_labels": paul_stages,
            "stage_order": ["Progenitor", "Differentiating lineage", "Specialized cell"],
            "stage_description": "broad curated maturation stage (exploratory grouping)",
            "display": "Paul15 myeloid progenitors",
            "feature_description": f"{n_hvg} HVGs -> {n_pcs} PCs (direct input)",
            "label_description": "curated Paul15 cluster",
            "root_index": paul_root,
        },
        "PBMC3k annotated": {
            "X": pbmc_X,
            "labels": pbmc_labels,
            "display": "PBMC3k annotated",
            "feature_description": f"{n_hvg} HVGs -> {n_pcs} PCs (direct input)",
            "label_description": "marker-annotated PBMC cell type",
        },
    }


def load_gse52529_myoblasts(*, random_state=42, n_hvg=2000, n_pcs=50):
    """Load clean GSE52529 myoblasts with log1p-FPKM HVG-PCA features."""
    import scanpy as sc

    if n_hvg < 1 or n_pcs < 1:
        raise ValueError("n_hvg and n_pcs must be positive integers.")

    myoblast_dir = Path(__file__).resolve().parents[2] / "data" / "gse52529"
    metadata = pd.read_csv(myoblast_dir / "GSE52529_cell_metadata.csv")
    metadata = metadata.loc[metadata["include_clean_single_cell"].astype(bool)].reset_index(drop=True)
    expression = pd.read_csv(
        myoblast_dir / "GSE52529_fpkm_matrix.txt.gz",
        sep="\t",
        index_col=0,
        compression="gzip",
    )
    adata = sc.AnnData(
        X=expression.loc[:, metadata["sample_id"]].T.to_numpy(dtype=np.float32, copy=True),
        obs=metadata.set_index("sample_id"),
        var=pd.DataFrame(index=expression.index.astype(str)),
    )
    # GEO supplies normalized FPKM values, so do not apply count-based
    # library-size normalization here.
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, n_top_genes=min(n_hvg, adata.n_vars), flavor="seurat", subset=True
    )
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=n_pcs, random_state=random_state)

    time_hours = adata.obs["time_hours"].to_numpy(dtype=int)
    stages = np.asarray([f"{hour} h" for hour in time_hours])
    return {
        "X": np.asarray(adata.obsm["X_pca"], dtype=np.float32),
        "labels": stages,
        "stage_labels": stages,
        "stage_order": [f"{hour} h" for hour in sorted(np.unique(time_hours))],
        "stage_description": "observed hours after serum switch",
        "time_hours": time_hours,
        "sample_metadata": adata.obs.copy(),
        "display": "Human myoblast differentiation (GSE52529)",
        "feature_description": f"log1p FPKM, {min(n_hvg, expression.shape[0])} HVGs -> {n_pcs} PCs",
        "label_description": "observed collection time",
    }


def make_imste(**parameters):
    from imste import IteratedMinimumSpanningTreeEmbedder

    defaults = {
        "n_components": 2,
        "n_msts": 5,
        "n_epochs": 1000,
        "mst_method": "famst",
        "mst_neighbors": 15,
        "random_state": 42,
        "device": "cpu",
    }
    defaults.update(parameters)
    return IteratedMinimumSpanningTreeEmbedder(**defaults)


def make_umap(**parameters):
    from umap import UMAP

    defaults = {
        "n_components": 2,
        "n_neighbors": 30,
        "min_dist": 0.1,
        "metric": "euclidean",
        "random_state": 42,
        "n_jobs": 1,
    }
    defaults.update(parameters)
    return UMAP(**defaults)


def fit_and_align_embeddings(datasets, *, imste_params=None, umap_params=None):
    """Fit IMSTE/UMAP once per dataset and align UMAP using signed axis swaps."""
    imste_parameters = {"random_state": 42}
    imste_parameters.update(imste_params or {})
    umap_parameters = {"random_state": 42}
    umap_parameters.update(umap_params or {})
    embeddings = {}
    alignment = {}
    for name, data in datasets.items():
        X = data["X"]
        embeddings[name] = {
            "IMSTE": make_imste(**imste_parameters).fit_transform(X),
            "UMAP": make_umap(**umap_parameters).fit_transform(X),
        }
        embeddings[name]["UMAP"], alignment[name] = align_by_axis_flips(
            embeddings[name]["IMSTE"], embeddings[name]["UMAP"]
        )
    return embeddings, alignment


def diagnose_umap_graph(X, labels, *, umap_params=None):
    """Report connected components and label mix in UMAP's fuzzy neighbor graph."""
    from scipy.sparse.csgraph import connected_components

    model = make_umap(**(umap_params or {})).fit(X)
    component_count, component_ids = connected_components(model.graph_, directed=False)
    labels = np.asarray(labels).astype(str)
    rows = []
    for component in range(component_count):
        indices = np.flatnonzero(component_ids == component)
        counts = pd.Series(labels[indices]).value_counts()
        rows.append({
            "graph_component": component,
            "cells": len(indices),
            "fraction": len(indices) / len(labels),
            "number_of_labels": int(counts.size),
            "largest_label_share": float(counts.iloc[0] / len(indices)),
            "most_common_labels": tuple((counts / len(indices)).head(3).round(3).items()),
        })
    return model.embedding_, pd.DataFrame(rows).sort_values("cells", ascending=False)


def _ordered_class_labels(labels):
    """Keep numbered cluster labels in numeric order; sort other labels plainly."""
    classes = sorted(set(np.asarray(labels).astype(str)))
    if classes and all(re.match(r"^\d+", label) for label in classes):
        return sorted(classes, key=lambda label: (int(re.match(r"^\d+", label).group()), label))
    return classes


def class_color_mapping(labels):
    """Return a stable palette, using a numeric jet gradient for numbered clusters."""
    import matplotlib.pyplot as plt

    classes = _ordered_class_labels(labels)
    if classes and all(re.match(r"^\d+", label) for label in classes):
        positions = np.linspace(0.02, 0.98, len(classes)) if len(classes) > 1 else [0.5]
        palette = plt.get_cmap("jet")(positions)
        return {label: palette[index] for index, label in enumerate(classes)}

    cmap_name = "tab20" if len(classes) > 10 else "tab10"
    palette = plt.get_cmap(cmap_name).colors
    return {label: palette[index] for index, label in enumerate(classes)}


def plot_embedding_comparisons(datasets, embeddings, alignment, *, point_size=10, point_alpha=0.78):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    figures = {}
    for name, data in datasets.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
        labels = np.asarray(data["labels"]).astype(str)
        classes = _ordered_class_labels(labels)
        class_codes = {label: index for index, label in enumerate(classes)}
        color_codes = np.asarray([class_codes[label] for label in labels])
        color_mapping = class_color_mapping(labels)
        cmap = ListedColormap([color_mapping[label] for label in classes])
        norm = BoundaryNorm(np.arange(-0.5, len(classes) + 0.5), len(classes))
        for ax, method in zip(axes, ("IMSTE", "UMAP")):
            Z = embeddings[name][method]
            ax.scatter(Z[:, 0], Z[:, 1], c=color_codes, cmap=cmap, norm=norm, s=point_size,
                       alpha=point_alpha, edgecolors="black", linewidths=0.15)
            note = f" (transform: {alignment[name]['transform']})" if method == "UMAP" else ""
            ax.set_title(f"{method}{note}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")
        handles = [
            plt.Line2D([], [], marker="o", linestyle="", markersize=5,
                       color=color_mapping[label], label=str(label))
            for i, label in enumerate(classes)
        ]
        fig.legend(handles=handles, title=data["label_description"],
                   loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize="small")
        fig.suptitle(name)
        figures[name] = fig
    return figures


def plot_cell_fate_stages(datasets, embeddings, *, point_size=10, point_alpha=0.78):
    """Plot existing embeddings colored by curated stages or observed timepoints."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    figures = {}
    for name, data in datasets.items():
        if "stage_labels" not in data:
            continue
        labels = np.asarray(data["stage_labels"]).astype(str)
        stages = data.get("stage_order", list(dict.fromkeys(labels)))
        stages = [stage for stage in stages if stage in set(labels)]
        codes = np.asarray([stages.index(label) for label in labels])
        paul_stage_colors = {
            "Progenitor": "#D73027",
            "Differentiating lineage": "#FFA500",
            "Specialized cell": "#FFFFFF",
        }
        if all(stage in paul_stage_colors for stage in stages):
            colors = [paul_stage_colors[stage] for stage in stages]
        else:
            positions = np.linspace(0.35, 0.9, len(stages)) if len(stages) > 1 else [0.65]
            colors = plt.get_cmap("Oranges")(positions)
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(-0.5, len(stages) + 0.5), len(stages))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
        for ax, method in zip(axes, ("IMSTE", "UMAP")):
            Z = embeddings[name][method]
            ax.scatter(Z[:, 0], Z[:, 1], c=codes, cmap=cmap, norm=norm, s=point_size,
                       alpha=point_alpha, edgecolors="black", linewidths=0.15)
            ax.set_title(method)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")
        handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=6,
                              color=colors[i], label=stage)
                   for i, stage in enumerate(stages)]
        fig.legend(handles=handles, title=data["stage_description"], loc="center left",
                   bbox_to_anchor=(1.01, 0.5), fontsize="small")
        fig.suptitle(f"{name}: {data['stage_description']}")
        # The notebook explicitly displays returned figures; unregister them
        # here to prevent Jupyter's end-of-cell auto-display from duplicating it.
        plt.close(fig)
        figures[name] = fig
    return figures


def plot_biological_class_graphs(datasets, embeddings, *, point_alpha=0.4, random_state=42):
    import matplotlib.pyplot as plt
    from imste.class_graph import plot_class_mixture_mst

    figures = {}
    tables = []
    for name, data in datasets.items():
        fig, _, edges = plot_class_mixture_mst(
            data["X"], data["labels"], embeddings=embeddings[name], title=name,
            n_msts=2, max_components_per_class=3, point_alpha=point_alpha,
            min_edge_alpha=0.2, max_edge_alpha=0.9, random_state=random_state,
            class_colors=class_color_mapping(data["labels"]),
        )
        edges.insert(0, "dataset", name)
        figures[name] = fig
        tables.append(edges)
    return figures, pd.concat(tables, ignore_index=True)


def compare_biological_class_relations(edge_scores, labels_by_dataset):
    """Compare unique graph relations and check a small expected-lineage set."""
    from imste.class_graph import compare_class_graph_relations

    differences = {
        name: compare_class_graph_relations(
            edge_scores, method_a="IMSTE", method_b="UMAP", dataset=name
        )
        for name in labels_by_dataset
    }

    def canonical_pair(a, b):
        return tuple(sorted((str(a), str(b))))

    def same_lineage_pairs(labels, suffix, description):
        members = sorted(label for label in set(labels) if label.endswith(suffix))
        return [(a, b, description) for a, b in itertools.combinations(members, 2)]

    paul_labels = labels_by_dataset["Paul15 myeloid progenitors"]
    expected = {
        "Paul15 myeloid progenitors": (
            same_lineage_pairs(paul_labels, "Ery", "erythroid states")
            + same_lineage_pairs(paul_labels, "Mo", "monocyte states")
            + same_lineage_pairs(paul_labels, "Neu", "neutrophil states")
            + same_lineage_pairs(paul_labels, "GMP", "GMP states")
            + same_lineage_pairs(paul_labels, "Baso", "basophil states")
            + [("7MEP", state, "MEP to erythroid/megakaryocyte")
               for state in sorted(set(paul_labels)) if state.endswith("Ery") or state == "8Mk"]
        ),
        "PBMC3k annotated": [
            ("CD4 T cells", "CD8 T cells", "T-cell subsets"),
            ("CD14+ Monocytes", "FCGR3A+ Monocytes", "monocyte subsets"),
            ("CD8 T cells", "NK cells", "cytotoxic lymphocyte populations"),
        ],
    }

    rows = []
    for dataset_name, relation_list in expected.items():
        dataset_edges = edge_scores.loc[edge_scores["dataset"] == dataset_name]
        for class_a, class_b, interpretation in relation_list:
            pair = canonical_pair(class_a, class_b)
            row = {"dataset": dataset_name, "relationship": interpretation,
                   "class_a": class_a, "class_b": class_b}
            for method in ("IMSTE", "UMAP"):
                subset = dataset_edges.loc[dataset_edges["embedding"] == method]
                subset = subset[[canonical_pair(a, b) == pair
                                 for a, b in zip(subset.class_a, subset.class_b)]]
                errors = np.clip(subset["symmetric_component_overlap"].to_numpy() / 0.5, 0, 1)
                row[f"{method}_edge_present"] = bool(len(subset))
                row[f"{method}_mean_edge_error"] = float(errors.mean()) if len(errors) else np.nan
            rows.append(row)
    expected_results = pd.DataFrame(rows)
    summary = expected_results.groupby("dataset").agg(
        expected_relations=("relationship", "size"),
        IMSTE_hits=("IMSTE_edge_present", "sum"),
        UMAP_hits=("UMAP_edge_present", "sum"),
    )
    return differences, expected_results, summary


def pairwise_cell_distance_rho(X, Z, *, seed=42, n_pairs=50000, chunk_size=256):
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(X), size=n_pairs)
    j = rng.integers(0, len(X), size=n_pairs)
    valid = i != j
    i, j = i[valid], j[valid]
    input_distances = np.empty(len(i), dtype=np.float64)
    embedded_distances = np.empty(len(i), dtype=np.float64)
    for start in range(0, len(i), chunk_size):
        stop = min(start + chunk_size, len(i))
        input_distances[start:stop] = np.linalg.norm(X[i[start:stop]] - X[j[start:stop]], axis=1)
        embedded_distances[start:stop] = np.linalg.norm(Z[i[start:stop]] - Z[j[start:stop]], axis=1)
    return float(spearmanr(input_distances, embedded_distances).statistic)


def population_position_scores(X, Z, labels, *, n_nearest=3):
    from scipy.stats import kendalltau, spearmanr

    labels = np.asarray(labels).astype(str)
    groups = np.unique(labels)
    centroids_X = np.vstack([X[labels == group].mean(axis=0) for group in groups])
    centroids_Z = np.vstack([Z[labels == group].mean(axis=0) for group in groups])
    distances_X = np.linalg.norm(centroids_X[:, None, :] - centroids_X[None, :, :], axis=2)
    distances_Z = np.linalg.norm(centroids_Z[:, None, :] - centroids_Z[None, :, :], axis=2)
    upper = np.triu_indices(len(groups), k=1)
    rho = float(spearmanr(distances_X[upper], distances_Z[upper]).statistic)
    tau = float(kendalltau(distances_X[upper], distances_Z[upper]).statistic)
    np.fill_diagonal(distances_X, np.inf)
    np.fill_diagonal(distances_Z, np.inf)
    k = min(n_nearest, len(groups) - 1)
    nearest_X = np.argsort(distances_X, axis=1)[:, :k]
    nearest_Z = np.argsort(distances_Z, axis=1)[:, :k]
    overlap = np.mean([len(set(a).intersection(b)) / k for a, b in zip(nearest_X, nearest_Z)])
    return rho, tau, float(overlap)


def global_position_metrics(datasets, embeddings, *, seed=42):
    rows = []
    for name, data in datasets.items():
        for method, coordinates in embeddings[name].items():
            rho, tau, overlap = population_position_scores(data["X"], coordinates, data["labels"])
            rows.append({
                "dataset": name,
                "method": method,
                "pairwise cell-distance rho": pairwise_cell_distance_rho(data["X"], coordinates, seed=seed),
                "population-centroid distance rho": rho,
                "population-centroid distance tau": tau,
                "nearest-population overlap@3": overlap,
            })
    return pd.DataFrame(rows)


def umap_sensitivity_metrics(datasets, *, seed=42, neighbors=(10, 20, 40)):
    rows = []
    for name, data in datasets.items():
        for n_neighbors in neighbors:
            coordinates = make_umap(random_state=seed, n_neighbors=n_neighbors).fit_transform(data["X"])
            rho, tau, overlap = population_position_scores(data["X"], coordinates, data["labels"])
            rows.append({
                "dataset": name,
                "UMAP n_neighbors": n_neighbors,
                "pairwise cell-distance rho": pairwise_cell_distance_rho(data["X"], coordinates, seed=seed),
                "population-centroid distance rho": rho,
                "population-centroid distance tau": tau,
                "nearest-population overlap@3": overlap,
            })
    return pd.DataFrame(rows)


def seed_robustness_metrics(datasets, *, seeds=(7, 123), n_epochs=1000, n_msts=5):
    rows = []
    for name, data in datasets.items():
        for seed in seeds:
            for method in ("IMSTE", "UMAP"):
                reducer = (
                    make_imste(random_state=seed, n_epochs=n_epochs, n_msts=n_msts)
                    if method == "IMSTE" else make_umap(random_state=seed)
                )
                coordinates = reducer.fit_transform(data["X"])
                rho, tau, overlap = population_position_scores(data["X"], coordinates, data["labels"])
                rows.append({
                    "dataset": name,
                    "seed": seed,
                    "method": method,
                    "pairwise cell-distance rho": pairwise_cell_distance_rho(data["X"], coordinates, seed=seed),
                    "population-centroid distance rho": rho,
                    "population-centroid distance tau": tau,
                    "nearest-population overlap@3": overlap,
                })
    return pd.DataFrame(rows)
