"""Data preparation and plotting helpers for the 20 Newsgroups notebooks."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

MODEL_ID = "google/embeddinggemma-2"
PROMPT_NAME = "Clustering"


def prepare_20newsgroups_dataset(
    data_dir: str | Path,
    *,
    embedding_method: str = "gemma2",
    max_per_category: int | None = None,
    random_state: int = 42,
    batch_size: int = 4,
    max_seq_length: int = 2048,
    tfidf_dimensions: int = 256,
    force_reembed: bool = False,
):
    """Fetch, stratify, embed or vectorize, and cache 20 Newsgroups messages."""
    from sklearn.datasets import fetch_20newsgroups

    embedding_method = embedding_method.lower()
    if embedding_method not in {"gemma2", "tfidf"}:
        raise ValueError("embedding_method must be either 'gemma2' or 'tfidf'.")
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    cache_dir = data_dir / "cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if max_per_category is not None and (
        not isinstance(max_per_category, int) or max_per_category < 1
    ):
        raise ValueError("max_per_category must be a positive integer or None.")
    if not isinstance(tfidf_dimensions, int) or tfidf_dimensions < 2:
        raise ValueError("tfidf_dimensions must be an integer greater than one.")
    sample_tag = "full" if max_per_category is None else f"{max_per_category}_per_category"
    method_tag = "embeddinggemma2" if embedding_method == "gemma2" else "tfidf"
    embedding_path = cache_dir / (
        f"20newsgroups_{sample_tag}_seed{random_state}_{method_tag}.npz"
    )
    manifest_path = cache_dir / f"{embedding_path.stem}_manifest.json"

    print("Fetching or loading the 20 Newsgroups corpus…")
    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        data_home=str(raw_dir),
        shuffle=False,
        download_if_missing=True,
    )
    all_texts = np.asarray(dataset.data, dtype=str)
    all_labels = np.asarray(dataset.target, dtype=np.int64)
    category_names = list(dataset.target_names)
    if len(all_texts) != len(all_labels) or len(category_names) != 20:
        raise ValueError("Unexpected 20 Newsgroups text, label, or category counts.")

    manifest = None
    if embedding_path.is_file() and manifest_path.is_file() and not force_reembed:
        manifest = json.loads(manifest_path.read_text())
        with np.load(embedding_path, allow_pickle=False) as saved:
            X = saved["embeddings"]
            texts = saved["texts"]
            labels = saved["labels"]
            sample_indices = saved["indices"]
            saved_categories = saved["category_names"].tolist()
            saved_method = (
                str(saved["embedding_method"].item())
                if "embedding_method" in saved.files
                else "gemma2"
            )
            saved_model_id = str(saved["model_id"].item()) if "model_id" in saved.files else MODEL_ID
            saved_prompt = str(saved["prompt_name"].item()) if "prompt_name" in saved.files else PROMPT_NAME
        if saved_method != embedding_method:
            raise ValueError("The cached embedding method does not match the current setting.")
        if embedding_method == "gemma2" and (saved_model_id != MODEL_ID or saved_prompt != PROMPT_NAME):
            raise ValueError("The cached model or prompt does not match the current settings.")
        if saved_categories != category_names:
            raise ValueError("Cached category names do not match the fetched dataset.")
        if not np.array_equal(texts, all_texts[sample_indices]) or not np.array_equal(
            labels, all_labels[sample_indices]
        ):
            raise ValueError("The cache rows do not match the fetched dataset.")
        print(f"Loaded cached vectors: {embedding_path} (embedding progress is skipped)")
    else:
        rng = np.random.default_rng(random_state)
        selected = []
        for category_id in range(len(category_names)):
            class_indices = np.flatnonzero(all_labels == category_id)
            n_selected = (
                len(class_indices)
                if max_per_category is None
                else min(max_per_category, len(class_indices))
            )
            selected.append(rng.choice(class_indices, size=n_selected, replace=False))
        sample_indices = np.concatenate(selected)
        rng.shuffle(sample_indices)
        texts, labels = all_texts[sample_indices], all_labels[sample_indices]

        started = time.perf_counter()
        if embedding_method == "gemma2":
            import torch
            from sentence_transformers import SentenceTransformer
            from tqdm.auto import tqdm

            if torch.cuda.is_available():
                device = "cuda"
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device, dtype = "mps", torch.float32
            else:
                device, dtype = "cpu", torch.float32
            print(f"Loading {MODEL_ID} on {device} ({dtype}); first load may take a while…")
            model = SentenceTransformer(
                MODEL_ID,
                device=device,
                model_kwargs={"torch_dtype": dtype},
                config_kwargs={"vision_config": None, "audio_config": None},
            )
            model.max_seq_length = max_seq_length
            X = None
            for start in tqdm(
                range(0, len(texts), batch_size),
                desc="Embedding 20 Newsgroups with Gemma 2",
                unit="batch",
            ):
                batch = texts[start : start + batch_size].tolist()
                vectors = model.encode(
                    batch,
                    prompt_name=PROMPT_NAME,
                    batch_size=len(batch),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                if X is None:
                    X = np.empty((len(texts), vectors.shape[1]), dtype=np.float32)
                X[start : start + len(batch)] = vectors
            model_id, prompt_name = MODEL_ID, PROMPT_NAME
        else:
            from scipy.sparse import vstack
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
            from sklearn.preprocessing import normalize
            from tqdm.auto import tqdm

            vectorizer = HashingVectorizer(
                n_features=2**16,
                alternate_sign=False,
                norm=None,
                stop_words="english",
                ngram_range=(1, 2),
                dtype=np.float32,
            )
            sparse_batches = [
                vectorizer.transform(texts[start : start + batch_size].tolist())
                for start in tqdm(
                    range(0, len(texts), batch_size),
                    desc="Vectorizing 20 Newsgroups with TF-IDF",
                    unit="batch",
                )
            ]
            sparse_tfidf = TfidfTransformer(sublinear_tf=True).fit_transform(
                vstack(sparse_batches, format="csr")
            )
            n_components = min(tfidf_dimensions, sparse_tfidf.shape[0] - 1, sparse_tfidf.shape[1] - 1)
            print(
                f"Reducing TF-IDF to {n_components} dimensions; "
                "the scikit-learn SVD step may take a while."
            )
            reducer = TruncatedSVD(
                n_components=n_components,
                n_iter=5,
                random_state=random_state,
            )
            X = reducer.fit_transform(sparse_tfidf).astype(np.float32, copy=False)
            X = normalize(X, norm="l2").astype(np.float32, copy=False)
            model_id, prompt_name = "HashingVectorizer + TF-IDF + TruncatedSVD", None
        if X is None or not np.isfinite(X).all():
            raise ValueError("Vectorization returned empty or non-finite values.")

        np.savez_compressed(
            embedding_path,
            embeddings=X,
            texts=texts,
            labels=labels,
            indices=sample_indices,
            category_names=np.asarray(category_names),
            model_id=np.asarray(model_id),
            prompt_name=np.asarray(prompt_name or ""),
            embedding_method=np.asarray(embedding_method),
        )
        manifest = {
            "dataset": "20 Newsgroups (all subsets)",
            "remove": ["headers", "footers", "quotes"],
            "embedding_method": embedding_method,
            "model_id": model_id,
            "prompt_name": prompt_name,
            "max_seq_length": max_seq_length if embedding_method == "gemma2" else None,
            "tfidf_dimensions": X.shape[1] if embedding_method == "tfidf" else None,
            "max_per_category": max_per_category,
            "random_state": random_state,
            "embedding_file": embedding_path.name,
            "shape": list(X.shape),
            "dtype": str(X.dtype),
            "vector_sha256": hashlib.sha256(
                np.ascontiguousarray(X).view(np.uint8)
            ).hexdigest(),
            "category_names": category_names,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Prepared {len(X):,} messages in {(time.perf_counter() - started) / 60:.1f} min")

    if X.ndim != 2 or X.shape[0] != len(labels) or len(labels) != len(texts):
        raise ValueError("Cached vectors, texts, and labels have inconsistent lengths.")
    if not np.isfinite(X).all():
        raise ValueError("The cached vectors contain NaN or infinite values.")
    fingerprint = hashlib.sha256(np.ascontiguousarray(X).view(np.uint8)).hexdigest()
    if fingerprint != manifest.get("vector_sha256"):
        raise ValueError("The cached vectors do not match the manifest fingerprint.")

    counts = np.bincount(labels, minlength=len(category_names))
    print(f"Embedding matrix: {X.shape} {X.dtype}")
    print("Messages per category:")
    for name, count in zip(category_names, counts):
        print(f"  {name}: {count:,}")
    print("Manifest:", manifest_path)
    return X, texts, labels, sample_indices, category_names, manifest


def load_20newsgroups_dataset(data_dir: str | Path, cache_index: int = -1):
    """Load and validate a cached embedding; default to the newest cache."""
    data_dir = Path(data_dir)
    cache_dir = data_dir / "cache"
    manifests = list(cache_dir.glob("20newsgroups_*_manifest.json"))
    if not manifests:
        raise FileNotFoundError(
            "Run dataset_setup/prepare_20newsgroups_dataset.ipynb first to prepare the vectors."
        )
    def created_at(path: Path) -> tuple[float, str]:
        stat = path.stat()
        # macOS exposes the true birth time; mtime is a portable fallback and
        # is set when this workflow writes the manifest.
        return getattr(stat, "st_birthtime", stat.st_mtime), path.name

    manifests.sort(key=created_at)
    if len(manifests) > 1:
        from datetime import datetime

        print("Available embedding caches:")
        for index, path in enumerate(manifests):
            info = json.loads(path.read_text())
            rows = info.get("shape", ["?"])[0]
            method = info.get("embedding_method", "gemma2")
            created = getattr(path.stat(), "st_birthtime", path.stat().st_mtime)
            created = datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  {index}: {method} · {rows:,} rows · created {created} · {path.name}")
    if not isinstance(cache_index, int) or not -len(manifests) <= cache_index < len(manifests):
        raise IndexError(f"cache_index must be in [{-len(manifests)}, {len(manifests) - 1}].")
    manifest_path = manifests[cache_index]
    manifest = json.loads(manifest_path.read_text())
    embedding_path = cache_dir / manifest["embedding_file"]
    with np.load(embedding_path, allow_pickle=False) as saved:
        X = saved["embeddings"]
        texts = saved["texts"]
        labels = saved["labels"]
        indices = saved["indices"]
        category_names = saved["category_names"].tolist()
    if X.ndim != 2 or X.shape[0] != len(labels) or len(labels) != len(texts):
        raise ValueError("Cached vectors, texts, and labels have inconsistent lengths.")
    if len(category_names) != 20 or labels.min() < 0 or labels.max() >= len(category_names):
        raise ValueError("Cached labels do not match the 20 Newsgroups categories.")
    if not np.isfinite(X).all():
        raise ValueError("The cached vectors contain NaN or infinite values.")
    fingerprint = hashlib.sha256(np.ascontiguousarray(X).view(np.uint8)).hexdigest()
    if fingerprint != manifest["vector_sha256"]:
        raise ValueError("The cached vectors do not match the manifest fingerprint.")
    method = manifest.get("embedding_method", "gemma2")
    print(f"Loaded {len(X):,} messages · {X.shape[1]} features · {method}: {manifest['model_id']}")
    return X, texts, labels, indices, category_names, manifest


def tune_20newsgroups_imste(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    random_state: int = 42,
    max_per_category: int = 250,
    n_epochs: int = 80,
    n_trials: int = 40,
    search_space: dict | None = None,
    base_settings: dict | None = None,
):
    """Tune IMSTE on a balanced, reproducible subset of the corpus."""
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) in sys.path:
        sys.path.remove(str(repo_root))
    sys.path.insert(0, str(repo_root))
    cached_imste = sys.modules.get("imste")
    cached_path = getattr(cached_imste, "__file__", None)
    if cached_imste is not None and (
        cached_path is None or not Path(cached_path).resolve().is_relative_to(repo_root)
    ):
        for module_name in [
            name for name in sys.modules if name == "imste" or name.startswith("imste.")
        ]:
            del sys.modules[module_name]

    from imste import IteratedMinimumSpanningTreeEmbedder

    y = np.asarray(labels)
    rng = np.random.default_rng(random_state)
    selected = []
    for category in np.unique(y):
        indices = np.flatnonzero(y == category)
        selected.append(rng.choice(indices, size=min(len(indices), max_per_category), replace=False))
    tune_indices = np.sort(np.concatenate(selected))
    print(
        f"Tuning on {len(tune_indices):,} messages across "
        f"{len(np.unique(y[tune_indices]))} categories"
    )

    if search_space is None:
        search_space = {
            "n_msts": (2, 64),
            "negative_ratio": (1, 10),
            "logistic_margin": (0.25, 2.0),
            "logistic_temperature": (0.1, 1.0),
            "lambda_rep": (0.25, 0.75),
            "gravity_strength": (0.01, 1.0),
            "gravity_resolution": (0.5, 2.0),
            "gravity_min_dist": (0.01, 1.0),
            "mean_shift_iterations": (1, 5),
        }
    settings = {
        "n_epochs": n_epochs,
        "random_state": random_state,
        "metric": "euclidean",
        "gravity_strength": 0.075,
        "mean_shift_iterations": 2,
        "verbose": False,
    }
    if base_settings:
        settings.update(base_settings)
    settings.update({"n_epochs": n_epochs, "verbose": False})
    tune_mapper = IteratedMinimumSpanningTreeEmbedder(**settings)
    result = tune_mapper.tune(
        X[tune_indices],
        y[tune_indices],
        search_space=search_space,
        n_samples=n_trials,
        n_initial_points=min(20, n_trials),
        coverage_weight=0.4,
        verbose=False,
    )
    return result, tune_indices


def fit_20newsgroups_umap(X: np.ndarray, random_state: int = 42):
    """Fit UMAP and return its coordinates and elapsed seconds."""
    import umap

    umap_settings = {
        "n_components": 2,
        "n_neighbors": 15,
        "min_dist": 0.1,
        "metric": "euclidean",
        "init": "recursive",
        "recursive_coarsening_ratio": 4,
        "compatibility_layout": False,
        "nearest_neighbor_backend": "rinnd",
        "random_state": random_state,
        "n_epochs": 200,
        "repulsion_strength": 1.0,
        "n_jobs": 1,
        "verbose": True,
    }
    print("Fitting UMAP (verbose progress enabled)")
    started = time.perf_counter()
    coordinates = umap.UMAP(**umap_settings).fit_transform(X)
    return coordinates, time.perf_counter() - started


def fit_20newsgroups_imste(
    X: np.ndarray,
    random_state: int = 42,
    settings: dict | None = None,
):
    """Fit IMSTE and return its coordinates, estimator, and stage timings."""
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    root_string = str(repo_root)
    if root_string in sys.path:
        sys.path.remove(root_string)
    sys.path.insert(0, root_string)

    # A notebook may already have resolved an unrelated namespace package as
    # ``imste``. Remove that incomplete import so Python loads this checkout.
    cached_imste = sys.modules.get("imste")
    cached_path = getattr(cached_imste, "__file__", None)
    if cached_imste is not None and (
        cached_path is None
        or not Path(cached_path).resolve().is_relative_to(repo_root)
    ):
        for module_name in [
            name for name in sys.modules if name == "imste" or name.startswith("imste.")
        ]:
            del sys.modules[module_name]

    from imste import IteratedMinimumSpanningTreeEmbedder

    imste_settings = {
        "n_msts": 64,
        "n_epochs": 200,
        "random_state": random_state,
        "metric": "euclidean",
        "gravity_strength": 0.075,
        "mean_shift_iterations": 2,
        "verbose": True,
    }
    if settings:
        imste_settings.update(settings)
    print("Fitting IMSTE (stage and epoch progress enabled)")
    estimator = IteratedMinimumSpanningTreeEmbedder(**imste_settings)
    coordinates = estimator.fit_transform(X)
    timings = {
        "imste": estimator.imste_fit_time_,
        "gravity": estimator.gravity_time_,
        "mean_shift": estimator.mean_shift_time_,
    }
    return coordinates, estimator, timings


def plot_20newsgroups_layouts(
    Z_umap,
    Z_imste,
    labels,
    category_names,
    texts,
    indices,
    timings,
    *,
    color_level: int = 3,
    show_marker_edge: bool = True,
    marker_edge_width: float = 0.35,
):
    """Display shared-palette density views and an interactive IMSTE scatter."""
    from html import escape
    from textwrap import fill
    import colorsys

    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    import plotly.graph_objects as go
    from imste.visualization import render_datashader_embedding

    if color_level not in {1, 2, 3}:
        raise ValueError("color_level must be 1, 2, or 3.")
    labels_text = np.asarray([category_names[int(label)] for label in labels])
    category_groups = {
        category: ".".join(category.split(".")[:color_level])
        for category in category_names
    }
    point_groups = np.asarray([category_groups[category] for category in labels_text])
    group_names = list(dict.fromkeys(category_groups.values()))
    root_names = list(dict.fromkeys(name.split(".")[0] for name in category_names))
    root_colors = {}
    for index, root_name in enumerate(root_names):
        red, green, blue = plt.get_cmap("tab10").colors[index % 10][:3]
        hue, _, _ = colorsys.rgb_to_hls(red, green, blue)
        root_colors[root_name] = hue

    color_key = {}
    for group_name in group_names:
        root_name = group_name.split(".")[0]
        siblings = [name for name in group_names if name.split(".")[0] == root_name]
        sibling_index = siblings.index(group_name)
        lightness_levels = np.linspace(0.34, 0.68, len(siblings))
        lightness = 0.52 if color_level == 1 else float(lightness_levels[sibling_index])
        color_key[group_name] = to_hex(
            colorsys.hls_to_rgb(root_colors[root_name], lightness, 0.72)
        )
    layouts = [
        ("UMAP", Z_umap, timings["umap"]),
        (
            "IMSTE",
            Z_imste,
            timings["imste"] + timings["gravity"] + timings["mean_shift"],
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    for ax, (name, coordinates, elapsed) in zip(axes, layouts):
        raster = render_datashader_embedding(
            coordinates,
            point_groups,
            width=1400,
            height=1100,
            color_key=color_key,
            min_alpha=45,
            spread_px=1,
            preserve_aspect=True,
        )
        ax.imshow(raster.to_pil())
        runtime = f"{elapsed / 60:.1f} min" if elapsed >= 60 else f"{elapsed:.1f} s"
        ax.set_title(f"20 Newsgroups — {name}, color level {color_level} ({runtime})")
        ax.set_axis_off()
    plt.show()

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color_key[name], label=name)
        for name in group_names
    ]
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.legend(
        handles=handles,
        loc="center",
        ncol=4,
        frameon=False,
        title=f"Newsgroup color level {color_level}",
    )
    ax.set_axis_off()
    plt.show()

    fig = go.Figure()
    for group_name in group_names:
        mask = point_groups == group_name
        original_categories = np.asarray(
            [category_names[int(label)] for label in labels[mask]], dtype=object
        )
        excerpts = np.asarray(
            [
                fill(escape(" ".join(text.split())[:560]), width=80).replace("\n", "<br>")
                for text in texts[mask]
            ],
            dtype=object,
        )
        customdata = np.column_stack(
            (
                np.full(int(mask.sum()), group_name, dtype=object),
                original_categories,
                indices[mask].astype(str),
                excerpts,
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=Z_imste[mask, 0],
                y=Z_imste[mask, 1],
                customdata=customdata,
                mode="markers",
                name=group_name,
                marker={
                    "color": color_key[group_name],
                    "size": 5,
                    "opacity": 0.72,
                    "line": {
                        "color": "gray",
                        "width": marker_edge_width if show_marker_edge else 0,
                    },
                },
                hoverlabel={"align": "left"},
                hovertemplate=(
                    f"Color group (level {color_level}): %{{customdata[0]}}<br>"
                    "Newsgroup: %{customdata[1]}<br>"
                    "Source row: %{customdata[2]}<br>"
                    "%{customdata[3]}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        template="plotly_white",
        title=f"20 Newsgroups — IMSTE · color level {color_level} · hover for message excerpts",
        width=1200,
        height=850,
        margin={"l": 20, "r": 20, "t": 70, "b": 150},
        legend={
            "title": f"Color level {color_level}",
            "orientation": "h",
            "x": 0.5,
            "y": -0.16,
            "xanchor": "center",
            "yanchor": "top",
            "entrywidth": 135,
            "entrywidthmode": "pixels",
            "groupclick": "togglegroup",
        },
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1},
    )
    return fig
