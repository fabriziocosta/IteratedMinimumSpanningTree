"""Prepare and reuse one size-aware arXiv text-vector cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_FORMAT_VERSION = 2


def _json_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def _source_identity(metadata_path: Path) -> dict[str, object]:
    stat = metadata_path.stat()
    return {
        "source_name": metadata_path.name,
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def _paths(data_dir: Path) -> dict[str, Path]:
    cache_dir = data_dir / "cache"
    return {
        "directory": cache_dir,
        "papers": cache_dir / "arxiv_papers.pkl",
        "vectors": cache_dir / "arxiv_vectors.npy",
        "manifest": cache_dir / "arxiv_manifest.json",
        "embeddings": cache_dir / "arxiv_embeddings.npz",
    }


def _read_manifest(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _cache_matches(
    paths: dict[str, Path],
    source_identity: dict[str, object],
    random_state: int,
    embedding_model: str,
) -> tuple[dict, pd.DataFrame, np.ndarray] | None:
    manifest = _read_manifest(paths["manifest"])
    if manifest is None or manifest.get("format_version") != CACHE_FORMAT_VERSION:
        return None
    if (
        manifest.get("source") != source_identity
        or manifest.get("random_state") != random_state
        or manifest.get("embedding_model") != embedding_model
        or not paths["papers"].is_file()
        or not paths["vectors"].is_file()
    ):
        return None
    try:
        papers = pd.read_pickle(paths["papers"])
        vectors = np.load(paths["vectors"], mmap_mode="r")
    except (OSError, ValueError, EOFError):
        return None
    if len(papers) != manifest.get("n_papers") or vectors.shape[0] != len(papers):
        return None
    return manifest, papers, vectors


def _manifest(
    source_identity: dict[str, object],
    random_state: int,
    embedding_model: str,
    papers: pd.DataFrame,
    vectors: np.ndarray,
) -> dict[str, object]:
    core = {
        "format_version": CACHE_FORMAT_VERSION,
        "source": source_identity,
        "random_state": random_state,
        "embedding_model": embedding_model,
        "n_papers": len(papers),
        "n_features": int(vectors.shape[1]),
        "dtype": str(vectors.dtype),
    }
    base_dataset_id = _json_hash(core)
    return {
        **core,
        "base_dataset_id": base_dataset_id,
        "active_n_papers": len(papers),
        "dataset_id": _json_hash({"base_dataset_id": base_dataset_id, "active_n_papers": len(papers)}),
    }


def _migrate_legacy_cache(
    paths: dict[str, Path],
    source_identity: dict[str, object],
    random_state: int,
    embedding_model: str,
) -> bool:
    """Promote the largest valid hashed cache created by the former notebook."""
    candidates = []
    directory = paths["directory"]
    for vector_path in directory.glob("arxiv_vectors_*.npy"):
        suffix = vector_path.stem.removeprefix("arxiv_vectors_")
        papers_path = directory / f"arxiv_sample_{suffix}.pkl"
        if not papers_path.is_file():
            continue
        try:
            vectors = np.load(vector_path, mmap_mode="r")
            papers = pd.read_pickle(papers_path)
        except (OSError, ValueError, EOFError):
            continue
        n_papers = len(papers)
        legacy_config = {
            **source_identity,
            "max_papers": n_papers,
            "random_state": random_state,
            "embedding_model": embedding_model,
            "cache_format": 1,
        }
        expected_suffix = hashlib.sha256(
            json.dumps(legacy_config, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        valid = (
            suffix == expected_suffix
            and vectors.shape[0] == n_papers
            and vectors.dtype == np.float32
        )
        if getattr(vectors, "_mmap", None) is not None:
            vectors._mmap.close()
        if valid:
            candidates.append((n_papers, papers_path, vector_path))

    if not candidates:
        return False
    _, papers_source, vectors_source = max(candidates, key=lambda item: item[0])
    papers = pd.read_pickle(papers_source)
    vectors = np.load(vectors_source, mmap_mode="r")
    promoted_manifest = _manifest(
        source_identity, random_state, embedding_model, papers, vectors
    )
    vectors._mmap.close()
    papers_source.replace(paths["papers"])
    vectors_source.replace(paths["vectors"])
    paths["manifest"].write_text(json.dumps(promoted_manifest, indent=2) + "\n")
    for old_path in directory.glob("arxiv_sample_*.pkl"):
        old_path.unlink(missing_ok=True)
    for old_path in directory.glob("arxiv_vectors_*.npy"):
        old_path.unlink(missing_ok=True)
    return True



def _remove_legacy_cache_files(directory: Path) -> None:
    """Remove per-size hashed files left by the former notebook workflow."""
    for old_path in directory.glob("arxiv_sample_*.pkl"):
        old_path.unlink(missing_ok=True)
    for old_path in directory.glob("arxiv_vectors_*.npy"):
        old_path.unlink(missing_ok=True)


def _save_cache(
    paths: dict[str, Path],
    source_identity: dict[str, object],
    random_state: int,
    embedding_model: str,
    papers: pd.DataFrame,
    vectors: np.ndarray,
) -> dict[str, object]:
    manifest = _manifest(
        source_identity, random_state, embedding_model, papers, vectors
    )
    papers_tmp = paths["directory"] / ".arxiv_papers.tmp.pkl"
    vectors_tmp = paths["directory"] / ".arxiv_vectors.tmp.npy"
    manifest_tmp = paths["directory"] / ".arxiv_manifest.tmp.json"
    papers.to_pickle(papers_tmp)
    with vectors_tmp.open("wb") as vector_file:
        np.save(vector_file, np.asarray(vectors, dtype=np.float32), allow_pickle=False)
    manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    papers_tmp.replace(paths["papers"])
    vectors_tmp.replace(paths["vectors"])
    manifest_tmp.replace(paths["manifest"])
    # Any previous 2D layouts describe the old input sample and must be rebuilt.
    paths["embeddings"].unlink(missing_ok=True)
    _remove_legacy_cache_files(paths["directory"])
    return manifest


def load_arxiv_dataset(data_dir: str | Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Load the single prepared arXiv cache, keeping vectors memory-mapped."""
    paths = _paths(Path(data_dir))
    manifest = _read_manifest(paths["manifest"])
    if manifest is None or not paths["papers"].is_file() or not paths["vectors"].is_file():
        raise FileNotFoundError(
            "No prepared arXiv cache exists. Run 03a_prepare_arxiv_dataset.ipynb first."
        )
    papers = pd.read_pickle(paths["papers"])
    vectors = np.load(paths["vectors"], mmap_mode="r")
    if len(papers) != vectors.shape[0] or len(papers) != manifest.get("n_papers"):
        raise ValueError("The prepared arXiv cache files do not match their manifest.")
    active_n = int(manifest.get("active_n_papers", manifest["n_papers"]))
    if not 1 <= active_n <= len(papers):
        raise ValueError("The active arXiv sample size is invalid in the cache manifest.")
    return papers.iloc[:active_n].reset_index(drop=True), vectors[:active_n], manifest


def prepare_arxiv_dataset(
    data_dir: str | Path,
    metadata_path: str | Path,
    *,
    max_papers: int = 80_000,
    random_state: int = 42,
    json_chunk_size: int = 50_000,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Create, enlarge, or reuse one nested arXiv sample and vector cache.

    Smaller requests reuse a prefix of the current largest cache without
    writing another dataset. Larger requests regenerate the deterministic
    nested sample and replace the one cache in place.
    """
    if isinstance(max_papers, bool) or not isinstance(max_papers, int) or max_papers < 1:
        raise ValueError("max_papers must be a positive integer.")
    if isinstance(json_chunk_size, bool) or not isinstance(json_chunk_size, int) or json_chunk_size < 1:
        raise ValueError("json_chunk_size must be a positive integer.")
    data_dir = Path(data_dir)
    metadata_path = Path(metadata_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"arXiv metadata JSONL not found: {metadata_path}")
    paths = _paths(data_dir)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    source_identity = _source_identity(metadata_path)

    cached = None if force_rebuild else _cache_matches(
        paths, source_identity, random_state, embedding_model
    )
    if cached is None and not force_rebuild and not paths["manifest"].exists():
        _migrate_legacy_cache(paths, source_identity, random_state, embedding_model)
        cached = _cache_matches(paths, source_identity, random_state, embedding_model)
    if cached is not None:
        manifest, cached_papers, cached_vectors = cached
        _remove_legacy_cache_files(paths["directory"])
        if manifest["n_papers"] >= max_papers:
            requested = min(max_papers, manifest["n_papers"])
            if manifest.get("active_n_papers", manifest["n_papers"]) != requested:
                manifest["active_n_papers"] = requested
                manifest["dataset_id"] = _json_hash({
                    "base_dataset_id": manifest.get("base_dataset_id", _json_hash(manifest)),
                    "active_n_papers": requested,
                })
                temp_manifest = paths["directory"] / ".arxiv_manifest.tmp.json"
                temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
                temp_manifest.replace(paths["manifest"])
                paths["embeddings"].unlink(missing_ok=True)
            print(
                f"Reusing the saved {manifest['n_papers']:,}-paper cache; "
                f"using {requested:,} rows without saving a second dataset."
            )
            return (
                cached_papers.iloc[:requested].reset_index(drop=True),
                cached_vectors[:requested],
                manifest,
            )
        if getattr(cached_vectors, "_mmap", None) is not None:
            cached_vectors._mmap.close()

    print(
        f"Sampling up to {max_papers:,} papers from {metadata_path.name} and "
        f"embedding with {embedding_model}..."
    )
    rng = np.random.default_rng(random_state)
    reservoir = pd.DataFrame(
        {
            "title": pd.Series(dtype="object"),
            "abstract": pd.Series(dtype="object"),
            "categories": pd.Series(dtype="object"),
            "_priority": pd.Series(dtype="float64"),
        }
    )
    rows_seen = 0
    for chunk in pd.read_json(metadata_path, lines=True, chunksize=json_chunk_size):
        chunk = chunk[["title", "abstract", "categories"]].copy()
        chunk["_priority"] = rng.random(len(chunk))
        reservoir = pd.concat([reservoir, chunk], ignore_index=True)
        if len(reservoir) > max_papers:
            reservoir = reservoir.nsmallest(max_papers, "_priority")
        rows_seen += len(chunk)
        print(
            f"Scanned {rows_seen:,} rows; retained {len(reservoir):,} papers",
            end="\r",
            flush=True,
        )
    papers = (
        reservoir.sort_values("_priority")
        .drop(columns="_priority")
        .reset_index(drop=True)
    )
    if not len(papers):
        raise ValueError("The metadata file contains no papers with the required fields.")

    from sentence_transformers import SentenceTransformer

    texts = (
        papers["title"].fillna("").str.strip()
        + "\n\n"
        + papers["abstract"].fillna("").str.strip()
    ).tolist()
    model = SentenceTransformer(embedding_model)
    vectors = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32, copy=False)
    manifest = _save_cache(
        paths,
        source_identity,
        random_state,
        embedding_model,
        papers,
        vectors,
    )
    papers, vectors, manifest = load_arxiv_dataset(data_dir)
    print(
        f"Saved one {manifest['n_papers']:,}-paper cache at {paths['directory']} "
        f"({vectors.shape[1]:,} features; {vectors.nbytes:,} vector bytes)."
    )
    return papers, vectors, manifest
