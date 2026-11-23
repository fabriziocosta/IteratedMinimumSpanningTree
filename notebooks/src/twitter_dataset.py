"""Stream, sample, embed, and cache ordered Twitter JSONL.GZ batches."""
from __future__ import annotations

import hashlib
import heapq
import json
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import pandas as pd

DEFAULT_EMBEDDING_MODEL = "google/embeddinggemma-2"
CACHE_FORMAT_VERSION = 3


def _paths(data_dir: Path) -> dict[str, Path]:
    cache_dir = data_dir / "cache"
    return {
        "directory": cache_dir,
        "tweets": cache_dir / "twitter_tweets.pkl",
        "vectors": cache_dir / "twitter_vectors.npy",
        "manifest": cache_dir / "twitter_manifest.json",
    }


def _source_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "source_name": path.name,
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def _normalise_sources(source_paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(source_paths, (str, Path)):
        paths = [Path(source_paths)]
    else:
        paths = sorted((Path(path) for path in source_paths), key=lambda p: p.name)
    if not paths:
        raise FileNotFoundError("No Twitter .json.gz batch files were found.")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Twitter batch not found: {missing[0]}")
    return paths


def _json_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def _read_manifest(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _sample_batch(
    source_path: Path,
    max_tweets: int,
    random_state: int,
    language: str | None,
    sample_strategy: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Collect the first N eligible rows quickly, or reservoir-sample a full batch."""
    import gzip
    from tqdm.auto import tqdm

    if sample_strategy not in {"first", "reservoir"}:
        raise ValueError("sample_strategy must be 'first' or 'reservoir'.")

    # Each archive gets its own stable random stream for the optional full scan.
    seed_material = f"{random_state}:{source_path.name}".encode("utf-8")
    batch_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
    rng = np.random.default_rng(batch_seed)
    heap: list[tuple[float, int, dict]] = []
    first_rows: list[dict] = []
    scanned = eligible = 0

    if sample_strategy == "first":
        progress = tqdm(
            total=max_tweets,
            desc="Reading batch",
            unit="tweets",
            mininterval=0.25,
        )
    else:
        progress = tqdm(
            total=source_path.stat().st_size,
            desc="Scanning batch",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            mininterval=0.25,
            smoothing=0.05,
        )

    compressed_size = source_path.stat().st_size
    last_compressed_position = 0
    try:
        with source_path.open("rb") as compressed_file:
            with gzip.GzipFile(fileobj=compressed_file, mode="rb") as stream:
                for raw_line in stream:
                    line = raw_line.decode("utf-8", errors="replace")
                    scanned += 1
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        record = None

                    if record is not None:
                        text = record.get("text")
                        lang = record.get("lang")
                        if isinstance(text, str) and text.strip() and (language is None or lang == language):
                            eligible += 1
                            entity_data = record.get("entities") or {}
                            hashtag_entities = entity_data.get("hashtags", []) if isinstance(entity_data, dict) else []
                            hashtags = [
                                str(item.get("text", "")).lower()
                                for item in hashtag_entities
                                if isinstance(item, dict) and item.get("text")
                            ]
                            row = {
                                "source_batch": source_path.name,
                                "tweet_id": str(record.get("id_str") or record.get("id") or ""),
                                "created_at": record.get("created_at"),
                                "lang": lang or "unknown",
                                "text": text.strip(),
                                "hashtags": hashtags,
                                "top_hashtag": hashtags[0] if hashtags else "(no hashtag)",
                                "retweet_count": record.get("retweet_count"),
                                "favorite_count": record.get("favorite_count"),
                                "is_retweet": text.startswith("RT @"),
                            }
                            if sample_strategy == "first":
                                first_rows.append(row)
                                progress.update(1)
                            else:
                                priority = float(rng.random())
                                item = (-priority, scanned, row)
                                if len(heap) < max_tweets:
                                    heapq.heappush(heap, item)
                                elif priority < -heap[0][0]:
                                    heapq.heapreplace(heap, item)

                    if sample_strategy == "first":
                        if scanned % 100_000 == 0:
                            progress.set_postfix(scanned=f"{scanned:,}", eligible=f"{eligible:,}")
                        if len(first_rows) >= max_tweets:
                            break
                    else:
                        compressed_position = compressed_file.tell()
                        progress.update(max(0, compressed_position - last_compressed_position))
                        last_compressed_position = compressed_position
                        if scanned % 100_000 == 0:
                            progress.set_postfix(
                                scanned=f"{scanned:,}", eligible=f"{eligible:,}",
                                retained=f"{len(heap):,}",
                            )
            if sample_strategy == "reservoir" and progress.n < compressed_size:
                progress.update(compressed_size - progress.n)
    finally:
        progress.close()

    if sample_strategy == "first":
        tweets = pd.DataFrame(first_rows)
    else:
        rows = [(-item[0], item[1], item[2]) for item in heap]
        rows.sort(key=lambda item: (item[0], item[1]))
        tweets = pd.DataFrame([item[2] for item in rows])
    if tweets.empty:
        raise ValueError(
            f"No usable tweets found in {source_path.name} "
            f"for language={language!r}; every batch must contribute rows."
        )
    stats = {
        "source": _source_identity(source_path),
        "scanned_rows": scanned,
        "eligible_rows_seen": eligible,
        "sampled_rows": len(tweets),
    }
    print(
        f"Batch complete: {source_path.name} — {scanned:,} scanned, "
        f"{eligible:,} eligible seen, {len(tweets):,} sampled."
    )
    return tweets, stats

def _sample_batches(
    source_paths: list[Path],
    quotas: dict[str, int],
    random_state: int,
    language: str | None,
    sample_strategy: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    samples = []
    batch_stats = []
    # Explicit loop preserves the date/name order and makes batch boundaries
    # visible in both progress output and the cached source_batch column.
    for source_path in source_paths:
        batch, stats = _sample_batch(
            source_path, quotas[source_path.name], random_state, language, sample_strategy
        )
        samples.append(batch)
        batch_stats.append(stats)
    tweets = pd.concat(samples, ignore_index=True)
    tweets.attrs["batch_stats"] = batch_stats
    return tweets, batch_stats


def _batch_quotas(batch_order: list[str], max_tweets: int) -> dict[str, int]:
    """Divide a total row budget evenly across ordered input batches."""
    base, remainder = divmod(max_tweets, len(batch_order))
    return {
        batch: base + (index < remainder)
        for index, batch in enumerate(batch_order)
    }


def _indices_per_batch(tweets: pd.DataFrame, quotas: dict[str, int]) -> np.ndarray:
    selected = []
    for batch, quota in quotas.items():
        if quota > 0:
            selected.extend(
                tweets.index[tweets["source_batch"] == batch][:quota].tolist()
            )
    return np.sort(np.asarray(selected, dtype=np.int64))


def load_twitter_dataset(data_dir: str | Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    paths = _paths(Path(data_dir))
    manifest = _read_manifest(paths["manifest"])
    if manifest is None or manifest.get("format_version") != CACHE_FORMAT_VERSION:
        raise FileNotFoundError(
            "No current Twitter batch cache exists. Run notebooks/dataset_setup/"
            "prepare_twitter_dataset.ipynb to rebuild it from all batches."
        )
    if not paths["tweets"].is_file() or not paths["vectors"].is_file():
        raise FileNotFoundError("Prepared Twitter cache files are missing.")
    tweets = pd.read_pickle(paths["tweets"])
    vectors = np.load(paths["vectors"], mmap_mode="r")
    if len(tweets) != vectors.shape[0] or len(tweets) != manifest.get("n_tweets"):
        raise ValueError("The prepared Twitter cache files do not match their manifest.")
    active_total = int(manifest.get("active_max_tweets", manifest["max_tweets"]))
    quotas = _batch_quotas(manifest["batch_order"], active_total)
    indices = _indices_per_batch(tweets, quotas)
    if not len(indices):
        raise ValueError("The active Twitter sample contains no rows.")
    return (
        tweets.iloc[indices].reset_index(drop=True),
        vectors[indices],
        manifest,
    )


def prepare_twitter_dataset(
    data_dir: str | Path,
    source_paths: str | Path | Iterable[str | Path],
    *,
    max_tweets: int = 100_000,
    random_state: int = 42,
    language: str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    sample_strategy: str = "first",
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Prepare ordered samples from every archive, then encode and cache them."""
    if (
        isinstance(max_tweets, bool)
        or not isinstance(max_tweets, int)
        or max_tweets < 1
    ):
        raise ValueError("max_tweets must be a positive integer.")
    data_dir = Path(data_dir)
    source_files = _normalise_sources(source_paths)
    if len({path.name for path in source_files}) != len(source_files):
        raise ValueError("Twitter batch filenames must be unique.")
    if max_tweets < len(source_files):
        raise ValueError(
            f"max_tweets must be at least the number of batches ({len(source_files)}) "
            "so every batch contributes at least one tweet."
        )
    if sample_strategy not in {"first", "reservoir"}:
        raise ValueError("sample_strategy must be 'first' or 'reservoir'.")
    batch_order = [path.name for path in source_files]
    source_identities = [_source_identity(path) for path in source_files]
    quotas = _batch_quotas(batch_order, max_tweets)
    paths = _paths(data_dir)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(paths["manifest"])

    matching = (
        not force_rebuild
        and manifest is not None
        and manifest.get("format_version") == CACHE_FORMAT_VERSION
        and manifest.get("sources") == source_identities
        and manifest.get("random_state") == random_state
        and manifest.get("language") == language
        and manifest.get("embedding_model") == embedding_model
        and manifest.get("sample_strategy") == sample_strategy
        and paths["tweets"].is_file()
        and paths["vectors"].is_file()
    )
    if matching:
        try:
            cached_tweets = pd.read_pickle(paths["tweets"])
            cached_vectors = np.load(paths["vectors"], mmap_mode="r")
            matching = (
                len(cached_tweets)
                == cached_vectors.shape[0]
                == manifest.get("n_tweets")
            )
        except (OSError, ValueError, EOFError):
            matching = False

    if matching and manifest["max_tweets"] >= max_tweets:
        indices = _indices_per_batch(cached_tweets, quotas)
        manifest["active_max_tweets"] = max_tweets
        manifest["active_n_tweets"] = len(indices)
        manifest["dataset_id"] = _json_hash(
            {
                "base_dataset_id": manifest["base_dataset_id"],
                "active_max_tweets": max_tweets,
            }
        )
        paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"Reusing cached embeddings for {len(source_files)} batches "
            f"({len(indices):,} active tweets; batch quotas {quotas})."
        )
        return (
            cached_tweets.iloc[indices].reset_index(drop=True),
            cached_vectors[indices],
            manifest,
        )

    tweets, batch_stats = _sample_batches(
        source_files, quotas, random_state, language, sample_strategy
    )
    print(
        f"Embedding {len(tweets):,} tweets from {len(source_files)} ordered batches "
        f"with {embedding_model}…"
    )
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embedding_model)
    vectors = model.encode(
        tweets["text"].tolist(),
        prompt_name="Clustering",
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    core = {
        "format_version": CACHE_FORMAT_VERSION,
        "sources": source_identities,
        "batches": batch_stats,
        "batch_order": [path.name for path in source_files],
        "random_state": random_state,
        "language": language,
        "embedding_model": embedding_model,
        "max_tweets": max_tweets,
        "batch_quotas": quotas,
        "sample_strategy": sample_strategy,
        "n_batches": len(source_files),
        "n_tweets": len(tweets),
        "n_features": int(vectors.shape[1]),
        "dtype": str(vectors.dtype),
    }
    base_id = _json_hash(core)
    manifest = {
        **core,
        "base_dataset_id": base_id,
        "active_max_tweets": max_tweets,
        "active_n_tweets": len(tweets),
        "dataset_id": _json_hash(
            {
                "base_dataset_id": base_id,
                "active_max_tweets": max_tweets,
            }
        ),
    }

    tweets_tmp = paths["directory"] / ".twitter_tweets.tmp.pkl"
    vectors_tmp = paths["directory"] / ".twitter_vectors.tmp.npy"
    manifest_tmp = paths["directory"] / ".twitter_manifest.tmp.json"
    tweets.to_pickle(tweets_tmp)
    with vectors_tmp.open("wb") as vector_file:
        np.save(vector_file, vectors, allow_pickle=False)
    manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tweets_tmp.replace(paths["tweets"])
    vectors_tmp.replace(paths["vectors"])
    manifest_tmp.replace(paths["manifest"])
    print(
        f"Saved {len(tweets):,} tweets × {vectors.shape[1]:,} features "
        f"from {len(source_files)} batches in {paths['directory']}."
    )
    return tweets.reset_index(drop=True), vectors, manifest
