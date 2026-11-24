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
CACHE_FORMAT_VERSION = 4

BALANCE_FIELDS = {"retweet_count", "language", "retweet", "hour", "weekday"}


def _paths(data_dir: Path) -> dict[str, Path]:
    cache_dir = data_dir / "cache"
    return {
        "directory": cache_dir,
        "tweets": cache_dir / "twitter_tweets.pkl",
        "vectors": cache_dir / "twitter_vectors.npy",
        "manifest": cache_dir / "twitter_manifest.json",
        "sample": cache_dir / "twitter_sample.pkl",
        "sample_manifest": cache_dir / "twitter_sample_manifest.json",
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


def _eligible_record(record: object, language: str | None) -> bool:
    if not isinstance(record, dict):
        return False
    text = record.get("text")
    return (
        isinstance(text, str)
        and bool(text.strip())
        and (language is None or record.get("lang") == language)
    )


def _balance_label(record: dict, balance_by: str) -> str:
    """Return a stable categorical stratum for one eligible source record."""
    if balance_by == "retweet_count":
        try:
            value = float(record.get("retweet_count") or 0)
            value = max(0, int(value)) if np.isfinite(value) else 0
        except (TypeError, ValueError, OverflowError):
            value = 0
        if value == 0:
            return "0"
        if value == 1:
            return "1"
        if value <= 5:
            return "2–5"
        if value <= 20:
            return "6–20"
        if value <= 100:
            return "21–100"
        return ">100"
    if balance_by == "language":
        return str(record.get("lang") or "unknown")
    if balance_by == "retweet":
        return "retweet" if str(record.get("text", "")).startswith("RT @") else "original"
    if balance_by in {"hour", "weekday"}:
        timestamp = pd.to_datetime(
            record.get("created_at"), format="mixed", errors="coerce", utc=True
        )
        if pd.isna(timestamp):
            return "unknown"
        return f"{timestamp.hour:02d}:00" if balance_by == "hour" else timestamp.day_name()
    raise ValueError(
        f"Unsupported balance_by={balance_by!r}; choose one of {sorted(BALANCE_FIELDS)}."
    )


def _tweet_row(source_path: Path, record: dict) -> dict[str, object]:
    text = record["text"]
    entity_data = record.get("entities") or {}
    hashtag_entities = entity_data.get("hashtags", []) if isinstance(entity_data, dict) else []
    hashtags = [
        str(item.get("text", "")).lower()
        for item in hashtag_entities
        if isinstance(item, dict) and item.get("text")
    ]
    return {
        "source_batch": source_path.name,
        "tweet_id": str(record.get("id_str") or record.get("id") or ""),
        "created_at": record.get("created_at"),
        "lang": record.get("lang") or "unknown",
        "text": text.strip(),
        "hashtags": hashtags,
        "top_hashtag": hashtags[0] if hashtags else "(no hashtag)",
        "retweet_count": record.get("retweet_count"),
        "favorite_count": record.get("favorite_count"),
        "is_retweet": text.startswith("RT @"),
    }


def _allocate_balanced_quotas(counts: dict[str, int], target: int) -> dict[str, int]:
    """Allocate a quota as evenly as possible, respecting each stratum's size."""
    labels = sorted(counts)
    if not labels or target <= 0:
        return {}
    capacities = np.asarray([counts[label] for label in labels], dtype=np.int64)
    allocation = np.zeros(len(labels), dtype=np.int64)
    remaining = min(int(target), int(capacities.sum()))
    while remaining:
        active = np.flatnonzero(allocation < capacities)
        if not len(active):
            break
        share, remainder = divmod(remaining, len(active))
        additions = np.full(len(active), share, dtype=np.int64)
        additions[:remainder] += 1
        additions = np.minimum(additions, capacities[active] - allocation[active])
        used = int(additions.sum())
        allocation[active] += additions
        remaining -= used
    return {label: int(allocation[i]) for i, label in enumerate(labels) if allocation[i]}


def _count_balance_strata(
    source_path: Path,
    language: str | None,
    balance_by: str,
    count_stride: int,
    random_state: int,
) -> tuple[dict[str, int], int, int]:
    """Estimate label counts from a systematic sample of one archive."""
    import gzip
    from tqdm.auto import tqdm

    if count_stride < 1:
        raise ValueError("balance_count_stride must be at least 1.")
    counts: dict[str, int] = {}
    scanned = sampled = eligible_estimate = 0
    seed_material = f"{random_state}:{source_path.name}:balance-count".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little") % count_stride
    compressed_size = source_path.stat().st_size
    last_compressed_position = 0
    progress = tqdm(
        total=compressed_size,
        desc="Counting labels",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        mininterval=0.5,
        smoothing=0.05,
    )
    try:
        with source_path.open("rb") as compressed_file:
            with gzip.GzipFile(fileobj=compressed_file, mode="rb") as stream:
                for raw_line in stream:
                    scanned += 1
                    if scanned % 100_000 == 0:
                        position = compressed_file.tell()
                        progress.update(max(0, position - last_compressed_position))
                        last_compressed_position = position
                        progress.set_postfix(
                            parsed=f"{sampled:,}",
                            eligible_est=f"{eligible_estimate:,}",
                            labels=len(counts),
                        )
                    if (scanned - 1) % count_stride != offset:
                        continue
                    sampled += 1
                    line = raw_line.decode("utf-8", errors="replace")
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        record = None
                    if _eligible_record(record, language):
                        eligible_estimate += count_stride
                        label = _balance_label(record, balance_by)
                        counts[label] = counts.get(label, 0) + count_stride
            position = compressed_file.tell()
            progress.update(max(0, position - last_compressed_position))
            if progress.n < compressed_size:
                progress.update(compressed_size - progress.n)
    finally:
        progress.close()
    print(
        f"Estimated {eligible_estimate:,} eligible tweets from {sampled:,} of "
        f"{scanned:,} rows across {len(counts)} {balance_by} labels in {source_path.name}."
    )
    if len(counts) == 1:
        only_label, only_count = next(iter(counts.items()))
        print(
            f"Only one {balance_by} label was found ({only_label}: about {only_count:,}); "
            "this batch cannot be balanced across labels."
        )
    return counts, scanned, eligible_estimate


def _sample_balanced_batch(
    source_path: Path,
    allocations: dict[str, int],
    random_state: int,
    language: str | None,
    balance_by: str,
    count_stats: tuple[int, int],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Second pass: reservoir-sample each allocated stratum from one archive."""
    import gzip
    from tqdm.auto import tqdm

    seed_material = f"{random_state}:{source_path.name}:balanced:{balance_by}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    heaps: dict[str, list[tuple[float, int, dict]]] = {
        label: [] for label, quota in allocations.items() if quota > 0
    }
    scanned = eligible = reported = 0
    compressed_size = source_path.stat().st_size
    progress = tqdm(
        total=compressed_size,
        desc="Sampling balanced labels",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        mininterval=0.5,
        smoothing=0.05,
    )
    last_compressed_position = 0
    try:
        with source_path.open("rb") as compressed_file:
            with gzip.GzipFile(fileobj=compressed_file, mode="rb") as stream:
                for raw_line in stream:
                    scanned += 1
                    try:
                        record = json.loads(raw_line.decode("utf-8", errors="replace"))
                    except (json.JSONDecodeError, TypeError):
                        record = None
                    if _eligible_record(record, language):
                        eligible += 1
                        label = _balance_label(record, balance_by)
                        quota = allocations.get(label, 0)
                        if quota:
                            priority = float(rng.random())
                            item = (-priority, scanned, _tweet_row(source_path, record))
                            heap = heaps[label]
                            if len(heap) < quota:
                                heapq.heappush(heap, item)
                            elif priority < -heap[0][0]:
                                heapq.heapreplace(heap, item)
                    if scanned % 100_000 == 0:
                        position = compressed_file.tell()
                        progress.update(max(0, position - last_compressed_position))
                        last_compressed_position = position
                        reported = scanned
                        progress.set_postfix(scanned=f"{scanned:,}", eligible=f"{eligible:,}")
            position = compressed_file.tell()
            progress.update(max(0, position - last_compressed_position))
    finally:
        progress.close()

    selected = [item for heap in heaps.values() for item in heap]
    selected.sort(key=lambda item: item[1])
    tweets = pd.DataFrame([item[2] for item in selected])
    if tweets.empty:
        raise ValueError(
            f"No usable tweets found in {source_path.name} for language={language!r}."
        )
    sampled_by_label = {label: len(heap) for label, heap in heaps.items()}
    scanned_first, eligible_first = count_stats
    stats = {
        "source": _source_identity(source_path),
        "scanned_rows": scanned,
        "eligible_rows_seen": eligible,
        "sampled_rows": len(tweets),
        "balance_counts": {label: int(count) for label, count in allocations.items()},
        "balance_sampled": sampled_by_label,
        "count_pass_scanned_rows": scanned_first,
        "count_pass_eligible_estimate": eligible_first,
    }
    print(
        f"Batch complete: {source_path.name} — {scanned:,} scanned, "
        f"{eligible:,} eligible, {len(tweets):,} sampled across {len(sampled_by_label)} labels."
    )
    return tweets, stats


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
    balance_by: str,
    balance_count_stride: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    samples = []
    batch_stats = []
    # Explicit loop preserves the date/name order and makes batch boundaries
    # visible in both progress output and the cached source_batch column.
    if sample_strategy == "balanced":
        counted = {}
        for source_path in source_paths:
            counts, scanned, eligible = _count_balance_strata(
                source_path, language, balance_by, balance_count_stride, random_state
            )
            counted[source_path.name] = (counts, (scanned, eligible))
        for source_path in source_paths:
            counts, count_stats = counted[source_path.name]
            allocations = _allocate_balanced_quotas(
                counts, quotas[source_path.name]
            )
            batch, stats = _sample_balanced_batch(
                source_path,
                allocations,
                random_state,
                language,
                balance_by,
                count_stats,
            )
            stats["balance_counts_estimated"] = counts
            samples.append(batch)
            batch_stats.append(stats)
        tweets = pd.concat(samples, ignore_index=True)
        tweets.attrs["batch_stats"] = batch_stats
        return tweets, batch_stats

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
    sample_strategy: str = "balanced",
    balance_by: str = "retweet_count",
    balance_count_stride: int = 10,
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
    if sample_strategy not in {"first", "reservoir", "balanced"}:
        raise ValueError("sample_strategy must be 'first', 'reservoir', or 'balanced'.")
    if sample_strategy == "balanced" and balance_by not in BALANCE_FIELDS:
        raise ValueError(
            f"balance_by must be one of {sorted(BALANCE_FIELDS)} for balanced sampling."
        )
    if (
        isinstance(balance_count_stride, bool)
        or not isinstance(balance_count_stride, int)
        or balance_count_stride < 1
    ):
        raise ValueError("balance_count_stride must be a positive integer.")
    batch_order = [path.name for path in source_files]
    source_identities = [_source_identity(path) for path in source_files]
    quotas = _batch_quotas(batch_order, max_tweets)
    paths = _paths(data_dir)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(paths["manifest"])

    sample_signature = {
        "format_version": CACHE_FORMAT_VERSION,
        "sources": source_identities,
        "random_state": random_state,
        "language": language,
        "max_tweets": max_tweets,
        "sample_strategy": sample_strategy,
        "balance_by": balance_by if sample_strategy == "balanced" else None,
        "balance_count_stride": balance_count_stride if sample_strategy == "balanced" else None,
    }
    sample_manifest = _read_manifest(paths["sample_manifest"])
    sample_cache_hit = (
        not force_rebuild
        and sample_manifest is not None
        and sample_manifest.get("signature") == sample_signature
        and paths["sample"].is_file()
    )
    cached_sample = None
    if sample_cache_hit:
        try:
            cached_sample = pd.read_pickle(paths["sample"])
            sample_cache_hit = len(cached_sample) == sample_manifest.get("n_tweets")
        except (OSError, ValueError, EOFError):
            sample_cache_hit = False
            cached_sample = None

    matching = (
        not force_rebuild
        and manifest is not None
        and manifest.get("format_version") == CACHE_FORMAT_VERSION
        and manifest.get("sources") == source_identities
        and manifest.get("random_state") == random_state
        and manifest.get("language") == language
        and manifest.get("embedding_model") == embedding_model
        and manifest.get("sample_strategy") == sample_strategy
        and (
            sample_strategy != "balanced"
            or (
                manifest.get("balance_by", "retweet_count") == balance_by
                and manifest.get("balance_count_stride", 1) == balance_count_stride
            )
        )
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

    if (
        matching
        and manifest["max_tweets"] >= max_tweets
        and (sample_strategy != "balanced" or manifest["max_tweets"] == max_tweets)
    ):
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

    if sample_cache_hit:
        tweets = cached_sample
        batch_stats = sample_manifest.get("batches", [])
        print(
            f"Reusing cached {sample_strategy} sample ({len(tweets):,} tweets); "
            "archive scans skipped."
        )
    else:
        # A completed sample in the final cache can be reused when only the
        # embedding model changes. Older manifests without a count stride are
        # accepted because their stored sample is already complete.
        final_sample_matches = (
            not force_rebuild
            and manifest is not None
            and manifest.get("format_version") == CACHE_FORMAT_VERSION
            and manifest.get("sources") == source_identities
            and manifest.get("random_state") == random_state
            and manifest.get("language") == language
            and manifest.get("max_tweets") == max_tweets
            and manifest.get("sample_strategy") == sample_strategy
            and manifest.get("balance_by", None)
            == (balance_by if sample_strategy == "balanced" else None)
            and (
                sample_strategy != "balanced"
                or "balance_count_stride" not in manifest
                or manifest.get("balance_count_stride") == balance_count_stride
            )
            and paths["tweets"].is_file()
        )
        if final_sample_matches:
            try:
                tweets = pd.read_pickle(paths["tweets"])
                final_sample_matches = len(tweets) == manifest.get("n_tweets")
            except (OSError, ValueError, EOFError):
                final_sample_matches = False
        if final_sample_matches:
            batch_stats = manifest.get("batches", [])
            print(
                f"Reusing {len(tweets):,} sampled tweets from the existing cache; "
                "archive scans skipped."
            )
        else:
            tweets, batch_stats = _sample_batches(
                source_files,
                quotas,
                random_state,
                language,
                sample_strategy,
                balance_by,
                balance_count_stride,
            )

        sample_tmp = paths["directory"] / ".twitter_sample.tmp.pkl"
        sample_manifest_tmp = paths["directory"] / ".twitter_sample_manifest.tmp.json"
        tweets.to_pickle(sample_tmp)
        sample_manifest_tmp.write_text(json.dumps({
            "signature": sample_signature,
            "n_tweets": len(tweets),
            "batches": batch_stats,
        }, indent=2) + "\n")
        sample_tmp.replace(paths["sample"])
        sample_manifest_tmp.replace(paths["sample_manifest"])
    print(
        f"Embedding {len(tweets):,} tweets from {len(source_files)} ordered batches "
        f"with {embedding_model}…"
    )
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embedding_model)
    available_prompts = getattr(model, "prompts", {}) or {}
    if "Clustering" in available_prompts:
        prompt_name = "Clustering"
    elif "query" in available_prompts and getattr(model, "default_prompt_name", None) is None:
        prompt_name = "query"
    else:
        prompt_name = None
    if prompt_name is not None:
        print(f"Using embedding prompt: {prompt_name}")
    vectors = model.encode(
        tweets["text"].tolist(),
        prompt_name=prompt_name,
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
        "balance_by": balance_by if sample_strategy == "balanced" else None,
        "balance_count_stride": balance_count_stride if sample_strategy == "balanced" else None,
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
