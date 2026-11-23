# Prepare and explore Twitter batches

The repository includes notebooks for preparing ordered Twitter archive batches
and comparing UMAP with a retweet-count-shaped IMSTE layout. The archive files
and generated vectors are local data; they are not included in the source
package or this documentation.

## Add the archive batches

Place one or more Twitter JSONL.GZ archives in `data/twitter/`. The preparation
notebook discovers `*.json.gz` files and processes them in filename order. It
splits `MAX_TWEETS` evenly across the discovered batches, assigning any
remainder to earlier filenames. It retains tweet text, batch name, timestamp,
language, hashtags, retweet count, favorite count, and retweet status, then
encodes text with `google/embeddinggemma-2`.

Open [`notebooks/dataset_setup/prepare_twitter_dataset.ipynb`](../notebooks/dataset_setup/prepare_twitter_dataset.ipynb)
and run its cells. Install the optional notebook and model dependencies from
the repository root, as the notebook's first setup cells do. The first
embedding run downloads the model weights. The output is stored under
`data/twitter/cache/` and reused when its source and configuration manifest
matches. Set `FORCE_REBUILD = True` to replace that cache.

## Choose the sampling strategy

`SAMPLE_STRATEGY = "first"` reads only until each batch quota is filled. It is
useful for a quick preview, but the earliest tweets may have identical target
values (for example, all-zero retweet counts). `SAMPLE_STRATEGY = "reservoir"`
selects a random sample from each full archive and therefore scans every source
row. The preparation notebook currently selects reservoir sampling to support
the retweet regression workflow. Choose `MAX_TWEETS` with available memory in
mind: each EmbeddingGemma vector has 3,072 float32 features, in addition to the
retained tweet data.

## Tune and view the layout

After preparing the vectors, open [`notebooks/twitter_exp.ipynb`](../notebooks/twitter_exp.ipynb).
It uses `log1p(retweet_count)` as the continuous target. IMSTE's
`target_strength` and `target_bandwidth` tune how target similarity changes
attraction between feature-graph neighbors. The notebook searches parameters
with five-fold regression tuning, presents the best settings, and applies any
values in `IMSTE_OVERRIDES` before fitting the final layout. Its color selector
can display batch, language, posting time, retweet status/count, hashtags, or
engagement.

The regression layout requires a non-constant target. If the selected cache
has constant retweet counts, use reservoir sampling or prepare a sample with
more target variation, then reload the visualization notebook.
