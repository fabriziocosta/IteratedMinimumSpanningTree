# Preprocessing raw inputs before IMSTE

`IteratedMinimumSpanningTreeEmbedder` normally accepts numeric feature matrices.
Attach a scikit-learn-compatible transformer to vectorize raw inputs first; the
resulting feature vectors are then used by IMSTE to build the embedding.

## Use a built-in modality preset

Install the optional text and image integration:

```bash
python -m pip install "imste[preprocessing]"
```

Install NSPPK separately for the graph preset:

```bash
python -m pip install git+https://github.com/Xfcosta/NSPPK.git
```

The text and image presets use Google's multimodal EmbeddingGemma 2 model via
Sentence Transformers. The first fit downloads the model weights if they are
not already cached. The graph preset uses NSPPK to vectorize a sequence of
NetworkX graphs. NSPPK is installed separately because it is distributed from
its upstream repository rather than PyPI.

```python
from imste import IteratedMinimumSpanningTreeEmbedder

mapper = IteratedMinimumSpanningTreeEmbedder(random_state=42)
mapper.preprocessor("text", batch_size=32, device="auto")
embedding = mapper.fit_transform([
    "A red fox crosses the snowy field.",
    "A fox runs through a winter landscape.",
    "A sailboat crosses a calm blue sea.",
    "Waves break along a sunny beach.",
])
```

`"text"` and `"image"` use `google/embeddinggemma-2` by default and accept
`model_name`, `batch_size`, and `device` options. Text inputs are strings or a
one-dimensional sequence of strings. Image inputs can be PIL images, image
paths, arrays, or a sequence/batch of images. The image preset requires the
`sentence-transformers[image]` dependencies included in the extra.

For graph-level embeddings, provide a sequence of NetworkX graphs:

```python
mapper = IteratedMinimumSpanningTreeEmbedder(random_state=42)
mapper.preprocessor("graph", radius=2, distance=3, nbits=12, parallel=False)
embedding = mapper.fit_transform(graphs)
```

Graph preset keyword arguments are passed to `nsppk.NSPPK`. Graph labels and
attributes are interpreted according to NSPPK's configuration.

## Use a custom transformer

Pass any scikit-learn-compatible transformer that implements `fit` and
`transform`. IMSTE clones the configured transformer and fits that clone as
part of `fit` or `fit_transform`.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

mapper = IteratedMinimumSpanningTreeEmbedder(random_state=42)
mapper.preprocessor(TfidfVectorizer(ngram_range=(1, 2)))
embedding = mapper.fit_transform(texts)
```

Sparse transformer output is accepted. When `y` is supplied to `fit`, it is
passed to the preprocessor; IMSTE uses labels to reweight graph-edge attraction
only when `label_strength` is positive. Otherwise labels are not used to
construct the embedding.

## Fit and transform behavior

Calling `.preprocessor(...)` configures the mapper and returns it, so it can be
chained with construction. `fit` stores the final coordinates in
`embedding_`; `fit_transform` returns them directly. The fitted transformer is
available as `preprocessor_`.

IMSTE is transductive: `transform` returns coordinates only for the original
fitted rows in their original order. It does not project new inputs. The
separate `tune()` method currently expects an already-vectorized numeric matrix
and does not run an attached preprocessor. For tuning, vectorize the inputs
separately and call `tune()` on a mapper without an attached preprocessor.

For complete text, image, graph, and custom-transformer examples, see
[`notebooks/preprocessing_demo.ipynb`](../notebooks/preprocessing_demo.ipynb).
