# Map new rows with InductiveIMSTE

`IteratedMinimumSpanningTreeEmbedder` is transductive: its `transform()` method
returns coordinates for the original fitted rows, in their original order. It
does not place new observations. `InductiveIMSTE` adds a learned MLP that maps
feature rows to approximate embedding coordinates.

Install the optional PyTorch dependency:

```bash
python -m pip install "imste[gpu]"
```

## Fit and project

The supervised example below assumes `X_train` and matching categorical labels
`y_train` are already prepared. For a continuous outcome, configure the
embedder with `target_strength` and pass the numeric training target as `y`.

```python
from imste import IteratedMinimumSpanningTreeEmbedder
from imste.inductive import InductiveIMSTE

model = InductiveIMSTE(
    embedder=IteratedMinimumSpanningTreeEmbedder(
        n_msts=10,
        label_strength=0.6,
        random_state=42,
    ),
    n_layers=6,
    layer_size=128,
    dropout=0.1,
    min_epochs=100,
    patience=20,
).fit(X_train, y=y_train)

Z_new = model.transform(X_new)
Z_reference = model.reference_embedding_
```

`X_train` and `X_new` must use the same features and column order. `fit()` first
fits IMSTE on all training rows, then trains an MLP to approximate those exact
coordinates. `reference_embedding_` stores the IMSTE coordinates;
`transform()` always returns MLP predictions, including for training rows.
Predictions can differ from the reference coordinates, and adding new rows does
not extend or re-optimize the fitted graph.
When the configured embedder has positive `label_strength` or
`target_strength`, pass the corresponding categorical labels or numeric targets
as `y` so its reference embedding uses the supervised edge weights.

## Training behavior and settings

The MLP standardizes its input features and target coordinates, reserves 10% of
the training rows for validation, and restores the checkpoint with the best
validation loss. The validation split controls MLP early stopping only; IMSTE
still uses all rows to create the reference embedding. The default projector
has six hidden layers with 128 units and dropout 0.1. `epochs`, `min_epochs`,
`patience`, `batch_size`, and `learning_rate` control its training. See the
complete :class:`~imste.inductive.InductiveIMSTE` parameter reference in the
API page.

The MLP must learn the geometry of the fitted layout from examples, so check
its predictions on held-out rows for your data before relying on them in a
downstream workflow. Install PyTorch through the `gpu` extra; the name reflects
that it also enables IMSTE's optional PyTorch/MPS optimizer, but
`InductiveIMSTE` can run with PyTorch on CPU. See the :doc:`api` for its
parameter reference.
