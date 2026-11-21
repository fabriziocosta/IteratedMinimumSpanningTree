API reference
=============

Estimators
----------

.. currentmodule:: imste

.. autoclass:: IteratedMinimumSpanningTreeEmbedder
   :members: fit, fit_transform, transform, outliers, fit_graph, fit_graph_transform, preprocessor, tune
   :show-inheritance:

.. py:class:: InductiveIMSTE

   Available from ``imste.inductive`` when PyTorch is installed. Install the
   optional dependency with ``python -m pip install "imste[gpu]"``. This
   estimator fits a transductive IMSTE embedding, then learns an MLP to map
   input feature rows to approximate embedding coordinates. Its
   ``reference_embedding_`` attribute holds the exact IMSTE coordinates;
   ``transform`` always returns MLP predictions, including for training rows.
   See the :doc:`inductive-guide` for a complete example and training details.

   Parameters
   ^^^^^^^^^^

   * ``embedder``: transductive ``IteratedMinimumSpanningTreeEmbedder``
     instance, or ``None`` to use its defaults.
   * ``n_layers=6``, ``layer_size=128``, ``dropout=0.1``: MLP architecture.
   * ``epochs=200``, ``min_epochs=100``, ``patience=20``: early-stopping
     training budget.
   * ``batch_size=256``, ``learning_rate=0.001``: MLP training settings.

Functions
---------

.. autofunction:: fit_graph_transform

.. autofunction:: spatial_coverage_score

.. autofunction:: detect_outliers

.. autoclass:: OutlierDetectionResult
   :members:

Differentiable losses
---------------------

These helpers operate on PyTorch tensors and require the ``gpu`` extra.

.. autofunction:: imste.losses.log_attraction_loss

.. autofunction:: imste.losses.logistic_repulsion_loss
