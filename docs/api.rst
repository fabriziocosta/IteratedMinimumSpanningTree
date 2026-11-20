API reference
=============

Estimators
----------

.. currentmodule:: imste

.. autoclass:: IteratedMinimumSpanningTreeEmbedder
   :members: fit, fit_transform, transform, fit_graph, fit_graph_transform, preprocessor, tune
   :show-inheritance:

.. py:class:: InductiveIMSTE

   Available from ``imste.inductive`` when PyTorch is installed. Install the
   optional dependency with ``python -m pip install "imste[gpu]"``. This
   estimator fits a transductive IMSTE embedding, then learns an MLP to map
   input feature rows to approximate embedding coordinates. Its
   ``reference_embedding_`` attribute holds the exact IMSTE coordinates.

   See the `README's inductive estimator guide
   <https://github.com/fabriziocosta/IteratedMinimumSpanningTree#inductive-map-new-rows-with-an-mlp>`_
   for the complete usage example and training behavior.

Functions
---------

.. autofunction:: fit_graph_transform

.. autofunction:: spatial_coverage_score

Differentiable losses
---------------------

These helpers operate on PyTorch tensors and require the ``gpu`` extra.

.. autofunction:: imste.losses.log_attraction_loss

.. autofunction:: imste.losses.logistic_repulsion_loss
