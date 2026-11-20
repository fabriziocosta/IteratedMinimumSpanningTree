API reference
=============

Estimators
----------

.. currentmodule:: imste

.. autoclass:: IteratedMinimumSpanningTreeEmbedder
   :members: fit, fit_transform, transform, fit_graph, fit_graph_transform, preprocessor, tune
   :show-inheritance:

.. autoclass:: imste.inductive.InductiveIMSTE
   :members: fit, transform, fit_transform
   :show-inheritance:

Functions
---------

.. autofunction:: fit_graph_transform

.. autofunction:: spatial_coverage_score

Differentiable losses
---------------------

These helpers operate on PyTorch tensors and require the ``gpu`` extra.

.. autofunction:: imste.losses.log_attraction_loss

.. autofunction:: imste.losses.logistic_repulsion_loss
