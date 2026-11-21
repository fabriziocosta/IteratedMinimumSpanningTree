IMSTE documentation
===================

Iterated Minimum Spanning Tree Embedding (IMSTE) learns a low-dimensional
layout from numeric feature rows or a supplied sparse distance graph. Install
the core package with ``python -m pip install imste``. This short example uses
scikit-learn's built-in digits dataset and does not download data:

.. code-block:: python

   from sklearn.datasets import load_digits
   from sklearn.preprocessing import StandardScaler
   from imste import IteratedMinimumSpanningTreeEmbedder

   digits = load_digits()
   X = StandardScaler().fit_transform(digits.data)
   embedding = IteratedMinimumSpanningTreeEmbedder(
       random_state=42,
   ).fit_transform(X)
   print(embedding.shape)  # (1797, 2)

With the default ``label_strength=0``, the coordinates are unsupervised; digit
labels are available separately for coloring or evaluation. Start with the
:doc:`getting-started` guide for a
plotting example and notes on scaling and graph construction.

Choose a path
-------------

* **Numeric feature rows:** :doc:`getting-started`
* **New rows after fitting:** :doc:`inductive-guide`
* **Text, images, or graph inputs:** :doc:`preprocessing-guide`
* **A supplied sparse distance graph:** see :doc:`api`
* **Faster initial layouts:** :doc:`multilevel-initialization`
* **Parameter search and visualization:** :doc:`tuning-guide` and
  :doc:`visualization-guide`

The core estimator is transductive: it learns coordinates for the rows used
to fit it. ``InductiveIMSTE`` adds an approximate MLP mapping for new rows.
The graph builder and optimizer are separate choices: approximate FAMST is the
default, exact Prim is available, and CPU optimization works without PyTorch.

.. toctree::
   :maxdepth: 2
   :caption: Get started

   getting-started
   inductive-guide

.. toctree::
   :maxdepth: 2
   :caption: Guides

   preprocessing-guide
   multilevel-initialization
   tuning-guide
   visualization-guide

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   whitepaper

Source code: `GitHub <https://github.com/fabriziocosta/IteratedMinimumSpanningTree>`_.
