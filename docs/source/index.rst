Welcome to toppred's documentation!
===================================
Extension to `sklearn.metrics`_ to allow metrics for classifiers that output a top ``n`` prediction.
Some classifiers output confidence levels for each class.
Oftentimes, you want to evaluate the performance of such classifiers assuming the correct prediction is the top ``n`` predictions with the highest confidence level.
This library serves as an extension to the functions provided by `sklearn.metrics`_ to allow for evaluating classifiers that do not output a single prediction per sample, but rather a range of top predictions per sample.

.. _`sklearn.metrics`: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   reference/reference
   contributors
   license
