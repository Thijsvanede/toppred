.. _Predictions:

Predictions
===========
When you want to compute metrics not supported in this library, we provide an iterator that yields ``y_pred`` arrays for each top ``i`` prediction that are compatible with ``sklearn.metrics``.
See :ref:`usage` for usage examples.

.. automethod:: toppred.predictions.top_predictions