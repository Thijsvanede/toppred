.. _Metrics:

Metrics
=======
As a substitute for ``sklearn.metrics.classification_report`` we offer the :py:meth:`top_classification_report` method to produce a classification report containing the metrics for all top ``n`` predictions given as ``y_pred``.
See :ref:`usage` for usage examples.

.. automethod:: toppred.metrics.top_classification_report