.. _usage:

Usage
=====
The main usage of this library is to compute metrics over the top-n predictions of a given classifier.
In the normal case, a classifier gives a single prediction per sample, often in the form of an array:

.. code:: python

    import numpy as np

    y_true = np.asarray([0, 1, 2, 1, 0]) # True labels
    y_pred = np.asarray([0, 1, 1, 0, 0]) # Predicted labels

However, a classifier could also return the top n most likely predictions, e.g.:

.. code:: python

    import numpy as np

    y_true = np.asarray([0, 1, 2, 1, 0]) # True labels
    y_pred = np.asarray([                # Predicted labels
        [0, 1, 2],
        [1, 0, 2],
        [1, 2, 0],
        [0, 1, 2],
        [0, 1, 2],
    ])

In this case, we would like to be able to compute the performance when:

 - The correct prediction is the most likely prediction (``y_pred[:, 0]``)
 - The correct prediction is in the top 2 most likely predictions (``y_pred[:, :2]``)
 - The correct prediction is in the top 3 most likely predictions (``y_pred[:, :3]``)

For this purpose, ``toppred`` provides two functions:

    :py:meth:`top_classification_report`, produces a classification report similar to ``sklearn.metrics.classification_report``. See usage example.
    :py:meth:`top_predictions`, provides an iterator over the top n most likely predictions and can be combined with many ``sklearn.metrics``. See usage example.

See the :ref:`Reference` API for detailed information, and our usage examples below.

Examples
^^^^^^^^

top_classification_report
-------------------------
One of the main functions of ``toppred`` is to create a :py:meth:`top_classification_report` similar to scikit-learn's `classification_report`_.
When we have a numpy array ``y_pred`` shape ``(n_samples, n_top_predictions)``, we can use it as follows to generate a :py:meth:`top_classification_report`.

.. _`classification_report`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

.. code:: python

    # Imports
    import numpy as np
    from toppred.metrics import top_classification_report

    # Define inputs
    y_true = np.asarray([1, 2, 3, 2, 1]) # Ground truth values
    y_pred = np.asarray([                # Sample prediction values
        [1, 2, 3],                       # We have a top 3 predictions for each
        [2, 1, 3],                       # input sample. I.e., 
        [1, 2, 3],                       # y_true.shape[0] == y_pred.shape[0].
        [3, 1, 2],
        [1, 2, 3],
    ])

    # Compute and print top classification report
    print(top_classification_report(
        y_true = y_true,
        y_pred = y_pred,
        labels = [0, 1, 2, 3],                 # Optional, defaults to None
        target_names = ['N/A', '1', '2', '3'], # Optional, defaults to None
        sample_weight = [1, 2, 3, 4, 5],       # Optional, defaults to None
        digits = 4,                            # Optional, int, defaults to 2
        output_dict = False,                   # Optional, If true, return as dictionary
        zero_division = "warn",                # Optional, defaults to "warn"
    ))

metrics
-------
Besides the :py:meth:`top_classification_report`, you may want to compute other metrics for the top ``n`` results in your prediction.
To this end, we provide the :py:meth:`top_predictions` which takes the ``y_pred`` of shape ``(n_samples, n_top_predictions)`` and yields a ``y_pred`` of shape ``(n_samples,)`` for each top ``i`` predictions that is compatible with most ``sklearn.metrics`` functions.

.. code:: python

    # Imports
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from toppred.predictions import top_predictions

    # Define inputs
    y_true = np.asarray([1, 2, 3, 2, 1]) # Ground truth values
    y_pred = np.asarray([                # Sample prediction values
        [1, 2, 3],                       # We have a top 3 predictions for each
        [2, 1, 3],                       # input sample. I.e., 
        [1, 2, 3],                       # y_true.shape[0] == y_pred.shape[0].
        [3, 1, 2],
        [1, 2, 3],
    ])

    # Use top_predictions to generate a y_pred value that is correct if the
    # prediction is in the top n predictions
    for top, prediction in top_predictions(y_true, y_pred):
        # Compute common metrics
        accuracy  = accuracy_score (y_true, prediction)
        precision = precision_score(y_true, prediction, average='macro')
        recall    = recall_score   (y_true, prediction, average='macro')
        f1        = f1_score       (y_true, prediction, average='macro')

        print(f"Metrics top {top+1} predictions:")
        print(f"    Accuracy : {accuracy}")
        print(f"    Precision: {precision}")
        print(f"    Recall   : {recall}")
        print(f"    F1_score : {f1}")
        print()

Probabilities
^^^^^^^^^^^^^
Some classifiers, including many neural networks do not give direct top n results, but instead provide a probability (confidence level) for each class, producing an output such as:

.. code:: python

    import numpy as np

    y_true = np.asarray([0, 1, 2, 1, 0]) # True labels
    y_prob = np.asarray([ # Prediction probability
        [0.7, 0.2, 0.1],  # class 0 -> 0.7, class 1 -> 0.2, class 2 -> 0.1
        [0.2, 0.7, 0.1],  # etc.
        [0.1, 0.7, 0.2],
        [0.8, 0.1, 0.1],
        [0.7, 0.2, 0.1],
    ])

In those cases, we can obtain a prediction for the top n most likely values:

.. code:: python

    # Get top n most likely values
    n = 3

    # Example: y_prob is numpy array
    y_pred = np.argsort(y_prob, axis=1)[:, -n:]

    # Example: y_prob is pytorch Tensor
    y_pred = torch.topk(y_prob, n).indices.cpu().numpy()

This results in the prediction array:

.. code:: python

    array([[0, 1, 2],
           [1, 0, 2],
           [1, 2, 0],
           [0, 1, 2],
           [0, 1, 2]])