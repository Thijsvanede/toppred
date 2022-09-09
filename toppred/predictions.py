from typing import Iterable
import numpy as np
from toppred.types import array_like

def top_predictions(
        y_true: array_like,
        y_pred: array_like,
    ) -> Iterable[np.ndarray]:
    """Iterates over the top predictions.

        Parameters
        ----------
        y_true : array_like of shape=(n_samples,)
            True labels corresponding to samples.

        y_pred : array_like of shape=(n_samples, n_predictions)
            Predicted labels for samples. Each column y_pred[:, i] indicates the
            i-th most likely prediction (0-indexed) for the given sample.

        Yields
        ------
        i : int
            Top in the i most likely predictions (0-indexed).

        y_pred : np.array of shape=(n_samples,)
            Prediction if the correct answer would be in the top i most likely 
            predictions (0-indexed).
        """
    # Cast input to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Perform checks
    if y_true.ndim != 1:
        raise ValueError(
            f"y_true should be a 1-D array, but was of shape '{y_true.shape}'."
        )
    if y_pred.ndim != 2:
        raise ValueError(
            f"y_pred should be a 2-D array, but was of shape '{y_pred.shape}'. "
            "Reshape your prediction either using array.reshape(-1, 1) if your "
            "data only contains a top 1 prediction or array.reshape(1, -1) if "
            "it contains a prediction for a single sample."
        )
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Number of samples in y_true ({y_true.shape[0]}) did not match "
            f"number of samples in y_pred ({y_pred.shape[0]}). Please make "
            "sure that y_pred is in the shape of (n_samples, n_predictions)."
        )

    # Initialise result
    result = y_pred[:, 0]

    # Loop over top predictions
    for top in range(y_pred.shape[1]):
        # Get correct prediction mask
        correct = y_pred[:, top] == y_true

        # Adjust correct predictions
        result[correct] = y_pred[correct, top]

        # Yield result
        yield top, result
