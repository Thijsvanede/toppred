import numpy as np
import unittest
from toppred.predictions import top_predictions

class PredictionTest(unittest.TestCase):
    """Tests the functionality of the toppred.predictions module."""

    def test_prediction(self):
        """Test the correctness of normal predictions."""
        # Test case
        y_true = np.asarray([1, 2, 3, 4, 5, 4, 3, 2, 1])
        y_pred = np.asarray([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ])
        error_message = "Incorrectly computed top_predictions"

        # Loop over all predictions
        for top, prediction in top_predictions(y_true=y_true, y_pred=y_pred):
            if top == 0:
                self.assertTrue(np.all(
                    prediction == np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1])
                ), error_message)
            elif top == 1:
                self.assertTrue(np.all(
                    prediction == np.asarray([1, 2, 1, 1, 1, 1, 1, 2, 1])
                ), error_message)
            elif top == 2:
                self.assertTrue(np.all(
                    prediction == np.asarray([1, 2, 3, 1, 1, 1, 3, 2, 1])
                ), error_message)


    def test_multi_dim_y_true(self):
        """Test whether we receive an error when y_true is multi dimensional."""
        # Test case
        y_true = np.asarray([1, 2, 3, 4, 5, 4, 3, 2, 1])
        y_pred = np.asarray([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ])
        error_message = (
            "y_true should have been detected as having incorrect dimensions."
        )

        # Test case when y_true and y_pred are accidentally swapped
        with self.assertRaises(ValueError, msg=error_message):
            list(top_predictions(
                y_true = y_pred,
                y_pred = y_true,
            ))

        # Test case when y_true is None
        with self.assertRaises(ValueError, msg=error_message):
            list(top_predictions(
                y_true = None,
                y_pred = y_pred,
            ))

        # Test case when y_true is equal to y_pred
        with self.assertRaises(ValueError, msg=error_message):
            list(top_predictions(
                y_true = y_pred,
                y_pred = y_pred,
            ))

        # Test case when y_true is random 2D array
        with self.assertRaises(ValueError, msg=error_message):
            list(top_predictions(
                y_true = np.zeros((10, 6)),
                y_pred = y_pred,
            ))


    def test_incorrect_dim_y_true(self):
        """Test whether we receive an error when y_pred is of incorrect
            dimensions."""
        # Test case
        y_true = np.asarray([1, 2, 3, 4, 5, 4, 3, 2, 1])
        y_pred = np.asarray([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ])
        error_message = (
            "y_pred should have been detected as having incorrect dimensions."
        )

        # Test case when y_true and y_pred are accidentally swapped
        with self.assertRaises(ValueError, msg=error_message):
            list(top_predictions(
                y_true = y_pred,
                y_pred = y_true,
            ))

        # Test case when y_pred is None
        with self.assertRaises(ValueError, msg=error_message):
            list(top_predictions(
                y_true = y_true,
                y_pred = None,
            ))

        # Test case when y_pred is equal to y_true
        with self.assertRaises(ValueError, msg=error_message):
            list(top_predictions(
                y_true = y_true,
                y_pred = y_true,
            ))

        # Test case when y_true is random 2D array
        with self.assertRaises(ValueError, msg=error_message):
            list(top_predictions(
                y_true = y_true,
                y_pred = np.zeros(10),
            ))


    def test_nonmaching_dimensions(self):
        """Test whether we receive an error when y_pred and y_true shapes do not
            match."""
        # Test case
        y_true = np.asarray([1, 2, 3, 4, 5, 4, 3, 2, 1])
        y_pred = np.asarray([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ])
        error_message = (
            "y_pred should have been detected as having incorrect dimensions."
        )

        # Test case when y_true and y_pred are missing values
        with self.assertRaises(ValueError, msg=error_message):
            list(top_predictions(
                y_true = y_true[1:],
                y_pred = y_pred,
            ))
        with self.assertRaises(ValueError, msg=error_message):
            list(top_predictions(
                y_true = y_true,
                y_pred = y_pred[1:],
            ))
        

if __name__ == "__main__":
    unittest.main()
