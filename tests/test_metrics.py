import numpy as np
import unittest
from toppred.metrics import top_classification_report

class PredictionTest(unittest.TestCase):
    """Tests the functionality of the toppred.metrics module."""

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

        # Generate classification report
        report = top_classification_report(
            y_true = y_true,
            y_pred = y_pred,
            labels = [0, 1, 2, 3, 4, 5, 6],
            target_names = ['N/A', 'one', 'two', 'three', 'four', 'five', 'six'],
            output_dict = True,
            digits = 4,
            zero_division = 0,
        )

        # Perform checks
        self.assertEqual(len(report), 3, "Not all top preds were returned")
        # Checks on individual reports
        for top, subreport in report.items():
            # Ensure all values are present in each report
            self.assertEqual(
                set(subreport),
                {'N/A', 'one', 'two', 'three', 'four', 'five', 'six'} | 
                {'micro avg', 'weighted avg', 'macro avg'},
                "Not all samples were present in each report",
            )

            # Check equivalence
            for label, metrics in subreport.items():
                # Check if metrics contain correct keys
                self.assertEqual(
                    set(metrics),
                    {'precision', 'recall', 'f1-score', 'support'},
                    "Not all required metrics are present in subreports."
                )

                # Check if support is equivalent for all metrics
                self.assertEqual(
                    metrics['support'],
                    report[0][label]['support'],
                    "Different support among top predictions."
                )

                # Check if metrics are strictly increasing
                if top != 0:
                    for metric, value in metrics.items():
                        # Skip support metric
                        if metric == 'support': continue
                        self.assertLessEqual(
                            report[top-1][label][metric],
                            value,
                            'Values should be strictly increasing'
                        )


if __name__ == "__main__":
    unittest.main()
