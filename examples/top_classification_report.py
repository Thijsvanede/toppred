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