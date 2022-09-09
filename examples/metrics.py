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