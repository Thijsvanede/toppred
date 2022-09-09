# toppred
Extension to [sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) to allow metrics for classifiers that output a top `n` prediction.
Some classifiers output confidence levels for each class.
Oftentimes, you want to evaluate the performance of such classifiers assuming the correct prediction is the top `n` predictions with the highest confidence level.
This library serves as an extension to the functions provided by [sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) to allow for evaluating classifiers that do not output a single prediction per sample, but rather a range of top predictions per sample.

## Installation
The most straightforward way of installing `toppred` is via pip:
```bash
pip3 install toppred
```

### From source
To install this library from source, simply clone the repository:
```bash
git clone https://github.com/Thijsvanede/toppred.git
```

Next, ensure that all [dependencies](#Dependencies) have been installed:

Using the `requirements.txt` file:
```bash
pip3 install -r /path/to/toppred/requirements.txt
```

Installing libraries independently:
```bash
pip3 install numpy pandas sklearn
```

Finally, install the library from source:
```bash
pip3 install -e /path/to/toppred/
```

## Usage
TODO

### Examples
For all directly executed examples see the `examples/` directory.

#### Top classification report
```python
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
```

#### Metrics
```python
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
```

## API
This library offers two main functions:
 * [top_classification_report()](#top_classification_report)
 * [top_predictions()](#top_classification_report)

### top_classification_report()
Create a classification report for a y_pred containing multiple top predictions. This function follows the same API as ``sklearn.metrics.classification_report`` with the exception that:
1. ``y_pred`` should be given as a 2D array instead of a 1D array.
2. If ``output_dict`` is ``True``, the output dictionary consists of a dictionary where ``key`` is 0-indexed top prediction and ``value`` is the ``sklearn.metrics.classification_report`` output dictionary for that top prediction.

#### Parameters
* `y_true` : `array_like_1d of shape=(n_samples,)`
  Ground truth (correct) target values.
* `y_pred` : `array_like_2d of shape=(n_samples, n_predictions)`
  Estimated targets as returned by a classifier. Each column y_pred[:, i] indicates the i-th most likely prediction (0-indexed) for the given sample.
* `labels` : `Optional[array_like_1d], default = None`
    Optional list of label indices to include in the report.
* `target_names` : `Optional[List[str]] = None`
  Optional display names matching the labels (same order).
* `sample_weight` : `Optional[array_like_1d], default = None`
  Sample weights.
* `digits` : `int, default = 2`
  Number of digits for formatting output floating point values. When ``output_dict`` is ``True``, this will be ignored and the returned values will not be rounded.
* `output_dict` : `bool, default = False`
  If True, return output as dict.
* `zero_division` : `Union[Literal["warn"], 0, 1], default = "warn"`
  Sets the value to return when there is a zero division. If set to “warn”, this acts as 0, but warnings are also raised.

#### Returns
* `report` : `Union[str, dict]`
  Text summary of the precision, recall, F1 score for each class. Dictionary returned if output_dict is True.

  The reported averages include macro average (averaging the unweighted mean per label), weighted average (averaging the support-weighted mean per label), and sample average (only for multilabel classification). Micro average (averaging the total true positives, false negatives and false positives) is only shown for multi-label or multi-class with a subset of classes, because it corresponds to accuracy otherwise and would be the same for all metrics. See also precision_recall_fscore_support for more details on averages.

  Note that in binary classification, recall of the positive class is also known as “sensitivity”; recall of the negative class is “specificity”.

### top_predictions()
Iterates over the top predictions.

#### Parameters
* `y_true` : `array_like_1d of shape=(n_samples,)`
  True labels corresponding to samples.

* `y_pred` : `array_like_2d of shape=(n_samples, n_predictions)`
  Predicted labels for samples. Each column y_pred[:, i] indicates the i-th most likely prediction (0-indexed) for the given sample.

#### Yields
* `i` : `int`
  Top in the i most likely predictions (0-indexed).

* `y_pred` : `np.array of shape=(n_samples,)`
  Prediction if the correct answer would be in the top i most likely predictions (0-indexed).
