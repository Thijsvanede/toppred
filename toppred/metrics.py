# Imports
from sklearn.metrics import classification_report
from toppred.predictions import top_predictions
from toppred.types import array_like_1d, array_like_2d
from toppred.utils import reports2string
from typing import List, Literal, Optional, Union

def top_classification_report(
        y_true       : array_like_1d,
        y_pred       : array_like_2d,
        labels       : Optional[array_like_1d] = None,
        target_names : Optional[List[str]] = None,
        sample_weight: Optional[array_like_1d] = None,
        digits       : int = 2,
        output_dict  : bool = False,
        zero_division: Literal["warn", 0, 1] = "warn",
    ) -> Union[str, dict]:
    """Create a classification report for a y_pred containing multiple top
        predictions. This function follows the same API as
        ``sklearn.metrics.classification_report`` with the exception that:
        
        1. ``y_pred`` should be given as a 2D array instead of a 1D array.
        2. If ``output_dict`` is ``True``, the output dictionary consists of a
        dictionary where ``key`` is 0-indexed top prediction and ``value`` is
        the ``sklearn.metrics.classification_report`` output dictionary for that
        top prediction.
        
        Parameters
        ----------
        y_true : array_like_1d of shape=(n_samples,)
            Ground truth (correct) target values.

        y_pred : array_like_2d of shape=(n_samples, n_predictions)
            Estimated targets as returned by a classifier. Each column
            y_pred[:, i] indicates the i-th most likely prediction (0-indexed)
            for the given sample.

        labels : Optional[array_like_1d], default = None
            Optional list of label indices to include in the report.

        target_names : Optional[List[str]] = None
            Optional display names matching the labels (same order).

        sample_weight : Optional[array_like_1d], default = None
            Sample weights.

        digits : int, default = 2
            Number of digits for formatting output floating point values. When
            ``output_dict`` is ``True``, this will be ignored and the returned
            values will not be rounded.

        output_dict : bool, default = False
            If True, return output as dict.

        zero_division : Union[Literal["warn"], 0, 1], default = "warn"
            Sets the value to return when there is a zero division. If set to
            “warn”, this acts as 0, but warnings are also raised.
        
        Returns
        -------
        report : Union[str, dict]
            Text summary of the precision, recall, F1 score for each class.
            Dictionary returned if output_dict is True.

            The reported averages include macro average (averaging the
            unweighted mean per label), weighted average (averaging the support-
            weighted mean per label), and sample average (only for multilabel
            classification). Micro average (averaging the total true positives,
            false negatives and false positives) is only shown for multi-label
            or multi-class with a subset of classes, because it corresponds to
            accuracy otherwise and would be the same for all metrics. See also
            precision_recall_fscore_support for more details on averages.

            Note that in binary classification, recall of the positive class is
            also known as “sensitivity”; recall of the negative class is
            “specificity”.
        """
    # Create dictionary_based classification reports for each top prediction
    reports = [
        classification_report(
            y_true        = y_true,
            y_pred        = y_pred_,
            labels        = labels,
            target_names  = target_names,
            sample_weight = sample_weight,
            digits        = digits,
            output_dict   = True,
            zero_division = zero_division,
        )
        for _, y_pred_ in top_predictions(y_true=y_true, y_pred=y_pred)
    ]

    # Return report as dictionary, if necessary
    if output_dict:
        return {i: report for i, report in enumerate(reports)}
    # Otherwise return reports as string
    else:
        return reports2string(reports, digits)
