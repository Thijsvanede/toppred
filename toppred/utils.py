# Imports
import numpy as np
import pandas as pd
from typing import List

def reports2string(reports: List[dict], digits: int = 2) -> str:
    """Convert a list of classification report dictionaries to a string.
    
        Parameters
        ----------
        reports : List[dict]
            List of classification reports to convert to string.
            Should be produced with ``sklearn.metrics.classification_report``
            where ``output_dict = True``.

        digits : int, default = 2
            Number of digits of precision to use.
            
        Returns
        -------
        report : str
            String representation of list of classification reports.
        """
    # Initialise dataframes
    dataframes = list()

    # Loop over all reports
    for top, report in enumerate(reports):
        # Initialise labels and metrics
        labels  = list()
        metrics = dict()

        # Collect all labels and metrics in report
        for label, performance in report.items():
            # Add space before micro avg
            if label == 'micro avg':
                labels.append('')
                for key in metrics:
                    metrics[key].append('')

            # Add label
            labels.append(label)
            # Add all metrics
            for metric, value in performance.items():

                # Set precision
                if metric != 'support':
                    # Set MultiIndex
                    metric = (f"Top {top+1}", metric, '')

                    # Set precision value
                    value = f"{value:.{digits}f}"

                    # Add metric
                    if metric not in metrics:
                        metrics[metric] = list()
                    metrics[metric].append(value)

            if top == len(reports) - 1:
                if ('', 'support', '') not in metrics:
                    metrics[('', 'support', '')] = list()
                metrics[('', 'support', '')].append(performance['support'])

        # Transform to dataframe
        dataframes.append(pd.DataFrame(metrics, index=labels))

    # Merge dataframes
    dataframe = pd.concat(dataframes, axis=1)

    # Style dataframe
    dataframe.style.set_properties(**{
        'text-align': 'right'
    })

    # Return dataframe as string
    return f"\n{dataframe}\n"
