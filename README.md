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

## Documentation
We provide an extensive documentation including installation instructions and reference at [toppred.readthedocs.io](https://toppred.readthedocs.io).