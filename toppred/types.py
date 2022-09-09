# Imports
import numpy as np
from typing import List, Tuple, Union

# Definition of https://numpy.org/doc/stable/reference/generated/numpy.asarray.html
array_like = Union[
    List,
    List[List],
    List[Tuple],
    Tuple[List],
    Tuple[Tuple],
    np.ndarray,
]
