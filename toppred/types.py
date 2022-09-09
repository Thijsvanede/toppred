# Imports
import numpy as np
from typing import List, Tuple, Union

# Definition of https://numpy.org/doc/stable/reference/generated/numpy.asarray.html
array_like_1d = Union[List, Tuple, np.ndarray]
array_like_2d = Union[
    List[array_like_1d],
    Tuple[array_like_1d],
    np.ndarray,
]
array_like = Union[
    array_like_1d,
    array_like_2d,
    List["array_like"],
    Tuple["array_like"],
]
