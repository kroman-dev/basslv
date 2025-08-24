import numpy as np

from typing import Union, Any, List
from numpy.typing import NDArray

FloatOrVectorType = Union[float, List[float], NDArray[float]]
FloatVectorType = Union[List[float], NDArray[float]]


def toNumpy(x: Any) -> NDArray:
    if isinstance(x, np.ndarray):
        return x
    return np.array([x]) if isNumber(x) else np.array(x)


def isNumber(x: Any) -> bool:
    return isinstance(x, int) or isinstance(x, float) or isinstance(x, complex)


def verifyFloat(**kwargs):
    for param_name, param_value in kwargs.items():
        if not isinstance(param_value, (float, int)):
            raise TypeError(
                f"'{param_name}' must be float, got {type(param_value).__name__}"
            )

