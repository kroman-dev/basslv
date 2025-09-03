import numpy as np

from abc import ABC, abstractmethod

from basslv.core.projectTyping import FloatVectorType, toNumpy


class GenericSolutionInterpolator(ABC):

    def __init__(self, x: FloatVectorType, y: FloatVectorType, tenor: float):
        self._x = x
        self._y = y
        self._tenor = tenor
        self._verifyCdfValues(self.y)
        self._buildInterpolator()

    @staticmethod
    def _verifyCdfValues(u: FloatVectorType):
        if isinstance(u, (list, np.ndarray)):
            if len(np.where(toNumpy(u) < 0.)[0]) > 0:
                raise ValueError("u contains negative values")
            elif len(np.where(toNumpy(u) > 1.)[0]) > 0:
                raise ValueError("u contains values bigger than 1.")
        else:
            TypeError('u must be FloatOrVector')

    @property
    def x(self) -> FloatVectorType:
        return self._x

    @property
    def y(self) -> FloatVectorType:
        return self._y

    @property
    def tenor(self) -> float:
        return self._tenor

    @abstractmethod
    def _buildInterpolator(self) -> None:
        pass

    @abstractmethod
    def _directCall(self, x: FloatVectorType) -> FloatVectorType:
        pass

    @abstractmethod
    def _inverseCall(self, u: FloatVectorType) -> FloatVectorType:
        pass

    def __call__(self, x: FloatVectorType) -> FloatVectorType:
        return self._directCall(x)

    def inverseCall(self, u: FloatVectorType) -> FloatVectorType:
        return self._inverseCall(u)
