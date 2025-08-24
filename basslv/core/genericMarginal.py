import numpy as np

from abc import ABC, abstractmethod

from basslv.core.projectTyping import FloatOrVectorType, verifyFloat, toNumpy

EPS = np.finfo(float).eps


class GenericMarginal(ABC):

    def __init__(self, tenor: float):
        verifyFloat(tenor=tenor)
        self.__tenor = tenor

    @property
    def tenor(self) -> float:
        return self.__tenor

    @staticmethod
    def _verifyCdfValues(u: FloatOrVectorType):
        if isinstance(u, (float, int)) and (u < 0. or u > 1.):
            raise ValueError(f"u must be in [0., 1.], got: {u}")
        elif isinstance(u, (list, np.ndarray)):
            if len(np.where(toNumpy(u) < 0.)[0]) > 0:
                raise ValueError("u contains negative values")
            elif len(np.where(toNumpy(u) > 1.)[0]) > 0:
                raise ValueError("u contains values bigger than 1.")
        else:
            TypeError('u must be FloatOrVector')

    def cdf(self, x: FloatOrVectorType) -> FloatOrVectorType:
        # return np.clip(self._cdf(x), a_min=EPS, a_max=1 - EPS)
        result = self._cdf(x)
        self._verifyCdfValues(result)
        return result

    def integralOfInverseCdf(self, u: FloatOrVectorType) -> FloatOrVectorType:
        self._verifyCdfValues(u)
        return self._integralOfInverseCdf(u)

    def inverseCdf(self, u: FloatOrVectorType) -> FloatOrVectorType:
        self._verifyCdfValues(u)
        # u = np.clip(u, a_min=EPS, a_max=1 - EPS)
        return self._inverseCdf(u)

    def derivativeOfInverseCdf(self, u: FloatOrVectorType) -> FloatOrVectorType:
        return self._derivativeOfInverseCdf(u)

    @abstractmethod
    def _cdf(self, x: FloatOrVectorType) -> FloatOrVectorType:
        pass

    @abstractmethod
    def _inverseCdf(self, u: FloatOrVectorType) -> FloatOrVectorType:
        pass

    @abstractmethod
    def _integralOfInverseCdf(self, u: FloatOrVectorType) -> FloatOrVectorType:
        """For eq. (12). It equals to digital asset-or-nothing put"""
        pass

    @abstractmethod
    def _derivativeOfInverseCdf(self, u: FloatOrVectorType) -> FloatOrVectorType:
        """For eq. (12)."""
        pass
