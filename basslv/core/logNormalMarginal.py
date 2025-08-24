import numpy as np

from scipy.stats import norm

from basslv.core.projectTyping import FloatOrVectorType, verifyFloat, toNumpy
from basslv.core.genericMarginal import GenericMarginal


EPS = np.finfo(float).eps


class LogNormalMarginal(GenericMarginal):

    def __init__(self, sigma: float, tenor: float):
        verifyFloat(sigma=sigma, tenor=tenor)
        super().__init__(tenor=tenor)
        # spot = 1. work in moneyness
        self._sigma = sigma

    @staticmethod
    def _verifyPositiveInput(x: FloatOrVectorType):
        if isinstance(x, (float, int)) and x <= 0.:
            raise ValueError("x must be bigger than 0.")
        elif isinstance(x, (list, np.ndarray)):
            if len(np.where(toNumpy(x) <= 0.)[0]) > 0:
                raise ValueError("x contains negative values")
        else:
            TypeError('x must be FloatOrVector')

    def getMappingFunction(self):
        mappingFunc = lambda t, w: np.exp(
            -0.5 * self._sigma ** 2 * t + self._sigma * w
        )
        return mappingFunc

    def _pdf(self, x):
        self._verifyPositiveInput(x)
        return 1 / (x * self._sigma * np.sqrt(self.tenor)) \
            * norm.pdf(
                (np.log(x) + 0.5 * self._sigma ** 2 * self.tenor) \
                / self._sigma / np.sqrt(self.tenor)
            )

    def _cdf(self, x: FloatOrVectorType) -> FloatOrVectorType:
        self._verifyPositiveInput(x)
        return norm.cdf(
            (np.log(x) + 0.5 * self._sigma ** 2 * self.tenor) \
            / self._sigma / np.sqrt(self.tenor)
        )

    def _inverseCdf(self, u: FloatOrVectorType) -> FloatOrVectorType:
        # Add np.clip else norm.ppf(1.) = infty
        _u = np.clip(u, EPS, 1 - EPS)
        return np.exp(
            - 0.5 * self._sigma ** 2 * self.tenor \
            + self._sigma * np.sqrt(self.tenor) * norm.ppf(_u)
        )

    def _integralOfInverseCdf(self, u: FloatOrVectorType) -> FloatOrVectorType:
        moneyness = self._inverseCdf(u)
        d_1 = (-np.log(moneyness) + 0.5 * self._sigma ** 2 * self.tenor) \
            / self._sigma / np.sqrt(self.tenor)
        return norm.cdf(-d_1)

    def _derivativeOfInverseCdf(self, u: FloatOrVectorType) -> FloatOrVectorType:
        # Gy = x
        Gy = self._inverseCdf(u)
        return 1 / self._pdf(Gy)

    def secondDerivativeOfInverseCdf(self, u: FloatOrVectorType) -> FloatOrVectorType:
        self._verifyPositiveInput(u)
        Gy = self._inverseCdf(u)
        normInverseCdf = (np.log(Gy) + self.tenor * self._sigma ** 2 / 2) \
                         / self._sigma / np.sqrt(self.tenor)
        Ay = 1 / norm.pdf(normInverseCdf)
        return self._sigma * np.sqrt(self.tenor) * Gy * Ay ** 2 \
            * (self._sigma * np.sqrt(self.tenor) + normInverseCdf)

    def __str__(self):
        return f"LogNormalMarginal_sigma={self._sigma}_tenor={self.tenor}"

    def __repr__(self):
        return self.__str__()
