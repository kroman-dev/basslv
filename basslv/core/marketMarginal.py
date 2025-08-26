import numpy as np

from scipy.interpolate import PchipInterpolator

from basslv.core.genericMarginal import GenericMarginal
from basslv.core.projectTyping import FloatOrVectorType


EPS = np.finfo(np.float64).eps


class MarketMarginal(GenericMarginal):

    def __init__(
            self,
            strikes: FloatOrVectorType,
            callPrices: FloatOrVectorType,
            tenor: float
    ):
        """
             F_{mu_i} = 1 + \partial_K Call(T_i; K) [1]
        """
        super().__init__(tenor=tenor)
        self._strikes = strikes
        self._callPrices = callPrices
        self._cdfValues = 1 + np.gradient(self._callPrices, self._strikes)

        if abs(self._cdfValues[-1] - (1. - EPS)) > EPS:
            # TODO add smart solution
            xMax = self._strikes[-1] + 0.1 * self._strikes[-1]
            xMin = self._strikes[0] - 0.1 * self._strikes[0]

            if xMin < 0.:
                raise ValueError('Fail augment the interpolation domain')

            self._strikes = np.concatenate([[xMin], self._strikes, [xMax]])
            self._cdfValues = np.concatenate([[0. + EPS], self._cdfValues, [1. - EPS]])

        self._interpolator = PchipInterpolator(self._strikes, self._cdfValues, extrapolate=False)
        self._inverseInterpolator = PchipInterpolator(self._cdfValues, self._strikes, extrapolate=False)

    def _derivativeOfInverseCdf(self, u: FloatOrVectorType) -> FloatOrVectorType:
        _u = np.clip(u, EPS, 1 - EPS)
        return self._inverseInterpolator.derivative()(_u)

    def _integralOfInverseCdf(self, u: FloatOrVectorType) -> FloatOrVectorType:
        _u = np.clip(u, EPS, 1 - EPS)
        return self._inverseInterpolator.antiderivative()(_u)

    def _inverseCdf(self, u: FloatOrVectorType) -> FloatOrVectorType:
        _u = np.clip(u, EPS, 1 - EPS)
        return self._inverseInterpolator(_u)

    def _cdf(self, x: FloatOrVectorType) -> FloatOrVectorType:
        _x = np.clip(x, a_min=self._strikes[0], a_max=self._strikes[-1])
        return self._interpolator(_x)

    def _pdf(self, x: FloatOrVectorType) -> FloatOrVectorType:
        _x = np.clip(x, a_min=self._strikes[0], a_max=self._strikes[-1])
        return self._interpolator.derivative()(_x)
