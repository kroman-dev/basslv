import numpy as np

from scipy.interpolate import PchipInterpolator
from scipy.stats import norm

from basslv.core.solutionInterpolator import SolutionInterpolator
from basslv.core.projectTyping import FloatVectorType

EPS = np.finfo(np.float64).eps


class SolutionFixedPointEquation(SolutionInterpolator):

    def __init__(
            self,
            x: FloatVectorType,
            y: FloatVectorType,
            tenor: float
    ):
        self._tenor = tenor
        super().__init__(x=x, y=y)

    def _buildInterpolator(self) -> None:
        """
            No 'extrapolation' guaranteeing that the output values
             respect the constraints of the image domain of a cdf.
             So we naively 'augment' the interpolation domain from (x, y)
             to ([x_inf, x, x_sup], [0, y, 1]), with x_inf and x_sup
             chosen as quantiles (e.g. 99.9% confidence interval).
        """

        shift = (1 - norm.cdf(self._x[-1] / np.sqrt(self.tenor))) / 20
        if (shift > EPS) and ((self._y[-1] - (1. - EPS)) > EPS):
            xMax = norm.ppf(1 - shift) * np.sqrt(self.tenor)
            if abs(xMax) > abs(self._x[0]):
                xMin = -xMax
            else:
                raise ValueError('Fail augment the interpolation domain')
            self._y[-1] - (1. - EPS)
            PchipInterpolator(
                np.concatenate([[0. + EPS], self._y, [1. - EPS]]),
                np.concatenate([[xMin], self._x, [xMax]]),
                extrapolate=False
            )
            self._x = np.concatenate([[xMin], self._x, [xMax]])
            self._y = np.concatenate([[0. + EPS], self._y, [1. - EPS]])

        self._interpolator = PchipInterpolator(self.x, self.y, extrapolate=False)
        self._inverseInterpolator = PchipInterpolator(self.y, self.x, extrapolate=False)

    @property
    def tenor(self) -> float:
        return self._tenor

    def _directCall(self, x: FloatVectorType) -> FloatVectorType:
        _x = np.clip(x, a_min=self._x[0], a_max=self._x[-1])
        return self._interpolator(_x)

    def _inverseCall(self, u: FloatVectorType) -> FloatVectorType:
        _u = np.clip(u, a_min=self._y[0], a_max=self._y[-1])
        return self._inverseInterpolator(_u)
