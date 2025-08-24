from scipy.interpolate import CubicSpline

from basslv.core.projectTyping import FloatVectorType
from basslv.interpolators.genericInterpolator import GenericInterpolator


class CubicSplineInterpolator(GenericInterpolator):
    """Interpolate data with a piecewise cubic polynomial"""
    _interpolator: CubicSpline
    _inverseInterpolator: CubicSpline

    def __init__(self, x: FloatVectorType, y: FloatVectorType):
        super().__init__(x=x, y=y)

    def _buildInterpolator(self) -> None:
        self._interpolator = CubicSpline(
            x=self._x,
            y=self._y,
        )

        self._inverseInterpolator = CubicSpline(
            x=self._y,
            y=self._x,
        )

    def _directCall(self, x: FloatVectorType) -> FloatVectorType:
        return self._interpolator(x)

    def _inverseCall(self, y: FloatVectorType) -> FloatVectorType:
        return self._inverseInterpolator(y)
