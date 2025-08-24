from scipy.interpolate import PchipInterpolator

from basslv.core.projectTyping import FloatVectorType
from basslv.interpolators.genericInterpolator import GenericInterpolator


class PiecewiseCubicHermiteInterpolator(GenericInterpolator):
    """Piecewise Cubic Hermite Interpolating Polynomial"""
    _interpolator: PchipInterpolator
    _inverseInterpolator: PchipInterpolator

    def __init__(
            self,
            x: FloatVectorType,
            y: FloatVectorType,
            extrapolate: bool = True
    ):
        self._extrapolate = extrapolate
        super().__init__(x=x, y=y)

    def _buildInterpolator(self) -> None:
        self._interpolator = PchipInterpolator(
            x=self._x,
            y=self._y,
            extrapolate=self._extrapolate
        )

        self._inverseInterpolator = PchipInterpolator(
            x=self._y,
            y=self._x,
            extrapolate=self._extrapolate
        )

    def _directCall(self, x: FloatVectorType) -> FloatVectorType:
        return self._interpolator(x)

    def _inverseCall(self, y: FloatVectorType) -> FloatVectorType:
        return self._inverseInterpolator(y)
