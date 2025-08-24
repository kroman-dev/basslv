from scipy.interpolate import interp1d

from basslv.core.projectTyping import FloatVectorType
from basslv.interpolators.genericInterpolator import GenericInterpolator


class LinearInterpolator(GenericInterpolator):
    _interpolator: interp1d
    _inverseInterpolator: interp1d

    def __init__(self, x: FloatVectorType, y: FloatVectorType):
        super().__init__(x=x, y=y)

    def _buildInterpolator(self) -> None:
        self._interpolator = interp1d(
            x=self._x,
            y=self._y,
            kind='linear'
        )

        self._inverseInterpolator = interp1d(
            x=self._y,
            y=self._x,
            kind='linear'
        )

    def _directCall(self, x: FloatVectorType) -> FloatVectorType:
        return self._interpolator(x)

    def _inverseCall(self, y: FloatVectorType) -> FloatVectorType:
        return self._inverseInterpolator(y)
