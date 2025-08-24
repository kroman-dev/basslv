from abc import ABC, abstractmethod

from basslv.core.projectTyping import FloatVectorType


class GenericInterpolator(ABC):

    def __init__(self, x: FloatVectorType, y: FloatVectorType):
        self._x = x
        self._y = y
        self._buildInterpolator()

    @property
    def x(self) -> FloatVectorType:
        return self._x

    @property
    def y(self) -> FloatVectorType:
        return self._y

    @abstractmethod
    def _buildInterpolator(self) -> None:
        pass

    @abstractmethod
    def _directCall(self, x: FloatVectorType) -> FloatVectorType:
        pass

    @abstractmethod
    def _inverseCall(self, y: FloatVectorType) -> FloatVectorType:
        pass

    def __call__(self, x: FloatVectorType) -> FloatVectorType:
        return self._directCall(x)

    def inverseCall(self, y: FloatVectorType) -> FloatVectorType:
        return self._inverseCall(y)
