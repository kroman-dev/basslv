import numpy as np

from typing import Callable
from numpy.typing import NDArray
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.hermite_e import hermegauss

from basslv.core.projectTyping import FloatVectorType, toNumpy
from basslv.core.genericHeatKernelConvolutionEngine import GenericHeatKernelConvolutionEngine


EPS = np.finfo(float).eps


class GaussHermitHeatKernelConvolutionEngine(GenericHeatKernelConvolutionEngine):

    _hermgaussPoints = 61

    @classmethod
    def setHermgaussPoints(cls, newHermgaussPoints: int):
        if not isinstance(newHermgaussPoints, int):
            raise TypeError(f'newHermgaussPoints must be int, got {newHermgaussPoints}')
        cls._hermgaussPoints = newHermgaussPoints

    @staticmethod
    def heatKernel(x: float, t: float) -> float:
        return np.exp(-0.5 * x ** 2 / t) / np.sqrt(2 * np.pi * t)

    @staticmethod
    def _simpleConvolution(
            time: float,
            func: Callable[[float], float],
            hermgaussPoints: int = 61
    ) -> Callable[[float], float]:
        _nodes, _weights = hermgauss(hermgaussPoints)
        def result(x: float):
            return 1 / np.sqrt(np.pi) * np.sum(
                _weights * func(x - np.sqrt(2 * time) * _nodes)
            )
        return result

    @staticmethod
    def _useGaussHermiteQuadrature(
            time: float,
            func: Callable[[NDArray], NDArray],
            hermgaussPoints: int = 61
    ) -> Callable[[FloatVectorType], FloatVectorType]:
        if not isinstance(time, (float, int)):
            raise ValueError('t is not float')

        _nodes, _weights = hermgauss(hermgaussPoints)
        def f(x: FloatVectorType):
            if isinstance(x, (float, int)):
                return 1 / np.sqrt(np.pi) * np.sum(
                    _weights * func(x - np.sqrt(2 * time) * _nodes)
                )
            if isinstance(x, list):
                x = toNumpy(x)
            # if x.shape = (m, 1) and nodes.shape = (n) => (x - nodes).shape = (m, n) <-> numpy broadcast
            # if x.shape = (m, n, 1) and nodes.shape = (n) => (x - nodes).shape = (m, n, n) <-> numpy broadcast
            # -1 as for exist dimension, 1 as new dimensions
            newShape = (-1, *[1] * x.ndim)
            nodes = _nodes.reshape(newShape)
            weights = _weights.reshape(newShape)
            newVariable = x[None] - np.sqrt(2 * time) * nodes

            return 1 / np.sqrt(np.pi) * np.sum(
                weights * func(newVariable),
                axis=0
            )

        return f

    @staticmethod
    def useGaussHermiteQuadrature(
            time: FloatVectorType,
            func: Callable[[FloatVectorType], FloatVectorType],
            hermgaussPoints: int = 61,
    ) -> Callable[[FloatVectorType], FloatVectorType]:
        _nodes, _weights = hermegauss(hermgaussPoints)

        def f(x):
            variableBroadcasted, timeBroadcasted = np.broadcast_arrays(x, time)
            targetShape = (-1, *[1] * variableBroadcasted.ndim)
            newVariable = variableBroadcasted[None] \
                          - np.sqrt(timeBroadcasted[None] + EPS) * _nodes.reshape(targetShape)
            return 1 / np.sqrt(2 * np.pi) * np.sum(
                func(newVariable) * _weights.reshape(targetShape),
                axis=0
            )

        return f

    def convolution(
            self,
            time: FloatVectorType,
            func: Callable[[FloatVectorType], FloatVectorType]
    ) -> Callable[[FloatVectorType], FloatVectorType]:
        return self.useGaussHermiteQuadrature(
            time=time,
            func=func,
            hermgaussPoints=self._hermgaussPoints
        )
