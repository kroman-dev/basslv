import time
import numpy as np

from typing import Callable
from functools import wraps

from basslv import FixedPointEquation, LogNormalMarginal, VisualVerification
from basslv import HeatKernelConvolutionEngine, FloatVectorType, SolutionFixedPointEquation
from basslv.core.genericMarginal import GenericMarginal, EPS
from basslv.core.solutionInterpolator import SolutionInterpolator


def loggedTime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        startTime = time.perf_counter()
        result = func(*args, **kwargs)
        endTime = time.perf_counter()
        print(
            f'Calculation time: {endTime - startTime:.4f} sec for: '
            f'{func.__name__}{args} {kwargs}'
        )
        return result
    return wrapper


class MappingFunction:

    _convolutionEngine = HeatKernelConvolutionEngine()

    def __init__(
            self,
            marginal1: GenericMarginal,
            marginal2: GenericMarginal,
            hermGaussPoints: int,
            solution: SolutionInterpolator,
            solutionInterpolatorConstructor: SolutionInterpolator,
            saveInternalConvolution: bool = False
    ):
        self._marginal1 = marginal1
        self._marginal2 = marginal2
        self._hermGaussPoints = hermGaussPoints
        self._internalConvolution = None
        self._solution = solution
        self._solutionConstructor = solutionInterpolatorConstructor
        self._saveInternalConvolution = saveInternalConvolution
        if saveInternalConvolution:
            self._internalConvolution = self._prepareInternalConvolution()

    def _prepareInternalConvolution(self):
        x = self._solution.x
        y = np.concatenate([
            [EPS],
            self._calculateInternalConvolution(self._solution)(x[1:-1])[0],
            [1 - EPS]
        ])
        return self._solutionConstructor(
            x=x,
            y=self._calculateInternalConvolution(self._solution)(x)[0],
            tenor=self._marginal1.tenor,
            extrapolation=False
        )

    def _calculateInternalConvolution(
            self,
            solution: SolutionInterpolator
    ) -> Callable[[FloatVectorType], FloatVectorType]:
        internalConvolution = \
            self._convolutionEngine.useGaussHermiteQuadrature(
                time=np.array([self._marginal2.tenor - self._marginal1.tenor])[None],
                func=solution,
                hermgaussPoints=self._hermGaussPoints
        )
        return internalConvolution

    def getMappingFunction(
            self,
            time: float
    ) -> Callable[[FloatVectorType], FloatVectorType]:
        """
            Equation (3) [1]
        """
        if self._saveInternalConvolution:
            internalConvolution = self._internalConvolution
        else:
            internalConvolution = \
                self._convolutionEngine.useGaussHermiteQuadrature(
                    time=np.array([self._marginal2.tenor - self._marginal1.tenor])[None],
                    func=self._solution,
                    hermgaussPoints=self._hermGaussPoints
            )

        def applySecondMarginalInverseCdf(x):
            u = internalConvolution(x)
            return marginal2.inverseCdf(u)

        externalConvolution = \
            self._convolutionEngine.useGaussHermiteQuadrature(
                time=self._marginal2.tenor - time,
                func=applySecondMarginalInverseCdf,
                hermgaussPoints=self._hermGaussPoints
        )
        return externalConvolution


if __name__ == '__main__':
    marginal1= LogNormalMarginal(sigma=0.2, tenor=2.)
    marginal2 = LogNormalMarginal(sigma=0.2, tenor=3.)
    wGrid = np.linspace(-5., 5., 2001, endpoint=True) * np.sqrt(2.)

    fixedPointEq = FixedPointEquation()
    solution = SolutionFixedPointEquation(
        x=wGrid,
        y=fixedPointEq.getExactSolutionOfFixedPointEquationInLogNormalCase(2.)(wGrid),
        tenor=2.
    )

    @loggedTime
    def calcMapping(mappingFunc, grid):
        return mappingFunc(grid)

    mappingFunctionSpeedUp = MappingFunction(
        marginal1=marginal1,
        marginal2=marginal2,
        hermGaussPoints=61,
        solution=solution,
        solutionInterpolatorConstructor=SolutionFixedPointEquation,
        saveInternalConvolution=True
    ).getMappingFunction(2.)

    mappingFunction = MappingFunction(
        marginal1=marginal1,
        marginal2=marginal2,
        hermGaussPoints=61,
        solution=solution,
        solutionInterpolatorConstructor=SolutionFixedPointEquation,
        saveInternalConvolution=False
    ).getMappingFunction(2.)

    VisualVerification.plotComparison(
        x=wGrid,
        funcValues1=calcMapping(mappingFunctionSpeedUp, wGrid),
        funcValues2=calcMapping(mappingFunction, wGrid),
        label1='mapping speed up',
        label2 = 'mapping common',
        generalTitle=''
    )

