import numpy as np

from typing import Callable

from basslv.core.genericMarginal import GenericMarginal
from basslv.core.solutionInterpolator import SolutionInterpolator
from basslv.core.projectTyping import FloatVectorType
from basslv.core.gaussHermitHeatKernelConvolutionEngine import GaussHermitHeatKernelConvolutionEngine


class MappingFunction:

    _convolutionEngine = GaussHermitHeatKernelConvolutionEngine()

    def __init__(
            self,
            marginal1: GenericMarginal,
            marginal2: GenericMarginal,
            hermGaussPoints: int,
            solutionOfFixedPointEquation: SolutionInterpolator,
            solutionInterpolatorConstructor: SolutionInterpolator,
            saveInternalConvolution: bool = False
    ):
        self._marginal1 = marginal1
        self._marginal2 = marginal2
        self._hermGaussPoints = hermGaussPoints
        self._internalConvolution = None
        self._solutionOfFixedPointEquation = solutionOfFixedPointEquation
        self._solutionConstructor = solutionInterpolatorConstructor
        self._saveInternalConvolution = saveInternalConvolution
        if saveInternalConvolution:
            self._internalConvolution = self._prepareInternalConvolution()

    def _prepareInternalConvolution(self):
        x = self._solutionOfFixedPointEquation.x
        return self._solutionConstructor(
            x=x,
            y=self._calculateInternalConvolution(self._solutionOfFixedPointEquation)(x)[0],
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
                    func=self._solutionOfFixedPointEquation,
                    hermgaussPoints=self._hermGaussPoints
            )

        def applySecondMarginalInverseCdf(x):
            u = internalConvolution(x)
            return self._marginal2.inverseCdf(u)

        externalConvolution = \
            self._convolutionEngine.useGaussHermiteQuadrature(
                time=self._marginal2.tenor - time,
                func=applySecondMarginalInverseCdf,
                hermgaussPoints=self._hermGaussPoints
        )
        return externalConvolution
