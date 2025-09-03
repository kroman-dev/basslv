import numpy as np

from typing import Callable

from basslv.core.genericMarginal import GenericMarginal
from basslv.core.genericSolutionInterpolator import GenericSolutionInterpolator
from basslv.core.projectTyping import FloatVectorType
from basslv.core.gaussHermitHeatKernelConvolutionEngine import GaussHermitHeatKernelConvolutionEngine
from basslv.core.genericMappingFunction import GenericMappingFunction


class MappingFunction(GenericMappingFunction):

    # TODO saveInternalConvolution accumulate an error
    _saveInternalConvolution = True
    # TODO details should depend on abstractions
    _convolutionEngine = GaussHermitHeatKernelConvolutionEngine()

    def __init__(
            self,
            marginal1: GenericMarginal,
            marginal2: GenericMarginal,
            solutionOfFixedPointEquation: GenericSolutionInterpolator,
            solutionInterpolatorConstructor: GenericSolutionInterpolator
    ):
        super().__init__(
            marginal1=marginal1,
            marginal2=marginal2,
            solutionOfFixedPointEquation=solutionOfFixedPointEquation,
            solutionInterpolatorConstructor=solutionInterpolatorConstructor
        )
        self._internalConvolution = None
        if self._saveInternalConvolution:
            self._internalConvolution = self._prepareInternalConvolution()

    @classmethod
    def setSaveInternalConvolution(cls, saveInternalConvolution: bool):
        cls._saveInternalConvolution = saveInternalConvolution

    def _prepareInternalConvolution(self):
        x = self._solutionOfFixedPointEquation.x / self._marginal1.tenor * self._marginal2.tenor

        return self._solutionConstructor(
            x=x,
            y=self._calculateInternalConvolution(self._solutionOfFixedPointEquation)(x)[0],
            tenor=self._marginal1.tenor,
            extrapolation=False
        )

    def _calculateInternalConvolution(
            self,
            solution: GenericSolutionInterpolator
    ) -> Callable[[FloatVectorType], FloatVectorType]:
        internalConvolution = \
            self._convolutionEngine.convolution(
                time=np.array([self._marginal2.tenor - self._marginal1.tenor])[None],
                func=solution
        )
        return internalConvolution

    def getMappingFunction(
            self,
            time: float
    ) -> Callable[[FloatVectorType], FloatVectorType]:
        """
            Equation (3) [1]
        """
        internalConvolution = self._internalConvolution \
            if self._saveInternalConvolution else self._calculateInternalConvolution(
                self._solutionOfFixedPointEquation
            )

        def applySecondMarginalInverseCdf(x):
            u = internalConvolution(x)
            return self._marginal2.inverseCdf(u)

        externalConvolution = \
            self._convolutionEngine.convolution(
                time=self._marginal2.tenor - time,
                func=applySecondMarginalInverseCdf
        )
        return externalConvolution
