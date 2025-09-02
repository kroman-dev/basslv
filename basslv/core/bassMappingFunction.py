import numpy as np

from abc import ABC, abstractmethod
from typing import Callable

from basslv.core.genericMarginal import GenericMarginal
from basslv.core.solutionInterpolator import SolutionInterpolator
from basslv.core.projectTyping import FloatVectorType
from basslv.core.heatKernelConvolutionEngine import HeatKernelConvolutionEngine


class MappingFunction:

    _convolutionEngine = None

    def __init__(
            self,
            marginal1: GenericMarginal,
            marginal2: GenericMarginal,
            solutionOfFixedPointEquation: SolutionInterpolator,
            solutionInterpolatorConstructor: SolutionInterpolator,
    ):
        self._marginal1 = marginal1
        self._marginal2 = marginal2
        self._solutionOfFixedPointEquation = solutionOfFixedPointEquation
        self._solutionConstructor = solutionInterpolatorConstructor

    @classmethod
    def setConvolutionEngine(cls, newConvolutionEngine: HeatKernelConvolutionEngine):
        cls._convolutionEngine = newConvolutionEngine

    @abstractmethod
    def getMappingFunction(
            self,
            time: FloatVectorType
    ) -> Callable[[FloatVectorType], FloatVectorType]:
        """
            Equation (3) [1]
        """
        pass
