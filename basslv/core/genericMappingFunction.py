from abc import ABC, abstractmethod
from typing import Callable

from basslv.core.genericMarginal import GenericMarginal
from basslv.core.genericSolutionInterpolator import GenericSolutionInterpolator
from basslv.core.projectTyping import FloatVectorType
from basslv.core.genericHeatKernelConvolutionEngine import GenericHeatKernelConvolutionEngine


class GenericMappingFunction(ABC):

    _convolutionEngine = None

    def __init__(
            self,
            marginal1: GenericMarginal,
            marginal2: GenericMarginal,
            solutionOfFixedPointEquation: GenericSolutionInterpolator,
            solutionInterpolatorConstructor: GenericSolutionInterpolator,
    ):
        self._marginal1 = marginal1
        self._marginal2 = marginal2
        self._solutionOfFixedPointEquation = solutionOfFixedPointEquation
        self._solutionConstructor = solutionInterpolatorConstructor

    @classmethod
    def setConvolutionEngine(
            cls,
            newConvolutionEngine: GenericHeatKernelConvolutionEngine
    ) -> None:
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
