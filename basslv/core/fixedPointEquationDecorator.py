from typing import Callable

from basslv.core.fixedPointEquation import FixedPointEquation
from basslv.core.solutionFixedPointEquation import SolutionFixedPointEquation
from basslv.core.mappingFunction import MappingFunction
from basslv.core.projectTyping import FloatVectorType
from basslv.core.solutionInterpolator import SolutionInterpolator
from basslv.core.genericMarginal import GenericMarginal


class FixedPointEquationDecorator(FixedPointEquation):

    @classmethod
    def getMappingFunction(
            cls,
            solution: SolutionInterpolator,
            marginal1: GenericMarginal,
            marginal2: GenericMarginal,
            time: float,
            hermGaussPoints: int
    ) -> Callable[[FloatVectorType], FloatVectorType]:
        """
            Equation (3) [1]
        """
        return MappingFunction(
            marginal1=marginal1,
            marginal2=marginal2,
            hermGaussPoints=hermGaussPoints,
            solutionOfFixedPointEquation=solution,
            solutionInterpolatorConstructor=SolutionFixedPointEquation,
            saveInternalConvolution=True
        ).getMappingFunction(time)
