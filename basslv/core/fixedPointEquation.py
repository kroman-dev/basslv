import numpy as np
from typing import Callable

from scipy import optimize
from scipy.stats import norm
from scipy.interpolate import CubicSpline

from basslv.core.genericMarginal import GenericMarginal
from basslv.core.genericHeatKernelConvolutionEngine import GenericHeatKernelConvolutionEngine
from basslv.core.genericSolutionInterpolator import GenericSolutionInterpolator
from basslv.core.genericMappingFunction import GenericMappingFunction
from basslv.core.mappingFunction import MappingFunction
from basslv.core.solutionFixedPointEquation import SolutionFixedPointEquation
from basslv.core.projectTyping import FloatVectorType


EPS = np.finfo(float).eps


class ConvergeError(Exception):
    def __init__(self, *args):
        super().__init__(*args)


# noinspection PyTypeChecker,PyCallingNonCallable
class FixedPointEquation:
    """
        [1] Antoine Conze and Henry-Labordere, "A new fast local volatility model"
    """
    # TODO details should depend on abstractions
    _mappingFunctionConstructor: GenericMappingFunction = MappingFunction
    _solutionInterpolatorConstructor: GenericSolutionInterpolator = \
        SolutionFixedPointEquation

    @classmethod
    def setMappingFunction(
            cls,
            newMappingFunction: GenericMappingFunction
    ) -> None:
        cls._mappingFunctionConstructor = newMappingFunction

    @classmethod
    def setConvolutionEngine(
            cls,
            newConvolutionEngine: GenericHeatKernelConvolutionEngine
    ) -> None:
        cls._mappingFunctionConstructor.setConvolutionEngine(
            newConvolutionEngine
        )

    @classmethod
    def setSolutionInterpolator(
            cls,
            newSolutionInterpolator: GenericSolutionInterpolator
    ) -> None:
        cls._solutionInterpolatorConstructor = newSolutionInterpolator

    @staticmethod
    def getExactSolutionOfFixedPointEquationInLogNormalCase(
            tenor: float
    ) -> Callable[[float], float]:
        """
            When applied to log-normal distributions,
             our construction yields the Black-Scholes model.
             So, exact solution of fixed point equation is N(w/sqrt(T))
        """
        return lambda x: norm.cdf(x / np.sqrt(tenor))

    @classmethod
    def getSolutionOfLinearisedFixedPointEquation(
            cls,
            marginal1: GenericMarginal,
            marginal2: GenericMarginal,
            wGrid: FloatVectorType
    ) -> GenericSolutionInterpolator:
        """
            Equation (12) [1]
        """
        uGrid = norm.cdf(wGrid / np.sqrt(marginal1.tenor))

        integrand = np.sqrt(
            marginal2.derivativeOfInverseCdf(uGrid) / (
                    marginal1.integralOfInverseCdf(uGrid)
                    - marginal2.integralOfInverseCdf(uGrid) + EPS
            )
        )
        integral = CubicSpline(uGrid, integrand).antiderivative()
        inverseCdfValues = np.sqrt(
            (marginal2.tenor - marginal1.tenor) / 2
        ) * (integral(uGrid) - integral(1 / 2))

        return cls._solutionInterpolatorConstructor(
            x=inverseCdfValues,
            y=uGrid,
            tenor=marginal1.tenor
        )

    @classmethod
    def getMappingFunction(
            cls,
            solution: GenericSolutionInterpolator,
            marginal1: GenericMarginal,
            marginal2: GenericMarginal,
            time: float
    ) -> Callable[[FloatVectorType], FloatVectorType]:
        """
            Equation (3) [1]
        """
        return cls._mappingFunctionConstructor(
            marginal1=marginal1,
            marginal2=marginal2,
            solutionOfFixedPointEquation=solution,
            solutionInterpolatorConstructor=cls._solutionInterpolatorConstructor
        ).getMappingFunction(time)

    @classmethod
    def applyOperatorA(
            cls,
            solution: GenericSolutionInterpolator,
            marginal1: GenericMarginal,
            marginal2: GenericMarginal
    ) -> Callable[[FloatVectorType], FloatVectorType]:
        """
            Equation (2) [1]
        """

        def applyFirstMarginalCdf(x):
            mappingFuncValues = cls.getMappingFunction(
                solution=solution,
                marginal1=marginal1,
                marginal2=marginal2,
                time=marginal1.tenor
            )(x)
            return marginal1.cdf(mappingFuncValues)

        return applyFirstMarginalCdf

    @staticmethod
    def LInfinityNorm(
            sequence1: FloatVectorType,
            sequence2: FloatVectorType
    ) -> float:
        return np.max(np.abs(sequence1 - sequence2))

    @classmethod
    def adjustCdfSolution(
            cls,
            solution: GenericSolutionInterpolator
    ) -> GenericSolutionInterpolator:
        """
            In current version (22/08/2025) a solution of the Fixed Point Equation,
             when an initial iteration is got as solution of Linearized Fixed Point Equation,
             is shifted to the right relative to the exact solution (verify in lognormal case).
             So, we know that solution must equal F(0.) = 0.5 (see Remark 1 [1]) and we just
             shift out solution so that equals 0.5.
        """
        wGrid = solution.x
        errorInZero = 0.5 - solution(0.)
        print(f'Error in zero: {errorInZero} for marginal.tenor={solution.tenor}')
        objective = lambda x: 0.5 - solution(x)
        adjustment = optimize.bisect(objective, a=-10., b=10.)
        adjustedSolution = cls._solutionInterpolatorConstructor(
            x=wGrid - adjustment,
            y=solution(wGrid),
            tenor=solution.tenor
        )
        print(f'adjusted error in zero: {0.5 - adjustedSolution(0.)}')
        print()
        return adjustedSolution

    @classmethod
    def solveFixedPointEquation(
            cls,
            marginal1: GenericMarginal,
            marginal2: GenericMarginal,
            maxIter: int = 61,
            tol: float = 1e-5,
            gridBound: float = 5.,
            gridPoints = 2001
    ) -> GenericSolutionInterpolator:
        wGrid = np.linspace(
            start=-gridBound,
            stop=gridBound,
            num=gridPoints,
            endpoint=True
        ) * np.sqrt(marginal1.tenor)

        solution = cls.getSolutionOfLinearisedFixedPointEquation(
            marginal1=marginal1,
            marginal2=marginal2,
            wGrid=wGrid
        )

        for iterationIndex in range(maxIter):
            solutionNextIteration = cls._solutionInterpolatorConstructor(
                x=wGrid,
                y=cls.applyOperatorA(
                    solution=solution,
                    marginal1=marginal1,
                    marginal2=marginal2
                )(wGrid),
                tenor=marginal1.tenor
            )

            lInftyValue = cls.LInfinityNorm(
                sequence1=solutionNextIteration(wGrid),
                sequence2=solution(wGrid)
            )

            solution = solutionNextIteration
            if lInftyValue < tol:
                print(
                    f'Solve fixed point eq: '
                    f'tenorStart={marginal1.tenor}, tenorEnd: {marginal2.tenor} \n'
                    f'current tol: {tol}, reach: {lInftyValue}, iter: {iterationIndex}'
                )
                break

            if iterationIndex == (maxIter - 1):
                print(f'Converge error: current tol: {tol}, reach {lInftyValue}')
                # raise ConvergeError(f'Current tol: {tol}, reach {lInftyValue}')

        solution = cls.adjustCdfSolution(solution=solution)
        return solution
