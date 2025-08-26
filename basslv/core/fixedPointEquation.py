import numpy as np
from typing import Callable

from scipy import optimize
from scipy.stats import norm
from scipy.interpolate import CubicSpline, PchipInterpolator

from basslv.core.genericMarginal import GenericMarginal
from basslv.core.heatKernelConvolutionEngine import HeatKernelConvolutionEngine
from basslv.core.solutionInterpolator import SolutionInterpolator
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
    _convolutionEngine = HeatKernelConvolutionEngine()

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

    @staticmethod
    def getSolutionOfLinearisedFixedPointEquation(
            marginal1: GenericMarginal,
            marginal2: GenericMarginal,
            wGrid: FloatVectorType,
            solutionInterpolator: SolutionInterpolator
    ) -> SolutionInterpolator:
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

        # PchipInterpolator(inverseCdfValues, uGrid, extrapolate=False)(wGrid)
        return solutionInterpolator(
            x=inverseCdfValues,
            y=uGrid,
            tenor=marginal1.tenor
        )

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
        internalConvolution = \
            cls._convolutionEngine.useGaussHermiteQuadrature(
                time=np.array([marginal2.tenor - marginal1.tenor])[None],
                func=solution,
                hermgaussPoints=hermGaussPoints
        )

        def applySecondMarginalInverseCdf(x):
            u = internalConvolution(x)
            return marginal2.inverseCdf(u)

        externalConvolution = \
            cls._convolutionEngine.useGaussHermiteQuadrature(
                time=marginal2.tenor - time,
                func=applySecondMarginalInverseCdf,
                hermgaussPoints=hermGaussPoints
        )
        return externalConvolution

    @classmethod
    def applyOperatorA(
            cls,
            solution: SolutionInterpolator,
            marginal1: GenericMarginal,
            marginal2: GenericMarginal,
            hermGaussPoints: int
    ) -> Callable[[FloatVectorType], FloatVectorType]:
        """
            Equation (2) [1]
        """

        def applyFirstMarginalCdf(x):
            mappingFuncValues = cls.getMappingFunction(
                solution=solution,
                marginal1=marginal1,
                marginal2=marginal2,
                time=marginal1.tenor,
                hermGaussPoints=hermGaussPoints
            )(x)
            return marginal1.cdf(mappingFuncValues)

        return applyFirstMarginalCdf

    @staticmethod
    def LInfinityNorm(
            sequence1: FloatVectorType,
            sequence2: FloatVectorType
    ) -> float:
        return np.max(np.abs(sequence1 - sequence2))

    @staticmethod
    def adjustCdfSolution(
            solution: SolutionInterpolator,
            solutionInterpolator: SolutionInterpolator
    ) -> SolutionInterpolator:
        """
            In current version (22/08/2025) a solution of the Fixed Point Equation,
             when an initial iteration is got as solution of Linearized Fixed Point Equation,
             is shifted to the right relative to the exact solution (verify in lognormal case).
             So, we know that solution must equal F(0.) = 0.5 (see Remark 1 [1]) and we just
             shift out solution so that equals 0.5.
        """
        wGrid = solution.x
        errorInZero = 0.5 - solution(0.)
        # if errorInZero > 0.3:
        #     raise ConvergeError('Incorrect fitting')
        print(f'Error in zero: {errorInZero}')
        objective = lambda x: 0.5 - solution(x)
        adjustment = optimize.bisect(objective, a=-7., b=7.)
        adjustedSolution = solutionInterpolator(
            x=wGrid - adjustment,
            y=solution(wGrid),
            tenor=solution.tenor
        )
        print(f'adjusted error in zero: {0.5 - adjustedSolution(0.)}')
        print()
        return adjustedSolution

    def solveFixedPointEquation(
            self,
            marginal1: GenericMarginal,
            marginal2: GenericMarginal,
            solutionInterpolator: SolutionInterpolator,
            maxIter: int = 61,
            tol: float = 1e-5,
            gridBound: float = 5.,
            gridPoints = 2001,
            hermGaussPoints: int = 61
    ) -> SolutionInterpolator:
        wGrid = np.linspace(
            start=-gridBound,
            stop=gridBound,
            num=gridPoints,
            endpoint=True
        ) * np.sqrt(marginal1.tenor)

        solution = self.getSolutionOfLinearisedFixedPointEquation(
            marginal1=marginal1,
            marginal2=marginal2,
            wGrid=wGrid,
            solutionInterpolator=solutionInterpolator
        )

        for iterationIndex in range(maxIter):
            solutionNextIteration = solutionInterpolator(
                x=wGrid,
                y=self.applyOperatorA(
                    solution=solution,
                    marginal1=marginal1,
                    marginal2=marginal2,
                    hermGaussPoints=hermGaussPoints
                )(wGrid),
                tenor=marginal1.tenor
            )

            lInftyValue = self.LInfinityNorm(
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
                print(f'Current tol: {tol}, reach {lInftyValue}')
                # return solution
                # raise ConvergeError(f'Current tol: {tol}, reach {lInftyValue}')

        solution = self.adjustCdfSolution(
            solution=solution,
            solutionInterpolator=solutionInterpolator
        )
        return solution
