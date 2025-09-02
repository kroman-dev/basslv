import numpy as np

from dataclasses import dataclass
from typing import Callable, List

from basslv.core.fixedPointEquation import FixedPointEquation
from basslv.core.solutionFixedPointEquation import SolutionFixedPointEquation
from basslv.core.heatKernelConvolutionEngine import HeatKernelConvolutionEngine
from basslv.core.logNormalMarginal import LogNormalMarginal
from basslv.core.projectTyping import FloatOrVectorType


EPS = np.finfo(float).eps


@dataclass
class _BassOneTenorModel:
    tenorStart: float
    tenorEnd: float
    mappingFunction: Callable[[float, float], float]
    solution: SolutionFixedPointEquation
    marginal: LogNormalMarginal

    def __str__(self):
        return f"Bass_{self.tenorStart}_{self.tenorEnd}"


# ONLY FOR LogNormalMarginal
class BassLocalVolatility:
    _convolutionEngine = HeatKernelConvolutionEngine()
    _fixedPointEquation = FixedPointEquation()

    @classmethod
    def setFixedPointEquation(cls, newFixedPointEquation):
        cls._fixedPointEquation = newFixedPointEquation

    @classmethod
    def _buildFirstMarginalMappingFunction(
            cls,
            marginal: LogNormalMarginal,
            hermgaussPoints: int
    ):
        exactSolution = cls._fixedPointEquation.\
            getExactSolutionOfFixedPointEquationInLogNormalCase(
                marginal.tenor
        )
        terminalConditionFunction = \
            lambda w: marginal.inverseCdf(exactSolution(w))
        mappingFunc = lambda t, w: \
            cls._convolutionEngine.useGaussHermiteQuadrature(
                time=marginal.tenor - t,
                func=terminalConditionFunction,
                hermgaussPoints=hermgaussPoints
        )(w)
        return mappingFunc

    @classmethod
    def _buildModels(
            cls,
            marginals: List[LogNormalMarginal],
            fixedPointEquationMaxIter: int,
            fixedPointEquationTolerance: float,
            fixedPointEquationGridBound: float,
            fixedPointEquationGridPoints: int,
            convolutionHermGaussPoints: int
    ) -> List[_BassOneTenorModel]:
        tenorModels: List[_BassOneTenorModel] = [
            _BassOneTenorModel(
                tenorStart=0.,
                tenorEnd=marginals[0].tenor,
                mappingFunction=cls._buildFirstMarginalMappingFunction(
                    marginal=marginals[0],
                    hermgaussPoints=convolutionHermGaussPoints
                ),
                solution=None,
                marginal=marginals[0]
            )
        ]

        for marginalIndex in range(len(marginals) - 1):
            marginal1, marginal2 = \
                marginals[marginalIndex], marginals[marginalIndex + 1]

            solution = cls._fixedPointEquation.solveFixedPointEquation(
                marginal1=marginal1,
                marginal2=marginal2,
                maxIter=fixedPointEquationMaxIter,
                tol=fixedPointEquationTolerance,
                gridBound=fixedPointEquationGridBound,
                gridPoints=fixedPointEquationGridPoints,
                hermGaussPoints=convolutionHermGaussPoints,
                solutionInterpolator=SolutionFixedPointEquation
            )
            mappingFunction = lambda t, w: \
                cls._fixedPointEquation.getMappingFunction(
                    solution=solution,
                    marginal1=marginal1,
                    marginal2=marginal2,
                    time=t,
                    hermGaussPoints=convolutionHermGaussPoints
            )(w)

            tenorModels.append(
                _BassOneTenorModel(
                    tenorStart=marginal1.tenor,
                    tenorEnd=marginal2.tenor,
                    mappingFunction=mappingFunction,
                    solution=solution,
                    marginal=marginal2
                )
            )

        return tenorModels

    @classmethod
    def sample(
            cls,
            t: FloatOrVectorType,
            pathsNumber: int,
            marginals: List[LogNormalMarginal],
            fixedPointEquationMaxIter: int = 61,
            fixedPointEquationTolerance: float = 1e-5,
            fixedPointEquationGridBound: float = 5.,
            fixedPointEquationGridPoints: int =2001,
            convolutionHermGaussPoints: int = 61,
            randomGenerator: np.random.Generator = None
    ):
        randomGenerator = randomGenerator or np.random.default_rng()

        brownianMotion = np.cumsum(
            randomGenerator.normal(size=(pathsNumber, len(t)))
                * np.sqrt(np.diff(t, prepend=0)),
            axis=1
        )
        underlyingPaths = np.full((pathsNumber, len(t)), np.nan)

        timeLeftBound = 0
        brownianStart = 0
        bassProcessStart = 0
        bassTenorModels = cls._buildModels(
            marginals=marginals,
            fixedPointEquationMaxIter=fixedPointEquationMaxIter,
            fixedPointEquationTolerance=fixedPointEquationTolerance,
            fixedPointEquationGridBound=fixedPointEquationGridBound,
            fixedPointEquationGridPoints=fixedPointEquationGridPoints,
            convolutionHermGaussPoints=convolutionHermGaussPoints
        )

        for bassModelIndex in range(len(bassTenorModels)):
            marginal = bassTenorModels[bassModelIndex].marginal
            mappingFunction = bassTenorModels[bassModelIndex].mappingFunction

            # retrieve relevant times and brownianMotion values
            fromIndex = np.searchsorted(t, timeLeftBound, "left")
            toIndex = np.searchsorted(t, bassTenorModels[bassModelIndex].tenorEnd, "right")
            currentTimeInInterval = t[fromIndex:toIndex]
            brownianOnInterval = brownianMotion[:, fromIndex:toIndex]

            bassProcessOnInterval = \
                bassProcessStart + brownianOnInterval - brownianStart
            underlyingPaths[:, fromIndex:toIndex] = \
                mappingFunction(currentTimeInInterval, bassProcessOnInterval)

            if toIndex == len(t):
                break

            # Brownian bridge
            leftTime, bridgeTime, rightTime = \
                t[toIndex - 1], bassTenorModels[bassModelIndex].tenorEnd, t[toIndex]
            B1, B2 = brownianMotion[:, toIndex - 1], brownianMotion[:, toIndex]
            mean = B1 * (rightTime - bridgeTime) / (rightTime - leftTime + EPS) \
                   + B2 * (bridgeTime - leftTime) / (rightTime - leftTime + EPS)
            std = np.sqrt(
                (rightTime - bridgeTime)
                * (bridgeTime - leftTime)
                / (rightTime - leftTime + EPS)
            )
            brownianStart = randomGenerator.normal(loc=mean, scale=std)[:, None]

            if bassModelIndex == (len(bassTenorModels) - 1):
                break

            solution = bassTenorModels[bassModelIndex + 1].solution
            bassProcessStart = solution.inverseCall(
                marginal.cdf(mappingFunction(bridgeTime, brownianStart))
            )
            timeLeftBound = marginal.tenor

        return underlyingPaths


