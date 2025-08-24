import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from dataclasses import dataclass, fields
from scipy.interpolate import CubicSpline
from typing import Callable, List, Any, Dict, Optional

from basslv.core.solutionFixedPointEquation import SolutionFixedPointEquation
from basslv.core.fixedPointEquation import FixedPointEquation
from basslv.core.logNormalMarginal import LogNormalMarginal
from basslv.core.projectTyping import FloatVectorType

EPS = np.finfo(float).eps


@dataclass
class _MultiPlotInput:
    plot: Dict[str, Any]
    set_title: Dict[str, str]
    set_yscale: Optional[Dict[str, List[str]]] = None

    def __post_init__(self):
        if self.set_yscale is None:
            self.set_yscale = {"args": ["linear"]}


# noinspection PyTypeChecker,PyCallingNonCallable
class VisualVerification:
    _fixedPointEquation = FixedPointEquation()

    @classmethod
    def verifyLogNormalMarginalFunctions(
            cls,
            x=np.linspace(1e-5, 5. - 1e-5, 1000),
            u=np.linspace(1e-5, 1. - 1e-5, 1000),
            marginal=LogNormalMarginal(sigma=0.2, tenor=2.)
    ):
        pdfSpline = CubicSpline(x, marginal._pdf(x))
        cdfSpline = CubicSpline(x, marginal.cdf(x))
        inverseCdfSpline = CubicSpline(u, marginal.inverseCdf(u))

        cls._multiPlot(
            [
                _MultiPlotInput(
                    plot={
                        "args": [
                            [x, x],
                            [marginal._pdf(x), cdfSpline.derivative()(x)]
                        ],
                        "kwargs": {"label": ['analytical', 'numerical']}
                    },
                    set_title={
                        "args": ["Compare numerical cdf derivative and pdf"]
                    },
                    set_yscale={
                        "args": ["linear"]
                    }
                ),
                _MultiPlotInput(
                    plot={
                        "args": [
                            [x, x],
                            [marginal.cdf(x), pdfSpline.antiderivative()(x)]
                        ],
                        "kwargs": {"label": ['analytical', 'numerical']}
                    },
                    set_title={
                        "args": ["Compare cdf and integral of pdf"]
                    }
                ),
                _MultiPlotInput(
                    plot={
                        "args": [
                            [u, marginal.cdf(x)],
                            [marginal.inverseCdf(u), x]
                        ],
                        "kwargs": {"label": ['analytical', 'numerical']}
                    },
                    set_title={
                        "args": ["Compare inverse cdf num and anl"]
                    }
                ),
                _MultiPlotInput(
                    plot={
                        "args": [
                            [u, u],
                            [marginal.derivativeOfInverseCdf(u), inverseCdfSpline.derivative()(u)]
                        ],
                        "kwargs": {"label": ['analytical', 'numerical']}
                    },
                    set_title={
                        "args": ["Compare derivative inverse cdf num and anl"]
                    },
                    set_yscale={
                        "args": ["log"]
                    }
                )
            ]
        )

    @classmethod
    def verifyMappingFunction(
            cls,
            marginal1: LogNormalMarginal = LogNormalMarginal(sigma=0.2, tenor=2.),
            marginal2: LogNormalMarginal = LogNormalMarginal(sigma=0.2, tenor=3.),
            wGrid: FloatVectorType = np.linspace(-5., 5., 2001, endpoint=True) * np.sqrt(2.),
            hermGaussPoints=61
    ):
        exactMappingFunction = lambda wiener: np.exp(
            -0.5 * marginal1._sigma ** 2 * marginal1.tenor \
            + marginal1._sigma * wiener
        )
        exactFixedPointSolution = \
            cls._fixedPointEquation. \
                getExactSolutionOfFixedPointEquationInLogNormalCase(
                marginal1.tenor
            )
        numericalSolution = cls._fixedPointEquation.solveFixedPointEquation(
            marginal1=marginal1,
            marginal2=marginal2,
            solutionInterpolator=SolutionFixedPointEquation,
            hermGaussPoints=hermGaussPoints
        )

        mappingFuncFromOperatorWithExactSolution = \
            cls._fixedPointEquation.getMappingFunction(
                solution=exactFixedPointSolution,
                marginal1=marginal1,
                marginal2=marginal2,
                time=marginal1.tenor,
                hermGaussPoints=hermGaussPoints
            )

        numericalMapping = cls._fixedPointEquation.getMappingFunction(
            solution=numericalSolution,
            marginal1=marginal1,
            marginal2=marginal2,
            time=marginal1.tenor,
            hermGaussPoints=hermGaussPoints
        )

        labelMappingAfterConvolution = \
            r'$f = K_{T_{2}-t} \star (' \
            r'F^{-1}_{\mu_{2}} \circ ' \
            r'(K_{T_{2}-T_{1}} \star F_{exact} ))$' \

        labelNumericalMapping = r'$f = K_{T_{2}-t} \star (' \
            r'F^{-1}_{\mu_{2}} \circ ' \
            r'(K_{T_{2}-T_{1}} \star F_{W_{T_1}} ))$'

        cls.plotFuncs(
            x=wGrid,
            funcs=[
                mappingFuncFromOperatorWithExactSolution,
                exactMappingFunction,
                numericalMapping
            ],
            labels=[
                labelMappingAfterConvolution,
                'f exact',
                labelNumericalMapping
            ],
            title='Mapping function comparison'
        )

    @classmethod
    def verifyConvolutionAccuracy(
            cls,
            func: Callable[[float], float],
            marginal1: LogNormalMarginal = LogNormalMarginal(sigma=0.2, tenor=2.),
            marginal2: LogNormalMarginal = LogNormalMarginal(sigma=0.2, tenor=3.),
            wGrid: FloatVectorType = np.linspace(-5., 5., 2001, endpoint=True) * np.sqrt(2.),
            cdfInterpolator: SolutionFixedPointEquation = SolutionFixedPointEquation,
            convolutionIterations: int = 10
    ):
        solutionsAfterConvolution = []
        solution = cdfInterpolator(
            x=wGrid,
            y=func(wGrid),
            tenor=marginal1.tenor
        )
        solutionsAfterConvolution.append(solution)

        for iterationNumber in range(convolutionIterations):
            print(f'iterationNumber={iterationNumber}')
            solution = cdfInterpolator(
                x=wGrid,
                y=cls._fixedPointEquation.applyOperatorA(
                    solution=solution,
                    marginal1=marginal1,
                    marginal2=marginal2,
                    hermGaussPoints=61
                )(wGrid),
                tenor=marginal1.tenor
            )
            solutionsAfterConvolution.append(deepcopy(solution))

        cls.plot(
            x=wGrid,
            funcsValues=[
                solutionsAfterConvolution[0](wGrid) - solution(wGrid)
                for solution in solutionsAfterConvolution
            ],
            labels=[
                f"Iter: {funcNumber}"
                for funcNumber in range(len(solutionsAfterConvolution))
            ],
            title="Successive convolutions"
        )

    @classmethod
    def verifySolutionOfLinearizedFixedPointEquation(
            cls,
            marginal1: LogNormalMarginal = LogNormalMarginal(sigma=0.2, tenor=2.),
            marginal2: LogNormalMarginal = LogNormalMarginal(sigma=0.2, tenor=3.),
            wGrid: FloatVectorType = np.linspace(-5., 5., 1000) * np.sqrt(2.),
            cdfInterpolator: SolutionFixedPointEquation = SolutionFixedPointEquation
    ):
        cdfSolution = cls._fixedPointEquation.getSolutionOfLinearisedFixedPointEquation(
            marginal1=marginal1,
            marginal2=marginal2,
            wGrid=wGrid,
            solutionInterpolator=cdfInterpolator
        )

        dw = wGrid[1] - wGrid[0]
        delta = marginal2.tenor - marginal1.tenor

        cdfValues = cdfSolution(wGrid)
        dF_dw = np.gradient(cdfValues, dw, edge_order=2)
        d2F_dw2 = np.gradient(dF_dw, dw, edge_order=2)

        G_1 = marginal1.inverseCdf(cdfValues)
        G_2 = marginal2.inverseCdf(cdfValues)

        dG2_dy = marginal2.derivativeOfInverseCdf(cdfValues)
        d2G2_dy2 = marginal2.secondDerivativeOfInverseCdf(cdfValues)

        lhs = G_2 + delta * dG2_dy * d2F_dw2 + 0.5 * delta * (dF_dw ** 2) * d2G2_dy2

        cls.plotComparison(
            x=cdfValues,
            func1=lhs,
            func2=G_1,
            label1='left hand side',
            label2='right hand side = G_1(F(w))',
            generalTitle=r"Check solution of linearized fixed point eq: "
                         r"$G_{2}(F(w)) + \Delta G_{2}'(F(w))F''(w)"
                         r" + \frac{1}{2}\Delta F'(w)^{2}G_{2}''(F(w)) = G_{1}(F(w))$",
            generalLabelX='u = F(w)',
            generalLabelY='S'
        )
        print(f"L^infty norm: {(lhs - G_1).max()}")


    @classmethod
    def plotComparison(
            cls,
            x,
            func1,
            func2,
            label1: str,
            label2: str,
            generalTitle: str,
            diffTitle: str = 'diff',
            generalLabelX: str = '',
            generalLabelY: str = '',
            diffLabelX: str = '',
            diffLabelY: str = ''
    ):
        _, (generalAxis, diffAxis) = plt.subplots(nrows=1, ncols=2, figsize=(15, 8), dpi=120)
        generalAxis.plot(x, func1, label=label1)
        generalAxis.plot(x, func2, label=label2)
        generalAxis.set_title(generalTitle)
        generalAxis.set_xlabel(generalLabelX)
        generalAxis.set_ylabel(generalLabelY)
        generalAxis.legend()
        generalAxis.grid()

        diffAxis.plot(x, func1 - func2, label='diff')
        diffAxis.set_xlabel(diffLabelX)
        diffAxis.set_ylabel(diffLabelY)
        diffAxis.legend()
        diffAxis.set_title(diffTitle)
        diffAxis.grid()
        plt.show()

    @classmethod
    def _multiPlot(
            cls,
            multiPlotInputs: List[_MultiPlotInput]
    ):
        if len(multiPlotInputs) <= 4:
            rowsNumber = 1
            colsNumber = len(multiPlotInputs)
        elif len(multiPlotInputs) < 9:
            rowsNumber = 2
            colsNumber = int(len(multiPlotInputs) / 2)
        else:
            raise NotImplementedError()

        _, axes = plt.subplots(
            nrows=rowsNumber,
            ncols=colsNumber,
            figsize=(15, 8),
            dpi=120
        )

        for multiPlotInput, axis in zip(multiPlotInputs, axes):
            plotFields = fields(multiPlotInput)
            for field in plotFields:
                axisMethod = getattr(axis, field.name)
                inputs = getattr(multiPlotInput, field.name)
                if field.name == 'plot':
                    xArgs = inputs.get('args')[0]
                    yArgs = inputs.get('args')[1]

                    if  len(xArgs) > 1:
                        for inputIndex, x, y in zip(range(len(xArgs)), xArgs, yArgs):
                            kwargs = {
                                key: values[inputIndex]
                                for key, values in inputs.get("kwargs").items()
                            }
                            axisMethod(x, y, **kwargs)

                else:
                    axisMethod(*inputs.get('args'))

            axis.legend()
            axis.grid(True)

        plt.show()

    @classmethod
    def multiPlot(
            cls,
            inputs: List[FloatVectorType],
            funcsCollection: List[List[Callable[[FloatVectorType], FloatVectorType]]],
            labelsCollection: List[List[str]],
            titles: List[str]
    ):
        if len(funcsCollection) <= 4:
            rowsNumber = 1
            colsNumber = len(funcsCollection)
        elif len(funcsCollection) < 9:
            rowsNumber = 2
            colsNumber = int(len(funcsCollection) / 2)
        else:
            raise NotImplementedError()

        _, axes = plt.subplots(
            nrows=rowsNumber,
            ncols=colsNumber,
            figsize=(15, 8),
            dpi=120
        )

        for collectionIndex, axis in enumerate(axes):
            funcs = funcsCollection[collectionIndex]
            labels = labelsCollection[collectionIndex]
            title = titles[collectionIndex]
            input = inputs[collectionIndex]

            for func, label in zip(funcs, labels):
                axis.plot(input, func(input), label=label)
                axis.set_title(title)
                axis.legend()
                axis.grid(True)

        plt.show()

    @classmethod
    def plot(cls, x, funcsValues, labels, title):
        _, generalAxis = plt.subplots(figsize=(15, 8), dpi=120)
        for funcValues, label in zip(funcsValues, labels):
            generalAxis.plot(x, funcValues, label=label)

        generalAxis.set_title(title)
        generalAxis.legend()
        generalAxis.grid()
        plt.show()

    @classmethod
    def plotFuncs(cls, x, funcs, labels, title):
        _, generalAxis = plt.subplots(figsize=(15, 8), dpi=120)
        for func, label in zip(funcs, labels):
            generalAxis.plot(x, func(x), label=label)

        generalAxis.set_title(title)
        generalAxis.legend()
        generalAxis.grid()
        plt.show()
