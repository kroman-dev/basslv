import numpy as np

from basslv import (
    MappingFunction, LogNormalMarginal,
    FixedPointEquation, SolutionFixedPointEquation,
    VisualVerification
)


if __name__ == '__main__':
    expiries = [2., 3.]
    vols = [0.5] * 2

    marginal1, marginal2 = [
        LogNormalMarginal(sigma=sigma, tenor=T)
        for sigma, T in zip(vols, expiries)
    ]
    _solution1 = FixedPointEquation\
        .getExactSolutionOfFixedPointEquationInLogNormalCase(expiries[0])
    _solution2 = FixedPointEquation\
        .getExactSolutionOfFixedPointEquationInLogNormalCase(expiries[1])

    x = np.linspace(-5., 5., 5001, endpoint=True) * np.sqrt(marginal1.tenor)
    solution = SolutionFixedPointEquation(
        x=x,
        y=_solution1(x),
        tenor=marginal1.tenor
    )

    mappingFunc = MappingFunction(
        marginal1=marginal1,
        marginal2=marginal2,
        solutionOfFixedPointEquation=solution,
        solutionInterpolatorConstructor=SolutionFixedPointEquation
    )
    solutionAfterConvolution = mappingFunc._calculateInternalConvolution(solution)

    solutionAfterConvolutionInterpolation = SolutionFixedPointEquation(
        x=x,
        y=solutionAfterConvolution(x)[0],
        tenor=marginal2.tenor,
        extrapolation=False
    )

    VisualVerification.plotComparison(
        x,
        funcValues1=_solution2(x),
        funcValues2=solutionAfterConvolution(x)[0],
        label1='exact',
        label2='slow',
        generalTitle='compare'
    )

    VisualVerification.plotComparison(
        x,
        funcValues1=solutionAfterConvolution(x)[0],
        funcValues2=solutionAfterConvolutionInterpolation(x),
        label1='exact',
        label2='slow',
        generalTitle='compare'
    )

    VisualVerification.plot(
        x,
        [_solution2(x), solutionAfterConvolution(x)[0], mappingFunc._internalConvolution(x)],
        labels=['exact', 'slow', 'fast']
    )
