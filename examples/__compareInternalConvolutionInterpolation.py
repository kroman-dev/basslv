import numpy as np

from basslv import (MappingFunction, LogNormalMarginal,
                    FixedPointEquation, SolutionFixedPointEquation, VisualVerification)


if __name__ == '__main__':
    expiries = [2., 3.]
    vols = [0.5] * 2

    marginal1, marginal2 = [
        LogNormalMarginal(sigma=sigma, tenor=T)
        for sigma, T in zip(vols, expiries)
    ]
    _solution = FixedPointEquation\
        .getExactSolutionOfFixedPointEquationInLogNormalCase(expiries[0])


    x = np.linspace(-5., 5., 5001, endpoint=True) * np.sqrt(marginal1.tenor)
    solution = SolutionFixedPointEquation(x, _solution(x), tenor=marginal1.tenor)

    mappingFuncFast = MappingFunction(
        marginal1,
        marginal2,
        solution,
        SolutionFixedPointEquation
    ).getMappingFunction(expiries[0])

    MappingFunction.setSaveInternalConvolution(False)

    mappingFuncSlow = MappingFunction(
        marginal1,
        marginal2,
        solution,
        SolutionFixedPointEquation
    ).getMappingFunction(expiries[0])

    exactMappingFunction = lambda wiener: np.exp(
        -0.5 * marginal1._sigma ** 2 * marginal1.tenor \
        + marginal1._sigma * wiener
    )

    VisualVerification.plotFuncs(
        x,
        [exactMappingFunction, mappingFuncFast, mappingFuncSlow],
        labels=['exact', 'fast', 'slow']
    )
