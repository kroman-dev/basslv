import numpy as np

from scipy.stats import norm

from basslv import FixedPointEquation
from basslv import VanillaCall, BlackPricingEngine, MarketMarginal
from basslv import SolutionFixedPointEquation
from basslv import VisualVerification


if __name__ == '__main__':

    fixedPointEquation = FixedPointEquation()
    exactFixedPointSolution = lambda x: norm.cdf(x / np.sqrt(marginal1.tenor))
    tenors = [1., 2.]
    forward = 1.
    discountFactor = 1
    strikes = np.linspace(0.25, 3, 100, endpoint=True)
    volatility = 0.2

    calls = [
        VanillaCall(
            forward=forward,
            discountFactor=discountFactor,
            timeToExpiry=tenor,
            pricingEngine=BlackPricingEngine()
        )
        for tenor in tenors
    ]

    callPrices = [
        [
            call.NPV(strike=strike, volatility=volatility)
            for strike in strikes
        ]
        for call in calls
    ]

    marginal1, marginal2 = [
        MarketMarginal(
            strikes=strikes,
            callPrices=callPrices[callIndex],
            tenor=tenors[callIndex]
        )
        for callIndex in range(len(calls))
    ]

    numericalSolution = fixedPointEquation.solveFixedPointEquation(
        marginal1=marginal1,
        marginal2=marginal2,
        maxIter=21,
        tol=1e-3,
        gridBound=5.,
        gridPoints=2001,
        hermGaussPoints=61,
        solutionInterpolator=SolutionFixedPointEquation
    )

    print(numericalSolution(0.))

    x = np.linspace(-5., 5., 5001, endpoint=True) * np.sqrt(marginal1.tenor)

    VisualVerification.plotComparison(
        x=x,
        func1=numericalSolution(x),
        func2=exactFixedPointSolution(x),
        label1='numericalSolution',
        label2='exact',
        generalTitle="title",
        diffTitle='diff',
    )

