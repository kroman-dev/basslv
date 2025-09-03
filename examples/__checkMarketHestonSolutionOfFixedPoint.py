import numpy as np

from scipy.stats import norm

from basslv import FixedPointEquation
from basslv import VanillaCall, HestonPricingEngine, MarketMarginal
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
            pricingEngine=HestonPricingEngine(
                kappa=0.1,
                theta=0.2,
                rho=-0.5,
                volOfVol=0.01,
                initialVariance=0.2
            )
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

    volatilitySmiles = [
        [
            call.getImpliedVolatility(
                strike=strikes[strikeIndex],
                optionPrice=callPrices[callIndex][strikeIndex]
            )
            for strikeIndex in range(len(strikes))
        ]
        for callIndex, call in enumerate(calls)
    ]

    VisualVerification.plot(
        x=strikes,
        funcsValues=volatilitySmiles,
        labels=['smile T=1', 'smile T=2'],
        title="Smiles from Heston"
    )

    marginal1, marginal2 = [
        MarketMarginal(
            strikes=strikes,
            callPrices=callPrices[tenorIndex],
            tenor=tenors[tenorIndex]
        )
        for tenorIndex in range(len(tenors))
    ]

    numericalSolution = fixedPointEquation.solveFixedPointEquation(
        marginal1=marginal1,
        marginal2=marginal2,
        maxIter=61,
        tol=1e-4,
        gridBound=5.,
        gridPoints=2001
    )

    print(numericalSolution(0.))

    x = np.linspace(-5., 5., 5001, endpoint=True) * np.sqrt(marginal1.tenor)

    VisualVerification.plot(
        x=x,
        funcsValues=[numericalSolution(x)],
        labels=['solution'],
        title="Solution of fixed point equation, with synthetic market marginals (heston)"
    )

    print(max(numericalSolution(x)))
