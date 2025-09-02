import time
import numpy as np

from basslv import BassLocalVolatility, VanillaCall, MarketMarginal
from basslv import HestonPricingEngine, VisualVerification

from basslv.core.fixedPointEquationDecorator import FixedPointEquationDecorator
from basslv.visualVerification.visualVerification import _MultiPlotInput


if __name__ == '__main__':

    startTime = time.time()

    sigma = 0.2
    timeGridPoints = 3
    # tenors = np.array([0., 1., 2.])
    tenors = [1., 2.]
    pathsNumber = 100000
    SEED = 42

    forward = 1.
    discountFactor = 1
    strikes = np.linspace(0.25, 3., 200, endpoint=True)
    volatility = 0.2

    t = np.linspace(0, tenors[-1], timeGridPoints, endpoint=True)

    calls = [
        VanillaCall(
            forward=forward,
            discountFactor=discountFactor,
            timeToExpiry=tenor,
            pricingEngine=HestonPricingEngine(
                kappa=0.1,
                theta=0.2,
                rho=-0.,
                volOfVol=0.01,
                initialVariance=0.1
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

    marginals = [
        MarketMarginal(
            strikes=strikes,
            callPrices=callPrices[tenorIndex],
            tenor=tenors[tenorIndex]
        )
        for tenorIndex in range(len(tenors))
    ]

    bassLocalVolatility = BassLocalVolatility()
    bassLocalVolatility.setFixedPointEquation(FixedPointEquationDecorator())

    pathsBassLv = bassLocalVolatility.sample(
        t=t,
        pathsNumber=pathsNumber,
        marginals=marginals,
        randomGenerator=np.random.default_rng(SEED),
        fixedPointEquationTolerance=1e-4,
        fixedPointEquationMaxIter=11
    )

    bassCallPrices = np.mean(np.maximum(pathsBassLv[:, 1][None] - strikes[None].T, 0), 1)
    absError = np.abs(bassCallPrices - np.array(callPrices[0]))
    relativeError = np.abs(bassCallPrices - np.array(callPrices[0])) / np.array(callPrices[0])

    print(time.time() - startTime)

    VisualVerification._multiPlot(
        [
            _MultiPlotInput(
                plot={
                    "args": [
                        [strikes],
                        [relativeError]
                    ]
                },
                set_title={
                    "args": ["relative error"]
                }
            ),
            _MultiPlotInput(
                plot={
                    "args": [
                        [strikes],
                        [absError]
                    ]
                },
                set_title={
                    "args": ["abs error"]
                }
            ),
        ]
    )
