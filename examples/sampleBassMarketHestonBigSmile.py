import time
import numpy as np

from basslv import BassLocalVolatility, VanillaCall, MarketMarginal
from basslv import HestonPricingEngine, VisualVerification

from basslv.core.fixedPointEquation import FixedPointEquation
from basslv.visualVerification.visualVerification import _MultiPlotInput


if __name__ == '__main__':

    startTime = time.time()

    timeGridPoints = 3
    # tenors = np.array([0., 1., 2.])
    tenors = [1., 2.]
    pathsNumber = 100000
    SEED = 42

    forward = 1.
    discountFactor = 1
    strikes = np.linspace(0.25, 2.55, 200, endpoint=True)
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
                rho=-0.5,
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
    bassLocalVolatility.setFixedPointEquation(FixedPointEquation())

    pathsBassLv = bassLocalVolatility.sample(
        t=t,
        pathsNumber=pathsNumber,
        marginals=marginals,
        randomGenerator=np.random.default_rng(SEED),
        fixedPointEquationTolerance=1e-4,
        fixedPointEquationMaxIter=61
    )

    bassCallPrices = np.mean(np.maximum(pathsBassLv[:, 1][None] - strikes[None].T, 0), 1)
    absError = np.abs(bassCallPrices - np.array(callPrices[0]))
    relativeError = np.abs(bassCallPrices - np.array(callPrices[0])) / np.array(callPrices[0])

    print(time.time() - startTime)

    volatilitySmilesTarget = [
        [
            call.getImpliedVolatility(
                strike=strikes[strikeIndex],
                optionPrice=callPrices[callIndex][strikeIndex]
            )
            for strikeIndex in range(30, len(strikes)-50)
        ]
        for callIndex, call in enumerate(calls)
    ][0]

    volatilitySmilesBass = [
        calls[0].getImpliedVolatility(
            strike=strikes[strikeIndex],
            optionPrice=bassCallPrices[strikeIndex]
        )
        for strikeIndex in range(30, len(strikes)-50)
    ]

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
            _MultiPlotInput(
                plot={
                    "args": [
                        [strikes[30:-50], strikes[30:-50]],
                        [volatilitySmilesBass, volatilitySmilesTarget]
                    ],
                    "kwargs": {"label": ['bass', 'target']}
                },
                set_title={
                    "args": ["abs error"]
                }
            ),
        ]
    )
