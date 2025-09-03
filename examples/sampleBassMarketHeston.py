import time
import numpy as np

from basslv import BassLocalVolatility, VanillaCall, VanillaPut, MarketMarginal
from basslv import HestonPricingEngine, VisualVerification

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
    strikes = np.linspace(0.25, 3., 200, endpoint=True)
    callStrikes = strikes[np.where(strikes >= 1.)[0]]
    putStrikes = strikes[np.where(strikes < 1.)[0]]

    t = np.linspace(0, tenors[-1], timeGridPoints, endpoint=True)

    calls = [
        VanillaCall(
            forward=forward,
            discountFactor=discountFactor,
            timeToExpiry=tenor,
            pricingEngine=HestonPricingEngine(
                kappa=2.,
                theta=0.1,
                rho=-0.2,
                volOfVol=0.7,
                initialVariance=0.1
            )
        )
        for tenor in tenors
    ]

    puts = [
        VanillaPut(
            forward=forward,
            discountFactor=discountFactor,
            timeToExpiry=tenor,
            pricingEngine=HestonPricingEngine(
                kappa=2.,
                theta=0.1,
                rho=-0.2,
                volOfVol=0.7,
                initialVariance=0.1
            )
        )
        for tenor in tenors
    ]

    callPrices = [
        [
            call.NPV(strike=strike, volatility=None)
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

    pathsBassLv = bassLocalVolatility.sample(
        t=t,
        pathsNumber=pathsNumber,
        marginals=marginals,
        randomGenerator=np.random.default_rng(SEED),
        fixedPointEquationTolerance=1e-4,
        fixedPointEquationMaxIter=61
    )

    bassCallPrices = np.mean(np.maximum(pathsBassLv[:, 1][None] - strikes[None].T, 0), 1)
    bassCalls = np.mean(np.maximum(pathsBassLv[:, 1][None] - callStrikes[None].T, 0), 1)
    bassPuts = np.mean(np.maximum(putStrikes[None].T - pathsBassLv[:, 1][None], 0), 1)
    absError = np.abs(bassCallPrices - np.array(callPrices[0]))
    relativeError = np.abs(bassCallPrices - np.array(callPrices[0])) / np.array(callPrices[0])

    print(time.time() - startTime)

    volatilitySmilesTarget = [
        [
            call.getImpliedVolatility(
                strike=strikes[strikeIndex],
                optionPrice=callPrices[callIndex][strikeIndex]
            )
            for strikeIndex in range(len(strikes))
        ]
        for callIndex, call in enumerate(calls)
    ][0]

    volatilityCallSmilesBass = [
        calls[0].getImpliedVolatility(
            strike=callStrikes[strikeIndex],
            optionPrice=bassCalls[strikeIndex]
        )
        for strikeIndex in range(len(callStrikes))
    ]
    volatilityPutSmilesBass = [
        puts[0].getImpliedVolatility(
            strike=putStrikes[strikeIndex],
            optionPrice=bassPuts[strikeIndex]
        )
        for strikeIndex in range(len(putStrikes))
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
                    "args": ["calls relative error"]
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
                    "args": ["calls abs error"]
                }
            ),
            _MultiPlotInput(
                plot={
                    "args": [
                        [putStrikes.tolist(), callStrikes.tolist(), strikes.tolist()],
                        [volatilityPutSmilesBass, volatilityCallSmilesBass, volatilitySmilesTarget]
                    ],
                    "kwargs": {"label": ['bassPuts', 'bassCalls', 'target']}
                },
                set_title={
                    "args": ["compare smiles"]
                }
            ),
        ]
    )
