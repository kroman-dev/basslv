import time
import numpy as np
import matplotlib.pyplot as plt

from basslv import BassLocalVolatility, VanillaCall
from basslv import MarketMarginal, HestonPricingEngine, VisualVerification


if __name__ == '__main__':
    startTime = time.time()

    sigma = 0.2
    tenors = [1., 2.]
    pathsNumber = 10000
    timeGridPoints = 3
    SEED = 42

    forward = 1.
    discountFactor = 1
    strikes = np.linspace(0.25, 3., 200, endpoint=True)
    volatility = 0.2

    # t = np.array(tenors)
    t = np.linspace(0, tenors[-1], timeGridPoints, endpoint=True)
    # t = np.array([1., 2.])

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

    pathsBassLv = BassLocalVolatility().sample(
        t=t,
        pathsNumber=pathsNumber,
        marginals=marginals,
        randomGenerator=np.random.default_rng(SEED),
        fixedPointEquationTolerance=1e-4,
        fixedPointEquationMaxIter=11
    )

    bassCallPrices = np.mean(np.maximum(pathsBassLv[:, 1][None] - strikes[None].T, 0), 1)
    absError = np.abs(bassCallPrices - np.array(callPrices[0]))

    print(time.time() - startTime)

    VisualVerification.plot(
        x=strikes,
        funcsValues=[absError],
        labels=['abs error'],
        title='abs error: bass vs exact (heston)'
    )

    exit()

    _, axis = plt.subplots(figsize=(15, 5), dpi=200)

    handleGroundTruth, *_ = axis.plot(t, pathsGroundTruth.T, c="C0")
    handleBassLv, *_ = axis.plot(t, pathsBassLv.T, ls=":", c="C1")
    for tenor in tenors:
        handleTime = axis.axvline(tenor, c="C2")

    axis.set_xlabel("$t$")
    axis.set_ylabel("$S_t$")
    axis.set_title("Path by Bass LV model fitted to syntatic (flat vola) Market marginals")
    axis.legend(
        handles=[handleGroundTruth, handleBassLv, handleTime],
        labels=["Ground Truth", "Bass LV", "Marginals"],
    )

    plt.show()
