import numpy as np
import matplotlib.pyplot as plt

from basslv import BassLocalVolatility, VanillaCall, BlackPricingEngine, MarketMarginal


if __name__ == '__main__':

    sigma = 0.2
    tenors = [1., 2.]
    timeGridPoints = 1000
    pathsNumber = 5
    SEED = 42

    forward = 1.
    discountFactor = 1
    strikes = np.linspace(0.25, 3., 200, endpoint=True)
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

    marginals = [
        MarketMarginal(
            strikes=strikes,
            callPrices=callPrices[callIndex],
            tenor=tenors[callIndex]
        )
        for callIndex in range(len(calls))
    ]

    t = np.linspace(0, tenors[-1], timeGridPoints)

    randomGenerator = np.random.default_rng(SEED)
    brownianMotion = np.cumsum(
        randomGenerator.normal(
            size=(pathsNumber, timeGridPoints)
        ) * np.sqrt(np.diff(t, prepend=0)),
        axis=1
    )
    pathsGroundTruth = np.exp(sigma * brownianMotion - sigma ** 2 * t / 2)

    pathsBassLv = BassLocalVolatility().sample(
        t=t,
        pathsNumber=pathsNumber,
        marginals=marginals,
        randomGenerator=np.random.default_rng(SEED),
        fixedPointEquationTolerance=3e-2,
        fixedPointEquationMaxIter=10
    )

    _, axis = plt.subplots(figsize=(15, 5), dpi=200)

    handleGroundTruth, *_ = axis.plot(t, pathsGroundTruth.T, c="C0")
    handleBassLv, *_ = axis.plot(t, np.exp(pathsBassLv.T), ls=":", c="C1")
    for tenor in tenors:
        handleTime = axis.axvline(tenor, c="C2")

    axis.set_xlabel("$t$")
    axis.set_ylabel("$S_t$")
    axis.set_title("Path by Bass LV model fitted to Black-Scholes marginals")
    axis.legend(
        handles=[handleGroundTruth, handleBassLv, handleTime],
        labels=["Ground Truth", "Bass LV", "Marginals"],
    )

    plt.show()
