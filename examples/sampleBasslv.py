import numpy as np
import matplotlib.pyplot as plt

from basslv import LogNormalMarginal
from basslv import BassLocalVolatility


if __name__ == '__main__':

    sigma = 0.2
    tenors = [1., 5., 10., 15., 20.]
    timeGridPoints = 1000
    pathsNumber = 5
    SEED = 0xB0BA

    marginals = [LogNormalMarginal(sigma=sigma, tenor=t) for t in tenors]

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
        fixedPointEquationTolerance=1e-4
    )

    _, axis = plt.subplots(figsize=(15, 5), dpi=200)

    handleGroundTruth, *_ = axis.plot(t, pathsGroundTruth.T, c="C0")
    handleBassLv, *_ = axis.plot(t, pathsBassLv.T, ls=":", c="C1")
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
