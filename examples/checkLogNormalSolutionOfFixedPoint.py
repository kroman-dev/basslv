import numpy as np

from scipy.stats import norm

from basslv import FixedPointEquation
from basslv import LogNormalMarginal
from basslv import SolutionFixedPointEquation
from basslv import VisualVerification


if __name__ == '__main__':

    expiries = [2., 3.]
    vols = [0.5] * 2

    marginal1, marginal2 = [
        LogNormalMarginal(sigma=sigma, tenor=T)
        for sigma, T in zip(vols, expiries)
    ]

    exactFixedPointSolution = lambda x: norm.cdf(x / np.sqrt(marginal1.tenor))

    numericalSolution = FixedPointEquation().solveFixedPointEquation(
        marginal1=marginal1,
        marginal2=marginal2,
        maxIter=61,
        tol=1e-5,
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

