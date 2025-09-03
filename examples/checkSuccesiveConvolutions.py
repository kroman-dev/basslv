import numpy as np

from basslv import VisualVerification
from basslv import SolutionFixedPointEquation
from basslv import LogNormalMarginal


if __name__ == '__main__':

    tenor = 2.
    VisualVerification.verifyConvolutionAccuracy(
        func=VisualVerification._fixedPointEquation.getExactSolutionOfFixedPointEquationInLogNormalCase(tenor),
        marginal1 = LogNormalMarginal(sigma=0.2, tenor=2.),
        marginal2 = LogNormalMarginal(sigma=0.2, tenor=3.),
        wGrid=np.linspace(-5., 5., 2001, endpoint=True) * np.sqrt(tenor),
        cdfInterpolator=SolutionFixedPointEquation,
        convolutionIterations=50
    )
