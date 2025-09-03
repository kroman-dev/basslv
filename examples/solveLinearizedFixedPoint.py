import numpy as np

from basslv import VisualVerification
from basslv import SolutionFixedPointEquation
from basslv import LogNormalMarginal


if __name__ == '__main__':
    tenor1 = 2.
    tenor2 = 3.
    VisualVerification.verifySolutionOfLinearizedFixedPointEquation(
        marginal1=LogNormalMarginal(sigma=0.2, tenor=tenor1),
        marginal2=LogNormalMarginal(sigma=0.2, tenor=tenor2),
        wGrid=np.linspace(-5., 5., 1000) * np.sqrt(tenor1)
    )
