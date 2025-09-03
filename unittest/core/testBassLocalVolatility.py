import numpy as np

from unittest import TestCase
from numpy.testing import assert_almost_equal

from basslv import BassLocalVolatility
from basslv import LogNormalMarginal


class TestConvolution(TestCase):

    def setUp(self):
        self._sampleBassLocalVolatility = BassLocalVolatility()
        expiries = [2., 3.]
        vols = [0.2] * 2

        self._marginal1, self._marginal2 = [
            LogNormalMarginal(sigma=sigma, tenor=T)
            for sigma, T in zip(vols, expiries)
        ]

    def testBuildOneMarginalMappingFunction(self):
        testedTime = 0.2
        sigma = 0.2
        sampleMarginal = LogNormalMarginal(sigma=sigma, tenor=0.5)
        exactMarginal = sampleMarginal.getMappingFunction()

        testedMappingFunction = \
            self._sampleBassLocalVolatility._buildFirstMarginalMappingFunction(
                sampleMarginal
        )

        wGrid = np.linspace(-3., 3., 2001, endpoint=True) \
                * np.sqrt(sampleMarginal.tenor)

        assert_almost_equal(
            testedMappingFunction(testedTime, wGrid),
            exactMarginal(testedTime, wGrid)
        )
