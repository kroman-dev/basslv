import numpy as np

from unittest import TestCase
from numpy.testing import assert_almost_equal

from basslv import FixedPointEquation
from basslv import LogNormalMarginal


class TestConvolution(TestCase):

    def setUp(self):
        self._sampleFixedPointEquation = FixedPointEquation()
        expiries = [2., 3.]
        vols = [0.2] * 2

        self._marginal1, self._marginal2 = [
            LogNormalMarginal(sigma=sigma, tenor=T)
            for sigma, T in zip(vols, expiries)
        ]

    def testApplyOperator(self):
        exactSolution = self._sampleFixedPointEquation \
            .getExactSolutionOfFixedPointEquationInLogNormalCase(
                self._marginal1.tenor
        )
        solutionAfterOperatorAction = \
            self._sampleFixedPointEquation.applyOperatorA(
                solution=exactSolution,
                marginal1=self._marginal1,
                marginal2=self._marginal2,
                hermGaussPoints=61
        )
        wGrid = np.linspace(-5., 5., 2001, endpoint=True) * np.sqrt(self._marginal1.tenor)
        with self.subTest("exact solution after operator action"):
            assert_almost_equal(
                solutionAfterOperatorAction(wGrid),
                exactSolution(wGrid)
            )
