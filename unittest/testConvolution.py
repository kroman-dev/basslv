import numpy as np

from unittest import TestCase
from scipy.stats import norm
from numpy.testing import assert_almost_equal

from basslv import HeatKernelConvolutionEngine


class TestConvolution(TestCase):

    def setUp(self):
        self._sampleEngine = HeatKernelConvolutionEngine()

    def testHermitGauss(self):
        getSquare = lambda x: x ** 2
        calcMethod = self._sampleEngine._useGaussHermiteQuadrature(
            time=1.,
            func=getSquare,
            hermgaussPoints=20
        )

        with self.subTest('square array'):
            # answer: x**2 + 1.
            for actual, expected in zip(
                    calcMethod(np.array([2., 3.])).tolist(), [5., 10.]
            ):
                self.assertAlmostEqual(
                    actual,
                    expected
                )

        calcMethod = self._sampleEngine._useGaussHermiteQuadrature(
            time=0.5,
            func=getSquare,
            hermgaussPoints=20
        )

        with self.subTest('square1'):
            # answer: x**2 + 0.5
            self.assertAlmostEqual(
                4.5,
                calcMethod([2]).tolist()[0]
            )

        with self.subTest("gaussian"):
            sigma = 0.5
            t = 0.1

            exactSolution = lambda x: 1.0 / np.sqrt(1 + 2 * t / sigma ** 2) \
                                      * np.exp(-x ** 2 / (2*t + sigma ** 2))

            calcMethod = self._sampleEngine._useGaussHermiteQuadrature(
                time=t,
                func=lambda x: np.exp(-(x / sigma) ** 2),
                hermgaussPoints=40
            )

            self.assertAlmostEqual(
                calcMethod([1.5]).tolist()[0],
                exactSolution(1.5)
            )

    def testHermitGaussVec(self):
        getSquare = lambda x: x ** 2
        calcMethod = self._sampleEngine.useGaussHermiteQuadrature(
            time=np.array([1., 1.]),
            func=getSquare,
            hermgaussPoints=20
        )
        assert_almost_equal(calcMethod([2., 3.]), [5., 10.])

    def testConvolutionWithGaussKernel(self):
        # N(T_1) * Kernel = N(T_2)
        # r"$K_{T_2 - T_1} \star N(\cdot/\sqrt{T_1}) = N(\cdot/\sqrt{T_2})$"
        tenor1 = 2.
        tenor2 = 3.
        timeDelta = tenor2 - tenor1
        testedFunction = lambda x: norm.cdf(x / np.sqrt(tenor1))
        desiredFunction = lambda x: norm.cdf(x / np.sqrt(tenor2))

        actualFunction = self._sampleEngine.useGaussHermiteQuadrature(
            time=timeDelta,
            func=testedFunction,
            hermgaussPoints=61
        )

        wGrid = np.linspace(-5., 5., 2001, endpoint=True) * np.sqrt(tenor1)
        assert_almost_equal(
            actualFunction(wGrid),
            desiredFunction(wGrid)
        )

