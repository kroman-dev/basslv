import numpy as np

from unittest import TestCase

from basslv import LogNormalMarginal

EPS = 1e-15


class TestLogNormalMarginal(TestCase):

    def setUp(self):
        self._sigma = 0.2
        self._tenor = 2.
        self._median = np.exp(-0.5 * self._sigma**2 * self._tenor)
        self._sampleMarginal = LogNormalMarginal(
            sigma=self._sigma,
            tenor=self._tenor,
        )

    def testCdf(self):
        with self.subTest("infty"):
            self.assertAlmostEqual(
                self._sampleMarginal.cdf(100000000.),
                1.
            )

        with self.subTest("zero"):
            self.assertAlmostEqual(
                self._sampleMarginal.cdf(EPS),
                0.
            )

        with self.subTest("median"):
            self.assertAlmostEqual(
                self._sampleMarginal.cdf(self._median),
                0.5
            )
            for actual, excpected in zip(
                self._sampleMarginal.cdf([self._median, self._median]),
                [0.5, 0.5]
            ):
                self.assertAlmostEqual(actual, excpected)

        with self.subTest("exceptions"):
            self._sampleMarginal.cdf([100000000., EPS])

            with self.assertRaises(ValueError):
                self._sampleMarginal.cdf(0.)

            with self.assertRaises(ValueError):
                self._sampleMarginal.cdf(-100000000.)

            with self.assertRaises(ValueError):
                self._sampleMarginal.cdf([-100000000., EPS])

            with self.assertRaises(TypeError):
                self._sampleMarginal.cdf(['-100000000.', EPS])

    def testInverseCdf(self):
        with self.subTest('median'):
            self.assertAlmostEqual(
                self._sampleMarginal.inverseCdf(0.5),
                self._median
            )
            self.assertAlmostEqual(
                self._sampleMarginal.inverseCdf([0.5, 0.5]).tolist(),
                [self._median, self._median]
            )

        with self.subTest("exceptions"):
            self._sampleMarginal.inverseCdf(0.)

            with self.assertRaises(ValueError):
                self._sampleMarginal.inverseCdf(-0.1)
            with self.assertRaises(ValueError):
                self._sampleMarginal.inverseCdf(1.1)
