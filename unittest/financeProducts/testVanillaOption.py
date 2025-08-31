from unittest import TestCase
from basslv.pricingEngine.blackPricingEngine import BlackPricingEngine
from basslv.financeProducts.vanillaOption import VanillaOption


class TestVanillaOption(TestCase):

    def setUp(self):

        self._forward = 15.
        self._discountFactor = 0.5
        self._strike = 10.
        self._volatility = 0.2

        self._blackVanillaCall = VanillaOption(
            forward=self._forward,
            discountFactor=self._discountFactor,
            timeToExpiry=2.,
            optionType=1,
            pricingEngine=BlackPricingEngine()
        )

        self._blackVanillaPut = VanillaOption(
            forward=self._forward,
            discountFactor=self._discountFactor,
            timeToExpiry=2.,
            optionType=-1,
            pricingEngine=BlackPricingEngine()
        )

    def testCallPutParity(self):
        callPrice = self._blackVanillaCall.NPV(self._strike, self._volatility)
        putPrice = self._blackVanillaPut.NPV(self._strike, self._volatility)

        self.assertAlmostEqual(
            self._discountFactor * (self._forward - self._strike),
            callPrice - putPrice
        )

    def testImpliedVolatility(self):
        callPrice = self._blackVanillaCall.NPV(self._strike, self._volatility)
        iv = self._blackVanillaCall.getImpliedVolatility(
            strike=self._strike,
            optionPrice=callPrice
        )

        self.assertAlmostEqual(
            self._volatility,
            iv
        )
