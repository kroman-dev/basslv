from functools import singledispatchmethod

from basslv.pricingEngine.genericPricingEngine import GenericPricingEngine
from basslv.pricingEngine.blackPricingEngine import BlackPricingEngine


class VanillaOption:

    def __init__(
            self,
            forward: float,
            discountFactor: float,
            timeToExpiry: float,
            optionType: int,
            pricingEngine: GenericPricingEngine
    ):
        self._forward = forward
        self._discountFactor = discountFactor
        self._timeToExpiry = timeToExpiry
        self._optionType = optionType
        self._pricingEngine = pricingEngine

    def NPV(self, strike: float, volatility: float) -> float:
        return self._NPV(self._pricingEngine, strike, volatility)

    @singledispatchmethod
    def _NPV(
            self,
            pricingEngine: GenericPricingEngine
    ) -> float:
        raise NotImplementedError(f"pricingEngine={pricingEngine}")

    @_NPV.register
    def _(
            self,
            pricingEngine: BlackPricingEngine,
            strike: float,
            volatility: float
    ) -> float:
        return pricingEngine.getOptionPrice(
            forward=self._forward,
            discountFactor=self._discountFactor,
            timeToExpiry=self._timeToExpiry,
            strike=strike,
            vol=volatility,
            optionType=self._optionType
        )

