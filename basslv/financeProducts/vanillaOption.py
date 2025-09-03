from functools import singledispatchmethod

from basslv.pricingEngine.genericPricingEngine import GenericPricingEngine
from basslv.pricingEngine.blackPricingEngine import BlackPricingEngine
from basslv.pricingEngine.hestonPricingEngine import HestonPricingEngine


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

    @property
    def timeToExpiry(self):
        return self._timeToExpiry

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
            volatility=volatility,
            optionType=self._optionType
        )

    @_NPV.register
    def _(
            self,
            pricingEngine: HestonPricingEngine,
            strike: float,
            volatility: float
    ) -> float:
        return pricingEngine.getOptionPrice(
            forward=self._forward,
            discountFactor=self._discountFactor,
            timeToExpiry=self._timeToExpiry,
            strike=strike,
            volatility=volatility,
            optionType=self._optionType
        )

    def getImpliedVolatility(
            self,
            strike: float,
            optionPrice: float
    ) -> float:
        return self._pricingEngine.getImpliedVolatility(
            optionPrice=optionPrice,
            forward=self._forward,
            discountFactor=self._discountFactor,
            timeToExpiry=self._timeToExpiry,
            strike=strike,
            optionType=self._optionType
        )
