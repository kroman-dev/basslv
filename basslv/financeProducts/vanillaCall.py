from basslv.financeProducts.vanillaOption import VanillaOption
from basslv.pricingEngine.genericPricingEngine import GenericPricingEngine

class VanillaCall(VanillaOption):

    def __init__(
            self,
            forward: float,
            discountFactor: float,
            timeToExpiry: float,
            pricingEngine: GenericPricingEngine
    ):
        super().__init__(
            forward=forward,
            discountFactor=discountFactor,
            timeToExpiry=timeToExpiry,
            optionType=1,
            pricingEngine=pricingEngine
        )
