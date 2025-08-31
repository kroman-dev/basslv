import py_lets_be_rational as lets_be_rational

from basslv.pricingEngine.blackImpliedVolatility import BlackImpliedVolatility
from basslv.pricingEngine.genericPricingEngine import GenericPricingEngine


class BlackPricingEngine(BlackImpliedVolatility, GenericPricingEngine):

    def getOptionPrice(
            self,
            forward: float,
            discountFactor: float,
            timeToExpiry: float,
            strike: float,
            volatility: float,
            optionType: int
    ) -> float:
        if optionType not in [-1, 1]:
            raise ValueError('optionType must be [-1, 1]')

        return discountFactor * lets_be_rational.black(
            F=forward,
            K=strike,
            sigma=volatility,
            T=timeToExpiry,
            q=optionType
        )
