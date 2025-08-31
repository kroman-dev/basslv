from py_vollib.black_scholes.implied_volatility import iv
from py_vollib.helpers.constants import FLOAT_MAX, MINUS_FLOAT_MAX
from py_vollib.helpers.exceptions import PriceIsAboveMaximum, PriceIsBelowIntrinsic


class BlackImpliedVolatility(object):

    @staticmethod
    def getImpliedVolatility(
            optionPrice: float,
            forward: float,
            discountFactor: float,
            timeToExpiry: float,
            strike: float,
            optionType: int
    ) -> float:

        if optionType not in [-1, 1]:
            raise ValueError('optionType be [-1, 1]')

        undiscountedOptionPrice = optionPrice / discountFactor
        impliedVolatility = iv(undiscountedOptionPrice, forward, strike, timeToExpiry, optionType)

        if impliedVolatility == FLOAT_MAX:
            raise PriceIsAboveMaximum()
        elif impliedVolatility == MINUS_FLOAT_MAX:
            raise PriceIsBelowIntrinsic()
        return impliedVolatility
