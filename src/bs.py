import numpy as np
import wrapt
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_lets_be_rational.exceptions import (
    BelowIntrinsicException,
    AboveMaximumException,
)


INTEREST_RATE = 0.0


@wrapt.decorator
def vectorized(wrapped, instance, args, kwargs):
    return np.vectorize(wrapped)(*args, **kwargs)


@vectorized
def call_price(spot, strike, tenor, vol):
    return black_scholes("c", spot, strike, tenor, INTEREST_RATE, vol)


@vectorized
def call_iv(spot, strike, tenor, call_price):
    try:
        return implied_volatility(call_price, spot, strike, tenor, INTEREST_RATE, "c")
    except (BelowIntrinsicException, AboveMaximumException):
        return np.nan


def markov_func(t, w, volatility):
    return np.exp(volatility * w - volatility**2 * t / 2)
