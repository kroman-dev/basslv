import numpy as np

from scipy.integrate import quad
from typing import Optional

from basslv.pricingEngine.genericPricingEngine import GenericPricingEngine
from basslv.pricingEngine.blackImpliedVolatility import BlackImpliedVolatility


class HestonPricingEngine(BlackImpliedVolatility, GenericPricingEngine):

    def __init__(
            self,
            kappa: float,
            theta: float,
            rho: float,
            volOfVol: float,
            initialVariance: float,
    ):
        self._kappa = kappa
        self._theta = theta
        self._rho = rho
        self._volvol = volOfVol
        self._initialVariance = initialVariance

    def calcCharacteristicFunction(
            self,
            u: float,
            timeToExpiry: float,
            initialVariance: float,
            logForward: float
    ) -> complex:
        # let timeToPricingDate = 0
        omegaSquare = np.power(self._volvol, 2)
        auxiliaryVariable = self._rho * self._volvol * 1j * u
        d = np.sqrt(np.power(auxiliaryVariable - self._kappa, 2) \
                    + 1j * u * omegaSquare + omegaSquare * np.power(u, 2))
        c = (self._kappa - d - auxiliaryVariable) \
            / (self._kappa + d - auxiliaryVariable)

        C = self._kappa * self._theta / omegaSquare * (
                (self._kappa - auxiliaryVariable - d) * timeToExpiry
                - 2 * np.log((1 - c * np.exp(-d * timeToExpiry)) / (1 - c))
        )
        D = (self._kappa - d - auxiliaryVariable) \
            / omegaSquare * (1 - np.exp(-d * timeToExpiry)) \
            / (1 - c * np.exp(-d * timeToExpiry))
        return np.exp(C + D * initialVariance + 1j * u * logForward)

    def integrand(
            self,
            u,
            forward,
            strike,
            initialVariance,
            timeToExpiry
    ) -> float:
        logStrike = np.log(strike)
        logForward = np.log(forward)
        firstTerm = np.exp(-1j * u * logStrike) / (1j * u * forward) \
            * self.calcCharacteristicFunction(
                  u=u - 1j,
                  timeToExpiry=timeToExpiry,
                  initialVariance=initialVariance,
                  logForward=logForward
              )
        secondTerm = np.exp(-1j * u * logStrike) / (1j * u) * self.calcCharacteristicFunction(
            u=u,
            timeToExpiry=timeToExpiry,
            initialVariance=initialVariance,
            logForward=logForward
        )
        return (forward * firstTerm - strike * secondTerm).real

    def getUndiscountedPrice(
            self,
            forward: float,
            strike: float,
            initialVariance: float,
            timeToExpiry: float
    ) -> float:
        inverse_pi = 1 / np.pi
        opt_price = 0.5 * (forward - strike)
        opt_price += inverse_pi * quad(
            self.integrand,
            0,
            np.inf,
            args=(forward, strike, initialVariance, timeToExpiry),
        )[0]
        return opt_price

    def getOptionPrice(
            self,
            forward: float,
            discountFactor: float,
            timeToExpiry: float,
            strike: float,
            volatility: Optional[float],
            optionType: int,
    ) -> float:
        if optionType != 1:
            raise NotImplementedError()

        undiscountedPrice = self.getUndiscountedPrice(
            forward=forward,
            strike=strike,
            initialVariance=self._initialVariance,
            timeToExpiry=timeToExpiry
        )
        return discountFactor * undiscountedPrice


if __name__ == '__main__':
    # TODO to del
    k = 2
    theta = 0.1
    rho = -0.2
    volvol = 0.7
    v_0 = 0.1

    rate = 0.02
    T = 1.
    df = np.exp(-rate * T)
    forward = 1.
    spot = forward * df
    strike = 1.0
    vol = v_0
    opt_type = 1

    heston_engine = HestonPricingEngine(k, theta, rho, volvol, v_0)
    heston_call_price = heston_engine.getOptionPrice(forward, df, T, strike, opt_type)
    print(heston_call_price)  # 0.11571599117319048
    print(0.11571599146960894 - heston_call_price)

    # test 2
    actual = 11.337773512150262
    forward = 100
    r = 0.02
    T = 1.5
    df = np.exp(-r * T)
    strike = 110
    v_0 = 0.15

    heston_engine = HestonPricingEngine(k, theta, rho, volvol, v_0)
    heston_call_price = heston_engine.getOptionPrice(forward, df, T, strike, opt_type)
    print(heston_call_price)
    print(actual - heston_call_price)
