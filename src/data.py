from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtr, ndtri


EPS = np.finfo(np.float64).eps
FloatArray = NDArray[np.float64]
Floats = FloatArray | float


class Marginal(ABC):

    @property
    @abstractmethod
    def tenor(self) -> float:
        pass

    @abstractmethod
    def cdf(self, x: Floats) -> Floats:
        pass

    @abstractmethod
    def icdf(self, u: Floats) -> Floats:
        pass

    @abstractmethod
    def iicdf(self, u: Floats) -> Floats:
        pass

    @abstractmethod
    def dicdf(self, u: Floats) -> Floats:
        pass


class MarginalInterp(Marginal):
    def __init__(
        self,
        strikes: np.ndarray,
        cdf_vals: np.ndarray,
        tenor: float,
    ):
        assert len(strikes > 3)
        self._t = tenor
        self._nodes = strikes
        self._cdf_vals = cdf_vals
        assert all(self._cdf_vals <= 1) and all(self._cdf_vals >= 0)
        assert all(np.gradient(self._cdf_vals) >= 0)

        # linear extrapolation of cdf
        k1, k2 = self._nodes[-2:]
        y1, y2 = self._cdf_vals[-2:]
        k_max = k1 + (1 - y1) / (y2 - y1) * (k2 - k1)
        self._nodes = np.append(self._nodes, k_max)
        self._cdf_vals = np.append(self._cdf_vals, 1 - EPS)

        k1, k2 = self._nodes[:2]
        y1, y2 = self._cdf_vals[:2]
        k_min = k1 + (0 - y1) / (y2 - y1) * (k2 - k1)
        self._nodes = np.insert(self._nodes, 0, k_min)
        self._cdf_vals = np.insert(self._cdf_vals, 0, EPS)

        self._cdf = PchipInterpolator(self._nodes, self._cdf_vals, extrapolate=True)
        self._icdf = PchipInterpolator(self._cdf_vals, self._nodes, extrapolate=True)
        self._iicdf = self._icdf.antiderivative()
        self._dicdf = self._icdf.derivative()

    @classmethod
    def from_call_prices(
        cls,
        strikes: np.ndarray,
        prices: np.ndarray,
        tenor: float,
    ):
        cdf_vals = 1 + np.gradient(prices, strikes)
        return cls(strikes, cdf_vals, tenor)

    @property
    def tenor(self):
        return self._t

    @property
    def nodes(self):
        return self._nodes[1:-1]

    def cdf(self, x):
        return np.clip(self._cdf(x), EPS, 1 - EPS)

    def icdf(self, u):
        return self._icdf(u)

    def iicdf(self, u):
        return self._iicdf(u) - self._iicdf(0)

    def dicdf(self, u):
        return self._dicdf(u)


class MarginalLogNormal(Marginal):
    def __init__(
        self,
        tenor: float,
        vola: float,
    ):
        self._t = tenor
        self._sigma = vola * np.sqrt(tenor)

    @property
    def tenor(self):
        return self._t

    def cdf(self, x):
        LB = EPS
        UB = 1 - EPS
        u = ndtr((np.log(x) + self._sigma**2 / 2) / self._sigma)
        return np.clip(u, LB, UB)

    def icdf(self, u):
        LB = EPS
        UB = 1 - EPS
        u = np.clip(u, LB, UB)
        return np.exp(self._sigma * ndtri(u) - self._sigma**2 / 2)

    def iicdf(self, u):
        k = self.icdf(u)
        d1 = (-np.log(k) + self._sigma**2 / 2) / self._sigma
        return 1 - ndtr(d1)

    def dicdf(self, u):
        x = self.icdf(u)
        return (
            np.sqrt(2 * np.pi)
            * self._sigma
            * np.exp((np.log(x) + self._sigma**2 / 2) ** 2 / self._sigma**2 / 2)
        )
