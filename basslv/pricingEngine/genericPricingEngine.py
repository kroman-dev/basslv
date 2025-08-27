from abc import ABC, abstractmethod


class GenericPricingEngine(ABC):

    @abstractmethod
    def getOptionPrice(
            self,
            forward: float,
            discountFactor: float,
            timeToExpiry: float,
            strike: float,
            vol: float,
            optionType: int
    ) -> float:
        pass

    @abstractmethod
    def getImpliedVolatility(
            self,
            optionPrice: float,
            forward: float,
            discountFactor: float,
            timeToExpiry: float,
            strike: float,
            optionType: int
    ):
        pass

