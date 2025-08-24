from basslv.core.genericMarginal import GenericMarginal
from basslv.core.projectTyping import FloatOrVectorType


class MarketMarginal(GenericMarginal):

    def __init__(self, tenor: float):
        super().__init__(tenor=tenor)

    def _derivativeOfInverseCdf(self, u: FloatOrVectorType) -> FloatOrVectorType:
        pass

    def _integralOfInverseCdf(self, u: FloatOrVectorType) -> FloatOrVectorType:
        pass

    def _inverseCdf(self, u: FloatOrVectorType) -> FloatOrVectorType:
        pass

    def _cdf(self, x: FloatOrVectorType) -> FloatOrVectorType:
        pass

