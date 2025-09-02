from abc import ABC, abstractmethod
from typing import Callable

from basslv.core.projectTyping import FloatVectorType


class HeatKernelConvolutionEngine(ABC):

    @abstractmethod
    def convolution(
            self,
            time: FloatVectorType,
            func: Callable[[FloatVectorType], FloatVectorType]
    ) -> Callable[[FloatVectorType], FloatVectorType]:
        pass
