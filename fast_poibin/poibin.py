import numpy as np
import numpy.typing as npt

from fast_poibin.pmf import FloatSequence, calc_pmf


class PoiBin:
    def __init__(self, probabilities: FloatSequence) -> None:
        self.pmf = calc_pmf(probabilities)
        self.cdf: npt.NDArray[np.float64] = np.minimum(np.cumsum(self.pmf), 1.0)
        self.cdf[-1] = 1.0

    def quantile(self, p: npt.ArrayLike):  # type: ignore
        return np.searchsorted(self.cdf, p)
