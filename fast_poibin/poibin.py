import numpy as np
import numpy.typing as npt

from fast_poibin.pmf import FloatSequence, calc_pmf


def _validate_probabilities(probabilities: FloatSequence) -> None:
    if isinstance(probabilities, np.ndarray):
        assert probabilities.ndim == 1
        assert np.all(probabilities <= 1)
        assert np.all(probabilities >= 0)
    else:
        assert all(0 <= p <= 1 for p in probabilities)


class PoiBin:
    """Class for storing PMF and CDF of Poisson binomial distribution.

    Complexity:
        - Time: O(N(logN)^2)
        - Space: O(N)

    Examples:
        >>> poibin = PoiBin([0.1, 0.2, 0.2])
        >>> poibin.pmf
        array([0.576, 0.352, 0.068, 0.004])
        >>> poibin.cdf
        array([0.576, 0.928, 0.996, 1.   ])

    Note:
        The internal algorithm is based on the well-known divide-and-conquer approach
        for convolving many polynomials, which seems to be equivalent to the one proposed
        in the following paper in the context of Poisson binomial distribution.

        Biscarri, William & Zhao, Sihai & Brunner, Robert. (2018). A simple and fast
        method for computing the Poisson binomial distribution function. Computational
        Statistics & Data Analysis. 122. 10.1016/j.csda.2018.01.007.
    """

    def __init__(self, probabilities: FloatSequence) -> None:
        """
        Args:
            probabilities: Success probabilities for each binomial trial.
        """
        _validate_probabilities(probabilities)
        self.pmf = calc_pmf(probabilities)
        self.cdf: npt.NDArray[np.float64] = np.minimum(np.cumsum(self.pmf), 1.0)
        self.cdf[-1] = 1.0

    def quantile(self, p: npt.ArrayLike):  # type: ignore
        """Quantile function (inverse of `cdf`).

        Args:
            p: probability value(s)

        Returns:
            int or array of ints
        """
        return np.searchsorted(self.cdf, p)
