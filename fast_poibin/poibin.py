from typing import Literal, Union

import numpy as np
import numpy.typing as npt

from fast_poibin.pmf import FloatSequence, calc_pmf, calc_pmf_dp


def _validate_probabilities(probabilities: FloatSequence) -> None:
    if isinstance(probabilities, np.ndarray):
        assert probabilities.ndim == 1
        assert np.all(probabilities <= 1)
        assert np.all(probabilities >= 0)
    else:
        assert all(0 <= p <= 1 for p in probabilities)


class PoiBin:
    """Class for storing PMF and CDF of Poisson binomial distribution.

    Attributes:
        pmf: array of probability mass function
        cdf: array of cumulative distribution function

    Examples:
        >>> poibin = PoiBin([0.1, 0.2, 0.2])
        >>> poibin.pmf
        array([0.576, 0.352, 0.068, 0.004])
        >>> poibin.cdf
        array([0.576, 0.928, 0.996, 1.   ])

    Note:
        This class stores data as arrays of `numpy.float64` no matter which type of
        float is passed at initialization. This is partly because `numpy.fft` deals
        with only `float64`, and partly because it's easier to implement. Please see
        also https://numpy.org/doc/1.24/reference/routines.fft.html#type-promotion.

    Notes:
        The internal algorithm  for `mixed` mode is based on the well-known divide-and-conquer
        approach for convolving many polynomials. Its complexities are as follows.

        - Time: O(N(logN)^2)
        - Space: O(N)

        In the context of Poisson binomial distribution, that seems to be equivalent to
        the one proposed in the following paper.

        Biscarri, William & Zhao, Sihai & Brunner, Robert. (2018). A simple and fast
        method for computing the Poisson binomial distribution function. Computational
        Statistics & Data Analysis. 122. 10.1016/j.csda.2018.01.007.

        A downside of this approach is numerical precision. Since convolution by FFT
        is unreliable in the world under the float epsilon (i.e. 2^-52 ~ 2 * 10^-16),
        the relative errors of the computed PMF could be large though it is still not
        bad w.r.t. the absolute errors. For details, please see the following issue:
        https://github.com/privet-kitty/fast-poibin/issues/9. For situations where the
        relative precision is required, you may want to use `dp` mode instead. You
        should also note that neither mode can express a probability mass below the
        least positive double float (~ 5 * 10^-324).
    """

    def __init__(
        self, probabilities: FloatSequence, mode: Union[Literal["mixed"], Literal["dp"]] = "mixed"
    ) -> None:
        """
        Args:
            probabilities: Success probabilities for each binomial trial.
        """
        _validate_probabilities(probabilities)
        if mode == "mixed":
            self.pmf = calc_pmf(probabilities)
        elif mode == "dp":
            self.pmf = calc_pmf_dp(np.array(probabilities, np.float64))
        else:
            raise ValueError(f"Invalid mode: {mode}")
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
