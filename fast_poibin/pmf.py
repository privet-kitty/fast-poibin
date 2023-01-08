import sys
from itertools import zip_longest
from typing import Any, Optional, Sequence, Union

import numba
import numpy as np
import numpy.typing as npt


def power_of_two_ceil(x: int) -> int:
    """Return the least power of two that is greater than or equal to `x`."""
    return 1 << max(0, x - 1).bit_length()


def convolve(
    vector1: npt.NDArray[np.float64], vector2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Convolve two vectors with FFT."""
    if vector1.size == 0 or vector2.size == 0:
        return np.array((), dtype=np.float64)
    prod_size = vector1.size + vector2.size - 1
    fft_size = power_of_two_ceil(prod_size) * 2
    res = np.fft.irfft(np.fft.rfft(vector1, fft_size) * np.fft.rfft(vector2, fft_size), fft_size)
    res.resize(prod_size, refcheck=False)
    # numpy.fft always deals with float64 and complex128, which is documented in
    # https://numpy.org/devdocs/reference/routines.fft.html#type-promotion.
    # Therefore it is guaranteed that dtype of res is float64.
    return res


def convolve_power_of_two_degree(
    vector1: npt.NDArray[np.float64], vector2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    DEPRECATED.

    Note:
        `vector1` and `vector2` must have the same power-of-two degree.
    """
    prod_size = vector1.size + vector2.size - 1
    res = np.zeros(prod_size, dtype=np.float64)
    # A polynomial of power-of-two degree has power-of-two-plus-one length as an array,
    # which is unfortunately the worst case of the FFT algorithm.
    # In that case, it seems to be faster to deal with one length shorter arrays.
    # Suppose we want to get f(x) * g(x). Let f(x) = a + x * f'(x) and g(x) = b + x * g'(x).
    # Then f(x) * g(x) = a * b + x * (b * f'(x) + a * g'(x)) + x^2 * f'(x) * g'(x).
    # I learned this technique from
    # https://maspypy.com/%E5%A4%9A%E9%A0%85%E5%BC%8F%E3%83%BB%E5%BD%A2%E5%BC%8F%E7%9A%84%E3%81%B9%E3%81%8D%E7%B4%9A%E6%95%B0-%E9%AB%98%E9%80%9F%E3%81%AB%E8%A8%88%E7%AE%97%E3%81%A7%E3%81%8D%E3%82%8B%E3%82%82%E3%81%AE#toc17
    a = vector1[0]
    b = vector2[0]
    res[0] = a * b
    res[2:] = convolve(vector1[1:], vector2[1:])
    res[1 : vector1.size] += vector1[1:] * b + vector2[1:] * a
    return res


@numba.njit(numba.float64[:](numba.float64[:]), cache=True)  # type: ignore
def calc_pmf_dp(probabilities: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Calculate PMF of Poisson binomial distribution by dynamic programming.

    Complexity:
        Time: O(N^2)
        Space: O(N)
    """
    n = len(probabilities)
    dp = np.zeros(n + 1, dtype=np.float64)
    dp[0] = 1.0
    for i, prob in enumerate(probabilities):
        for j in range(i + 1, 0, -1):
            dp[j] = dp[j] * (1 - prob) + dp[j - 1] * prob
        dp[0] *= 1 - prob
    return dp


# calc_pmf uses numpy.convolve instead of FFT under this threshold. This value was decided
# based on the experiment in https://github.com/privet-kitty/fast-poibin/issues/1.
FFT_THRESHOLD = 1024

# calc_pmf first performs DP on each subarray of this length. This value was decided
# based on the experiment in https://github.com/privet-kitty/fast-poibin/issues/3
DP_STEP = 255


# FIXME: This type definition is a compromise.
# 1. 1D np.ndarray is not Sequence. I couldn't find an appropriate iterable type that
# contains np.ndarray.
# 2. I'm not sure whether np.floating[Any] is a decent type for generic float.
if sys.version_info[0:2] <= (3, 8):
    FloatSequence = Union[Sequence[float], npt.NDArray[np.floating]]
else:
    FloatSequence = Union[Sequence[float], npt.NDArray[np.floating[Any]]]


def calc_pmf(probabilities: FloatSequence, dp_step: int = DP_STEP) -> npt.NDArray[np.float64]:
    """Calculate PMF of Poisson binomial distribution.

    Complexity:
        Time: O(N(logN)^2)
        Space: O(N)
    """
    size = len(probabilities)
    # Just performing DP is usually better than convolving an array of dp_step length and
    # another short one, at least for dp_step=DP_STEP. However, this idea should further
    # be refined. Maybe I'd better introduce an independent threshold.
    if size < dp_step * 2:
        res: npt.NDArray[np.float64] = calc_pmf_dp(np.array(probabilities, np.float64))
        return res
    if dp_step > 0:
        # FIXME: Is min() really necessary?
        # FIXME: I copy the returned values of calc_pmf_dp here, because they are sometimes
        # just a view of another array, which can't be resized.
        polynomials = [
            calc_pmf_dp(np.array(probabilities[i : min(i + dp_step, size)], np.float64)).copy()
            for i in range(0, size, dp_step)
        ]
    else:
        polynomials = [np.array((1 - p, p), dtype=np.float64) for p in probabilities]

    def _convolve(
        poly1: npt.NDArray[np.float64], poly2: Optional[npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        if poly2 is None:
            return poly1
        if poly1.size >= FFT_THRESHOLD:
            return convolve(poly1, poly2)
        else:
            return np.convolve(poly1, poly2)

    while len(polynomials) > 1:
        it = iter(polynomials)
        polynomials = [_convolve(p1, p2) for p1, p2 in zip_longest(it, it)]

    if not polynomials:
        return np.array((1.0,), np.float64)
    res = polynomials[0]
    res.resize(size + 1, refcheck=False)
    return np.maximum(res, 0.0)


if __name__ == "__main__":
    pmf = calc_pmf([0.1, 0.2, 0.3, 0.1, 0.2])
    cdf = np.cumsum(pmf)
    print(pmf, cdf)
    pmf = calc_pmf_dp(np.array([0.1, 0.2, 0.3, 0.1, 0.2]))
    cdf = np.cumsum(pmf)
    print(pmf, cdf)
