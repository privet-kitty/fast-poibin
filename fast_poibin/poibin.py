from itertools import zip_longest
from typing import Any, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt


def power_of_two_ceil(x: int) -> int:
    """Return the least power of two that is greater than or equal to `x`."""
    return 1 << max(0, x - 1).bit_length()


def convolve(
    vector1: npt.NDArray[np.float64], vector2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    if vector1.size == 0 or vector2.size == 0:
        return np.array([], dtype=np.float64)
    prod_size = vector1.size + vector2.size - 1
    fft_size = power_of_two_ceil(prod_size) * 2
    res = np.fft.irfft(np.fft.rfft(vector1, fft_size) * np.fft.rfft(vector2, fft_size), fft_size)
    res.resize(prod_size, refcheck=False)
    return res


def convolve_power_of_two_degree(
    vector1: npt.NDArray[np.float64], vector2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
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
    # https://maspypy.com/%E5%A4%9A%E9%A0%85%E5%BC%8F%E3%83%BB%E5%BD%A2%E5%BC%8F%E7%9A%84%E3%81%B9%E3%81%8D%E7%B4%9A%E6%95%B0-%E9%AB%98%E9%80%9F%E3%81%AB%E8%A8%88%E7%AE%97%E3%81%A7%E3%81%8D%E3%82%8B%E3%82%82%E3%81%AE
    a = vector1[0]
    b = vector2[0]
    res[0] = a * b
    res[2:] = convolve(vector1[1:], vector2[1:])
    res[1 : vector1.size] += vector1[1:] * b + vector2[1:] * a
    return res


# FIXME: This type definition is a compromise.
# 1. np.ndarray is not Sequence. I couldn't find an appropriate iterable type that
# contains np.ndarray.
# 2. I'm not sure whether np.floating[Any] is a decent type.
RealSequence = Union[Sequence[float], npt.NDArray[np.floating[Any]]]


def calc_pmf_dp(probabilities: RealSequence) -> npt.NDArray[np.float64]:
    """Calculate PMF of Poisson binomial distribution by dynamic programming.

    Time complexity: O(N^2)
    Space complexity: O(N)
    """
    dp: npt.NDArray[np.float64] = np.zeros(len(probabilities) + 1, dtype=np.float64)
    dp[0] = 1.0
    for prob in probabilities:
        # np.roll works because the trailing element of dp is always zero.
        dp = dp * np.float64(1 - prob) + np.roll(dp, 1) * np.float64(prob)
    return dp


# Use numpy.convolve instead under this threshold. This value was decided
# based on the experiment https://github.com/privet-kitty/fast-poibin/issues/1.
FFT_THRESHOLD = 1024


def calc_pmf(probabilities: RealSequence, dp_threshold: int = 0) -> npt.NDArray[np.float64]:
    """Calculate PMF of Poisson binomial distribution.

    Time complexity: O(N(logN)^2)
    Space comlexity: O(N)
    """
    step = power_of_two_ceil(dp_threshold)
    size = len(probabilities)
    if size == 0:
        return np.array([1.0], dtype=np.float64)
    if step > 1:
        # FIXME: Is min() really necessary?
        polynomials = [
            calc_pmf_dp(probabilities[i : min(i + step, size)]) for i in range(0, size, step)
        ]
    else:
        polynomials = [np.array((1 - p, p), dtype=np.float64) for p in probabilities]

    def _convolve(
        poly1: npt.NDArray[np.float64], poly2: Optional[npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        if poly2 is None:
            return poly1
        if poly1.size >= FFT_THRESHOLD:
            if poly1.size != poly2.size:
                poly2.resize(poly1.size, refcheck=False)
            return convolve_power_of_two_degree(poly1, poly2)
        else:
            return np.convolve(poly1, poly2)

    while len(polynomials) > 1:
        it = iter(polynomials)
        polynomials = [_convolve(p1, p2) for p1, p2 in zip_longest(it, it)]
    res = polynomials[0]
    res.resize(size + 1, refcheck=False)
    return np.maximum(res, 0.0)


if __name__ == "__main__":
    pmf = calc_pmf([0.1, 0.2, 0.3, 0.1, 0.2])
    cdf = np.cumsum(pmf)
    print(pmf, cdf)
    pmf = calc_pmf_dp([0.1, 0.2, 0.3, 0.1, 0.2])
    cdf = np.cumsum(pmf)
    print(pmf, cdf)
