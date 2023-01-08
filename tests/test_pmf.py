from itertools import product

import numpy as np
import numpy.testing as nptest

from fast_poibin.pmf import (
    FFT_THRESHOLD,
    calc_pmf,
    calc_pmf_dp,
    convolve,
    convolve_power_of_two_degree,
)


def test_convolve_zero() -> None:
    nptest.assert_array_equal(convolve(np.array([1.0, 2.0]), np.array([])), [])
    nptest.assert_array_equal(convolve(np.array([]), np.array([10.0])), [])
    nptest.assert_array_equal(convolve(np.array([]), np.array([])), [])


def test_convolve_coincides_with_numpy_convolve() -> None:
    rng = np.random.default_rng(2235)
    for size1, size2 in product(range(1, 30), range(1, 30)):
        vector1 = rng.random(size1, np.float64)
        vector2 = rng.random(size2, np.float64)
        nptest.assert_allclose(convolve(vector1, vector2), np.convolve(vector1, vector2))


def test_convolve_preserves_args() -> None:
    rng = np.random.default_rng(2235)
    for size1, size2 in product(range(1, 10), range(1, 10)):
        vector1 = rng.random(size1, np.float64)
        vector2 = rng.random(size2, np.float64)
        vector1_clone = vector1.copy()
        vector2_clone = vector2.copy()
        convolve(vector1, vector2)
        nptest.assert_array_equal(vector1, vector1_clone)
        nptest.assert_array_equal(vector2, vector2_clone)


def test_convolve_power_of_two_degree_coincides_with_numpy_convolve() -> None:
    rng = np.random.default_rng(2235)
    for deg in [1, 2, 4, 8, 16]:
        vector1 = rng.random(deg + 1, np.float64)
        vector2 = rng.random(deg + 1, np.float64)
        nptest.assert_allclose(
            convolve_power_of_two_degree(vector1, vector2), np.convolve(vector1, vector2)
        )


def test_convolve_power_of_two_degree_preserves_args() -> None:
    rng = np.random.default_rng(2235)
    for deg in [1, 2, 4, 8, 16]:
        vector1 = rng.random(deg + 1, np.float64)
        vector2 = rng.random(deg + 1, np.float64)
        vector1_clone = vector1.copy()
        vector2_clone = vector2.copy()
        convolve_power_of_two_degree(vector1, vector2)
        nptest.assert_array_equal(vector1, vector1_clone)
        nptest.assert_array_equal(vector2, vector2_clone)


def test_calc_pmf_small_handmade_case() -> None:
    probs = [0.1, 0.2, 0.7, 0.2, 0.2]
    # (0.9 + 0.1x)(0.8 + 0.2x)^3(0.3+0.7x)
    # = 0.00056 x^5 + 0.012 x^4 + 0.0924 x^3 + 0.3152 x^2 + 0.4416 x + 0.13824
    # (By WolframAlpha)
    for step in [0, 2, 8, 32, 64]:
        nptest.assert_allclose(
            calc_pmf(probs, step),
            [0.13824, 0.4416, 0.3152, 0.0924, 0.012, 0.00056],
        )


def test_calc_pmf_dp_coincide_with_calc_pmf_fft() -> None:
    rng = np.random.default_rng(2235)
    for size, step in product(list(range(10)) + [10, 20, 29, 41, 51], [0, 2, 8, 32, 64]):
        for _ in range(50):
            probs = rng.random(size, np.float64)
            nptest.assert_allclose(calc_pmf(probs, step), calc_pmf_dp(probs), rtol=1e-9, atol=1e-9)

    # larger instance
    for size, step in product([2000], [0, 64]):
        assert size >= FFT_THRESHOLD
        for _ in range(10):
            probs = rng.random(size, np.float64)
            nptest.assert_allclose(calc_pmf(probs, step), calc_pmf_dp(probs), rtol=1e-9, atol=1e-9)


def test_calc_pmf_fft_non_negativity() -> None:
    rng = np.random.default_rng(2235)
    size = 2021
    assert size >= FFT_THRESHOLD
    for _ in range(50):
        probs = rng.random(size, np.float64)
        assert np.all(calc_pmf(probs, 0) >= 0)


def test_calc_pmf_non_float64_ndarray() -> None:
    for step in [0, 4]:
        assert calc_pmf(np.array([0.1, 0.2, 0.3], np.float16), step).dtype == np.float64
        assert calc_pmf(np.array([0.1, 0.2, 0.3], np.float32), step).dtype == np.float64


def test_calc_pmf_non_ndarray() -> None:
    for step in [0, 4]:
        assert calc_pmf([0, 1, 0, 0, 1], step).dtype == np.float64
        assert calc_pmf([0.1, 0.2, 0.1, 0.2, 0.3], step).dtype == np.float64


def test_calc_pmf_zero() -> None:
    for step in [0, 1, 2, 4, 8]:
        nptest.assert_array_equal(calc_pmf([], step), [1.0])


def test_calc_pmf_dp_zero() -> None:
    nptest.assert_array_equal(calc_pmf_dp(np.array([])), [1.0])


def test_calc_pmf_one() -> None:
    for step in [0, 1, 2, 4, 8]:
        nptest.assert_array_equal(calc_pmf([0.3], step), [0.7, 0.3])


def test_calc_pmf_dp_one() -> None:
    nptest.assert_array_equal(calc_pmf_dp(np.array([0.3])), [0.7, 0.3])
