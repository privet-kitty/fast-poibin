from itertools import product

import numpy as np
import numpy.testing as nptest

from fast_poibin.poibin import calc_pmf, calc_pmf_dp, convolve, convolve_power_of_two_degree


def test_convolve_zero() -> None:
    nptest.assert_array_equal(convolve(np.array([1.0, 2.0]), np.array([])), np.array([]))
    nptest.assert_array_equal(convolve(np.array([]), np.array([10.0])), np.array([]))
    nptest.assert_array_equal(convolve(np.array([]), np.array([])), np.array([]))


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


def test_calc_pmf_dp_coincide_with_calc_pmf_fft() -> None:
    rng = np.random.default_rng(2235)
    for size, threshold in product(list(range(10)) + [10, 20, 29, 41, 51], [0, 2, 8, 32, 64]):
        for _ in range(50):
            probs = rng.random(size, np.float64)
            nptest.assert_allclose(
                calc_pmf(probs, threshold), calc_pmf_dp(probs), rtol=1e-9, atol=1e-9
            )


def test_calc_pmf_fft_non_negativity() -> None:
    rng = np.random.default_rng(2235)
    for size in range(0, 51):
        for _ in range(50):
            probs = rng.random(size, np.float64)
            assert np.all(calc_pmf(probs, dp_threshold=0) >= 0)


def test_calc_pmf_dp_non_float64_ndarray() -> None:
    assert calc_pmf_dp(np.array([0.1, 0.2, 0.3], dtype=np.float16)).dtype == np.float64
    assert calc_pmf_dp(np.array([0.1, 0.2, 0.3], dtype=np.float32)).dtype == np.float64


def test_calc_pmf_fft_non_float64_ndarray() -> None:
    assert calc_pmf(np.array([0.1, 0.2, 0.3], dtype=np.float16), 0).dtype == np.float64
    assert calc_pmf(np.array([0.1, 0.2, 0.3], dtype=np.float32), 0).dtype == np.float64


def test_calc_pmf_dp_sequence() -> None:
    assert calc_pmf_dp([0, 1, 0, 0, 1]).dtype == np.float64
    assert calc_pmf_dp([0.1, 0.2, 0.1, 0.2, 0.3]).dtype == np.float64


def test_calc_pmf_fft_sequence() -> None:
    assert calc_pmf([0, 1, 0, 0, 1], 0).dtype == np.float64
    assert calc_pmf([0.1, 0.2, 0.1, 0.2, 0.3], 0).dtype == np.float64


def test_calc_pmf_zero() -> None:
    for threshold in [0, 1, 2, 4, 8]:
        nptest.assert_array_equal(calc_pmf([], threshold), np.array([1.0], np.float64))


def test_calc_pmf_dp_zero() -> None:
    nptest.assert_array_equal(calc_pmf_dp([]), np.array([1.0], np.float64))


def test_calc_pmf_one() -> None:
    for threshold in [0, 1, 2, 4, 8]:
        nptest.assert_array_equal(calc_pmf([0.3], threshold), np.array([0.7, 0.3], np.float64))


def test_calc_pmf_dp_one() -> None:
    nptest.assert_array_equal(calc_pmf_dp([0.3]), np.array([0.7, 0.3], np.float64))
