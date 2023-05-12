import numpy as np
import numpy.testing as nptest
import pytest

from fast_poibin.pmf import FloatSequence
from fast_poibin.poibin import PoiBin


def test_poibin_empty() -> None:
    for mode in ("mixed", "dp"):
        poibin = PoiBin([], mode=mode)  # type: ignore
        nptest.assert_array_almost_equal(poibin.pmf, [1])
        nptest.assert_array_almost_equal(poibin.cdf, [1])
        assert poibin.quantile(0) == 0
        assert poibin.quantile(0.05) == 0
        assert poibin.quantile(0.1) == 0
        assert poibin.quantile(0.9) == 0
        assert poibin.quantile(1) == 0
        nptest.assert_array_equal(poibin.quantile([0.1, 0.2, 0.05, 1, 0]), [0] * 5)


def test_poibin_single() -> None:
    for mode in ("mixed", "dp"):
        poibin = PoiBin([0.9], mode=mode)  # type: ignore
        nptest.assert_array_almost_equal(poibin.pmf, [0.1, 0.9])
        nptest.assert_array_almost_equal(poibin.cdf, [0.1, 1])
        assert poibin.quantile(0) == 0
        assert poibin.quantile(0.05) == 0
        assert poibin.quantile(0.1) == 1
        assert poibin.quantile(0.9) == 1
        assert poibin.quantile(1) == 1
        nptest.assert_array_equal(poibin.quantile([0.1, 0.2, 0.05, 1, 0]), [1, 1, 0, 1, 0])


def test_poibin_all_zero() -> None:
    for mode in ("mixed", "dp"):
        poibin = PoiBin([0, 0, 0, 0, 0], mode=mode)  # type: ignore
        nptest.assert_array_equal(poibin.pmf, [1, 0, 0, 0, 0, 0])
        nptest.assert_array_equal(poibin.cdf, [1] * 6)
        assert poibin.quantile(0) == 0
        assert poibin.quantile(0.5) == 0
        assert poibin.quantile(1) == 0


def test_poibin_all_one() -> None:
    for mode in ("mixed", "dp"):
        poibin = PoiBin([1, 1, 1], mode=mode)  # type: ignore
        nptest.assert_array_equal(poibin.pmf, [0, 0, 0, 1])
        nptest.assert_array_equal(poibin.cdf, [0, 0, 0, 1])
        assert poibin.quantile(0) == 0
        assert poibin.quantile(0.5) == 3
        assert poibin.quantile(1) == 3


def test_poibin_non_float64() -> None:
    """PoiBin stores float64 even when the argument is not."""
    for mode in ("mixed", "dp"):
        poibin = PoiBin(np.array([0.2, 0.5], np.float32), mode=mode)  # type: ignore
        assert poibin.pmf.dtype == np.float64
        assert poibin.cdf.dtype == np.float64


def test_poibin_non_1d_array_is_forbidden() -> None:
    with pytest.raises(AssertionError) as e:
        PoiBin(np.array([[0.3], [0.4]]))
    assert e.traceback[-1].name == "_validate_probabilities"


def test_poibin_non_probablities() -> None:
    def check(x: FloatSequence) -> None:
        with pytest.raises(AssertionError) as e:
            PoiBin(x)
        assert e.traceback[-1].name == "_validate_probabilities"

    check(np.array([1.5]))
    check(np.array([-0.5]))
    check([1.5])
    check([-0.5])


def test_poibin_dp_mode_mixed_mode_equivalence() -> None:
    poibin_dp = PoiBin([0.1, 0.2, 0.3, 0.4], mode="dp")
    poibin_mixed = PoiBin([0.1, 0.2, 0.3, 0.4], mode="mixed")
    nptest.assert_array_almost_equal(poibin_dp.pmf, poibin_mixed.pmf)
    nptest.assert_array_almost_equal(poibin_dp.cdf, poibin_mixed.cdf)


def test_poibin_invalid_mode() -> None:
    with pytest.raises(ValueError):
        PoiBin([], mode="asdf")  # type: ignore
