"""Unit tests for smefit.credible_intervals — ETI/HDI estimators on raw
sample arrays, independent of Fit/FitResult."""

import numpy as np
import pytest

from smefit.credible_intervals import equal_tailed_interval, highest_density_interval

# ---------------------------------------------------------------------------
# equal_tailed_interval
# ---------------------------------------------------------------------------


def test_equal_tailed_interval_matches_percentile_logic():
    """Regression against the pre-refactor Fit.confidence_bounds logic: the
    [tail, 100-tail] percentiles."""
    values = list(range(0, 1001))  # percentile p is 10*p

    low, high = equal_tailed_interval(values, 68)

    assert (low, high) == pytest.approx((160.0, 840.0))


@pytest.mark.parametrize("level", [0, 100, -5, 0.95, 120])
def test_equal_tailed_interval_rejects_a_level_that_is_not_a_percentage(level):
    with pytest.raises(ValueError, match="between 1 and 100"):
        equal_tailed_interval([0.0, 1.0, 2.0], level)


# ---------------------------------------------------------------------------
# highest_density_interval
# ---------------------------------------------------------------------------


def test_hdi_on_a_unimodal_gaussian_is_one_segment_containing_the_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.0, scale=1.0, size=5000)

    segments = highest_density_interval(values, 68)

    assert len(segments) == 1
    low, high = segments[0]
    assert low < 0.0 < high


def test_hdi_is_no_wider_than_the_eti_at_the_same_level():
    """The HDI is the *narrowest* interval covering the target mass, so on a
    unimodal posterior it can only be narrower than or equal to the ETI."""
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.0, scale=1.0, size=5000)

    hdi_low, hdi_high = highest_density_interval(values, 68)[0]
    eti_low, eti_high = equal_tailed_interval(values, 68)

    assert (hdi_high - hdi_low) <= (eti_high - eti_low) + 1e-9


def test_hdi_on_a_bimodal_sample_returns_two_disjoint_segments():
    """Two well-separated Gaussians: exactly two segments, neither of which
    contains the gap between the modes — mimics the linear+quadratic EFT
    degeneracy that produces two disconnected allowed regions."""
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(-5.0, 0.2, 3000), rng.normal(5.0, 0.2, 3000)])

    segments = highest_density_interval(values, 68)

    assert len(segments) == 2
    for low, high in segments:
        assert not (low < 0.0 < high)


def test_hdi_respects_a_hard_lower_bound():
    """A positivity-bound coefficient: the lower edge of the single segment
    is pinned exactly at the bound, not left to leak below it."""
    rng = np.random.default_rng(0)
    values = np.abs(rng.normal(0.0, 1.0, 5000))

    segments = highest_density_interval(values, 95, bounds=(0.0, None))

    assert len(segments) == 1
    low, _high = segments[0]
    assert low == pytest.approx(0.0, abs=1e-9)
    assert low >= 0.0


@pytest.mark.parametrize("level", [0, 100, -5, 0.95, 120])
def test_hdi_rejects_a_level_that_is_not_a_percentage(level):
    with pytest.raises(ValueError, match="between 1 and 100"):
        highest_density_interval([0.0, 1.0, 2.0], level)
