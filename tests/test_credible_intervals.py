"""Unit tests for smefit.credible_intervals — the HDI estimator on raw sample
arrays, independent of Fit/FitResult. The ETI is exercised through
Fit.confidence_bounds in test_fit_result.py and serves here as the reference
the HDI is compared against."""

import numpy as np
import pytest

from smefit.credible_intervals import equal_tailed_interval, highest_density_interval


@pytest.fixture
def gaussian():
    return np.random.default_rng(0).normal(loc=0.0, scale=1.0, size=5000)


def test_hdi_of_a_unimodal_gaussian_is_one_interval_around_the_mode(gaussian):
    pieces = highest_density_interval(gaussian, 68)

    assert len(pieces) == 1
    low, high = pieces[0]
    assert low < 0.0 < high
    assert (low, high) == pytest.approx((-1.0, 1.0), abs=0.1)


@pytest.mark.parametrize("level", [68, 95])
def test_hdi_of_a_gaussian_agrees_with_the_eti(gaussian, level):
    """On a symmetric unimodal posterior the two constructions are the same
    interval, up to sampling noise."""
    [hdi] = highest_density_interval(gaussian, level)
    [eti] = equal_tailed_interval(gaussian, level)

    assert hdi == pytest.approx(eti, abs=0.05)


def test_hdi_of_a_skewed_posterior_is_narrower_and_closer_to_the_mode():
    """Where they differ: the HDI is the narrowest interval holding the
    level's mass, so on a right-skewed pdf (Gamma(2, 1): mode 1, mean 2)
    HDI is narrower than the ETI and shifted towards the mode."""
    values = np.random.default_rng(0).gamma(shape=2.0, scale=1.0, size=5000)

    [(hdi_low, hdi_high)] = highest_density_interval(values, 68)
    [(eti_low, eti_high)] = equal_tailed_interval(values, 68)

    assert hdi_high - hdi_low < 0.95 * (eti_high - eti_low)
    assert hdi_low < eti_low
    assert hdi_high < eti_high


def test_hdi_of_a_bimodal_posterior_is_two_disjoint_intervals():
    """Two separated modes, as a linear+quadratic degeneracy gives: one piece
    per mode, neither bridging the gap between them."""
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(-5.0, 0.2, 3000), rng.normal(5.0, 0.2, 3000)])

    pieces = highest_density_interval(values, 68)

    assert len(pieces) == 2
    assert pieces[0][1] < 0.0 < pieces[1][0]


def test_hdi_is_pinned_at_a_hard_lower_bound():
    """A posterior piled up against a positivity bound reaches it exactly,
    instead of leaking past it."""
    values = np.abs(np.random.default_rng(0).normal(0.0, 1.0, 5000))

    [(low, high)] = highest_density_interval(values, 95, bounds=(0.0, None))

    assert low == 0.0
    assert high == pytest.approx(1.96, abs=0.1)


def test_hdi_is_pinned_at_a_hard_upper_bound():
    values = -np.abs(np.random.default_rng(0).normal(0.0, 1.0, 5000))

    [(low, high)] = highest_density_interval(values, 95, bounds=(None, 0.0))

    assert high == 0.0
    assert low == pytest.approx(-1.96, abs=0.1)


def test_several_modes_take_precedence_over_a_hard_bound():
    """A sign-definite coefficient with two modes is still split into one
    piece per mode; the bound only shapes a single interval."""
    rng = np.random.default_rng(0)
    values = np.abs(
        np.concatenate([rng.normal(1.0, 0.2, 3000), rng.normal(5.0, 0.2, 3000)])
    )

    assert highest_density_interval(
        values, 68, bounds=(0.0, None)
    ) == highest_density_interval(values, 68)
    assert len(highest_density_interval(values, 68)) == 2


def test_hdi_ignores_nans(gaussian):
    with_nan = np.concatenate([gaussian, [np.nan, np.nan]])

    assert highest_density_interval(with_nan, 68) == highest_density_interval(
        gaussian, 68
    )
