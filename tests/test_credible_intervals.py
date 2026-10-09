"""Unit tests for smefit.credible_intervals — the HDI estimator on raw sample
arrays, independent of Fit/FitResult. The ETI is exercised through
Fit.confidence_bounds in test_fit_result.py and serves here as the reference
the HDI is compared against."""

import numpy as np
import pytest
from scipy.stats import norm

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


# The two tests below bracket the KDE bandwidth used to count modes: arviz's
# ISJ default fails the first, Silverman's and Scott's rules the second.


@pytest.mark.parametrize("level", [68, 95])
def test_hdi_of_a_resampled_unimodal_posterior_is_one_interval(level):
    """Nested sampling stores its weighted points resampled with replacement,
    so a unimodal posterior arrives full of repeated draws; they must not be
    read as modes. Mimicked by importance-resampling N(0, 4) proposals to an
    N(0, 1) target: 6400 draws, ~2700 distinct."""
    rng = np.random.default_rng(0)
    proposals = rng.normal(0.0, 4.0, 8000)
    log_weights = -0.5 * proposals**2 + 0.5 * (proposals / 4.0) ** 2
    weights = np.exp(log_weights - log_weights.max())
    values = rng.choice(proposals, size=6400, p=weights / weights.sum())

    assert len(highest_density_interval(values, level)) == 1


def test_hdi_resolves_a_small_narrow_second_mode():
    """80% N(0, 1) + 20% N(2.5, 0.2): the narrow mode is its own piece of the
    exact 95% region, which a bandwidth set by the overall spread smooths
    away."""
    rng = np.random.default_rng(0)
    values = np.where(
        rng.random(6400) < 0.8, rng.normal(0.0, 1.0, 6400), rng.normal(2.5, 0.2, 6400)
    )

    [(_, main_high), (narrow_low, narrow_high)] = highest_density_interval(values, 95)

    assert main_high < narrow_low < 2.5 < narrow_high


def _resampled(target_pdf, size, rng, proposal_sd=3.0, n_proposals=8000):
    """Nested-sampling-like draws: N(0, proposal_sd) proposals importance-
    resampled with replacement to ``target_pdf``, so many draws repeat."""
    proposals = rng.normal(0.0, proposal_sd, n_proposals)
    weights = target_pdf(proposals) / norm.pdf(proposals, 0.0, proposal_sd)
    return rng.choice(proposals, size=size, p=weights / weights.sum())


def _mass_inside(values, pieces):
    return np.mean([any(low <= x <= high for low, high in pieces) for x in values])


@pytest.mark.parametrize("level", [68, 95])
def test_hdi_edges_of_two_narrow_separated_modes_match_the_exact_region(level):
    """A sign-flipped Yukawa: two equal, narrow modes far apart. The exact
    region is each mode's central level-percent interval; the bandwidth that
    counts the modes is set by the distance between them and would blur each
    mode, so the edges must come from a finer one."""
    centres, sigma = (-0.368, 0.0), 0.0045
    rng = np.random.default_rng(0)
    values = np.where(
        rng.random(6400) < 0.5,
        rng.normal(centres[0], sigma, 6400),
        rng.normal(centres[1], sigma, 6400),
    )
    z = norm.ppf(0.5 + level / 200)

    pieces = highest_density_interval(values, level)

    assert len(pieces) == 2
    for (low, high), centre in zip(pieces, centres):
        assert low == pytest.approx(centre - z * sigma, abs=0.3 * sigma)
        assert high == pytest.approx(centre + z * sigma, abs=0.3 * sigma)
    assert _mass_inside(values, pieces) == pytest.approx(level / 100, abs=0.02)


@pytest.mark.parametrize("level", [68, 95])
def test_hdi_of_a_heavily_resampled_bimodal_posterior_keeps_every_mode(level):
    """With only ~2000 distinct draws out of 6400, the fine bandwidth breaks
    each mode into fragments. Keeping only as many fragments as there are
    modes loses mass, and in this sample a whole mode; gluing them per mode
    keeps both modes and at least the level's mass."""

    def target(x):
        return 0.5 * norm.pdf(x, -2.0, 0.3) + 0.5 * norm.pdf(x, 2.0, 0.3)

    values = _resampled(target, 6400, np.random.default_rng(6))

    [(low_a, high_a), (low_b, high_b)] = highest_density_interval(values, level)

    assert low_a < -2.0 < high_a < low_b < 2.0 < high_b
    assert (
        _mass_inside(values, [(low_a, high_a), (low_b, high_b)]) >= level / 100 - 0.01
    )


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
