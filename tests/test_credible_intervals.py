"""Unit tests for smefit.credible_intervals — the HDI estimator on raw sample
arrays, independent of Fit/FitResult. The ETI is exercised through
Fit.confidence_bounds in test_fit_result.py and serves here as the reference
the HDI is compared against."""

import numpy as np
import pytest
from scipy.stats import norm

from smefit import credible_intervals
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


# The tests below pin the KDE bandwidth used to count modes, Silverman's rule:
# arviz's ISJ default and its ``experimental`` mean of ISJ and Silverman split
# resampled unimodal posteriors, and the price is the merge of a small narrow
# mode at a level where only a shallow dip separates it.


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


@pytest.fixture
def narrow_second_mode():
    """80% N(0, 1) + 20% N(2.5, 0.2)."""
    rng = np.random.default_rng(0)
    return np.where(
        rng.random(6400) < 0.8, rng.normal(0.0, 1.0, 6400), rng.normal(2.5, 0.2, 6400)
    )


def test_hdi_resolves_a_small_narrow_second_mode(narrow_second_mode):
    """At 68% the narrow mode is its own piece, with a deep gap before it."""
    [(_, main_high), (narrow_low, narrow_high)] = highest_density_interval(
        narrow_second_mode, 68
    )

    assert main_high < narrow_low < 2.5 < narrow_high


def test_hdi_merges_a_narrow_mode_behind_a_shallow_dip(narrow_second_mode):
    """At 95% the exact region still has a gap before the narrow mode, but
    the threshold is low and the dip above it shallow, and Silverman's rule,
    sized by the overall spread, smooths it away. The region is then one
    interval covering both modes: wider than exact, never narrower."""
    pieces = highest_density_interval(narrow_second_mode, 95)

    [(low, high)] = pieces
    assert low < 0.0 < 2.5 < high
    assert _mass_inside(narrow_second_mode, pieces) >= 0.95


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
    mode, so the edges must come from each mode's own density."""
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
    """With only ~2000 distinct draws out of 6400, a bandwidth that counts
    every draw as independent breaks each mode into fragments. Each mode is
    one piece, and together they hold the level's mass: no more, no less."""

    def target(x):
        return 0.5 * norm.pdf(x, -2.0, 0.3) + 0.5 * norm.pdf(x, 2.0, 0.3)

    values = _resampled(target, 6400, np.random.default_rng(6))

    pieces = highest_density_interval(values, level)

    [(low_a, high_a), (low_b, high_b)] = pieces
    assert low_a < -2.0 < high_a < low_b < 2.0 < high_b
    assert _mass_inside(values, pieces) == pytest.approx(level / 100, abs=0.02)


@pytest.mark.parametrize("level", [68, 95])
def test_hdi_of_a_heavily_resampled_unimodal_posterior_holds_the_level(level):
    """Regression for the review of PR #137: a nested-sampling posterior with
    about a third of its draws distinct, like the ``ultranest_fit`` template's.
    The KDE that counted modes split it at a narrow dip into two touching
    pieces holding 93% (68% level) and 99.9% (95% level) of the draws. It is
    one interval holding the level's mass."""
    values = _resampled(norm.pdf, 3460, np.random.default_rng(5), n_proposals=2500)

    pieces = highest_density_interval(values, level)

    assert len(pieces) == 1
    assert _mass_inside(values, pieces) == pytest.approx(level / 100, abs=0.02)


@pytest.mark.parametrize("level", [68, 95])
def test_the_mode_count_is_sized_by_the_effective_number_of_draws(level):
    """With only ~17% of the draws distinct, Silverman's rule sized by the
    number of draws still splits this unimodal posterior at both levels;
    sized by their effective number, it does not."""
    values = _resampled(norm.pdf, 3460, np.random.default_rng(49), n_proposals=1000)

    pieces = highest_density_interval(values, level)

    assert len(pieces) == 1
    assert _mass_inside(values, pieces) == pytest.approx(level / 100, abs=0.02)


def test_a_mode_of_one_repeated_draw_is_a_point(gaussian):
    """A dead point of large weight, resampled 170 times far from the rest:
    at 95% it is a mode of its own, with no width to estimate a density
    over. Its piece is that point."""
    values = np.concatenate([gaussian, np.full(170, 8.0)])

    pieces = highest_density_interval(values, 95)

    [(low, high), point] = pieces
    assert low < 0.0 < high
    assert point == (8.0, 8.0)
    assert _mass_inside(values, pieces) == pytest.approx(0.95, abs=0.02)


@pytest.mark.parametrize("level", [68, 95])
def test_a_cut_through_a_mode_does_not_split_it(monkeypatch, gaussian, level):
    """If the mode count puts a cut through a mode, the densities on either
    side are still above the threshold at the cut, and the two pieces are
    one: the interval found without the cut."""
    monkeypatch.setattr(
        credible_intervals.az,
        "hdi",
        lambda _values, **_kwargs: np.array([[-1.0, -0.1], [0.1, 1.0]]),
    )

    pieces = highest_density_interval(gaussian, level)

    [(low, high)] = pieces
    z = norm.ppf(0.5 + level / 200)
    assert (low, high) == pytest.approx((-z, z), abs=0.1)
    assert _mass_inside(gaussian, pieces) == pytest.approx(level / 100, abs=0.02)


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


def test_a_hard_bound_pins_the_mode_next_to_it():
    """A sign-definite coefficient with two modes, one piled up against the
    bound: still one piece per mode, and the one at the bound reaches it
    exactly."""
    rng = np.random.default_rng(0)
    values = np.concatenate(
        [np.abs(rng.normal(0.0, 0.5, 3000)), rng.normal(5.0, 0.2, 3000)]
    )

    [(low, high), (far_low, far_high)] = highest_density_interval(
        values, 68, bounds=(0.0, None)
    )

    assert low == 0.0
    assert high < far_low < 5.0 < far_high


def test_hdi_ignores_nans(gaussian):
    with_nan = np.concatenate([gaussian, [np.nan, np.nan]])

    assert highest_density_interval(with_nan, 68) == highest_density_interval(
        gaussian, 68
    )
