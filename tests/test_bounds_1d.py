"""Unit tests for smefit.bounds_1d — the 1D confidence bounds every posterior
bounds figure and table is computed from."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from smefit.bounds_1d import (
    Bounds,
    coeff_bounds,
    confidence_bounds,
    mass_reach,
    split_solution,
)
from smefit.fit_result import Fit, FitResult, FitResultGroup

# ---------------------------------------------------------------------------
# confidence_bounds
# ---------------------------------------------------------------------------


def test_confidence_bounds_are_equal_tailed_percentiles() -> None:
    """68% means 16/84, 95% means 2.5/97.5 — the convention the old report
    pipeline used, and the one the tables have always quoted."""
    samples = np.arange(0.0, 1001.0)  # percentile p of 0..1000 is 10 * p

    low, mid, high = confidence_bounds(samples, 68)

    assert (low, high) == pytest.approx((160.0, 840.0))
    assert mid == pytest.approx(500.0)


def test_confidence_bounds_at_95_percent() -> None:
    low, _, high = confidence_bounds(np.arange(0.0, 1001.0), 95)

    assert (low, high) == pytest.approx((25.0, 975.0))


def test_confidence_bounds_central_value_is_the_mean() -> None:
    """Not the median: an asymmetric posterior has to report the same central
    value as the error bars are drawn around."""
    samples = np.array([0.0, 0.0, 0.0, 4.0])

    assert confidence_bounds(samples, 68).mid == pytest.approx(1.0)


def test_confidence_bounds_are_named() -> None:
    """Consumers unpack these three all over the report; a plain tuple would
    make (low, mid, high) a positional convention to remember."""
    bounds = confidence_bounds([1.0, 2.0, 3.0], 68)

    assert isinstance(bounds, Bounds)
    assert (bounds.low, bounds.mid, bounds.high) == tuple(bounds)


def test_confidence_bounds_ignore_nans() -> None:
    """A sampler that wrote a NaN should shrink the sample, not the bounds to
    NaN."""
    with_nan = np.array([0.0, 1.0, 2.0, np.nan])

    assert confidence_bounds(with_nan, 68) == pytest.approx(
        confidence_bounds([0.0, 1.0, 2.0], 68)
    )


def test_confidence_bounds_reproduce_a_gaussian_sigma() -> None:
    """The percentile bounds of a Gaussian posterior are the familiar
    +-sigma, which is what makes the reach definition of feature C usable."""
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=2.0, scale=0.5, size=200_000)

    low, mid, high = confidence_bounds(samples, 68.27)

    assert mid == pytest.approx(2.0, abs=0.01)
    assert (high - low) / 2 == pytest.approx(0.5, rel=0.02)


@pytest.mark.parametrize("level", [0, 100, -5, 0.95, 120])
def test_confidence_bounds_reject_a_level_that_is_not_a_percentage(
    level: float,
) -> None:
    """`confidence_level: 0.95` is the plausible mistake, and it would
    silently draw a 0.95% interval."""
    with pytest.raises(ValueError, match="between 1 and 100"):
        confidence_bounds([1.0, 2.0], level)


def test_confidence_bounds_reject_empty_samples() -> None:
    with pytest.raises(ValueError, match="empty"):
        confidence_bounds([], 68)


# ---------------------------------------------------------------------------
# split_solution
# ---------------------------------------------------------------------------


def test_split_solution_cuts_at_the_midpoint_of_the_range() -> None:
    samples = np.array([0.0, 0.1, 0.2, 5.0, 5.1, 5.2])

    solution1, solution2 = split_solution(samples)

    assert solution1 == pytest.approx([0.0, 0.1, 0.2])
    assert solution2 == pytest.approx([5.0, 5.1, 5.2])


def test_split_solution_returns_the_branch_nearest_zero_first() -> None:
    """Solution 1 is the SM-like one wherever this is used, whichever side of
    the origin it happens to sit on."""
    samples = np.array([-5.2, -5.1, -5.0, -0.2, -0.1, 0.0])

    solution1, solution2 = split_solution(samples)

    assert solution1 == pytest.approx([-0.2, -0.1, 0.0])
    assert solution2 == pytest.approx([-5.2, -5.1, -5.0])


def test_split_solution_keeps_every_sample() -> None:
    """A sample landing exactly on the midpoint belongs to one branch or the
    other; dropping it would bias the bounds of a small posterior."""
    samples = np.array([0.0, 1.0, 2.0])  # midpoint is exactly 1.0

    solution1, solution2 = split_solution(samples)

    assert solution1.size + solution2.size == samples.size


def test_split_solution_bounds_avoid_the_gap_between_the_modes() -> None:
    """The reason this exists: equal-tailed percentiles of the unsplit samples
    span the empty region and put the mean where there is no posterior mass."""
    samples = np.concatenate([np.linspace(0.0, 0.2, 500), np.linspace(5.0, 5.2, 500)])

    unsplit = confidence_bounds(samples, 68)
    first = confidence_bounds(split_solution(samples)[0], 68)

    assert unsplit.low < 1.0 < unsplit.high  # spans the gap
    assert first.high < 1.0


def test_split_solution_rejects_a_single_mode() -> None:
    """A coefficient listed in double_solution whose posterior has one mode is
    a stale runcard, not an empty branch to plot."""
    with pytest.raises(ValueError, match="double_solution"):
        split_solution(np.array([1.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# coeff_bounds — the same numbers, off a Fit
# ---------------------------------------------------------------------------


def make_joint_fit(name: str, samples: dict[str, list[float]]) -> Fit:
    """A joint fit as the bounds routines see one: samples for every free
    coefficient, drawn together."""
    return Fit(
        fit_results=FitResult(
            free_parameters=list(samples),
            best_fit_point={},
            max_loglikelihood=-1.0,
            num_data=10,
            samples={key: jnp.array(vals) for key, vals in samples.items()},
        ),
        fit_name=name,
    )


def make_individual_fit(name: str, samples: dict[str, list[float]]) -> Fit:
    """A one-coefficient-at-a-time fit: one single-parameter FitResult each,
    which is what a `fits:` entry pointing at an individual_fits output
    loads as."""
    return Fit(
        fit_results=FitResultGroup(
            [
                FitResult(
                    free_parameters=[key],
                    best_fit_point={key: 0.0},
                    max_loglikelihood=-1.0,
                    num_data=10,
                    samples={key: jnp.array(vals)},
                )
                for key, vals in samples.items()
            ]
        ),
        fit_name=name,
    )


@pytest.fixture
def joint_fit() -> Fit:
    return make_joint_fit(
        "fit_a", {"OtG": list(np.arange(0.0, 1001.0)), "OpQM": [1.0, 2.0, 3.0]}
    )


def test_coeff_bounds_reads_every_requested_coefficient(joint_fit: Fit) -> None:
    bounds = coeff_bounds(joint_fit, ["OtG", "OpQM"], 68)

    assert list(bounds) == ["OtG", "OpQM"]


def test_coeff_bounds_follows_the_requested_order(joint_fit: Fit) -> None:
    """The caller has already put the coefficients in params_to_plot order;
    the bounds keep it, so consumers can iterate the result directly."""
    assert list(coeff_bounds(joint_fit, ["OpQM", "OtG"], 68)) == ["OpQM", "OtG"]


def test_coeff_bounds_are_the_percentiles_of_the_samples(joint_fit: Fit) -> None:
    (single_solution,) = coeff_bounds(joint_fit, ["OtG"], 68)["OtG"]

    assert single_solution[68.0] == pytest.approx((160.0, 500.0, 840.0))


def test_coeff_bounds_holds_one_entry_per_level(joint_fit: Fit) -> None:
    """A bounds plot draws a thin 95% bar behind a thick 68% one, so both come
    out of one pass over the samples."""
    (single_solution,) = coeff_bounds(joint_fit, ["OtG"], [68, 95])["OtG"]

    assert sorted(single_solution) == [68.0, 95.0]
    assert single_solution[95.0].low < single_solution[68.0].low


def test_coeff_bounds_accepts_a_single_level(joint_fit: Fit) -> None:
    """`confidence_level: 95` is a scalar in the runcard, not a one-element
    list."""
    (single_solution,) = coeff_bounds(joint_fit, ["OtG"], 95)["OtG"]

    assert list(single_solution) == [95.0]


def test_coeff_bounds_gives_one_solution_by_default(joint_fit: Fit) -> None:
    """Bimodality is declared, never detected: a posterior is bimodal because
    of the physics, not because a criterion fired on these samples."""
    assert len(coeff_bounds(joint_fit, ["OtG"], 68)["OtG"]) == 1


def test_coeff_bounds_splits_a_declared_double_solution() -> None:
    fit = make_joint_fit(
        "fit", {"OtG": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2], "OpQM": [0.0, 1.0]}
    )

    solutions = coeff_bounds(fit, ["OtG", "OpQM"], 68, double_solution=["OtG"])

    assert len(solutions["OtG"]) == 2
    assert solutions["OtG"][0][68.0].mid < 1.0  # nearest zero first
    assert solutions["OtG"][1][68.0].mid > 4.0
    assert len(solutions["OpQM"]) == 1


def test_coeff_bounds_accepts_a_bare_double_solution_name() -> None:
    """`double_solution: OtG` is a plausible way to write a one-element
    list, as it is for params_to_plot."""
    fit = make_joint_fit("fit", {"OtG": [0.0, 0.1, 5.0, 5.1]})

    assert len(coeff_bounds(fit, ["OtG"], 68, double_solution="OtG")["OtG"]) == 2


def test_coeff_bounds_skips_a_coefficient_the_fit_never_sampled(
    joint_fit: Fit,
) -> None:
    """One runcard-wide params_to_plot serves several fits, which need not
    have fitted the same coefficients: the consumer draws a gap."""
    bounds = coeff_bounds(joint_fit, ["OtG", "OpMissing"], 68)

    assert "OpMissing" not in bounds
    assert "OtG" in bounds


def test_coeff_bounds_reads_an_individual_fit() -> None:
    """The whole point of going through Fit: individual bounds are a fits:
    entry pointing at an individual_fits output, not a mode switch."""
    fit = make_individual_fit(
        "individual", {"OtG": list(np.arange(0.0, 1001.0)), "OpQM": [1.0, 2.0, 3.0]}
    )

    (single_solution,) = coeff_bounds(fit, ["OtG"], 68)["OtG"]

    assert single_solution[68.0] == pytest.approx((160.0, 500.0, 840.0))


def test_coeff_bounds_of_an_individual_fit_match_the_joint_reading() -> None:
    """Same samples, same numbers, whichever container they arrived in."""
    samples = {"OtG": [0.0, 1.0, 2.0, 3.0], "OpQM": [-1.0, 0.0, 1.0]}

    joint = coeff_bounds(make_joint_fit("j", samples), ["OtG", "OpQM"], [68, 95])
    individual = coeff_bounds(
        make_individual_fit("i", samples), ["OtG", "OpQM"], [68, 95]
    )

    assert joint == individual


def test_coeff_bounds_rejects_a_fit_without_samples() -> None:
    """A Hessian fit that kept no posterior has nothing to take a percentile
    of, and the error has to name which fit that was."""
    fit = Fit(
        fit_results=FitResult(
            free_parameters=["OtG"],
            best_fit_point={"OtG": 1.0},
            max_loglikelihood=-1.0,
            num_data=10,
        ),
        fit_name="sample_less",
    )

    with pytest.raises(ValueError, match="sample_less.*no posterior samples"):
        coeff_bounds(fit, ["OtG"], 68)


# ---------------------------------------------------------------------------
# mass_reach
# ---------------------------------------------------------------------------


def test_mass_reach_is_the_inverse_root_of_the_semi_interval() -> None:
    """A coefficient is a ratio c/Lambda^2 in TeV^-2, so the inverse square
    root of a bound on it is the scale in TeV that bound probes."""
    assert mass_reach(Bounds(-0.25, 0.0, 0.25)) == pytest.approx(1 / np.sqrt(0.25))


def test_mass_reach_can_take_the_whole_interval() -> None:
    assert mass_reach(Bounds(-0.25, 0.0, 0.25), full_interval=True) == pytest.approx(
        1 / np.sqrt(0.5)
    )


def test_mass_reach_reads_an_asymmetric_interval_through_its_width() -> None:
    """Not through max(|low|, |high|): the width is what the fit constrained,
    and where the interval sits is the central value's business."""
    assert mass_reach(Bounds(0.0, 0.25, 0.5)) == pytest.approx(
        mass_reach(Bounds(-0.25, 0.0, 0.25))
    )


def test_mass_reach_of_a_tighter_bound_is_larger() -> None:
    """The whole point of the plot: the better the constraint, the further
    the scale it probes."""
    assert mass_reach(Bounds(-0.1, 0.0, 0.1)) > mass_reach(Bounds(-1.0, 0.0, 1.0))


def test_mass_reach_matches_n_sigma_for_a_gaussian_posterior() -> None:
    """Percentile-based, so it stays right for the non-Gaussian posterior a
    quadratic fit produces — but it must still reproduce the familiar
    1/sqrt(sigma) where the posterior is Gaussian."""
    rng = np.random.default_rng(0)
    samples = rng.normal(0.0, 0.5, size=200_000)

    reach = mass_reach(confidence_bounds(samples, 68.27))

    assert reach == pytest.approx(1 / np.sqrt(0.5), rel=0.01)


@pytest.mark.parametrize(
    "bounds",
    [Bounds(0.5, 0.5, 0.5), Bounds(0.5, 0.0, -0.5), Bounds(-np.inf, 0.0, np.inf)],
)
def test_mass_reach_of_an_unconstrained_coefficient_is_nan(bounds: Bounds) -> None:
    """A bar of infinite height, or none at all, would read as a result; a
    gap says the fit did not constrain the coefficient."""
    assert np.isnan(mass_reach(bounds))
