"""Unit tests for smefit.plot_utils — select_params and the contour helpers."""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np
import pytest

from smefit.fit_result import Fit, FitResult
from smefit.plot_utils import (
    baseline_point,
    best_fit_pair,
    coeff_limits,
    common_free_coefficients,
    per_fit_option,
    select_params,
)

# ---------------------------------------------------------------------------
# select_params
# ---------------------------------------------------------------------------


def test_select_params_without_a_list_keeps_everything() -> None:
    """No params_to_plot in the runcard means the routine plots all it has,
    in its own order."""
    assert select_params(["OpZZ", "OpA"], None) == ["OpZZ", "OpA"]


def test_select_params_follows_the_requested_order() -> None:
    """The list is a layout as much as a filter: operators are written in the
    order they should be read, which is rarely the alphabetical one."""
    assert select_params(["OpA", "OpB", "OpC"], ["OpC", "OpA"]) == ["OpC", "OpA"]


def test_select_params_skips_names_that_are_not_there() -> None:
    """One runcard-wide list serves several fits, which need not have fitted
    the same coefficients: each keeps the largest subset it can."""
    assert select_params(["OpA", "OpB"], ["OpA", "OpMissing", "OpB"]) == [
        "OpA",
        "OpB",
    ]


def test_select_params_says_which_names_it_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Expected with several fits, so it is reported rather than warned — but
    it still has to be reported, or a typo shrinks a plot in silence."""
    with caplog.at_level(logging.INFO):
        select_params(["OpA"], ["OpA", "OpMissing"], context="my_fit")

    assert "OpMissing" in caplog.text
    assert "my_fit" in caplog.text


def test_select_params_keeps_a_repeated_name_once() -> None:
    """A coefficient listed twice would otherwise be drawn twice."""
    assert select_params(["OpA", "OpB"], ["OpA", "OpB", "OpA"]) == ["OpA", "OpB"]


def test_select_params_accepts_a_bare_name() -> None:
    """`params_to_plot: OtG` is a plausible way to write a one-element list."""
    assert select_params(["OpA", "OtG"], "OtG") == ["OtG"]


def test_select_params_rejects_an_empty_selection() -> None:
    """Nothing left to plot is a misspelt list, not a subset anybody asked
    for, so it says what was available instead of drawing an empty figure."""
    with pytest.raises(ValueError, match="OpA"):
        select_params(["OpA", "OpB"], ["OpTypo"])


# ---------------------------------------------------------------------------
# The contour helpers — all consuming Fit objects
# ---------------------------------------------------------------------------


def make_fit(
    name: str,
    samples: dict[str, list[float]],
    free: list[str],
    best: dict[str, float] | None = None,
    baselines: dict[str, float] | None = None,
) -> Fit:
    """A Fit as the contour helpers see one: joint samples, a name, maybe a
    best-fit point, and the coefficients block of the runcard it was run with
    when a coefficient's baseline is not the origin."""
    runcard: dict = {}
    if baselines is not None:
        runcard["coefficients"] = {
            coeff: {"free": True, "baseline_value": value}
            for coeff, value in baselines.items()
        }
    return Fit(
        fit_results=FitResult(
            free_parameters=free,
            best_fit_point=best or {},
            max_loglikelihood=-1.0,
            num_data=10,
            samples={key: jnp.array(vals) for key, vals in samples.items()},
        ),
        fit_name=name,
        fit_runcard=runcard,
    )


@pytest.fixture
def fit_pair() -> list[Fit]:
    """Two fits sharing OtG and OpQM; only the first also fitted OtW."""
    first = make_fit(
        "fit_a",
        {"OtG": [0.1, 0.2, 0.3], "OpQM": [1.0, 2.0, 3.0], "OtW": [5.0, 6.0, 7.0]},
        ["OtG", "OpQM", "OtW"],
        best={"OtG": 0.2, "OpQM": 2.0},
    )
    second = make_fit(
        "fit_b",
        {"OtG": [-0.1, 0.0, 0.1], "OpQM": [0.5, 1.5, 2.5]},
        ["OtG", "OpQM"],
    )
    return [first, second]


# --- common_free_coefficients ----------------------------------------------


def test_common_free_coefficients_takes_the_intersection(
    fit_pair: list[Fit],
) -> None:
    """Panels overlay every fit, so only coefficients free in all of them can
    be drawn — in the order of the first fit."""
    assert common_free_coefficients(fit_pair) == ["OtG", "OpQM"]


def test_common_free_coefficients_warns_about_dropped_ones(
    fit_pair: list[Fit], caplog: pytest.LogCaptureFixture
) -> None:
    """A coefficient silently missing from a comparison misleads; dropping
    OtW is warned, not just logged."""
    with caplog.at_level(logging.WARNING):
        common_free_coefficients(fit_pair)

    assert "OtW" in caplog.text


def test_common_free_coefficients_applies_params_to_plot(
    fit_pair: list[Fit],
) -> None:
    """params_to_plot restricts and orders, exactly as in every other report
    routine."""
    assert common_free_coefficients(fit_pair, ["OpQM", "OtG"]) == ["OpQM", "OtG"]


def test_common_free_coefficients_intersects_before_selecting(
    fit_pair: list[Fit],
) -> None:
    """Asking for a coefficient only some fits have skips it: the selection
    runs on the intersection, not on the first fit."""
    selected = common_free_coefficients(fit_pair, ["OpQM", "OtW", "OtG"])

    assert selected == ["OpQM", "OtG"]


def test_common_free_coefficients_ignores_non_free_samples() -> None:
    """Samples also hold fixed and derived coefficients; only free ones are
    plotted."""
    fits = [
        make_fit(
            "fit",
            {"OtG": [0.1, 0.2], "OpQM": [1.0, 2.0], "OpDerived": [9.0, 9.0]},
            ["OtG", "OpQM"],
        )
    ] * 2

    assert common_free_coefficients(fits) == ["OtG", "OpQM"]


def test_common_free_coefficients_rejects_fewer_than_two(
    fit_pair: list[Fit],
) -> None:
    """Contours are pairwise: one coefficient has no panel to draw."""
    with pytest.raises(ValueError, match="at least 2"):
        common_free_coefficients(fit_pair, ["OtG"])


def test_common_free_coefficients_rejects_disjoint_fits(fit_pair: list[Fit]) -> None:
    """Fits with nothing in common cannot be overlaid, and the error names
    them."""
    other = make_fit("fit_c", {"OpD": [1.0, 2.0]}, ["OpD"])

    with pytest.raises(ValueError, match="fit_a.*fit_c.*in common"):
        common_free_coefficients([fit_pair[0], other])


# --- coeff_limits -----------------------------------------------------------


def test_coeff_limits_pool_the_samples_of_every_fit(fit_pair: list[Fit]) -> None:
    """A coefficient spans the same range in every panel it appears in, so
    the limits cover both fits' samples."""
    limits = coeff_limits(fit_pair, ["OtG"])

    low, high = limits["OtG"]
    assert low == pytest.approx(-0.1 - 0.1 * 0.4)
    assert high == pytest.approx(0.3 + 0.1 * 0.4)


def test_coeff_limits_stretch_to_an_included_point(fit_pair: list[Fit]) -> None:
    """OpQM is sampled strictly positive, so the SM marker at the origin would
    fall outside a frame drawn from the samples alone."""
    low, high = coeff_limits(fit_pair, ["OpQM"], include_points={"OpQM": 0.0})["OpQM"]

    assert low < 0.0 < high


def test_coeff_limits_stretch_to_a_non_zero_included_point(
    fit_pair: list[Fit],
) -> None:
    """The point is not always the origin: a coefficient whose SM sits away
    from zero must still be inside the frame."""
    low, high = coeff_limits(fit_pair, ["OtG"], include_points={"OtG": 5.0})["OtG"]

    assert high > 5.0
    assert low == pytest.approx(-0.1 - 0.1 * (5.0 + 0.1))


def test_coeff_limits_without_a_point_keep_the_sample_range(
    fit_pair: list[Fit],
) -> None:
    low, _ = coeff_limits(fit_pair, ["OpQM"])["OpQM"]

    assert low > 0.0


def test_coeff_limits_pad_a_zero_width_range_by_one() -> None:
    """A fraction of a zero-wide range would leave nothing to draw in."""
    flat = make_fit("flat", {"OtG": [2.0, 2.0]}, ["OtG"])

    assert coeff_limits([flat], ["OtG"])["OtG"] == (1.0, 3.0)


def test_coeff_limits_padding_is_a_fraction_of_the_range() -> None:
    fit = make_fit("fit", {"OtG": [0.0, 1.0]}, ["OtG"])

    low, high = coeff_limits([fit], ["OtG"], padding=0.5)["OtG"]

    assert (low, high) == pytest.approx((-0.5, 1.5))


# --- per_fit_option ---------------------------------------------------------


def test_per_fit_option_none_falls_back_to_the_defaults(
    fit_pair: list[Fit],
) -> None:
    assert per_fit_option(None, fit_pair, [False, True]) == [False, True]


def test_per_fit_option_broadcasts_a_single_value(fit_pair: list[Fit]) -> None:
    assert per_fit_option(True, fit_pair, [False, False]) == [True, True]


def test_per_fit_option_broadcasts_a_list_value(fit_pair: list[Fit]) -> None:
    """A list is a value like any other — double_solution takes one — not the
    per-fit form, which is a dict keyed by fit name."""
    assert per_fit_option(["OtG"], fit_pair, [[], []]) == [["OtG"], ["OtG"]]


def test_per_fit_option_dict_sets_the_named_fits_only(fit_pair: list[Fit]) -> None:
    assert per_fit_option({"fit_b": True}, fit_pair, [False, False]) == [False, True]


def test_per_fit_option_warns_about_unknown_fit_names(
    fit_pair: list[Fit], caplog: pytest.LogCaptureFixture
) -> None:
    """A misspelt fit name would otherwise change nothing in silence."""
    with caplog.at_level(logging.WARNING):
        resolved = per_fit_option({"fit_typo": True}, fit_pair, [False, False])

    assert resolved == [False, False]
    assert "fit_typo" in caplog.text


# --- baseline_point ---------------------------------------------------------


def test_baseline_point_is_the_origin_without_a_baseline(fit_pair: list[Fit]) -> None:
    """A runcard that never mentions baseline_value puts the SM at zero, which
    is where the marker has always been drawn."""
    assert baseline_point(fit_pair, ["OtG", "OpQM"]) == {"OtG": 0.0, "OpQM": 0.0}


def test_baseline_point_reads_baseline_value_from_the_runcard() -> None:
    """A coefficient parametrised around a non-zero SM value carries it in the
    runcard the fit was run with, not in fit_results.json."""
    fit = make_fit(
        "fit",
        {"OtG": [0.1, 0.2], "OpQM": [1.0, 2.0]},
        ["OtG", "OpQM"],
        baselines={"OtG": 1.5},
    )

    assert baseline_point([fit], ["OtG", "OpQM"]) == {"OtG": 1.5, "OpQM": 0.0}


def test_baseline_point_takes_the_first_fit_and_warns_on_disagreement(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One marker is drawn for every fit overlaid, so a second baseline cannot
    be honoured — but it must not pass unnoticed either."""
    samples = {"OtG": [0.1, 0.2], "OpQM": [1.0, 2.0]}
    first = make_fit("fit_a", samples, ["OtG", "OpQM"], baselines={"OtG": 1.5})
    second = make_fit("fit_b", samples, ["OtG", "OpQM"], baselines={"OtG": -2.0})

    with caplog.at_level(logging.WARNING):
        baselines = baseline_point([first, second], ["OtG", "OpQM"])

    assert baselines == {"OtG": 1.5, "OpQM": 0.0}
    assert "OtG" in caplog.text
    assert "fit_b" in caplog.text


# --- best_fit_pair ----------------------------------------------------------


def test_best_fit_pair_reads_the_best_fit_point(fit_pair: list[Fit]) -> None:
    assert best_fit_pair(fit_pair[0], "OtG", "OpQM") == (0.2, 2.0)


def test_best_fit_pair_is_none_when_a_coefficient_is_missing(
    fit_pair: list[Fit],
) -> None:
    """None rather than an error: the consumer falls back to the posterior
    means then."""
    assert best_fit_pair(fit_pair[1], "OtG", "OpQM") is None


def test_best_fit_pair_returns_plain_floats(fit_pair: list[Fit]) -> None:
    """The pair is scattered by matplotlib, which wants numbers, not arrays."""
    pair = best_fit_pair(fit_pair[0], "OtG", "OpQM")

    assert pair is not None
    assert all(isinstance(value, float) for value in pair)


def test_helpers_leave_the_fit_samples_untouched(fit_pair: list[Fit]) -> None:
    """Limits are computed on copies: plotting must not mutate a fit."""
    results = fit_pair[0].fit_results
    assert isinstance(results, FitResult) and results.samples is not None
    before = {name: np.asarray(vals).copy() for name, vals in results.samples.items()}

    coeff_limits(fit_pair, common_free_coefficients(fit_pair))

    for name, vals in results.samples.items():
        np.testing.assert_array_equal(np.asarray(vals), before[name])
