"""Unit tests for smefit.plot_utils — select_params and the contour helpers."""

from __future__ import annotations

import logging
import re

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.ticker import ScalarFormatter

from smefit.fit_result import Fit, FitResult, FitResultGroup
from smefit.plot_utils import (
    baseline_point,
    best_fit_pair,
    coeff_limits,
    common_free_coefficients,
    compact_tick_labels,
    contour_coefficients,
    per_fit_option,
    select_params,
    stuck_values,
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
    fixed: dict[str, float] | None = None,
) -> Fit:
    """A Fit as the contour helpers see one: joint samples, a name, maybe a
    best-fit point, and the coefficients block of the runcard it was run with
    when a coefficient's baseline is not the origin.

    ``fixed`` names coefficients the runcard froze, as smefit records them: a
    ``free: False, value: v`` entry in the runcard, and the constant ``v``
    resolved into every posterior sample."""
    coefficients: dict[str, dict] = {}
    if baselines is not None:
        coefficients.update(
            {
                coeff: {"free": True, "baseline_value": value}
                for coeff, value in baselines.items()
            }
        )
    if fixed is not None:
        coefficients.update(
            {coeff: {"free": False, "value": value} for coeff, value in fixed.items()}
        )
        length = len(next(iter(samples.values()))) if samples else 2
        samples = {
            **samples,
            **{coeff: [value] * length for coeff, value in fixed.items()},
        }

    runcard: dict = {"coefficients": coefficients} if coefficients else {}
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


def make_individual_fit(name: str, samples: dict[str, list[float]]) -> Fit:
    """A fit run one coefficient at a time: a group of single-parameter
    results rather than one joint result."""
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


def test_common_free_coefficients_keeps_one_when_min_count_allows_it(
    fit_pair: list[Fit],
) -> None:
    """The 1D bounds routines have a panel to draw of a single coefficient,
    so they ask for one."""
    assert common_free_coefficients(fit_pair, ["OtG"], min_count=1) == ["OtG"]


def test_common_free_coefficients_reports_the_count_it_needed(
    fit_pair: list[Fit],
) -> None:
    """The error says how many the caller wanted, since it is no longer
    always two."""
    with pytest.raises(ValueError, match="at least 3"):
        common_free_coefficients(fit_pair, min_count=3)


def test_common_free_coefficients_reads_an_individual_fit() -> None:
    """An individual fit reports the coefficients it fitted one by one, so a
    fits: entry pointing at an individual_fits output can be overlaid on a
    joint one by the routines that read 1D posteriors."""
    individual = make_individual_fit(
        "individual", {"OtG": [0.1, 0.2], "OpQM": [1.0, 2.0]}
    )
    joint = make_fit("joint", {"OtG": [0.0, 0.3], "OpQM": [0.5, 2.5]}, ["OtG", "OpQM"])

    assert common_free_coefficients([individual, joint]) == ["OtG", "OpQM"]


def test_common_free_coefficients_rejects_disjoint_fits(fit_pair: list[Fit]) -> None:
    """Fits with nothing in common cannot be overlaid, and the error names
    them."""
    other = make_fit("fit_c", {"OpD": [1.0, 2.0]}, ["OpD"])

    with pytest.raises(ValueError, match="fit_a.*fit_c.*in common"):
        common_free_coefficients([fit_pair[0], other])


# --- contour_coefficients ---------------------------------------------------


@pytest.fixture
def stuck_pair() -> list[Fit]:
    """Two fits over the same operators, the second having frozen OpQM at 0.5
    and never having heard of OtW."""
    first = make_fit(
        "fit_a",
        {"OtG": [0.1, 0.2, 0.3], "OpQM": [1.0, 2.0, 3.0], "OtW": [5.0, 6.0, 7.0]},
        ["OtG", "OpQM", "OtW"],
        baselines={"OtG": 0.0, "OpQM": 0.0, "OtW": 0.0},
    )
    second = make_fit(
        "fit_b",
        {"OtG": [-0.1, 0.0, 0.1]},
        ["OtG"],
        fixed={"OpQM": 0.5},
    )
    return [first, second]


def test_contour_coefficients_take_the_union(fit_pair: list[Fit]) -> None:
    """A coefficient only one fit floated still has a panel: the other is
    drawn where it held it, which is the comparison worth seeing."""
    assert contour_coefficients(fit_pair) == ["OtG", "OpQM", "OtW"]


def test_contour_coefficients_follow_the_first_fit_then_the_newcomers() -> None:
    """Order is the first fit's, and a coefficient only a later fit floated
    joins at the end rather than reshuffling the grid."""
    first = make_fit("fit_a", {"OtG": [0.1, 0.2]}, ["OtG"])
    second = make_fit("fit_b", {"OpQM": [1.0, 2.0], "OtG": [0.0, 0.3]}, ["OpQM", "OtG"])

    assert contour_coefficients([first, second]) == ["OtG", "OpQM"]


def test_contour_coefficients_do_not_warn_about_a_frozen_one(
    stuck_pair: list[Fit], caplog: pytest.LogCaptureFixture
) -> None:
    """What common_free_coefficients warns about dropping is now drawn."""
    with caplog.at_level(logging.WARNING):
        assert contour_coefficients(stuck_pair) == ["OtG", "OpQM", "OtW"]

    assert caplog.text == ""


def test_contour_coefficients_apply_params_to_plot(fit_pair: list[Fit]) -> None:
    assert contour_coefficients(fit_pair, ["OpQM", "OtG"]) == ["OpQM", "OtG"]


def test_contour_coefficients_accept_one_no_fit_floated(
    stuck_pair: list[Fit],
) -> None:
    """OpDerived is free nowhere, so it is not drawn by default — but naming
    it is a legitimate request, and every fit is then drawn stuck on it."""
    frozen = make_fit("fit_c", {"OtG": [0.1, 0.2]}, ["OtG"], fixed={"OpDerived": 2.0})

    assert contour_coefficients([frozen], ["OtG", "OpDerived"]) == [
        "OtG",
        "OpDerived",
    ]


def test_contour_coefficients_reject_fewer_than_two(fit_pair: list[Fit]) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        contour_coefficients(fit_pair, ["OtG"])


def test_contour_coefficients_reject_fits_without_a_free_one() -> None:
    """Nothing was sampled anywhere, so there is no posterior in the figure at
    all — a runcard mistake rather than a figure."""
    frozen = make_fit("fit_a", {}, [], fixed={"OtG": 1.0, "OpQM": 2.0})

    with pytest.raises(ValueError, match="fit_a.*free coefficient"):
        contour_coefficients([frozen])


# --- stuck_values -----------------------------------------------------------


def test_stuck_values_report_a_frozen_coefficient(stuck_pair: list[Fit]) -> None:
    """A fixed coefficient reaches the posterior as its constant value, which
    is where the fit is stuck."""
    first, second = stuck_values(stuck_pair, ["OtG", "OpQM"])

    assert first == {}  # both sampled
    assert second == {"OpQM": 0.5}


def test_stuck_values_place_an_undeclared_coefficient_at_its_baseline(
    stuck_pair: list[Fit],
) -> None:
    """fit_b never mentions OtW, so it sits where its SM does — the baseline
    the fit that does declare it gives."""
    first, second = stuck_values(stuck_pair, ["OtW"], {"OtW": 1.5})

    assert first == {}
    assert second == {"OtW": 1.5}


def test_stuck_values_compute_the_baselines_when_not_given() -> None:
    """The default is the same SM point the marker is drawn at."""
    declaring = make_fit(
        "fit_a",
        {"OtG": [0.1, 0.2], "OtW": [5.0, 6.0]},
        ["OtG", "OtW"],
        baselines={"OtW": -2.0},
    )
    other = make_fit("fit_b", {"OtG": [0.0, 0.3]}, ["OtG"])

    assert stuck_values([declaring, other], ["OtW"]) == [{}, {"OtW": -2.0}]


def test_stuck_values_keep_a_derived_coefficient_that_moves() -> None:
    """An expr-constrained coefficient follows the free ones: its samples are
    a posterior to contour, not a point to mark."""
    fit = make_fit(
        "fit",
        {"OtG": [0.1, 0.2, 0.3], "OpDerived": [0.01, 0.04, 0.09]},
        ["OtG"],
    )

    assert stuck_values([fit], ["OtG", "OpDerived"]) == [{}]


def test_stuck_values_warn_about_a_free_coefficient_that_never_moved(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A sampler that left a free coefficient where it started has no density
    to estimate; it is drawn as fixed, but not silently."""
    fit = make_fit("fit", {"OtG": [0.1, 0.2], "OpQM": [1.0, 1.0]}, ["OtG", "OpQM"])

    with caplog.at_level(logging.WARNING):
        assert stuck_values([fit], ["OtG", "OpQM"]) == [{"OpQM": 1.0}]

    assert "OpQM" in caplog.text
    assert "fit" in caplog.text


def test_stuck_values_return_plain_floats(stuck_pair: list[Fit]) -> None:
    """The values are handed to matplotlib as coordinates, not as arrays."""
    _, second = stuck_values(stuck_pair, ["OpQM"])

    assert isinstance(second["OpQM"], float)


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


def test_coeff_limits_pad_a_zero_width_range_by_its_own_scale() -> None:
    """A fraction of a zero-wide range would leave nothing to draw in, and a
    flat unit would swamp a coefficient of 1e-3 — a coefficient every fit held
    fixed is one of the cases this covers."""
    flat = make_fit("flat", {"OtG": [2.0, 2.0]}, ["OtG"])

    assert coeff_limits([flat], ["OtG"])["OtG"] == (0.0, 4.0)


def test_coeff_limits_pad_a_zero_width_range_at_the_origin_by_one() -> None:
    """Nothing to scale by there, so the unit comes back."""
    flat = make_fit("flat", {"OtG": [0.0, 0.0]}, ["OtG"])

    assert coeff_limits([flat], ["OtG"])["OtG"] == (-1.0, 1.0)


def test_coeff_limits_skip_a_fit_that_lacks_the_coefficient(
    fit_pair: list[Fit],
) -> None:
    """Only fit_a sampled OtW; fit_b is drawn at a point instead, which
    reaches the range as an included point, not as samples it does not have."""
    low, high = coeff_limits(fit_pair, ["OtW"])["OtW"]

    assert (low, high) == pytest.approx((5.0 - 0.2, 7.0 + 0.2))


def test_coeff_limits_stretch_to_every_included_point(fit_pair: list[Fit]) -> None:
    """Several points per coefficient: the SM marker and the value each fit
    that did not sample it was stuck at."""
    low, high = coeff_limits(fit_pair, ["OtW"], include_points={"OtW": [0.0, 9.0]})[
        "OtW"
    ]

    assert low < 0.0
    assert high > 9.0


def test_coeff_limits_frame_a_coefficient_no_fit_sampled(
    fit_pair: list[Fit],
) -> None:
    """Every fit stuck on it: the frame comes from the points alone."""
    low, high = coeff_limits(fit_pair, ["OpFrozen"], include_points={"OpFrozen": 0.5})[
        "OpFrozen"
    ]

    assert (low, high) == pytest.approx((0.0, 1.0))


def test_coeff_limits_reject_a_coefficient_with_nothing_to_place_it_by(
    fit_pair: list[Fit],
) -> None:
    with pytest.raises(ValueError, match="OpFrozen"):
        coeff_limits(fit_pair, ["OpFrozen"])


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
    """A list is a value like any other, not the per-fit form — that is a dict
    keyed by fit name."""
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


def test_baseline_point_asks_the_first_fit_that_declares_the_coefficient(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A fit that never declared a coefficient has no baseline to offer: it
    must not drag the SM point back to the origin, nor be warned about."""
    samples = {"OtG": [0.1, 0.2]}
    silent = make_fit("fit_a", samples, ["OtG"])
    declaring = make_fit("fit_b", samples, ["OtG"], baselines={"OtG": 0.0, "OtW": 1.5})

    with caplog.at_level(logging.WARNING):
        assert baseline_point([silent, declaring], ["OtW"]) == {"OtW": 1.5}

    assert caplog.text == ""


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


# --- compact_tick_labels ----------------------------------------------------


def test_compact_tick_labels_factors_the_power_out_of_small_numbers(draw) -> None:
    """A coefficient of 1e-4 would otherwise print five leading zeros in every
    label, and a panel has room for about four characters."""
    fig, ax = plt.subplots()
    ax.set_xlim(-5e-4, 5e-4)

    compact_tick_labels(ax)
    draw(fig)

    # the labels come out as mathtext, so the digits are what to look at
    digits = [
        re.sub(r"[^0-9.]", "", t.get_text())
        for t in ax.get_xticklabels()
        if t.get_text()
    ]
    assert digits, "no tick labels at all"
    assert not any("0.00" in label for label in digits), digits
    assert all(len(label) <= 3 for label in digits), digits
    # the power is written once, at the end of the axis
    assert "10" in ax.xaxis.get_offset_text().get_text()
    plt.close(fig)


def test_compact_tick_labels_leave_ordinary_numbers_alone(draw) -> None:
    """Nothing is factored out of a range that reads perfectly well as it
    is — the power would be one more thing to look up."""
    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, 0.5)

    compact_tick_labels(ax)
    draw(fig)

    assert not ax.xaxis.get_offset_text().get_text()
    plt.close(fig)


def test_compact_tick_labels_reformat_both_axes() -> None:
    """A coefficient sits on the x-axis of one panel and the y-axis of
    another, and has to read the same way in both."""
    fig, ax = plt.subplots()

    compact_tick_labels(ax)

    assert isinstance(ax.xaxis.get_major_formatter(), ScalarFormatter)
    assert isinstance(ax.yaxis.get_major_formatter(), ScalarFormatter)
    plt.close(fig)


def test_compact_tick_labels_can_hide_the_power() -> None:
    """matplotlib draws the power whatever `labelbottom` says, so an inner
    panel of a grid would carry a stray one of its own."""
    fig, ax = plt.subplots()

    compact_tick_labels(ax, show_x_offset=False, show_y_offset=True)

    assert not ax.xaxis.offsetText.get_visible()
    assert ax.yaxis.offsetText.get_visible()
    plt.close(fig)


def test_compact_tick_labels_can_reformat_one_axis_only() -> None:
    """The bounds plot names its rows after coefficients: a numeric formatter
    on that axis would replace the names with the positions they sit at."""
    fig, ax = plt.subplots()
    default = ax.yaxis.get_major_formatter()

    compact_tick_labels(ax, axis="x")

    assert isinstance(ax.xaxis.get_major_formatter(), ScalarFormatter)
    assert ax.yaxis.get_major_formatter() is default
    plt.close(fig)


def test_compact_tick_labels_rejects_an_unknown_axis() -> None:
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="'both', 'x' or 'y'"):
        compact_tick_labels(ax, axis="z")
    plt.close(fig)
