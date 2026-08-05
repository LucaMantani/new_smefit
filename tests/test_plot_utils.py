"""Unit tests for smefit.plot_utils — helpers shared by the plotting routines.

Loading fits from disk is :class:`~smefit.fit_result.Fit`'s job and is covered
by ``tests/test_fit_result.py``; here every fit is built in memory.
"""

import jax.numpy as jnp
import pytest

from smefit.fit_result import Fit
from smefit.plot_utils import (
    best_fit_pair,
    coeff_limits,
    common_coefficients,
    default_labels,
    latex_label,
    per_fit_option,
    plot_coefficients,
    require_joint_posterior,
    require_samples,
)


def make_fit(names, fit_name="fit", free=None, best_fit=None, **kwargs):
    """A Fit sampling *names*, free in all of them unless *free* says otherwise."""
    free = list(names) if free is None else list(free)
    return Fit(
        free_parameters=free,
        best_fit_point=best_fit if best_fit is not None else {},
        max_loglikelihood=-1.0,
        num_data=10,
        samples={n: jnp.array([0.0, 1.0]) for n in names},
        fit_name=fit_name,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# latex_label / default_labels
# ---------------------------------------------------------------------------


def test_latex_label_escapes_underscores():
    assert latex_label("my_fit_lin") == r"$\mathrm{my\_fit\_lin}$"


def test_default_labels_fall_back_to_the_escaped_name():
    assert default_labels([make_fit(["OpA"], fit_name="my_fit")]) == [
        r"$\mathrm{my\_fit}$"
    ]


def test_default_labels_use_the_runcard_label_verbatim():
    """An explicit label is already LaTeX and must not be escaped or wrapped."""
    label = r"$\mathrm{FCC}\textnormal{-}\mathrm{ee\ descoped,\ 2IP}$"
    fits = [
        make_fit(["OpA"], fit_name="fit_a", label=label),
        make_fit(["OpA"], fit_name="fit_b"),
    ]

    assert default_labels(fits) == [label, r"$\mathrm{fit\_b}$"]


def test_default_labels_ignore_an_empty_label():
    fit = make_fit(["OpA"], fit_name="my_fit", label="")

    assert default_labels([fit]) == [r"$\mathrm{my\_fit}$"]


# ---------------------------------------------------------------------------
# plot_coefficients
# ---------------------------------------------------------------------------


def test_plot_coefficients_are_the_free_ones():
    fit = make_fit(["OpA", "OpDerived"], free=["OpA"])

    assert plot_coefficients(fit) == ["OpA"]


def test_plot_coefficients_fall_back_to_every_sampled_one():
    fit = make_fit(["OpA", "OpB"], free=[])

    assert plot_coefficients(fit) == ["OpA", "OpB"]


# ---------------------------------------------------------------------------
# require_samples / require_joint_posterior
# ---------------------------------------------------------------------------


def test_require_samples_passes_with_samples():
    require_samples([make_fit(["OpA"])])


def test_require_samples_raises_without_samples():
    fit = Fit(
        free_parameters=["OpA"],
        best_fit_point={"OpA": 0.0},
        max_loglikelihood=-1.0,
        num_data=10,
        samples=None,
        fit_name="empty_fit",
    )

    with pytest.raises(ValueError, match="no posterior samples"):
        require_samples([fit])


def test_require_joint_posterior_rejects_individual_fits():
    fit = make_fit(["OpA", "OpB"], fit_name="ind", individual_fit=True)

    with pytest.raises(ValueError, match="individual"):
        require_joint_posterior([fit])


def test_require_joint_posterior_passes_for_a_joint_fit():
    require_joint_posterior([make_fit(["OpA", "OpB"])])


# ---------------------------------------------------------------------------
# common_coefficients
# ---------------------------------------------------------------------------


def test_common_coefficients_intersects_and_keeps_first_order():
    first = make_fit(["OpB", "OpA", "OpC"])
    second = make_fit(["OpA", "OpB"])

    assert common_coefficients([first, second]) == ["OpB", "OpA"]


def test_common_coefficients_warns_about_dropped_coefficients(caplog):
    with caplog.at_level("WARNING"):
        common_coefficients([make_fit(["OpA", "OpC"]), make_fit(["OpA"])])

    assert "OpC" in caplog.text


def test_common_coefficients_raises_when_nothing_is_shared():
    with pytest.raises(ValueError, match="no free coefficient in common"):
        common_coefficients([make_fit(["OpA"]), make_fit(["OpB"])])


def test_common_coefficients_dofs_show_selects_and_orders():
    fit = make_fit(["OpA", "OpB", "OpC"])

    assert common_coefficients([fit], ["OpC", "OpA"]) == ["OpC", "OpA"]


def test_common_coefficients_dofs_show_may_include_derived_coefficients():
    """dofs_show is checked against the samples, not only the free parameters."""
    fit = make_fit(["OpA", "OpDerived"], free=["OpA"])

    assert common_coefficients([fit], ["OpA", "OpDerived"]) == ["OpA", "OpDerived"]


def test_common_coefficients_dofs_show_missing_coefficient_raises():
    fit = make_fit(["OpA"], fit_name="my_fit")

    with pytest.raises(ValueError, match="OpZ"):
        common_coefficients([fit], ["OpA", "OpZ"])


# ---------------------------------------------------------------------------
# coeff_limits
# ---------------------------------------------------------------------------


def test_coeff_limits_span_every_fit():
    first = Fit(
        free_parameters=["OpA"],
        best_fit_point={},
        max_loglikelihood=-1.0,
        num_data=10,
        samples={"OpA": jnp.array([1.0, 2.0])},
        fit_name="a",
    )
    second = Fit(
        free_parameters=["OpA"],
        best_fit_point={},
        max_loglikelihood=-1.0,
        num_data=10,
        samples={"OpA": jnp.array([3.0, 4.0])},
        fit_name="b",
    )

    low, high = coeff_limits([first, second], ["OpA"], padding=0.0, include_sm=False)[
        "OpA"
    ]
    assert (low, high) == pytest.approx((1.0, 4.0))


def test_coeff_limits_keep_the_sm_point_in_frame():
    fit = Fit(
        free_parameters=["OpA"],
        best_fit_point={},
        max_loglikelihood=-1.0,
        num_data=10,
        samples={"OpA": jnp.array([1.0, 2.0])},
        fit_name="a",
    )

    low, _ = coeff_limits([fit], ["OpA"], padding=0.0, include_sm=True)["OpA"]
    assert low == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# per_fit_option
# ---------------------------------------------------------------------------


def test_per_fit_option_none_keeps_the_defaults():
    fits = [make_fit(["OpA"], fit_name="a"), make_fit(["OpA"], fit_name="b")]

    assert per_fit_option(None, fits, [True, False]) == [True, False]


def test_per_fit_option_scalar_applies_to_every_fit():
    fits = [make_fit(["OpA"], fit_name="a"), make_fit(["OpA"], fit_name="b")]

    assert per_fit_option(True, fits, [False, False]) == [True, True]


def test_per_fit_option_dict_is_keyed_by_fit_name():
    fits = [make_fit(["OpA"], fit_name="a"), make_fit(["OpA"], fit_name="b")]

    assert per_fit_option({"b": True}, fits, [False, False]) == [False, True]


def test_per_fit_option_warns_about_unknown_fit_names(caplog):
    fits = [make_fit(["OpA"], fit_name="a")]

    with caplog.at_level("WARNING"):
        per_fit_option({"nope": True}, fits, [False])

    assert "nope" in caplog.text


# ---------------------------------------------------------------------------
# best_fit_pair
# ---------------------------------------------------------------------------


def test_best_fit_pair_returns_the_pair():
    fit = make_fit(["OpA", "OpB"], best_fit={"OpA": 0.5, "OpB": -0.5})

    assert best_fit_pair(fit, "OpA", "OpB") == (0.5, -0.5)


def test_best_fit_pair_is_none_when_a_coefficient_is_missing():
    fit = make_fit(["OpA", "OpB"], best_fit={"OpA": 0.5})

    assert best_fit_pair(fit, "OpA", "OpB") is None
