"""Unit tests for smefit.figures — heatmaps and corner plots.

_plot_heatmap enables matplotlib's usetex globally, which would require a
LaTeX installation to actually render text. Tests here never draw/save the
figure (get_text() etc. only read stored attributes), and the module-level
`rc` call is neutralised so running the suite never depends on LaTeX being
installed (e.g. on CI runners).
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from smefit import figures as figures_mod
from smefit.figures import (
    _plot_heatmap,
    corner_plot,
    plot_corner,
    plot_corner_analytic,
    plot_corner_blackjax,
    plot_corner_hessian,
    plot_corner_ultranest,
    plot_fisher_diagonals_heatmap,
)
from smefit.fit_result import Fit


@pytest.fixture(autouse=True)
def _no_matplotlib_rc(monkeypatch):
    monkeypatch.setattr(figures_mod, "rc", lambda *args, **kwargs: None)
    yield
    plt.close("all")


def test_plot_heatmap_axis_labels():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    fig = _plot_heatmap(matrix, ["OpA", "OpZZ"], ["DS_A", "DS_B"], vmin=0, vmax=10)

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["DS_A", "DS_B"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["OpA", "OpZZ"]


def test_plot_heatmap_uses_latex_label_when_known():
    """Coefficient names present in coeff_info_latex are relabelled on the y-axis."""
    matrix = np.array([[1.0]])
    fig = _plot_heatmap(matrix, ["OQQ1"], ["DS_A"], vmin=0, vmax=10)

    ax = fig.axes[0]
    assert ax.get_yticklabels()[0].get_text() == r"$c_{QQ}^{\scriptscriptstyle 1}$"


def test_plot_heatmap_skips_zero_cells():
    """Cells equal to 0.0 get no text annotation."""
    matrix = np.array([[0.0, 5.0]])
    fig = _plot_heatmap(matrix, ["OpA"], ["DS_A", "DS_B"], vmin=0, vmax=10)

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.texts] == ["5.0"]


def test_plot_heatmap_text_color_threshold():
    """Cell text is white above 60% of vmax, black otherwise."""
    matrix = np.array([[8.0, 2.0]])
    fig = _plot_heatmap(matrix, ["OpA"], ["DS_A", "DS_B"], vmin=0, vmax=10)

    ax = fig.axes[0]
    colors = {t.get_text(): t.get_color() for t in ax.texts}
    assert colors["8.0"] == "white"
    assert colors["2.0"] == "black"


def test_plot_heatmap_figsize_scales_with_shape():
    small = _plot_heatmap(np.zeros((1, 1)), ["OpA"], ["DS_A"], vmin=0, vmax=1)
    large = _plot_heatmap(
        np.zeros((8, 10)),
        [f"Op{i}" for i in range(8)],
        [f"DS_{i}" for i in range(10)],
        vmin=0,
        vmax=1,
    )

    assert large.get_size_inches()[0] > small.get_size_inches()[0]
    assert large.get_size_inches()[1] > small.get_size_inches()[1]


def test_plot_fisher_diagonals_heatmap_scales_values_to_percent():
    fd = pd.DataFrame(
        {"DS_A": [0.3, 0.7], "DS_B": [0.6, 0.4]},
        index=["OpA", "OpZZ"],
    )

    fig = plot_fisher_diagonals_heatmap(fd)

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["DS_A", "DS_B"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["OpA", "OpZZ"]
    texts = {t.get_text() for t in ax.texts}
    assert texts == {"30.0", "70.0", "60.0", "40.0"}


# ---------------------------------------------------------------------------
# corner_plot
# ---------------------------------------------------------------------------


def _fit(fit_name="fit", names=("OpX", "OpY", "OpZ"), seed=0, shift=0.0, **kwargs):
    """A Fit with Gaussian samples for every coefficient in *names*."""
    rng = np.random.default_rng(seed)
    values = (
        rng.multivariate_normal(np.zeros(len(names)), np.eye(len(names)), size=500)
        + shift
    )
    kwargs.setdefault("best_fit_point", {})
    return Fit(
        free_parameters=list(names),
        max_loglikelihood=-1.0,
        num_data=10,
        samples={n: jnp.array(values[:, i]) for i, n in enumerate(names)},
        fit_name=fit_name,
        **kwargs,
    )


def _legend(fig):
    return [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None][0]


def test_corner_plot_builds_the_lower_triangle():
    """n coefficients give n(n-1)/2 panels plus the legend axes."""
    fig = corner_plot([_fit()])

    # 3 coefficients -> 3 pairs, plus the legend axes
    assert len(fig.axes) == 4


def test_corner_plot_panels_are_placed_below_the_diagonal():
    fig = corner_plot([_fit()])

    gridspecs = [ax.get_subplotspec().get_topmost_subplotspec() for ax in fig.axes[:3]]
    positions = {(gs.rowspan.start, gs.colspan.start) for gs in gridspecs}
    assert positions == {(0, 0), (1, 0), (1, 1)}


def test_corner_plot_labels_only_the_outer_panels():
    fig = corner_plot([_fit(names=("OpX", "OpY", "OpZ"))])

    labelled_x = {ax.get_xlabel() for ax in fig.axes if ax.get_xlabel()}
    labelled_y = {ax.get_ylabel() for ax in fig.axes if ax.get_ylabel()}
    # bottom row carries OpX and OpY, left column carries OpY and OpZ
    assert labelled_x == {"OpX", "OpY"}
    assert labelled_y == {"OpY", "OpZ"}


def test_corner_plot_axis_limits_are_shared_per_coefficient():
    fig = corner_plot([_fit()])

    # panel (row 1, col 0) has OpX on x, panel (row 1, col 1) has OpY on x,
    # and panel (row 0, col 0) has OpY on y: the OpY range must agree
    by_position = {
        (
            ax.get_subplotspec().rowspan.start,
            ax.get_subplotspec().colspan.start,
        ): ax
        for ax in fig.axes[:3]
    }
    assert by_position[(1, 1)].get_xlim() == by_position[(0, 0)].get_ylim()


def test_corner_plot_limits_always_contain_the_sm_point():
    # push every sample far away from the SM
    fig = corner_plot([_fit(names=("OpX", "OpY"), shift=100.0)])

    low, high = fig.axes[0].get_xlim()
    assert low <= 0 <= high


def test_corner_plot_dofs_show_selects_coefficients():
    fig = corner_plot([_fit()], dofs_show=["OpX", "OpZ"])

    # a single pair: one panel, and the legend goes inside it
    assert len(fig.axes) == 1
    assert fig.axes[0].get_xlabel() == "OpX"
    assert fig.axes[0].get_ylabel() == "OpZ"


def test_corner_plot_two_confidence_levels_draw_an_extra_contour():
    single = corner_plot([_fit(names=("OpX", "OpY"))])
    double = corner_plot([_fit(names=("OpX", "OpY"))], confidence_level=[68, 95])

    assert len(double.axes[0].patches) == len(single.axes[0].patches) + 1


def test_corner_plot_confidence_level_shown_is_the_outer_one():
    fig = corner_plot([_fit(names=("OpX", "OpY"))], confidence_level=[68, 95])

    assert any("95" in text.get_text() for text in fig.axes[0].texts)


def test_corner_plot_subplot_size_scales_the_figure():
    small = corner_plot([_fit()], subplot_size=2)
    large = corner_plot([_fit()], subplot_size=4)

    assert large.get_size_inches()[0] == 2 * small.get_size_inches()[0]


def test_corner_plot_requires_at_least_two_coefficients():
    with pytest.raises(ValueError, match="at least 2 coefficients"):
        corner_plot([_fit(names=("OpX",))])


def test_corner_plot_without_fits_raises():
    with pytest.raises(ValueError, match="No fits to plot"):
        corner_plot([])


def test_corner_plot_rejects_a_fit_without_samples():
    fit = Fit(
        free_parameters=["OpX", "OpY"],
        best_fit_point={},
        max_loglikelihood=-1.0,
        num_data=10,
        samples=None,
        fit_name="no_samples",
    )

    with pytest.raises(ValueError, match="no posterior samples"):
        corner_plot([fit])


def test_corner_plot_rejects_an_individual_fit():
    """One-at-a-time posteriors are independent: their 2D contours are artefacts."""
    fit = _fit(names=("OpX", "OpY"), individual_fit=True)

    with pytest.raises(ValueError, match="individual"):
        corner_plot([fit])


# ---------------------------------------------------------------------------
# legend labels
# ---------------------------------------------------------------------------


def test_corner_plot_legend_lists_every_fit_and_the_sm():
    fits = [_fit(fit_name="fit_a"), _fit(fit_name="fit_b", seed=1)]

    fig = corner_plot(fits)

    assert [t.get_text() for t in _legend(fig).get_texts()] == [
        r"$\mathrm{fit\_a}$",
        r"$\mathrm{fit\_b}$",
        r"$\mathrm{SM}$",
    ]


def test_corner_plot_legend_uses_the_label_of_the_fits_entry():
    """A runcard 'label' is raw LaTeX and reaches the legend untouched."""
    label = r"$\mathrm{FCC}\textnormal{-}\mathrm{ee\ descoped,\ 2IP}$"
    fits = [
        _fit(fit_name="a_very_long_fit_name", label=label),
        _fit(fit_name="fit_b", seed=1),
    ]

    fig = corner_plot(fits)

    assert [t.get_text() for t in _legend(fig).get_texts()] == [
        label,
        r"$\mathrm{fit\_b}$",
        r"$\mathrm{SM}$",
    ]


def test_corner_plot_labels_setting_overrides_the_fits_entry_label():
    fits = [_fit(fit_name="fit_a", label="$A$"), _fit(fit_name="fit_b", seed=1)]

    fig = corner_plot(fits, labels={"fit_a": "$override$"})

    assert [t.get_text() for t in _legend(fig).get_texts()] == [
        "$override$",
        r"$\mathrm{fit\_b}$",
        r"$\mathrm{SM}$",
    ]


# ---------------------------------------------------------------------------
# contour style
# ---------------------------------------------------------------------------


def test_corner_plot_kde_uses_contours_not_ellipses():
    fig = corner_plot([_fit(names=("OpX", "OpY"))], kde=True)

    assert not fig.axes[0].patches  # no ellipse patches
    assert fig.axes[0].collections  # filled + outlined KDE contours


def test_corner_plot_kde_defaults_to_use_quad_of_the_fit():
    """Quadratic fits generally have non-Gaussian posteriors."""
    quad = corner_plot([_fit(names=("OpX", "OpY"), use_quad=True)])
    lin = corner_plot([_fit(names=("OpX", "OpY"), use_quad=False)])

    assert not quad.axes[0].patches
    assert len(lin.axes[0].patches) == 2  # outlined + filled ellipse


def test_corner_plot_kde_setting_overrides_use_quad():
    fig = corner_plot([_fit(names=("OpX", "OpY"), use_quad=True)], kde=False)

    assert len(fig.axes[0].patches) == 2


def test_corner_plot_double_solution_is_keyed_by_fit_name():
    """Only the named fit is split into two solutions."""
    best_fit = {"OpX": 0.5, "OpY": -0.5}
    fits = [
        _fit(
            fit_name="fit_a",
            names=("OpX", "OpY"),
            use_quad=True,
            best_fit_point=best_fit,
        ),
        _fit(
            fit_name="fit_b",
            names=("OpX", "OpY"),
            seed=1,
            use_quad=True,
            best_fit_point=best_fit,
        ),
    ]

    fig = corner_plot(
        fits, double_solution={"fit_a": ["OpX"]}, show_best_fit=True, show_sm=False
    )

    # fit_a is marked once per solution, fit_b once at its best-fit point
    assert len(_markers(fig.axes[0])) == 3


# ---------------------------------------------------------------------------
# same-run corner plots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action, sampler",
    [
        (plot_corner_ultranest, "UltraNest"),
        (plot_corner_blackjax, "BlackJAX"),
        (plot_corner_hessian, "Hessian"),
        (plot_corner_analytic, "Analytic"),
    ],
)
def test_plot_corner_from_fit_labels_the_sampler(action, sampler):
    """A fit run in the same session is named after its sampler."""
    fig = action(_fit(fit_name=sampler, names=("OpX", "OpY")))

    assert [t.get_text() for t in _legend(fig).get_texts()] == [
        rf"$\mathrm{{{sampler}}}$",
        r"$\mathrm{SM}$",
    ]


def test_plot_corner_from_fit_forwards_the_settings():
    fig = plot_corner_analytic(
        _fit(fit_name="Analytic", names=("OpX", "OpY")), {"show_sm": False}
    )

    assert [t.get_text() for t in _legend(fig).get_texts()] == [r"$\mathrm{Analytic}$"]


def test_plot_corner_from_fits_forwards_the_settings():
    fig = plot_corner([_fit()], {"dofs_show": ["OpX", "OpY"]})

    assert len(fig.axes) == 1
    assert fig.axes[0].get_xlabel() == "OpX"


# ---------------------------------------------------------------------------
# show_sm / show_best_fit
# ---------------------------------------------------------------------------


def _markers(ax):
    """Scatter markers on a panel, in draw order.

    Filtered by type: contour sets are Collections too and also answer
    get_offsets(), so they would otherwise be counted as markers.
    """
    from matplotlib.collections import PathCollection

    return [c for c in ax.collections if isinstance(c, PathCollection)]


def test_corner_plot_draws_the_sm_point_by_default():
    fig = corner_plot([_fit(names=("OpX", "OpY"))])

    offsets = [tuple(m.get_offsets()[0]) for m in _markers(fig.axes[0])]
    assert (0.0, 0.0) in offsets


def test_corner_plot_show_sm_false_removes_the_marker():
    fig = corner_plot([_fit(names=("OpX", "OpY"))], show_sm=False)

    offsets = [tuple(m.get_offsets()[0]) for m in _markers(fig.axes[0])]
    assert (0.0, 0.0) not in offsets


def test_corner_plot_show_sm_false_removes_the_legend_entry():
    fig = corner_plot([_fit(fit_name="fit_a", names=("OpX", "OpY"))], show_sm=False)

    assert [t.get_text() for t in _legend(fig).get_texts()] == [r"$\mathrm{fit\_a}$"]


def test_corner_plot_show_sm_false_frees_the_axis_limits():
    """With no SM marker the frame no longer has to contain the origin."""
    with_sm = corner_plot([_fit(names=("OpX", "OpY"), shift=100.0)])
    without_sm = corner_plot([_fit(names=("OpX", "OpY"), shift=100.0)], show_sm=False)

    assert with_sm.axes[0].get_xlim()[0] <= 0
    assert without_sm.axes[0].get_xlim()[0] > 0


def test_corner_plot_best_fit_marker_is_off_by_default():
    fit = _fit(names=("OpX", "OpY"), best_fit_point={"OpX": 0.5, "OpY": -0.5})

    fig = corner_plot([fit])

    offsets = [tuple(m.get_offsets()[0]) for m in _markers(fig.axes[0])]
    assert offsets == [(0.0, 0.0)]  # SM only


def test_corner_plot_show_best_fit_uses_the_stored_best_fit_point():
    fit = _fit(names=("OpX", "OpY"), best_fit_point={"OpX": 0.5, "OpY": -0.5})

    fig = corner_plot([fit], show_best_fit=True, show_sm=False)

    offsets = [tuple(m.get_offsets()[0]) for m in _markers(fig.axes[0])]
    assert offsets == [(0.5, -0.5)]


def test_corner_plot_show_best_fit_falls_back_to_the_posterior_mean():
    fit = _fit(names=("OpX", "OpY"))  # no best-fit point recorded

    fig = corner_plot([fit], show_best_fit=True, show_sm=False)

    marker = _markers(fig.axes[0])[0].get_offsets()[0]
    assert marker[0] == pytest.approx(fit.samples_frame["OpX"].mean())
    assert marker[1] == pytest.approx(fit.samples_frame["OpY"].mean())


def test_corner_plot_show_best_fit_marks_every_fit():
    fits = [
        _fit(
            fit_name="fit_a",
            names=("OpX", "OpY"),
            best_fit_point={"OpX": 0.5, "OpY": -0.5},
        ),
        _fit(
            fit_name="fit_b",
            names=("OpX", "OpY"),
            seed=1,
            best_fit_point={"OpX": -1.0, "OpY": 1.0},
        ),
    ]

    fig = corner_plot(fits, show_best_fit=True, show_sm=False)

    offsets = [tuple(m.get_offsets()[0]) for m in _markers(fig.axes[0])]
    assert offsets == [(0.5, -0.5), (-1.0, 1.0)]


def test_corner_plot_double_solution_ignores_the_stored_best_fit():
    """Two disjoint modes cannot be summarised by one stored value."""
    fit = _fit(
        names=("OpX", "OpY"),
        use_quad=True,
        best_fit_point={"OpX": 0.5, "OpY": -0.5},
    )

    fig = corner_plot([fit], double_solution=["OpX"], show_best_fit=True, show_sm=False)

    assert len(_markers(fig.axes[0])) == 2  # one marker per solution
