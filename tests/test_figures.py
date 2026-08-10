"""Unit tests for smefit.figures — _plot_heatmap and the heatmaps built on it.

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
    plot_fisher_diagonals_heatmap,
    plot_posterior_correlations,
)
from smefit.fit_result import Fit, FitResult, FitResultGroup


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


def test_plot_heatmap_blanks_cells_that_are_not_numbers():
    """A nan cell is left blank rather than annotated 'nan'."""
    matrix = np.array([[np.nan, 5.0]])
    fig = _plot_heatmap(matrix, ["OpA"], ["DS_A", "DS_B"], vmin=0, vmax=10)

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.texts] == ["5.0"]


def test_plot_heatmap_can_keep_zero_cells():
    """With mask_zeros off, a zero is a value like any other."""
    matrix = np.array([[0.0, 5.0]])
    fig = _plot_heatmap(
        matrix, ["OpA"], ["DS_A", "DS_B"], vmin=0, vmax=10, mask_zeros=False
    )

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.texts] == ["0.0", "5.0"]


def test_plot_heatmap_text_color_threshold_uses_the_magnitude():
    """A strongly negative cell is as saturated as a strongly positive one, so
    its text goes white too."""
    matrix = np.array([[-0.9, 0.9, -0.1]])
    fig = _plot_heatmap(
        matrix,
        ["OpA"],
        ["DS_A", "DS_B", "DS_C"],
        vmin=-1,
        vmax=1,
        mask_zeros=False,
        value_fmt="{:.2f}",
        text_threshold=0.6,
    )

    ax = fig.axes[0]
    colors = {t.get_text(): t.get_color() for t in ax.texts}
    assert colors["-0.90"] == "white"
    assert colors["0.90"] == "white"
    assert colors["-0.10"] == "black"


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


def test_plot_heatmap_title_goes_above_the_column_labels():
    """The column labels sit on top, so the heading is the x label — which
    matplotlib keeps clear of them — rather than a hand-padded title."""
    fig = _plot_heatmap(
        np.array([[1.0]]), ["OpA"], ["DS_A"], vmin=0, vmax=10, title="My fit"
    )

    ax = fig.axes[0]
    assert ax.get_xlabel() == "My fit"
    assert ax.xaxis.get_label_position() == "top"


def _fit(
    free=("OpA", "OpZZ"),
    samples=None,
    label=None,
    fit_name="my_fit",
    action="run_analytic_fit",
):
    """A joint fit holding the given posterior samples.

    The default pair is perfectly anti-correlated, which keeps the expected
    matrix obvious in tests that are about the drawing rather than the numbers.
    """
    if samples is None:
        samples = {"OpA": [0.0, 1.0, 2.0, 3.0], "OpZZ": [0.0, -1.0, -2.0, -3.0]}
    return Fit(
        fit_results=FitResult(
            free_parameters=list(free),
            best_fit_point={name: 0.0 for name in free},
            max_loglikelihood=-1.0,
            num_data=10,
            samples={name: jnp.array(vals) for name, vals in samples.items()},
        ),
        fit_name=fit_name,
        label=label,
        fit_runcard={"actions_": [action]},
    )


def test_plot_posterior_correlations_labels_both_axes_with_the_coefficients():
    """Both axes are the free coefficients, LaTeX-labelled where the operator
    is known — unlike the Fisher heatmap, whose columns are data sources."""
    fig = plot_posterior_correlations(
        _fit(
            free=["OQQ1", "NotARealOp"],
            samples={"OQQ1": [0.0, 1.0, 2.0], "NotARealOp": [2.0, 0.0, 1.0]},
        )
    )

    ax = fig.axes[0]
    labels = [r"$c_{QQ}^{\scriptscriptstyle 1}$", "NotARealOp"]
    assert [t.get_text() for t in ax.get_xticklabels()] == labels
    assert [t.get_text() for t in ax.get_yticklabels()] == labels


def test_plot_posterior_correlations_covers_the_free_coefficients_only():
    """Derived coefficients are in `samples` too: they are functions of the
    free ones, so they say nothing new about how the posterior is correlated."""
    fig = plot_posterior_correlations(
        _fit(
            free=["OpA", "OpZZ"],
            samples={
                "OpA": [0.0, 1.0, 2.0, 3.0],
                "OpZZ": [0.0, -1.0, -2.0, -3.0],
                "OpDerived": [0.0, 1.0, 4.0, 9.0],
            },
        )
    )

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["OpA", "OpZZ"]
    assert sorted(t.get_text() for t in ax.texts) == [
        "-1.00",
        "-1.00",
        "1.00",
        "1.00",
    ]


def test_plot_posterior_correlations_is_headed_by_the_fit_it_is_drawn_from():
    """Several fits produce several heatmaps; each has to say which it is."""
    labelled = plot_posterior_correlations(_fit(label=r"$\mathrm{Analytic}$"))
    unlabelled = plot_posterior_correlations(_fit(fit_name="my_fit"))

    assert labelled.axes[0].get_xlabel() == r"$\mathrm{Analytic}$"
    assert unlabelled.axes[0].get_xlabel() == "my_fit"


def test_plot_posterior_correlations_presentation_is_overridable():
    """These are the parameters a runcard drives through reportengine's own
    resolution — a top-level `cmap:` key, or
    ``{@plot_posterior_correlations(cmap="PuOr")@}`` — so each has to be a
    parameter, and passing one has to actually change the figure."""
    fig = plot_posterior_correlations(
        _fit(label=r"$\mathrm{Analytic}$"),
        cmap="PuOr",
        value_fmt="{:.3f}",
        colorbar=False,
    )

    ax = fig.axes[0]
    # The heading stays the fit's, whatever else is overridden: it is the only
    # thing saying which fit the heatmap is drawn from.
    assert ax.get_xlabel() == r"$\mathrm{Analytic}$"
    assert ax.images[0].cmap.name == "PuOr"
    assert sorted(t.get_text() for t in ax.texts) == [
        "-1.000",
        "-1.000",
        "1.000",
        "1.000",
    ]
    # The colorbar is an Axes of its own, so dropping it leaves just the heatmap
    assert len(fig.axes) == 1


def test_plot_posterior_correlations_annotates_uncorrelated_pairs_too():
    """An uncorrelated pair is a result, not a gap: zero cells are drawn and
    annotated rather than blanked as they are in the Fisher heatmap."""
    fig = plot_posterior_correlations(
        _fit(samples={"OpA": [1.0, -1.0, 1.0, -1.0], "OpZZ": [1.0, 1.0, -1.0, -1.0]})
    )

    ax = fig.axes[0]
    assert sorted(t.get_text() for t in ax.texts) == ["0.00", "0.00", "1.00", "1.00"]


def test_plot_posterior_correlations_blanks_a_frozen_coefficient():
    """A coefficient whose samples never moved correlates with nothing."""
    fig = plot_posterior_correlations(
        _fit(samples={"OpA": [0.0, 1.0, 2.0], "OpZZ": [1.0, 1.0, 1.0]})
    )

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.texts] == ["1.00"]


def test_plot_posterior_correlations_is_drawn_on_a_fixed_scale():
    """The colour scale spans the full range of a correlation whatever the
    matrix happens to contain, so two fits' heatmaps can be compared by eye."""
    fig = plot_posterior_correlations(
        _fit(samples={"OpA": [0.0, 1.0, 2.0, 5.0], "OpZZ": [2.0, 0.0, 1.0, 1.5]})
    )

    image = fig.axes[0].images[0]
    assert image.get_clim() == (-1, 1)


def test_plot_posterior_correlations_rejects_an_individual_fit():
    """One-at-a-time fits never sampled their coefficients together."""
    fit = Fit(
        fit_results=FitResultGroup(
            [
                FitResult(
                    free_parameters=["OpA"],
                    best_fit_point={"OpA": 0.0},
                    max_loglikelihood=-1.0,
                    num_data=10,
                    samples={"OpA": jnp.array([0.0, 1.0])},
                )
            ]
        ),
        fit_name="my_individual_fit",
        fit_runcard={"actions_": ["run_individual_analytic_fits"]},
    )

    with pytest.raises(ValueError, match="my_individual_fit"):
        plot_posterior_correlations(fit)
