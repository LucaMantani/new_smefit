"""Unit tests for smefit.figures — the report figures and the heatmap they share.

The figures enable matplotlib's usetex, which needs a LaTeX installation to
render any text. CI has none, and installing one is expensive, so the suite must
never invoke it — see the `_no_latex` fixture below.

Most tests here also avoid drawing at all (get_text() and friends only read
stored attributes), but that is not something to rely on — anything that
measures text, `tight_layout()` above all, renders it.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.stats
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.contour import ContourSet
from matplotlib.patches import Ellipse, Rectangle

from smefit import credible_intervals
from smefit import figures as figures_mod
from smefit.figures import (
    _density_level,
    _gaussian_ellipse,
    _kde_grid,
    _plot_heatmap,
    plot_chi2_scan,
    plot_coefficient_bounds,
    plot_coefficient_bounds_comparison,
    plot_fisher_diagonals_heatmap,
    plot_global_information_heatmap,
    plot_mass_reach,
    plot_mass_reach_comparison,
    plot_pca_components_heatmap,
    plot_pca_spectrum,
    plot_posterior_contours,
    plot_posterior_contours_comparison,
    plot_posterior_correlations,
    plot_posterior_histograms,
    plot_posterior_histograms_comparison,
)
from smefit.fisher import _fisher_in_pca_basis
from smefit.fit_result import Fit, FitResult, FitResultGroup
from smefit.latex_labels import make_latex_label
from smefit.pca import PCA
from smefit.tables import fisher_diagonals_normalised


def _plain(name):
    """A ``latex_label`` for tests that are not about labels: names stay readable."""
    return name


@pytest.fixture(autouse=True)
def _no_latex(monkeypatch):
    """Keep matplotlib off LaTeX for the duration of a test.

    Stubbing out `set_plot_style` is what stops the figures switching usetex on.
    The rcParam is then forced off as well, so the invariant the suite needs —
    "no LaTeX is invoked" — is asserted directly rather than inferred from
    nobody having turned it on; that inference is exactly what broke when
    `smefit.latex_labels` used to do it at import time.
    """
    monkeypatch.setattr(figures_mod, "set_plot_style", lambda: None)
    monkeypatch.setitem(matplotlib.rcParams, "text.usetex", False)
    yield
    plt.close("all")


def test_plot_heatmap_axis_labels():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    fig = _plot_heatmap(matrix, ["OpA", "OpZZ"], ["DS_A", "DS_B"], vmin=0, vmax=10)

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["DS_A", "DS_B"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["OpA", "OpZZ"]


def test_plot_heatmap_draws_labels_as_given():
    """The callers have already turned names into labels, so the heatmap does
    not look them up again — that would bypass the runcard's latex_labels."""
    matrix = np.array([[1.0]])
    fig = _plot_heatmap(matrix, ["OQQ1"], ["DS_A"], vmin=0, vmax=10)

    ax = fig.axes[0]
    assert ax.get_yticklabels()[0].get_text() == "OQQ1"


def test_plot_heatmap_skips_zero_cells():
    """Cells equal to 0.0 get no text annotation."""
    matrix = np.array([[0.0, 5.0]])
    fig = _plot_heatmap(matrix, ["OpA"], ["DS_A", "DS_B"], vmin=0, vmax=10)

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.texts] == ["5.0"]


def test_plot_heatmap_text_contrasts_with_the_cell():
    """Cell text is white on a dark cell and black on a light one."""
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


def test_plot_heatmap_text_contrast_follows_a_diverging_map_both_ways():
    """On a diverging map both ends are dark, so a strongly negative cell gets
    white text just like a strongly positive one, and the pale middle gets
    black. A rule based on the value rather than the colour would only get one
    end right."""
    matrix = np.array([[-0.9, 0.9, -0.1]])
    fig = _plot_heatmap(
        matrix,
        ["OpA"],
        ["DS_A", "DS_B", "DS_C"],
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        mask_zeros=False,
        value_fmt="{:.2f}",
    )

    ax = fig.axes[0]
    colors = {t.get_text(): t.get_color() for t in ax.texts}
    assert colors["-0.90"] == "white"
    assert colors["0.90"] == "white"
    assert colors["-0.10"] == "black"


def test_plot_heatmap_text_contrast_follows_a_map_that_lightens_with_value():
    """viridis runs dark-to-light, the opposite way to Blues: the *low* cells
    are the dark ones, and that is where the white text has to go. This is the
    case a value threshold got backwards, and it is reachable now that the
    colormap is a runcard setting."""
    matrix = np.array([[1.0, 99.0]])
    fig = _plot_heatmap(
        matrix, ["OpA"], ["DS_A", "DS_B"], vmin=0, vmax=100, cmap="viridis"
    )

    ax = fig.axes[0]
    colors = {t.get_text(): t.get_color() for t in ax.texts}
    assert colors["1.0"] == "white"
    assert colors["99.0"] == "black"


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


def test_plot_global_information_heatmap_scales_values_to_percent():
    table = pd.DataFrame(
        {"DS_A": [0.3, 0.7], "DS_B": [0.7, 0.3]},
        index=["OpA", "OpZZ"],
    )

    fig = plot_global_information_heatmap(table)

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["DS_A", "DS_B"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["OpA", "OpZZ"]
    assert {t.get_text() for t in ax.texts} == {"30.0", "70.0"}
    assert ax.images[0].get_clim() == (0.0, 100.0)


def test_plot_global_information_heatmap_widens_range_for_negative_shares():
    """Shares outside 0-100% widen the colour range instead of being clipped."""
    table = pd.DataFrame({"DS_A": [0.5, 1.1], "DS_B": [0.5, -0.1]}, index=["a", "b"])

    fig = plot_global_information_heatmap(table)

    ax = fig.axes[0]
    np.testing.assert_allclose(ax.images[0].get_clim(), (-10.0, 110.0))
    assert {"-10.0", "110.0"} <= {t.get_text() for t in ax.texts}


def test_plot_chi2_scan_yields_one_figure_per_coefficient():
    scans = [
        {"OpA": {"points": [-1.0, 0.0, 1.0], "chi2": [4.0, 0.0, 4.0]}},
        {"OpZZ": {"points": [-2.0, 2.0], "chi2": [1.0, 1.0]}},
    ]

    result = plot_chi2_scan(scans, latex_label=_plain)

    assert [name for _, name in result] == ["OpA", "OpZZ"]
    for fig, _ in result:
        ax = fig.axes[0]
        assert ax.get_ylabel() == r"$\chi^2$"


def test_plot_chi2_scan_plots_points_against_chi2():
    scans = [{"OpA": {"points": [-1.0, 0.0, 1.0], "chi2": [4.0, 0.0, 4.0]}}]

    result = plot_chi2_scan(scans, latex_label=_plain)

    fig, name = result[0]
    assert name == "OpA"
    line = fig.axes[0].lines[0]
    assert line.get_xdata().tolist() == [-1.0, 0.0, 1.0]
    assert line.get_ydata().tolist() == [4.0, 0.0, 4.0]


def test_plot_chi2_scan_uses_latex_label_when_known():
    scans = [{"OQQ1": {"points": [0.0, 1.0], "chi2": [0.0, 1.0]}}]

    result = plot_chi2_scan(scans, latex_label=make_latex_label())

    fig, _ = result[0]
    assert fig.axes[0].get_xlabel() == r"$c_{QQ}^{\scriptscriptstyle 1}$"


def test_plot_chi2_scan_applies_latex_labels():
    scans = [{"OQQ1": {"points": [0.0, 1.0], "chi2": [0.0, 1.0]}}]

    result = plot_chi2_scan(scans, latex_label=make_latex_label({"OQQ1": "$c_1$"}))

    fig, name = result[0]
    assert fig.axes[0].get_xlabel() == "$c_1$"
    assert name == "OQQ1"


def test_plot_fisher_diagonals_heatmap_presentation_is_overridable():
    """Same runcard-driven keywords as the correlation heatmap, so a report
    can style both in the same idiom."""
    fd = pd.DataFrame({"DS_A": [0.3], "DS_B": [0.7]}, index=["OpA"])

    fig = plot_fisher_diagonals_heatmap(
        fd, cmap="viridis", value_fmt="{:.2f}", colorbar=True
    )

    ax = fig.axes[0]
    assert ax.images[0].cmap.name == "viridis"
    assert sorted(t.get_text() for t in ax.texts) == ["30.00", "70.00"]
    # The colorbar is an Axes of its own, so asking for it adds one
    assert len(fig.axes) == 2


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
    use_quad=False,
    contours="kde",
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
        contours=contours,
        fit_runcard={"actions_": [action], "use_quad": use_quad},
    )


def test_plot_posterior_correlations_labels_both_axes_with_the_coefficients():
    """Both axes are the free coefficients, LaTeX-labelled where the operator
    is known — unlike the Fisher heatmap, whose columns are data sources."""
    fig = plot_posterior_correlations(
        _fit(
            free=["OQQ1", "NotARealOp"],
            samples={"OQQ1": [0.0, 1.0, 2.0], "NotARealOp": [2.0, 0.0, 1.0]},
        ),
        latex_label=make_latex_label(),
    )

    ax = fig.axes[0]
    labels = [r"$c_{QQ}^{\scriptscriptstyle 1}$", r"$\mathrm{NotARealOp}$"]
    assert [t.get_text() for t in ax.get_xticklabels()] == labels
    assert [t.get_text() for t in ax.get_yticklabels()] == labels


def test_plot_posterior_correlations_applies_latex_labels():
    fig = plot_posterior_correlations(
        _fit(), latex_label=make_latex_label({"OpA": "$c_A$"})
    )

    ax = fig.axes[0]
    labels = ["$c_A$", make_latex_label()("OpZZ")]
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
        ),
        latex_label=_plain,
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
    labelled = plot_posterior_correlations(
        _fit(label=r"$\mathrm{Analytic}$"), latex_label=_plain
    )
    unlabelled = plot_posterior_correlations(
        _fit(fit_name="my_fit"), latex_label=_plain
    )

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
        latex_label=_plain,
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
        _fit(samples={"OpA": [1.0, -1.0, 1.0, -1.0], "OpZZ": [1.0, 1.0, -1.0, -1.0]}),
        latex_label=_plain,
    )

    ax = fig.axes[0]
    assert sorted(t.get_text() for t in ax.texts) == ["0.00", "0.00", "1.00", "1.00"]


def test_plot_posterior_correlations_blanks_a_frozen_coefficient():
    """A coefficient whose samples never moved correlates with nothing."""
    fig = plot_posterior_correlations(
        _fit(samples={"OpA": [0.0, 1.0, 2.0], "OpZZ": [1.0, 1.0, 1.0]}),
        latex_label=_plain,
    )

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.texts] == ["1.00"]


def test_plot_posterior_correlations_is_drawn_on_a_fixed_scale():
    """The colour scale spans the full range of a correlation whatever the
    matrix happens to contain, so two fits' heatmaps can be compared by eye."""
    fig = plot_posterior_correlations(
        _fit(samples={"OpA": [0.0, 1.0, 2.0, 5.0], "OpZZ": [2.0, 0.0, 1.0, 1.5]}),
        latex_label=_plain,
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
        plot_posterior_correlations(fit, latex_label=_plain)


def test_plot_posterior_correlations_restricts_to_params_to_plot():
    """A global fit has too many coefficients to read at once; the runcard's
    params_to_plot picks the block worth looking at, and its order is the
    order they are drawn in."""
    fig = plot_posterior_correlations(
        _fit(
            free=["OpA", "OpB", "OpZZ"],
            samples={
                "OpA": [0.0, 1.0, 2.0, 3.0],
                "OpB": [0.0, 2.0, 1.0, 4.0],
                "OpZZ": [0.0, -1.0, -2.0, -3.0],
            },
        ),
        params_to_plot=["OpZZ", "OpA"],
        latex_label=_plain,
    )

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["OpZZ", "OpA"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["OpZZ", "OpA"]
    # The pair is perfectly anti-correlated, and OpB is gone rather than blank.
    assert sorted(t.get_text() for t in ax.texts) == [
        "-1.00",
        "-1.00",
        "1.00",
        "1.00",
    ]


def test_plot_posterior_correlations_skips_a_coefficient_this_fit_lacks():
    """params_to_plot is one runcard-wide list over several fits, so a name a
    given fit never fitted drops out of that fit's heatmap rather than
    stopping the report."""
    fig = plot_posterior_correlations(
        _fit(free=["OpA", "OpZZ"]),
        params_to_plot=["OpA", "OpNotInThisFit"],
        latex_label=_plain,
    )

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["OpA"]


# ---------------------------------------------------------------------------
# plot_pca_components_heatmap / plot_pca_spectrum
# ---------------------------------------------------------------------------


def _pca(**kwargs):
    return PCA(
        eigenvalues=np.array([4.0, 1.0, 0.0]),
        eigenvectors=np.array([[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]]),
        coeff_names=["OpA", "OpB", "OpC"],
        **{"threshold": 1.0e-3, "min_weight": 0.01, **kwargs},
    )


def test_plot_pca_components_heatmap_axis_labels():
    fig = plot_pca_components_heatmap(_pca().as_frame())

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["PC1", "PC2", "PC3"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["OpA", "OpB", "OpC"]


def test_plot_pca_components_heatmap_keeps_zero_cells():
    """A zero weight is a value, not an absent source, so it is drawn."""
    fig = plot_pca_components_heatmap(_pca().as_frame())

    ax = fig.axes[0]
    assert len(ax.texts) == 9
    assert "0.00" in [t.get_text() for t in ax.texts]


def test_plot_pca_components_heatmap_uses_a_symmetric_scale():
    fig = plot_pca_components_heatmap(_pca().as_frame())

    im = fig.axes[0].images[0]
    assert (im.norm.vmin, im.norm.vmax) == (-1, 1)


def test_plot_pca_spectrum_axis_and_threshold():
    fig = plot_pca_spectrum(_pca(), latex_label=_plain)

    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["PC1", "PC2", "PC3"]


def test_plot_pca_spectrum_labels_the_components():
    fig = plot_pca_spectrum(_pca(), latex_label=make_latex_label({"PC2": "$v_2$"}))

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        r"$\mathrm{PC1}$",
        "$v_2$",
        r"$\mathrm{PC3}$",
    ]
    # the dashed line sits at the flat-direction threshold
    thresholds = [
        line.get_ydata()[0] for line in ax.lines if line.get_linestyle() == "--"
    ]
    assert thresholds == [_pca().threshold]


def _bar_sizes(ax):
    return [len(container) for container in ax.containers]


def test_plot_pca_spectrum_separates_flat_from_constrained():
    """The flat direction is drawn as its own, labelled, set of bars."""
    fig = plot_pca_spectrum(_pca(), latex_label=_plain)

    ax = fig.axes[0]
    assert _bar_sizes(ax) == [2, 1]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "constrained",
        "flat",
    ]


def test_plot_pca_spectrum_marks_the_exactly_flat_direction():
    """A zero eigenvalue leaves a bar of no length, so it gets a marker too."""
    fig = plot_pca_spectrum(_pca(), latex_label=_plain)

    ax = fig.axes[0]
    markers = [coll for coll in ax.collections if isinstance(coll, PathCollection)]
    assert [len(coll.get_offsets()) for coll in markers] == [1]


def test_plot_pca_spectrum_without_flat_directions_draws_one_set():
    pca_obj = PCA(
        eigenvalues=np.array([4.0, 1.0]),
        eigenvectors=np.eye(2),
        coeff_names=["OpA", "OpB"],
        threshold=1.0e-3,
        min_weight=0.01,
    )
    fig = plot_pca_spectrum(pca_obj, latex_label=_plain)

    ax = fig.axes[0]
    assert _bar_sizes(ax) == [2]
    assert not [coll for coll in ax.collections if isinstance(coll, PathCollection)]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["constrained"]


# ---------------------------------------------------------------------------
# plot_posterior_histograms / plot_posterior_histograms_comparison
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)
_GAUSS = _RNG.normal(size=500)


def _visible(fig):
    return [ax for ax in fig.axes if ax.get_visible()]


def test_plot_posterior_histograms_one_panel_per_free_coefficient():
    fig = plot_posterior_histograms(
        _fit(samples={"OpA": _GAUSS, "OpZZ": _GAUSS}), latex_label=_plain
    )

    assert [ax.get_xlabel() for ax in _visible(fig)] == ["OpA", "OpZZ"]


def test_plot_posterior_histograms_includes_constrained_coefficients():
    """An expr coefficient's posterior is induced by the free ones, and is
    worth seeing on its own; a fixed one is a constant, and is left out."""
    fit = _fit(
        free=["OpA"],
        samples={"OpA": _GAUSS, "OpSq": _GAUSS**2, "OpFixed": np.ones_like(_GAUSS)},
    )
    fit.fit_runcard["coefficients"] = {
        "OpA": {"free": True},
        "OpSq": {"free": False, "vars": ["OpA"], "expr": "OpA**2"},
        "OpFixed": {"free": False, "value": 1.0},
    }
    fig = plot_posterior_histograms(fit, latex_label=_plain)

    assert [ax.get_xlabel() for ax in _visible(fig)] == ["OpA", "OpSq"]


def test_plot_posterior_histograms_comparison_takes_the_union_of_the_fits():
    """A coefficient one fit lacks still gets a panel, drawn by the other."""
    first = _fit(free=["OpA"], samples={"OpA": _GAUSS}, fit_name="a")
    second = _fit(
        free=["OpA", "OpB"], samples={"OpA": _GAUSS, "OpB": _GAUSS}, fit_name="b"
    )
    fig = plot_posterior_histograms_comparison([first, second], latex_label=_plain)

    panel_a, panel_b = _visible(fig)
    assert [panel_a.get_xlabel(), panel_b.get_xlabel()] == ["OpA", "OpB"]
    # filled bars plus a step outline per fit
    assert len(panel_a.patches) > len(panel_b.patches)


def test_plot_posterior_histograms_comparison_can_share_bin_edges_between_fits():
    first = _fit(free=["OpA"], samples={"OpA": _GAUSS}, fit_name="a")
    second = _fit(free=["OpA"], samples={"OpA": _GAUSS + 3.0}, fit_name="b")
    fig = plot_posterior_histograms_comparison(
        [first, second], latex_label=_plain, n_bins=10, shared_bins=True
    )

    (ax,) = _visible(fig)
    bars = [p for p in ax.patches if isinstance(p, Rectangle)]
    # ten bars per fit, on the same ten edges
    assert len(bars) == 20
    assert len({round(p.get_x(), 12) for p in bars}) == 10


def test_plot_posterior_histograms_comparison_bins_each_fit_over_its_own_range_by_default():
    """A posterior ten times narrower still gets all its bins."""
    wide = _fit(free=["OpA"], samples={"OpA": _GAUSS}, fit_name="wide")
    narrow = _fit(free=["OpA"], samples={"OpA": 0.1 * _GAUSS}, fit_name="narrow")
    fig = plot_posterior_histograms_comparison(
        [wide, narrow], latex_label=_plain, n_bins=10
    )

    (ax,) = _visible(fig)
    bars = [p for p in ax.patches if isinstance(p, Rectangle)]
    wide_bars, narrow_bars = bars[:10], bars[10:]
    assert len(narrow_bars) == 10
    assert narrow_bars[0].get_width() == pytest.approx(
        wide_bars[0].get_width() * np.ptp(0.1 * _GAUSS) / np.ptp(_GAUSS)
    )


def test_plot_posterior_histograms_comparison_legend_names_each_fit_by_its_label():
    fits = [
        _fit(samples={"OpA": _GAUSS, "OpZZ": _GAUSS}, fit_name="a", label="$A$"),
        _fit(samples={"OpA": _GAUSS, "OpZZ": _GAUSS}, fit_name="b"),
    ]
    fig = plot_posterior_histograms_comparison(fits, latex_label=_plain)

    (legend,) = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == ["$A$", "b"]


def test_plot_posterior_histograms_comparison_rejects_no_fits():
    with pytest.raises(ValueError, match="fits is empty"):
        plot_posterior_histograms_comparison([], latex_label=_plain)


def test_plot_posterior_histograms_rejects_a_fit_without_samples():
    fit = _fit()
    fit.fit_results.samples = None
    with pytest.raises(ValueError, match="my_fit"):
        plot_posterior_histograms(fit, latex_label=_plain)


def test_plot_posterior_histograms_restricts_to_params_to_plot():
    fig = plot_posterior_histograms(
        _fit(
            free=["OpA", "OpB", "OpZZ"],
            samples=dict.fromkeys(["OpA", "OpB", "OpZZ"], _GAUSS),
        ),
        latex_label=_plain,
        params_to_plot=["OpZZ", "OpA"],
    )

    assert [ax.get_xlabel() for ax in _visible(fig)] == ["OpZZ", "OpA"]


def test_plot_posterior_histograms_lays_panels_out_n_cols_wide():
    names = [f"Op{i}" for i in range(5)]
    fig = plot_posterior_histograms(
        _fit(free=names, samples=dict.fromkeys(names, _GAUSS)),
        latex_label=_plain,
        n_cols=2,
    )

    assert len(fig.axes) == 6
    assert len(_visible(fig)) == 5


def test_plot_posterior_histograms_marks_a_point_mass_with_a_line():
    fig = plot_posterior_histograms(
        _fit(free=["OpA"], samples={"OpA": np.zeros(10)}), latex_label=_plain
    )

    (line,) = _visible(fig)[0].get_lines()
    assert list(line.get_xdata()) == [0.0, 0.0]


def test_plot_posterior_histograms_reads_an_individual_fit():
    """Unlike correlations, marginals exist for one-at-a-time fits."""
    fit = Fit(
        fit_results=FitResultGroup(
            [
                FitResult(
                    free_parameters=[name],
                    best_fit_point={name: 0.0},
                    max_loglikelihood=-1.0,
                    num_data=10,
                    samples={name: jnp.array(_GAUSS)},
                )
                for name in ["OpA", "OpB"]
            ]
        ),
        fit_name="my_individual_fit",
        fit_runcard={"actions_": ["run_individual_analytic_fits"]},
    )
    fig = plot_posterior_histograms(fit, latex_label=_plain)

    assert [ax.get_xlabel() for ax in _visible(fig)] == ["OpA", "OpB"]


def test_plot_posterior_histograms_comparison_overlays_an_individual_fit_on_a_joint_one():
    """Only the samples matter: a joint fit and a one-at-a-time fit share
    panels, and each draws the coefficients it has."""
    joint = _fit(
        free=["OpA", "OpB"], samples={"OpA": _GAUSS, "OpB": _GAUSS}, fit_name="joint"
    )
    individual = Fit(
        fit_results=FitResultGroup(
            [
                FitResult(
                    free_parameters=["OpA"],
                    best_fit_point={"OpA": 0.0},
                    max_loglikelihood=-1.0,
                    num_data=10,
                    samples={"OpA": jnp.array(0.5 * _GAUSS)},
                )
            ]
        ),
        fit_name="individual",
        fit_runcard={"actions_": ["run_individual_analytic_fits"]},
    )
    fig = plot_posterior_histograms_comparison([joint, individual], latex_label=_plain)

    panel_a, panel_b = _visible(fig)
    assert [panel_a.get_xlabel(), panel_b.get_xlabel()] == ["OpA", "OpB"]
    # two fits on OpA, one on OpB: filled bars plus a step outline per fit
    assert len(panel_a.patches) == 2 * len(panel_b.patches)
    (legend,) = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == ["joint", "individual"]


def test_plot_posterior_histograms_graph_resolves(tmp_path):
    """The template runcard resolves through reportengine, which checks the
    annotations that calling the provider directly skips."""
    import yaml
    from reportengine.resourcebuilder import FuzzyTarget, ResourceBuilder

    from smefit.api import smefitAPI

    template = Path(__file__).parents[1] / "template_runcards/posterior_histograms.yaml"
    runcard = yaml.safe_load(template.read_text())
    template_text = runcard.pop("template_text")
    assert "{@plot_posterior_histograms_comparison@}" in template_text
    assert "{@with fits@}" in template_text
    assert "{@plot_posterior_histograms@}" in template_text
    for entry in runcard["fits"]:
        folder = tmp_path / entry["name"]
        _fit(samples={"OpA": _GAUSS, "OpZZ": _GAUSS}).fit_results.write(folder)
        (folder / "input").mkdir()
        (folder / "input" / "runcard.yaml").write_text(
            yaml.safe_dump({"actions_": ["run_analytic_fit"]})
        )
        entry["path"] = str(tmp_path)

    config = smefitAPI.config_class(runcard, environment=smefitAPI.loadedenv)
    builder = ResourceBuilder(
        config,
        smefitAPI.provider_loaded,
        [
            FuzzyTarget("plot_posterior_histograms_comparison", (), (), ()),
            FuzzyTarget("plot_posterior_histograms", ("fits",), (), ()),
        ],
        perform_final=False,
    )
    builder.rootns.update(smefitAPI.loadedenv.ns_dump())
    builder.resolve_fuzzytargets()


# ---------------------------------------------------------------------------
# plot_coefficient_bounds / plot_coefficient_bounds_comparison
# ---------------------------------------------------------------------------


def _bounds_fit(
    samples: dict[str, list[float]] | None = None,
    fit_name: str = "my_fit",
    label: str | None = None,
) -> Fit:
    """A joint fit whose posterior is uniform on [0, 1] for every coefficient,
    so its equal-tailed intervals are known by heart."""
    if samples is None:
        grid = list(np.linspace(0.0, 1.0, 1001))
        samples = {"OpA": grid, "OpZZ": grid}
    return _fit(free=list(samples), samples=samples, fit_name=fit_name, label=label)


def _bars(ax: matplotlib.axes.Axes) -> list[tuple[float, float, float, float]]:
    """Every bar drawn, as ``(y, low, high, linewidth)``."""
    bars = []
    for coll in ax.collections:
        if not isinstance(coll, LineCollection):
            continue
        for seg, width in zip(coll.get_segments(), coll.get_linewidths()):
            (low, y), (high, _) = seg
            bars.append((float(y), float(low), float(high), float(width)))
    return bars


def _one_figure(action, *args, **kwargs) -> matplotlib.figure.Figure:
    """The only figure a bounds action yields: one interval type asked for."""
    [(fig, _)] = list(action(*args, **kwargs))
    return fig


def test_plot_coefficient_bounds_draws_one_row_per_coefficient_top_to_bottom() -> None:
    fig = _one_figure(
        plot_coefficient_bounds,
        _bounds_fit(),
        latex_label=make_latex_label({"OpA": "$c_A$"}),
    )

    ax = fig.axes[0]
    ticks = dict(zip([t.get_text() for t in ax.get_yticklabels()], ax.get_yticks()))
    assert list(ticks) == ["$c_A$", r"$\mathrm{OpZZ}$"]
    assert ticks["$c_A$"] > ticks[r"$\mathrm{OpZZ}$"]


def test_plot_coefficient_bounds_draws_every_level_widest_thinnest() -> None:
    """The bars are the fit's own bounds, and the narrower interval is drawn
    thicker so it reads on top of the wider one."""
    fit = _bounds_fit()
    fig = _one_figure(
        plot_coefficient_bounds,
        fit,
        latex_label=_plain,
        params_to_plot="OpA",
        bounds_levels=[68, 95],
    )

    bars = sorted(_bars(fig.axes[0]), key=lambda bar: bar[3])
    # flat on both sides: pytest.approx does not compare nested sequences
    expected = [
        edge
        for level in (95.0, 68.0)
        for edge in fit.confidence_bounds(level)["OpA"][0]
    ]
    assert [edge for _, low, high, _ in bars for edge in (low, high)] == (
        pytest.approx(expected)
    )
    assert bars[0][3] < bars[1][3]


def test_plot_coefficient_bounds_keeps_params_to_plot_in_its_order() -> None:
    fig = _one_figure(
        plot_coefficient_bounds,
        _bounds_fit(),
        latex_label=_plain,
        params_to_plot=["OpZZ", "OpA"],
    )

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["OpZZ", "OpA"]


def test_plot_coefficient_bounds_draws_every_piece_of_a_split_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multimodal posterior has a credible region in several pieces; each is
    a bar on the coefficient's row, and nothing spans the gap between them."""
    fit = _bounds_fit(samples={"OpA": [0.0, 1.0]})
    pieces = [(-2.0, -1.0), (1.0, 2.0)]
    monkeypatch.setattr(
        fit, "confidence_bounds", lambda level, interval_type, params: {"OpA": pieces}
    )

    fig = _one_figure(
        plot_coefficient_bounds, fit, latex_label=_plain, bounds_levels=95
    )

    bars = _bars(fig.axes[0])
    assert sorted((low, high) for _, low, high, _ in bars) == pieces
    assert len({y for y, *_ in bars}) == 1


def test_plot_coefficient_bounds_passes_the_interval_type_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = _bounds_fit(samples={"OpA": [0.0, 1.0]})
    asked = []

    def bounds(level: float, interval_type: str, params: list[str]) -> dict:
        asked.append(interval_type)
        return {"OpA": [(0.0, 1.0)]}

    monkeypatch.setattr(fit, "confidence_bounds", bounds)
    fig = _one_figure(
        plot_coefficient_bounds, fit, latex_label=_plain, interval_types="hdi"
    )

    assert set(asked) == {"hdi"}
    legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any("HDI" in text for text in legend)


@pytest.mark.parametrize("comparison", [False, True])
def test_plot_coefficient_bounds_only_bounds_the_coefficients_drawn(
    monkeypatch: pytest.MonkeyPatch, comparison: bool
) -> None:
    """``params_to_plot`` is applied before any bound is computed: an HDI is a
    KDE per coefficient, too slow to spend on rows never drawn."""
    bounded = []

    def recording(values: np.ndarray, _level: float, _bounds: object) -> list:
        bounded.append(float(values[-1]))
        return [(0.0, 1.0)]

    monkeypatch.setitem(credible_intervals._INTERVAL_TYPES, "recording", recording)
    # tell the coefficients apart by their last sample
    fit = _bounds_fit(samples={"OpA": [0.0, 1.0], "OpZZ": [0.0, 2.0]})
    action, fits = (
        (plot_coefficient_bounds_comparison, [fit])
        if comparison
        else (plot_coefficient_bounds, fit)
    )

    _one_figure(
        action,
        fits,
        latex_label=_plain,
        params_to_plot=["OpZZ"],
        bounds_levels=[68, 95],
        interval_types="recording",
    )

    assert bounded == [2.0, 2.0]


@pytest.mark.parametrize("comparison", [False, True])
def test_plot_coefficient_bounds_yields_one_figure_per_interval_type(
    monkeypatch: pytest.MonkeyPatch, comparison: bool
) -> None:
    """``interval_types`` is the tables' key: a list of types is one figure
    each, in the order given and suffixed with the type, so the plots of a
    report show the very intervals its tables quote."""
    monkeypatch.setitem(
        credible_intervals._INTERVAL_TYPES, "fake", lambda _v, _l, _b: [(-1.0, 1.0)]
    )
    action, fits = (
        (plot_coefficient_bounds_comparison, [_bounds_fit()])
        if comparison
        else (plot_coefficient_bounds, _bounds_fit())
    )

    result = list(action(fits, latex_label=_plain, interval_types=["eti", "fake"]))

    assert [suffix for _, suffix in result] == ["eti", "fake"]
    legends = [
        " ".join(t.get_text() for t in fig.axes[0].get_legend().get_texts())
        for fig, _ in result
    ]
    assert "ETI" in legends[0] and "FAKE" not in legends[0]
    assert "FAKE" in legends[1] and "ETI" not in legends[1]
    assert {(low, high) for _, low, high, _ in _bars(result[1][0].axes[0])} == {
        (-1.0, 1.0)
    }


def test_plot_coefficient_bounds_defaults_to_one_eti_figure() -> None:
    [(_, suffix)] = list(plot_coefficient_bounds(_bounds_fit(), latex_label=_plain))
    assert suffix == "eti"


def test_plot_coefficient_bounds_rejects_empty_interval_types() -> None:
    with pytest.raises(ValueError, match="interval_types is empty"):
        list(
            plot_coefficient_bounds(
                _bounds_fit(), latex_label=_plain, interval_types=[]
            )
        )


def test_plot_coefficient_bounds_reads_an_individual_fit() -> None:
    """A bound is about one coefficient at a time, which is what a
    one-at-a-time fit has."""
    fit = Fit(
        fit_results=FitResultGroup(
            [
                FitResult(
                    free_parameters=[name],
                    best_fit_point={name: 0.0},
                    max_loglikelihood=-1.0,
                    num_data=10,
                    samples={name: jnp.linspace(0.0, 1.0, 101)},
                )
                for name in ("OpA", "OpZZ")
            ]
        ),
        fit_name="my_individual_fit",
        fit_runcard={"actions_": ["run_individual_analytic_fits"]},
    )

    fig = _one_figure(
        plot_coefficient_bounds, fit, latex_label=_plain, bounds_levels=95
    )

    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["OpA", "OpZZ"]
    assert len(_bars(fig.axes[0])) == 2


def test_plot_coefficient_bounds_symlog_axis_and_limits() -> None:
    fig = _one_figure(
        plot_coefficient_bounds,
        _bounds_fit(),
        latex_label=_plain,
        x_log=True,
        x_min=-5.0,
        x_max=5.0,
    )

    ax = fig.axes[0]
    assert ax.get_xscale() == "symlog"
    assert ax.get_xlim() == (-5.0, 5.0)


def _unit_and_narrow_fit(narrow_width: float, fit_name: str = "my_fit") -> Fit:
    """``OpA`` uniform on [0, 1] and ``OpZZ`` uniform on [0, narrow_width]: the
    second row is that much narrower than the first."""
    unit = np.linspace(0.0, 1.0, 1001)
    return _bounds_fit(
        samples={"OpA": list(unit), "OpZZ": list(narrow_width * unit)},
        fit_name=fit_name,
    )


def _row_bars(
    ax: matplotlib.axes.Axes, label: str
) -> list[tuple[float, float, float, float]]:
    """The bars of the row labelled *label*, thinnest (widest level) first."""
    ticks = dict(zip([t.get_text() for t in ax.get_yticklabels()], ax.get_yticks()))
    row = [bar for bar in _bars(ax) if abs(bar[0] - ticks[label]) < 0.5]
    return sorted(row, key=lambda bar: bar[3])


def test_plot_coefficient_bounds_rescales_a_row_much_narrower_than_the_widest() -> None:
    """A row 300 times narrower than the widest would be a dot on its axis:
    it is drawn 100 times larger, every level by the same factor, and its
    label says so. The widest row is the reference and is left alone."""
    fit = _unit_and_narrow_fit(0.003)
    fig = _one_figure(
        plot_coefficient_bounds, fit, latex_label=_plain, bounds_levels=[68, 95]
    )

    ax = fig.axes[0]
    scaled = r"$10^{2}$ OpZZ"
    assert [t.get_text() for t in ax.get_yticklabels()] == ["OpA", scaled]
    for label, name, factor in [("OpA", "OpA", 1.0), (scaled, "OpZZ", 100.0)]:
        drawn = [
            edge for _, low, high, _ in _row_bars(ax, label) for edge in (low, high)
        ]
        expected = [
            factor * edge
            for level in (95.0, 68.0)
            for edge in fit.confidence_bounds(level)[name][0]
        ]
        assert drawn == pytest.approx(expected)


def test_plot_coefficient_bounds_writes_a_factor_of_ten_without_an_exponent() -> None:
    fig = _one_figure(
        plot_coefficient_bounds, _unit_and_narrow_fit(0.03), latex_label=_plain
    )

    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels == ["OpA", r"$10$ OpZZ"]


def test_plot_coefficient_bounds_rescaling_can_be_switched_off() -> None:
    """Without it every row is drawn as the fit has it, under its plain label."""
    fit = _unit_and_narrow_fit(0.003)
    fig = _one_figure(
        plot_coefficient_bounds,
        fit,
        latex_label=_plain,
        bounds_levels=95,
        rescale_bounds=False,
    )

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["OpA", "OpZZ"]
    [(_, low, high, _)] = _row_bars(ax, "OpZZ")
    assert (low, high) == pytest.approx(fit.confidence_bounds(95.0)["OpZZ"][0])


def test_plot_coefficient_bounds_does_not_rescale_on_a_symlog_axis() -> None:
    """The symmetric log axis is the other way to read bounds decades apart:
    rows keep their values there, or the axis would no longer mean anything."""
    fit = _unit_and_narrow_fit(0.003)
    fig = _one_figure(
        plot_coefficient_bounds, fit, latex_label=_plain, bounds_levels=95, x_log=True
    )

    ax = fig.axes[0]
    assert ax.get_xscale() == "symlog"
    assert [t.get_text() for t in ax.get_yticklabels()] == ["OpA", "OpZZ"]
    [(_, low, high, _)] = _row_bars(ax, "OpZZ")
    assert (low, high) == pytest.approx(fit.confidence_bounds(95.0)["OpZZ"][0])


def test_plot_coefficient_bounds_rejects_empty_levels() -> None:
    with pytest.raises(ValueError, match="bounds_levels"):
        _one_figure(
            plot_coefficient_bounds, _bounds_fit(), latex_label=_plain, bounds_levels=[]
        )


def test_plot_coefficient_bounds_comparison_rows_are_the_union_of_the_fits() -> None:
    """A coefficient one fit did not sample is a gap for that fit, not an
    error, and each fit sits on its own offset within the row."""
    grid = list(np.linspace(0.0, 1.0, 101))
    both = _bounds_fit(fit_name="both", label="$\\mathrm{Both}$")
    only_b = _bounds_fit(samples={"OpB": grid}, fit_name="only_b")

    fig = _one_figure(
        plot_coefficient_bounds_comparison,
        [both, only_b],
        latex_label=_plain,
        bounds_levels=95,
    )

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["OpA", "OpZZ", "OpB"]
    bars = _bars(ax)
    assert len(bars) == 3
    colors = [tuple(c) for coll in ax.collections for c in coll.get_colors()]
    assert len(set(colors)) == 2
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend[:2] == ["$\\mathrm{Both}$", "only_b"]


def test_plot_coefficient_bounds_comparison_rescales_a_row_by_one_factor() -> None:
    """Two fits on the same row are there to be compared, so they share the
    row's factor: the one its widest fit needs. On its own the tighter fit
    would have been drawn 100 times larger, and looked as loose as the other."""
    loose = _unit_and_narrow_fit(0.03, fit_name="loose")
    tight = _unit_and_narrow_fit(0.003, fit_name="tight")

    fig = _one_figure(
        plot_coefficient_bounds_comparison,
        [loose, tight],
        latex_label=_plain,
        bounds_levels=95,
    )

    ax = fig.axes[0]
    scaled = r"$10$ OpZZ"
    assert [t.get_text() for t in ax.get_yticklabels()] == ["OpA", scaled]
    # the first fit sits on the upper offset of the row
    upper, lower = sorted(_row_bars(ax, scaled), key=lambda bar: -bar[0])
    for (_, low, high, _), fit in [(upper, loose), (lower, tight)]:
        expected = [10.0 * edge for edge in fit.confidence_bounds(95.0)["OpZZ"][0]]
        assert [low, high] == pytest.approx(expected)


def test_plot_coefficient_bounds_comparison_rescaling_can_be_switched_off() -> None:
    loose = _unit_and_narrow_fit(0.03, fit_name="loose")
    tight = _unit_and_narrow_fit(0.003, fit_name="tight")

    fig = _one_figure(
        plot_coefficient_bounds_comparison,
        [loose, tight],
        latex_label=_plain,
        bounds_levels=95,
        rescale_bounds=False,
    )

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["OpA", "OpZZ"]
    upper, lower = sorted(_row_bars(ax, "OpZZ"), key=lambda bar: -bar[0])
    for (_, low, high, _), fit in [(upper, loose), (lower, tight)]:
        assert (low, high) == pytest.approx(fit.confidence_bounds(95.0)["OpZZ"][0])


def test_plot_coefficient_bounds_comparison_rejects_no_fits() -> None:
    with pytest.raises(ValueError, match="fits is empty"):
        _one_figure(plot_coefficient_bounds_comparison, [], latex_label=_plain)


# ---------------------------------------------------------------------------
# plot_mass_reach / plot_mass_reach_comparison
# ---------------------------------------------------------------------------


def _reach_bars(ax: matplotlib.axes.Axes) -> list[tuple[float, float, float]]:
    """Every bar drawn, as ``(x centre, height, alpha)``."""
    return [
        (p.get_x() + p.get_width() / 2, p.get_height(), p.get_alpha())
        for p in ax.patches
        if isinstance(p, Rectangle)
    ]


@pytest.mark.parametrize(
    "pieces, expected",
    [
        ([(-0.25, 0.25)], 2.0),  # half-width 1/4
        ([(1.0, 3.0)], 1.0),  # off zero: the width counts, not the position
        ([(-2.0, -1.0), (1.0, 2.0)], 2**-0.5),  # outermost edges, gap included
        ([], np.nan),
        ([(1.0, 1.0)], np.nan),  # no width
    ],
)
def test_reach_from_pieces_is_one_over_root_half_width(pieces, expected) -> None:
    assert figures_mod._reach_from_pieces(pieces) == pytest.approx(
        expected, nan_ok=True
    )


def test_plot_mass_reach_draws_one_bar_per_coefficient_and_level() -> None:
    """The narrowest level is the tallest bar, drawn faintest under the wider."""
    fit = _bounds_fit()
    fig = _one_figure(
        plot_mass_reach,
        fit,
        latex_label=make_latex_label({"OpA": "$c_A$"}),
        bounds_levels=[95, 68],
    )

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["$c_A$", r"$\mathrm{OpZZ}$"]
    assert ax.get_yscale() == "log"
    bars = _reach_bars(ax)
    assert len(bars) == 4
    expected = {
        level: figures_mod._reach_from_pieces(fit.confidence_bounds(level)["OpA"])
        for level in (68.0, 95.0)
    }
    op_a = sorted((b for b in bars if b[0] < 0.5), key=lambda b: b[2])
    assert [h for _, h, _ in op_a] == pytest.approx([expected[68.0], expected[95.0]])
    assert expected[68.0] > expected[95.0]


def test_plot_mass_reach_skips_a_coefficient_without_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = _bounds_fit()
    monkeypatch.setattr(
        fit,
        "confidence_bounds",
        lambda level, interval_type, params: {"OpA": [(-1.0, 1.0)], "OpZZ": [(0, 0)]},
    )

    fig = _one_figure(plot_mass_reach, fit, latex_label=_plain, bounds_levels=95)

    [(x, height, _)] = _reach_bars(fig.axes[0])
    assert x == pytest.approx(0.0)
    assert height == pytest.approx(1.0)


def test_plot_mass_reach_keeps_params_to_plot_and_axis_options() -> None:
    fig = _one_figure(
        plot_mass_reach,
        _bounds_fit(),
        latex_label=_plain,
        params_to_plot=["OpZZ", "OpA"],
        y_log=False,
        y_min=0.0,
        y_max=5.0,
    )

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["OpZZ", "OpA"]
    assert ax.get_yscale() == "linear"
    assert ax.get_ylim() == (0.0, 5.0)


@pytest.mark.parametrize(
    "shortest_interval, expected_bottom",
    [
        ((-0.04, 0.04), 1.0),  # 5 TeV -> starts at 1 TeV
        ((-1.0, 1.0), 0.1),  # 1 TeV (on a power of 10) -> drops to 0.1 TeV
    ],
)
def test_plot_mass_reach_log_axis_starts_a_decade_below_the_shortest_bar(
    monkeypatch: pytest.MonkeyPatch,
    shortest_interval: tuple[float, float],
    expected_bottom: float,
) -> None:
    """Autoscaling a log axis would cut every bar at the shortest one."""
    fit = _bounds_fit()
    monkeypatch.setattr(
        fit,
        "confidence_bounds",
        lambda level, interval_type, params: {
            "OpA": [shortest_interval],
            "OpZZ": [(-1e-4, 1e-4)],  # 100 TeV
        },
    )

    fig = _one_figure(plot_mass_reach, fit, latex_label=_plain, bounds_levels=95)

    assert fig.axes[0].get_ylim()[0] == pytest.approx(expected_bottom)


def test_plot_mass_reach_yields_one_figure_per_interval_type() -> None:
    suffixes = [
        suffix
        for _, suffix in plot_mass_reach(
            _bounds_fit(), latex_label=_plain, interval_types=["eti", "hdi"]
        )
    ]
    assert suffixes == ["eti", "hdi"]


def test_plot_mass_reach_comparison_groups_are_the_union_of_the_fits() -> None:
    grid = list(np.linspace(0.0, 1.0, 101))
    both = _bounds_fit(fit_name="both", label="$\\mathrm{Both}$")
    only_b = _bounds_fit(samples={"OpB": grid}, fit_name="only_b")

    fig = _one_figure(
        plot_mass_reach_comparison,
        [both, only_b],
        latex_label=_plain,
        bounds_levels=95,
    )

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["OpA", "OpZZ", "OpB"]
    bars = _reach_bars(ax)
    assert len(bars) == 3
    # each fit off the group centre, on opposite sides
    xs = sorted(x for x, *_ in bars)
    assert xs[0] < 0 < xs[1] < 1 < xs[2]
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend[:2] == ["$\\mathrm{Both}$", "only_b"]


def test_plot_mass_reach_comparison_rejects_no_fits() -> None:
    with pytest.raises(ValueError, match="fits is empty"):
        _one_figure(plot_mass_reach_comparison, [], latex_label=_plain)


def test_plot_mass_reach_graph_resolves(tmp_path):
    """The template runcard resolves through reportengine, which checks the
    annotations that calling the provider directly skips."""
    import yaml
    from reportengine.resourcebuilder import FuzzyTarget, ResourceBuilder

    from smefit.api import smefitAPI

    template = Path(__file__).parents[1] / "template_runcards/mass_reach_plot.yaml"
    runcard = yaml.safe_load(template.read_text())
    template_text = runcard.pop("template_text")
    assert "{@plot_mass_reach_comparison@}" in template_text
    assert "{@plot_mass_reach@}" in template_text
    for entry in runcard["fits"]:
        folder = tmp_path / entry["name"]
        _fit(samples={"OpA": _GAUSS, "OpZZ": _GAUSS}).fit_results.write(folder)
        (folder / "input").mkdir()
        (folder / "input" / "runcard.yaml").write_text(
            yaml.safe_dump({"actions_": ["run_analytic_fit"]})
        )
        entry["path"] = str(tmp_path)

    config = smefitAPI.config_class(runcard, environment=smefitAPI.loadedenv)
    builder = ResourceBuilder(
        config,
        smefitAPI.provider_loaded,
        [
            FuzzyTarget("plot_mass_reach_comparison", (), (), ()),
            FuzzyTarget("plot_mass_reach", ("fits",), (), ()),
        ],
        perform_final=False,
    )
    builder.rootns.update(smefitAPI.loadedenv.ns_dump())
    builder.resolve_fuzzytargets()


# ---------------------------------------------------------------------------
# 2D contour helpers: _kde_grid, _density_level, _gaussian_ellipse
# ---------------------------------------------------------------------------

_COV_2D = np.array([[1.0, 0.6], [0.6, 0.5]])


def _gaussian_pair(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``n`` draws of a correlated 2D Gaussian centred on (1, -2), covariance `_COV_2D`."""
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal([1.0, -2.0], _COV_2D, size=n)
    return draws[:, 0], draws[:, 1]


@pytest.mark.parametrize("level", [68.0, 95.0])
def test_density_level_encloses_the_requested_fraction_of_samples(
    level: float,
) -> None:
    x, y = _gaussian_pair()
    *_, sample_density = _kde_grid(x, y)

    inside = np.mean(sample_density >= _density_level(sample_density, level))

    assert inside == pytest.approx(level / 100.0, abs=1e-3)


def test_density_level_encloses_the_requested_fraction_of_a_bimodal_posterior() -> None:
    """The quadratic case the KDE is there for: two disjoint solutions."""
    x, y = _gaussian_pair(n=2000)
    x = np.concatenate([x - 5.0, x + 5.0])
    y = np.concatenate([y, y])
    *_, sample_density = _kde_grid(x, y)

    inside = np.mean(sample_density >= _density_level(sample_density, 95.0))

    assert inside == pytest.approx(0.95, abs=1e-3)


def test_density_level_is_lower_for_a_wider_level() -> None:
    *_, sample_density = _kde_grid(*_gaussian_pair())

    assert _density_level(sample_density, 95.0) < _density_level(sample_density, 68.0)


def test_kde_grid_covers_the_samples_with_a_square_grid() -> None:
    x, y = _gaussian_pair()

    xx, yy, density, sample_density = _kde_grid(x, y)

    shape = (figures_mod._KDE_GRIDSIZE, figures_mod._KDE_GRIDSIZE)
    assert xx.shape == yy.shape == density.shape == shape
    assert sample_density.shape == x.shape
    assert xx.min() < x.min() and xx.max() > x.max()
    assert yy.min() < y.min() and yy.max() > y.max()


def test_kde_grid_rejects_mismatched_sizes() -> None:
    """Checked before thinning, which would otherwise erase the difference."""
    n = figures_mod._KDE_MAX_SAMPLES + 1000
    with pytest.raises(ValueError, match="same size"):
        _kde_grid(np.zeros(2 * n), np.zeros(2 * n - 1))


def test_kde_grid_thins_large_posteriors(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []
    real_kde = figures_mod.scipy.stats.gaussian_kde

    def spy(dataset: np.ndarray) -> object:
        seen.append(dataset.shape[1])
        return real_kde(dataset)

    monkeypatch.setattr(figures_mod.scipy.stats, "gaussian_kde", spy)
    x, y = _gaussian_pair(n=3 * figures_mod._KDE_MAX_SAMPLES)

    *_, sample_density = _kde_grid(x, y)

    assert seen == [figures_mod._KDE_MAX_SAMPLES]
    assert sample_density.size == figures_mod._KDE_MAX_SAMPLES


def test_kde_grid_thinning_keeps_every_interleaved_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NUTS stores its chains interleaved (sample k is chain k mod 4). Four
    chains each in its own mode, thinned by 4: a stride would keep chain 0
    alone, so every mode must still be in what the KDE sees."""
    seen: list[np.ndarray] = []
    real_kde = figures_mod.scipy.stats.gaussian_kde

    def spy(dataset: np.ndarray) -> object:
        seen.append(dataset)
        return real_kde(dataset)

    monkeypatch.setattr(figures_mod.scipy.stats, "gaussian_kde", spy)
    n_chains = 4
    n = n_chains * figures_mod._KDE_MAX_SAMPLES
    modes = 10.0 * np.arange(n_chains)
    rng = np.random.default_rng(0)
    x = modes[np.arange(n) % n_chains] + rng.normal(size=n)
    y = rng.normal(size=n)

    _kde_grid(x, y)

    [dataset] = seen
    kept_chain = np.rint(dataset[0] / 10.0).astype(int)
    counts = np.bincount(kept_chain, minlength=n_chains)
    # about a quarter each, far from the 5000 / 0 / 0 / 0 of a stride
    assert np.all(counts > 0.2 * figures_mod._KDE_MAX_SAMPLES)


def test_kde_grid_leaves_small_posteriors_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int] = []
    real_kde = figures_mod.scipy.stats.gaussian_kde

    def spy(dataset: np.ndarray) -> object:
        seen.append(dataset.shape[1])
        return real_kde(dataset)

    monkeypatch.setattr(figures_mod.scipy.stats, "gaussian_kde", spy)

    _kde_grid(*_gaussian_pair(n=1234))

    assert seen == [1234]


def test_gaussian_ellipse_axes_match_the_sample_covariance() -> None:
    x, y = _gaussian_pair(n=200_000)

    _, (width, height), angle = _gaussian_ellipse(x, y)

    eig_val, eig_vec = np.linalg.eigh(_COV_2D)
    assert width / 2 == pytest.approx(np.sqrt(eig_val[1]), rel=1e-2)
    assert height / 2 == pytest.approx(np.sqrt(eig_val[0]), rel=1e-2)
    # the major axis is an eigenvector, so its angle is fixed modulo 180 degrees
    expected = np.degrees(np.arctan2(eig_vec[1, 1], eig_vec[0, 1]))
    assert (angle - expected + 90.0) % 180.0 - 90.0 == pytest.approx(0.0, abs=1.0)


def test_gaussian_ellipse_is_centred_on_the_mean() -> None:
    """The mean, about which the covariance is taken: on skewed samples, where
    it parts from the median, the ellipse follows it."""
    rng = np.random.default_rng(0)
    x, y = rng.exponential(size=5000), rng.exponential(size=5000)

    center, _, _ = _gaussian_ellipse(x, y)

    assert center == pytest.approx((x.mean(), y.mean()))
    assert abs(center[0] - np.median(x)) > 0.2


def test_gaussian_ellipse_rejects_mismatched_sizes() -> None:
    with pytest.raises(ValueError, match="same size"):
        _gaussian_ellipse(np.zeros(10), np.zeros(9))


# ---------------------------------------------------------------------------
# plot_posterior_contours / plot_posterior_contours_comparison
# ---------------------------------------------------------------------------


def _contour_fit(
    names: tuple[str, ...] = ("OpA", "OpB", "OpC"),
    fit_name: str = "my_fit",
    label: str | None = None,
    use_quad: bool = False,
    contours: str = "kde",
    n: int = 400,
    seed: int = 0,
) -> Fit:
    """A joint fit with a correlated Gaussian posterior over ``names``."""
    rng = np.random.default_rng(seed)
    cov = 0.5 * np.eye(len(names)) + 0.3
    draws = rng.multivariate_normal(np.zeros(len(names)), cov, size=n)
    samples = {name: list(draws[:, k]) for k, name in enumerate(names)}
    return _fit(
        free=list(names),
        samples=samples,
        fit_name=fit_name,
        label=label,
        use_quad=use_quad,
        contours=contours,
    )


def _panels(fig: matplotlib.figure.Figure) -> list[matplotlib.axes.Axes]:
    """The contour panels, the legend cell (axis off) left out."""
    return [ax for ax in fig.axes if ax.axison]


def _cell(ax: matplotlib.axes.Axes) -> tuple[int, int]:
    spec = ax.get_subplotspec()
    return spec.rowspan.start, spec.colspan.start


def _ellipses(ax: matplotlib.axes.Axes) -> list[Ellipse]:
    return [p for p in ax.patches if isinstance(p, Ellipse)]


def _filled(ax: matplotlib.axes.Axes) -> list[Ellipse]:
    """The filled region of each fit: the patch drawn at alpha 0.3."""
    return [p for p in _ellipses(ax) if p.get_alpha() == pytest.approx(0.3)]


def _inner(ax: matplotlib.axes.Axes) -> list[Ellipse]:
    """The outlines of the levels inside the filled one, widest first: the
    unfilled ellipses not drawn solid."""
    return sorted(
        (
            p
            for p in _ellipses(ax)
            if not p.get_fill() and p.get_linestyle() not in ("-", "solid")
        ),
        key=lambda p: p.width,
        reverse=True,
    )


def test_plot_posterior_contours_two_coefficients_is_one_panel_with_its_legend() -> (
    None
):
    fig = plot_posterior_contours(
        _contour_fit(names=("OpA", "OpB")), latex_label=_plain
    )

    [ax] = fig.axes
    assert ax.get_legend() is not None
    assert fig.get_suptitle()
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("OpA", "OpB")


def test_plot_posterior_contours_three_coefficients_fill_the_lower_triangle() -> None:
    label = make_latex_label({"OpA": "$c_A$"})
    fig = plot_posterior_contours(_contour_fit(), latex_label=label)

    panels = {_cell(ax): ax for ax in _panels(fig)}
    # (row j - 1, column i) holds coefficient i against coefficient j
    assert set(panels) == {(0, 0), (1, 0), (1, 1)}
    [legend_cell] = [ax for ax in fig.axes if not ax.axison]
    assert legend_cell.get_legend() is not None
    assert _cell(legend_cell) == (0, 1)

    # axis labels on the outer panels only
    assert panels[1, 0].get_xlabel() == "$c_A$"
    assert panels[1, 1].get_xlabel() == label("OpB")
    assert panels[0, 0].get_ylabel() == label("OpB")
    assert panels[1, 0].get_ylabel() == label("OpC")
    assert panels[0, 0].get_xlabel() == ""
    assert panels[1, 1].get_ylabel() == ""


def test_plot_posterior_contours_follow_the_params_to_plot_order() -> None:
    fig = plot_posterior_contours(
        _contour_fit(),
        latex_label=_plain,
        params_to_plot=["OpC", "NotThere", "OpA", "OpB"],
    )

    panels = {_cell(ax): ax for ax in _panels(fig)}
    assert [panels[1, i].get_xlabel() for i in range(2)] == ["OpC", "OpA"]
    assert [panels[j, 0].get_ylabel() for j in range(2)] == ["OpA", "OpB"]


def test_plot_posterior_contours_rejects_fewer_than_two_coefficients() -> None:
    with pytest.raises(ValueError, match="at least two coefficients"):
        plot_posterior_contours(
            _contour_fit(), latex_label=_plain, params_to_plot=["OpA", "Nope"]
        )


def test_plot_posterior_contours_fill_the_widest_level_and_outline_the_rest() -> None:
    fig = plot_posterior_contours(
        _contour_fit(names=("OpA", "OpB"), contours="gaussian"), latex_label=_plain
    )

    [ax] = fig.axes
    [filled] = _filled(ax)
    [inner] = _inner(ax)
    assert filled.width > inner.width
    assert inner.get_linestyle() == "dashed"


def test_plot_posterior_contours_give_each_inner_level_its_own_line_style() -> None:
    fig = plot_posterior_contours(
        _contour_fit(names=("OpA", "OpB"), contours="gaussian"),
        latex_label=_plain,
        bounds_levels=[68, 90, 95, 99],
    )

    [ax] = fig.axes
    [filled] = _filled(ax)
    inner = _inner(ax)
    assert all(filled.width > p.width for p in inner)
    # 95, 90, 68: the cycle in order, from the widest inner level down
    assert [p.get_linestyle() for p in inner] == ["dashed", "dotted", "dashdot"]


@pytest.mark.parametrize("names", [("OpA", "OpB"), ("OpA", "OpB", "OpC")])
def test_plot_posterior_contours_legend_a_grey_entry_per_level(
    names: tuple[str, ...],
) -> None:
    """After the fit, one entry per level in the style it is drawn with: a
    filled patch for the widest, a line in its own style for the others —
    for the single panel's legend and the legend cell's alike."""
    fig = plot_posterior_contours(
        _contour_fit(names=names, contours="gaussian"),
        latex_label=_plain,
        bounds_levels=[68, 90, 95],
    )

    [legend] = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    labels = [t.get_text() for t in legend.get_texts()]
    assert labels == [
        "my_fit",
        *(rf"${level}\%\ \mathrm{{CL}}$" for level in (95, 90, 68)),
    ]
    _, widest, *inner = legend.legend_handles
    grey = matplotlib.colors.to_rgba("grey")
    assert widest.get_facecolor()[:3] == pytest.approx(grey[:3])
    assert [h.get_linestyle() for h in inner] == ["--", ":"]
    assert all(h.get_color() == "grey" for h in inner)


def test_plot_posterior_contours_scale_one_gaussian_per_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every level and layer of a fit's ellipses is one Gaussian scaled by
    sqrt(chi2_2(level)): one eigendecomposition per fit and panel."""
    calls: list[int] = []
    real_eigh = np.linalg.eigh

    def spy(matrix: np.ndarray) -> object:
        calls.append(1)
        return real_eigh(matrix)

    fit = _contour_fit(names=("OpA", "OpB"), contours="gaussian")
    samples = fit.fit_results.samples
    center, axes, angle = _gaussian_ellipse(samples["OpA"], samples["OpB"])
    monkeypatch.setattr(figures_mod.np.linalg, "eigh", spy)

    fig = plot_posterior_contours(fit, latex_label=_plain, bounds_levels=[68, 95])

    assert len(calls) == 1
    [ax] = fig.axes
    for patch in _ellipses(ax):
        assert patch.center == pytest.approx(center)
        assert patch.angle == pytest.approx(angle)
    [filled] = _filled(ax)
    [inner] = _inner(ax)
    for patch, level in [(filled, 95.0), (inner, 68.0)]:
        scale = np.sqrt(scipy.stats.chi2.ppf(level / 100.0, 2))
        assert (patch.width, patch.height) == pytest.approx(tuple(scale * axes))


def test_plot_posterior_contours_cycle_the_line_styles_past_their_end() -> None:
    levels = [50, 60, 70, 80, 90]
    fig = plot_posterior_contours(
        _contour_fit(names=("OpA", "OpB"), contours="gaussian"),
        latex_label=_plain,
        bounds_levels=levels,
    )

    [ax] = fig.axes
    cycle = figures_mod._LEVEL_LINESTYLE_CYCLE
    assert [p.get_linestyle() for p in _inner(ax)] == [
        cycle[idx % len(cycle)] for idx in range(len(levels) - 1)
    ]


def test_plot_posterior_contours_kde_fills_the_widest_level() -> None:
    fig = plot_posterior_contours(
        _contour_fit(names=("OpA", "OpB")),
        latex_label=_plain,
        hatch=False,
    )

    [ax] = fig.axes
    sets = [c for c in ax.collections if isinstance(c, ContourSet)]
    [fill] = [c for c in sets if c.filled]
    outlines = sorted((c for c in sets if not c.filled), key=lambda c: c.levels[0])
    # the wider region sits at the lower density: filled, and solid
    assert len(outlines) == 2
    assert fill.levels[0] == pytest.approx(outlines[0].levels[0])
    assert outlines[1].linestyles == "dashed"
    assert outlines[0].linestyles != "dashed"


def test_plot_posterior_contours_kde_gives_each_inner_level_its_own_line_style() -> (
    None
):
    fig = plot_posterior_contours(
        _contour_fit(names=("OpA", "OpB")),
        latex_label=_plain,
        bounds_levels=[68, 90, 95],
        hatch=False,
    )

    [ax] = fig.axes
    outlines = sorted(
        (c for c in ax.collections if isinstance(c, ContourSet) and not c.filled),
        key=lambda c: c.levels[0],
    )
    # increasing density is decreasing level: 95 (filled, matplotlib's default
    # solid line), then 90 and 68
    widest, *inner = outlines
    assert widest.linestyles not in figures_mod._LEVEL_LINESTYLE_CYCLE
    assert [c.linestyles for c in inner] == ["dashed", "dotted"]


@pytest.mark.parametrize("use_quad", [True, False])
@pytest.mark.parametrize(
    ("contours", "expect_kde"), [("kde", True), ("gaussian", False)]
)
def test_plot_posterior_contours_draw_in_the_fits_contour_style(
    monkeypatch: pytest.MonkeyPatch,
    use_quad: bool,
    contours: str,
    expect_kde: bool,
) -> None:
    """KDE by default whatever ``use_quad``: a linear fit's posterior need not
    be Gaussian, so ellipses are only ever asked for."""
    calls: list[int] = []
    real = figures_mod._kde_grid

    def spy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
        calls.append(1)
        return real(x, y)

    monkeypatch.setattr(figures_mod, "_kde_grid", spy)

    fig = plot_posterior_contours(
        _contour_fit(use_quad=use_quad, contours=contours), latex_label=_plain
    )

    # one KDE evaluation per panel, shared by both levels
    assert len(calls) == (3 if expect_kde else 0)
    has_ellipses = any(_ellipses(ax) for ax in _panels(fig))
    assert has_ellipses is not expect_kde


def test_plot_posterior_contours_comparison_draws_each_fit_its_own_way() -> None:
    """``contours`` belongs to the fit: one comparison can mix the two."""
    fig = plot_posterior_contours_comparison(
        [
            _contour_fit(names=("OpA", "OpB"), fit_name="kde"),
            _contour_fit(names=("OpA", "OpB"), fit_name="gauss", contours="gaussian"),
        ],
        latex_label=_plain,
        hatch=False,
    )

    [ax] = fig.axes
    contour_colors = {
        matplotlib.colors.to_hex(c.get_facecolor()[0])
        for c in ax.collections
        if isinstance(c, ContourSet) and c.filled
    }
    ellipse_colors = {matplotlib.colors.to_hex(p.get_facecolor()) for p in _filled(ax)}
    assert contour_colors == {matplotlib.colors.to_hex("C0")}
    assert ellipse_colors == {matplotlib.colors.to_hex("C1")}


def test_plot_posterior_contours_hatch_one_pattern_per_fit() -> None:
    fig = plot_posterior_contours_comparison(
        [
            _contour_fit(fit_name="a", contours="gaussian"),
            _contour_fit(fit_name="b", seed=1, contours="gaussian"),
        ],
        latex_label=_plain,
    )

    hatches = {p.get_hatch() for ax in _panels(fig) for p in ax.patches} - {None}
    assert len(hatches) == 2


def test_plot_posterior_contours_without_hatch_draw_no_hatch_layer() -> None:
    fig = plot_posterior_contours(
        _contour_fit(contours="gaussian"), latex_label=_plain, hatch=False
    )

    assert all(p.get_hatch() is None for ax in _panels(fig) for p in ax.patches)


def test_plot_posterior_contours_comparison_leaves_a_fit_out_of_panels_it_lacks() -> (
    None
):
    """A coefficient one fit did not sample is a gap for that fit, not an error."""
    fig = plot_posterior_contours_comparison(
        [
            _contour_fit(
                names=("OpA", "OpB"), fit_name="ab", label="$AB$", contours="gaussian"
            ),
            _contour_fit(names=("OpA", "OpC"), fit_name="ac", contours="gaussian"),
        ],
        latex_label=_plain,
    )

    by_cell = {_cell(ax): ax for ax in _panels(fig)}
    colors = {
        cell: {matplotlib.colors.to_hex(p.get_facecolor()) for p in _filled(ax)}
        for cell, ax in by_cell.items()
    }
    c0, c1 = matplotlib.colors.to_hex("C0"), matplotlib.colors.to_hex("C1")
    # coefficients OpA, OpB, OpC: (OpA, OpB) at (0, 0), (OpA, OpC) at (1, 0),
    # (OpB, OpC) at (1, 1)
    assert colors == {(0, 0): {c0}, (1, 0): {c1}, (1, 1): set()}
    [legend_cell] = [ax for ax in fig.axes if not ax.axison]
    labels = [t.get_text() for t in legend_cell.get_legend().get_texts()]
    # the fits, then the default levels
    assert labels == ["$AB$", "ac", r"$95\%\ \mathrm{CL}$", r"$68\%\ \mathrm{CL}$"]


def test_plot_posterior_contours_comparison_leave_out_a_coefficient_a_fit_fixed() -> (
    None
):
    """The stored samples hold fixed coefficients too: one free in a fit but
    fixed in another is a gap for the latter, not a constant fed to the KDE
    (a singular covariance) or pooled into the limits."""
    free = _contour_fit(fit_name="free")
    fixed = _contour_fit(names=("OpA", "OpB"), fit_name="fixed", seed=1)
    fixed.fit_results.samples["OpC"] = [50.0] * 400

    fig = plot_posterior_contours_comparison([free, fixed], latex_label=_plain)

    by_cell = {_cell(ax): ax for ax in _panels(fig)}
    # (OpA, OpC) at (1, 0), (OpB, OpC) at (1, 1): OpC on the y-axis
    for cell in [(1, 0), (1, 1)]:
        low, high = by_cell[cell].get_ylim()
        assert high < 50.0
        assert len(by_cell[cell].collections) == len(by_cell[(0, 0)].collections) // 2


def test_plot_posterior_contours_comparison_rejects_no_fits() -> None:
    with pytest.raises(ValueError, match="fits is empty"):
        plot_posterior_contours_comparison([], latex_label=_plain)


def test_plot_posterior_contours_comparison_rejects_an_individual_fit() -> None:
    individual = Fit(
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
        plot_posterior_contours_comparison(
            [_contour_fit(), individual], latex_label=_plain
        )


def test_plot_posterior_contours_comparison_rejects_a_fit_without_samples() -> None:
    empty = Fit(
        fit_results=FitResult(
            free_parameters=["OpA", "OpB"],
            best_fit_point={"OpA": 0.0, "OpB": 0.0},
            max_loglikelihood=-1.0,
            num_data=10,
        ),
        fit_name="no_samples",
        fit_runcard={"actions_": ["run_analytic_fit"]},
    )

    with pytest.raises(ValueError, match="no_samples"):
        plot_posterior_contours_comparison([_contour_fit(), empty], latex_label=_plain)


def test_plot_coefficient_bounds_labels_the_axis_with_a_coefficient() -> None:
    fig = _one_figure(plot_coefficient_bounds, _bounds_fit(), latex_label=_plain)

    assert fig.axes[0].get_xlabel().startswith("$c_i")


# ---------------------------------------------------------------------------
# A fit in the PCA basis: the same routine, on Fit.in_pca_basis
# ---------------------------------------------------------------------------

_ROOT_HALF: float = 0.5**0.5


def _rotation_pca() -> PCA:
    """PCA whose directions are OpA + OpB and OpA - OpB, normalised."""
    return PCA(
        eigenvalues=np.array([2.0, 1.0]),
        eigenvectors=np.array([[1.0, 1.0], [1.0, -1.0]]) * _ROOT_HALF,
        coeff_names=["OpA", "OpB"],
        threshold=1.0e-3,
        min_weight=0.01,
    )


def test_plot_fisher_diagonals_heatmap_draws_the_fisher_in_the_pca_basis() -> None:
    """DS_A constrains OpA + OpB alone, DS_B OpA - OpB alone: one per component."""
    total = np.array([[1.5, 0.5], [0.5, 1.5]])  # the spectrum of _rotation_pca
    along_pc1 = np.full((2, 2), 1.0)
    fim = {
        source: pd.DataFrame(F, index=["OpA", "OpB"], columns=["OpA", "OpB"])
        for source, F in (("DS_A", along_pc1), ("DS_B", total - along_pc1))
    }
    rotated = _fisher_in_pca_basis(fim, {}, _rotation_pca(), None)
    table = fisher_diagonals_normalised(rotated, _plain)
    np.testing.assert_allclose(table.to_numpy(), np.eye(2), atol=1e-12)

    fig = plot_fisher_diagonals_heatmap(table)

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["PC1", "PC2"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["DS_A", "DS_B"]
    assert [t.get_text() for t in ax.texts] == ["100.0", "100.0"]  # zeros not drawn


def _rotated_fit() -> Fit:
    """OpA = OpB on every sample: PC1 = sqrt(2) OpA, PC2 = 0."""
    grid = list(np.linspace(0.0, 1.0, 1001))
    fit = _bounds_fit(samples={"OpA": grid, "OpB": grid})
    return fit.in_pca_basis(_rotation_pca())


def test_plot_coefficient_bounds_draws_a_fit_in_the_pca_basis() -> None:
    fit = _rotated_fit()
    fig = _one_figure(plot_coefficient_bounds, fit, _plain, bounds_levels=95)

    ax = fig.axes[0]
    ticks = dict(zip([t.get_text() for t in ax.get_yticklabels()], ax.get_yticks()))
    assert list(ticks) == ["PC1", "PC2"]
    assert ticks["PC1"] > ticks["PC2"]
    drawn = {y: (low, high) for y, low, high, _ in _bars(ax)}
    assert drawn[ticks["PC1"]] == pytest.approx(
        (0.025 / _ROOT_HALF, 0.975 / _ROOT_HALF)
    )
    assert drawn[ticks["PC2"]] == pytest.approx((0.0, 0.0), abs=1e-12)
