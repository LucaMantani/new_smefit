"""Unit tests for smefit.figures — the report figures and the heatmap they share.

The figures enable matplotlib's usetex, which needs a LaTeX installation to
render any text. CI has none, and installing one is expensive, so the suite must
never invoke it — see the `_no_latex` fixture below.

Most tests here also avoid drawing at all (get_text() and friends only read
stored attributes), but that is not something to rely on — anything that
measures text, `tight_layout()` above all, renders it.
"""

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection

from smefit import figures as figures_mod
from smefit.figures import (
    _plot_heatmap,
    plot_chi2_scan,
    plot_fisher_diagonals_heatmap,
    plot_pca_components_heatmap,
    plot_pca_spectrum,
    plot_posterior_correlations,
)
from smefit.fit_result import Fit, FitResult, FitResultGroup
from smefit.op_to_latex import coeff_info_latex
from smefit.pca import PCA


@pytest.fixture(autouse=True)
def _no_latex(monkeypatch):
    """Keep matplotlib off LaTeX for the duration of a test.

    Stubbing out `set_plot_style` is what stops the figures switching usetex on.
    The rcParam is then forced off as well, so the invariant the suite needs —
    "no LaTeX is invoked" — is asserted directly rather than inferred from
    nobody having turned it on; that inference is exactly what broke when
    `smefit.op_to_latex` used to do it at import time.
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


def test_plot_chi2_scan_yields_one_figure_per_coefficient():
    scans = [
        {"OpA": {"points": [-1.0, 0.0, 1.0], "chi2": [4.0, 0.0, 4.0]}},
        {"OpZZ": {"points": [-2.0, 2.0], "chi2": [1.0, 1.0]}},
    ]

    result = plot_chi2_scan(scans)

    assert [name for _, name in result] == ["OpA", "OpZZ"]
    for fig, _ in result:
        ax = fig.axes[0]
        assert ax.get_ylabel() == r"$\chi^2$"


def test_plot_chi2_scan_plots_points_against_chi2():
    scans = [{"OpA": {"points": [-1.0, 0.0, 1.0], "chi2": [4.0, 0.0, 4.0]}}]

    result = plot_chi2_scan(scans)

    fig, name = result[0]
    assert name == "OpA"
    line = fig.axes[0].lines[0]
    assert line.get_xdata().tolist() == [-1.0, 0.0, 1.0]
    assert line.get_ydata().tolist() == [4.0, 0.0, 4.0]


def test_plot_chi2_scan_uses_latex_label_when_known():
    scans = [{"OQQ1": {"points": [0.0, 1.0], "chi2": [0.0, 1.0]}}]

    result = plot_chi2_scan(scans)

    fig, _ = result[0]
    assert fig.axes[0].get_xlabel() == r"$c_{QQ}^{\scriptscriptstyle 1}$"


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
        _fit(free=["OpA", "OpZZ"]), params_to_plot=["OpA", "OpNotInThisFit"]
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
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        coeff_info_latex.get(name, name) for name in ["OpA", "OpB", "OpC"]
    ]


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
    fig = plot_pca_spectrum(_pca())

    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["PC1", "PC2", "PC3"]
    # the dashed line sits at the flat-direction threshold
    thresholds = [
        line.get_ydata()[0] for line in ax.lines if line.get_linestyle() == "--"
    ]
    assert thresholds == [_pca().threshold]


def _bar_sizes(ax):
    return [len(container) for container in ax.containers]


def test_plot_pca_spectrum_separates_flat_from_constrained():
    """The flat direction is drawn as its own, labelled, set of bars."""
    fig = plot_pca_spectrum(_pca())

    ax = fig.axes[0]
    assert _bar_sizes(ax) == [2, 1]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "constrained",
        "flat",
    ]


def test_plot_pca_spectrum_marks_the_exactly_flat_direction():
    """A zero eigenvalue leaves a bar of no length, so it gets a marker too."""
    fig = plot_pca_spectrum(_pca())

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
    fig = plot_pca_spectrum(pca_obj)

    ax = fig.axes[0]
    assert _bar_sizes(ax) == [2]
    assert not [coll for coll in ax.collections if isinstance(coll, PathCollection)]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["constrained"]
