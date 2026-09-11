"""Unit tests for smefit.figures — _plot_heatmap and the heatmaps built on it.

_plot_heatmap enables matplotlib's usetex globally, which would require a
LaTeX installation to actually render text. Tests here never draw/save the
figure (get_text() etc. only read stored attributes), and the module-level
`rc` call is neutralised so running the suite never depends on LaTeX being
installed (e.g. on CI runners).
"""

import re

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib import patches
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter

from smefit import figures as figures_mod
from smefit.core import ReferencePoint
from smefit.figures import (
    _ROW_GAP_RATIO,
    _plot_heatmap,
    plot_coefficient_bounds,
    plot_fisher_diagonals_heatmap,
    plot_fits_coefficient_bounds,
    plot_fits_mass_reach,
    plot_fits_posterior_contours,
    plot_fits_posterior_histograms,
    plot_mass_reach,
    plot_posterior_contours,
    plot_posterior_correlations,
    plot_posterior_histograms,
)
from smefit.fit_result import Fit, FitResult, FitResultGroup
from smefit.op_to_latex import coeff_info_latex


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
    baselines=None,
    fixed=None,
):
    """A joint fit holding the given posterior samples.

    The default pair is perfectly anti-correlated, which keeps the expected
    matrix obvious in tests that are about the drawing rather than the numbers.

    ``fixed`` names coefficients the runcard froze, as smefit records them: a
    ``free: False, value: v`` runcard entry, and the constant ``v`` resolved
    into every posterior sample.
    """
    if samples is None:
        samples = {"OpA": [0.0, 1.0, 2.0, 3.0], "OpZZ": [0.0, -1.0, -2.0, -3.0]}

    coefficients = {}
    if baselines is not None:
        coefficients.update(
            {
                name: {"free": True, "baseline_value": value}
                for name, value in baselines.items()
            }
        )
    if fixed is not None:
        coefficients.update(
            {name: {"free": False, "value": value} for name, value in fixed.items()}
        )
        length = len(next(iter(samples.values())))
        samples = {
            **samples,
            **{name: [value] * length for name, value in fixed.items()},
        }

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
        fit_runcard={
            "actions_": [action],
            "use_quad": use_quad,
            **({"coefficients": coefficients} if coefficients else {}),
        },
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
# plot_fits_posterior_contours / plot_posterior_contours
# ---------------------------------------------------------------------------


def _three_coeff_samples() -> dict[str, list[float]]:
    """Three non-degenerate coefficients, enough for a 2x2 panel grid."""
    rng = np.random.default_rng(0)
    return {
        name: rng.normal(loc, 1.0, size=50).tolist()
        for name, loc in [("OpA", 0.5), ("OpZZ", -0.5), ("OpC", 1.0)]
    }


def _two_coeff_gaussians() -> dict[str, list[float]]:
    """Two non-degenerate coefficients: the _fit default pair is perfectly
    anti-correlated, whose singular covariance a KDE cannot estimate."""
    rng = np.random.default_rng(1)
    return {
        "OpA": rng.normal(0.5, 1.0, size=50).tolist(),
        "OpZZ": rng.normal(-0.5, 1.0, size=50).tolist(),
    }


def test_contours_overlay_every_fit_in_one_panel() -> None:
    """Two fits and two coefficients: one panel holding both fits' ellipses
    (two patches each) and the SM marker, beside the legend's own cell."""
    fig = plot_fits_posterior_contours(
        [
            _fit(fit_name="fit_a", samples=_two_coeff_gaussians()),
            _fit(fit_name="fit_b", samples=_two_coeff_gaussians()),
        ]
    )

    assert len(fig.axes) == 2  # the panel and the legend cell
    ax = fig.axes[0]
    assert len(ax.patches) == 6  # outline, fill and hatch layer, for each fit
    assert len(ax.collections) == 1  # the SM marker


def test_contours_mark_the_sm_at_the_coefficient_baselines() -> None:
    """The SM is not always the origin: a coefficient parametrised around a
    non-zero baseline_value must be marked where that puts it."""
    fig = plot_fits_posterior_contours(
        [_fit(samples=_two_coeff_gaussians(), baselines={"OpA": 1.5, "OpZZ": -2.0})]
    )

    ax = fig.axes[0]
    sm_marker = ax.collections[-1]
    assert sm_marker.get_offsets().tolist() == [[1.5, -2.0]]


def test_contours_draw_the_runcard_reference_points_beside_the_sm() -> None:
    """reference_points add to the SM marker rather than replacing it, each
    with its own legend entry."""
    points = [
        ReferencePoint(label="$A$", values={"OpA": 2.0}),
        ReferencePoint(label="$B$", values={"OpA": -2.0, "OpZZ": 1.0}),
    ]

    fig = plot_fits_posterior_contours(
        [_fit(samples=_two_coeff_gaussians())], reference_points=points
    )

    ax = fig.axes[0]
    drawn = [c.get_offsets().tolist()[0] for c in ax.collections[-3:]]
    assert drawn == [[0.0, 0.0], [2.0, 0.0], [-2.0, 1.0]]
    legend_labels = [t.get_text() for t in fig.axes[-1].get_legend().get_texts()]
    assert legend_labels[-3:] == [r"$\mathrm{SM}$", "$A$", "$B$"]


def test_contours_reference_point_replaces_the_sm_when_it_is_off() -> None:
    """show_sm: False with one entry is how a runcard moves the marker."""
    points = [ReferencePoint(label="$A$", values={"OpA": 2.0, "OpZZ": 1.0})]

    fig = plot_fits_posterior_contours(
        [_fit(samples=_two_coeff_gaussians())],
        reference_points=points,
        show_sm=False,
    )

    assert fig.axes[0].collections[-1].get_offsets().tolist() == [[2.0, 1.0]]
    legend_labels = [t.get_text() for t in fig.axes[-1].get_legend().get_texts()]
    assert legend_labels[-1] == "$A$"


def test_contours_draw_an_ellipse_when_both_coefficients_have_a_std() -> None:
    """A point given with an uncertainty on both axes of a panel also gets the
    confidence ellipse of the uncorrelated Gaussian it describes."""
    points = [
        ReferencePoint(
            label="$A$", values={"OpA": 1.0, "OpZZ": 0.0}, std={"OpA": 0.4, "OpZZ": 0.2}
        )
    ]

    fig = plot_fits_posterior_contours(
        [_fit(samples=_two_coeff_gaussians())], reference_points=points, show_sm=False
    )

    ax = fig.axes[0]
    # the fit's three patches, then the reference point's, drawn the same way
    assert len(ax.patches) == 6
    outline, fill, _hatch = ax.patches[-3:]
    assert fill.get_center() == (1.0, 0.0)
    assert fill.get_width() > fill.get_height()  # 0.4 against 0.2
    assert fill.get_alpha() == pytest.approx(0.3)
    assert outline.get_facecolor()[3] == 0.0


def test_contours_hatch_every_filled_contour_differently() -> None:
    """Two fits and a reference point: three fills, three textures, so the
    figure survives greyscale and colour blindness."""
    points = [
        ReferencePoint(label="$A$", values={"OpA": 3.0}, std={"OpA": 0.4, "OpZZ": 0.4})
    ]

    fig = plot_fits_posterior_contours(
        [
            _fit(fit_name="fit_a", samples=_two_coeff_gaussians()),
            _fit(fit_name="fit_b", samples=_three_coeff_samples()),
        ],
        params_to_plot=["OpA", "OpZZ"],
        reference_points=points,
        show_sm=False,
    )

    hatched = [p.get_hatch() for p in fig.axes[0].patches if p.get_hatch()]
    assert len(hatched) == 3  # one per fill: two fits and the point
    assert len(set(hatched)) == 3


def test_contours_hatch_can_be_turned_off() -> None:
    fig = plot_fits_posterior_contours(
        [_fit(samples=_two_coeff_gaussians())], hatch=False, show_sm=False
    )

    assert all(p.get_hatch() is None for p in fig.axes[0].patches)


def test_contours_legend_key_of_a_point_with_a_contour_is_a_filled_patch() -> None:
    """A point that draws a contour is legended like the fits — a filled patch
    — with its own marker in the middle, rather than by a bare marker."""
    values = {"OpA": 1.0, "OpZZ": 0.0}
    keyed = plot_fits_posterior_contours(
        [_fit(samples=_two_coeff_gaussians())],
        reference_points=[
            ReferencePoint(label="$A$", values=values, std={"OpA": 0.4, "OpZZ": 0.2})
        ],
        show_sm=False,
    )
    bare = plot_fits_posterior_contours(
        [_fit(samples=_two_coeff_gaussians())],
        reference_points=[ReferencePoint(label="$A$", values=values)],
        show_sm=False,
    )

    keyed_legend = keyed.axes[-1].get_legend()
    bare_legend = bare.axes[-1].get_legend()
    # the outline, fill and hatch layer the fits' keys are made of, on top of
    # what the bare marker key already draws
    assert (
        len(keyed_legend.findobj(Rectangle)) == len(bare_legend.findobj(Rectangle)) + 3
    )
    # and the marker itself survives, drawn over them
    assert len(keyed_legend.findobj(PathCollection)) == 1


def test_contours_legend_marker_sits_at_the_centre_of_its_key() -> None:
    """Overlaid on a patch, a marker placed anywhere but the middle reads as a
    mistake — and matplotlib's default for a single scatter key is 3/8 up."""
    fig = plot_fits_posterior_contours(
        [_fit(samples=_two_coeff_gaussians())],
        reference_points=[
            ReferencePoint(
                label="$A$", values={"OpA": 1.0}, std={"OpA": 0.4, "OpZZ": 0.2}
            )
        ],
        show_sm=False,
    )
    fig.canvas.draw()  # the key artists are positioned at draw time

    legend = fig.axes[-1].get_legend()
    key = [r for r in legend.findobj(Rectangle) if r.get_width() > 0][-1]
    marker = legend.findobj(PathCollection)[0]

    box = key.get_window_extent()
    position = marker.get_offset_transform().transform(marker.get_offsets())[0]
    assert position[0] == pytest.approx((box.x0 + box.x1) / 2, abs=0.5)
    assert position[1] == pytest.approx((box.y0 + box.y1) / 2, abs=0.5)


def test_contours_skip_the_ellipse_when_a_std_is_missing() -> None:
    """One std describes a band, not an ellipse: the panel keeps the marker
    and draws nothing around it."""
    points = [
        ReferencePoint(label="$A$", values={"OpA": 1.0}, std={"OpA": 0.4}),
    ]

    fig = plot_fits_posterior_contours(
        [_fit(samples=_two_coeff_gaussians())], reference_points=points, show_sm=False
    )

    ax = fig.axes[0]
    assert len(ax.patches) == 3  # the fit's three, none of the point's
    assert ax.collections[-1].get_offsets().tolist() == [[1.0, 0.0]]


def test_contours_ellipse_follows_the_confidence_level() -> None:
    """The ellipse is drawn at the level the panels are read with, so it can
    be compared against the contours beside it."""
    points = [
        ReferencePoint(label="$A$", values={"OpA": 0.0}, std={"OpA": 0.4, "OpZZ": 0.2})
    ]
    kwargs = {
        "reference_points": points,
        "show_sm": False,
    }

    narrow = plot_fits_posterior_contours(
        [_fit(samples=_two_coeff_gaussians())], confidence_level=68, **kwargs
    )
    wide = plot_fits_posterior_contours(
        [_fit(samples=_two_coeff_gaussians())], confidence_level=95, **kwargs
    )

    assert wide.axes[0].patches[-1].get_width() > narrow.axes[0].patches[-1].get_width()


def test_contours_keep_the_whole_reference_ellipse_inside_the_frame() -> None:
    """Stretching to the centre alone would clip an ellipse whose edge reaches
    further than the samples."""
    points = [
        ReferencePoint(label="$A$", values={"OpA": 8.0}, std={"OpA": 2.0, "OpZZ": 0.5})
    ]

    fig = plot_fits_posterior_contours(
        [_fit(samples=_two_coeff_gaussians())], reference_points=points, show_sm=False
    )

    ellipse = fig.axes[0].patches[-1]
    assert fig.axes[0].get_xlim()[1] > 8.0 + ellipse.get_width() / 2


def test_contours_keep_a_reference_point_inside_the_frame() -> None:
    """Points stretch the axes exactly as the SM marker does."""
    points = [ReferencePoint(label="$A$", values={"OpA": 14.0})]

    fig = plot_fits_posterior_contours(
        [_fit(samples=_two_coeff_gaussians())], reference_points=points
    )

    assert fig.axes[0].get_xlim()[1] > 14.0


def test_contours_keep_a_non_zero_sm_point_inside_the_frame() -> None:
    """Stretching the axes to the marker is what makes it visible: these
    samples sit nowhere near the baseline."""
    fig = plot_fits_posterior_contours(
        [_fit(samples=_two_coeff_gaussians(), baselines={"OpA": 12.0})]
    )

    ax = fig.axes[0]
    assert ax.get_xlim()[1] > 12.0


def test_contours_three_coefficients_form_the_lower_triangle() -> None:
    """Three coefficients make three pairwise panels plus the legend axis in
    the free upper-right corner."""
    fits = [_fit(free=["OpA", "OpZZ", "OpC"], samples=_three_coeff_samples())]

    fig = plot_fits_posterior_contours(fits)

    assert len(fig.axes) == 4
    panels = fig.axes[:3]
    # only the outer panels carry axis labels: x on the bottom row, y on the
    # left column
    assert [bool(ax.get_xlabel()) for ax in panels] == [False, True, True]
    assert [bool(ax.get_ylabel()) for ax in panels] == [True, True, False]


def test_contours_per_fit_action_draws_a_single_fit() -> None:
    """The per-fit action wraps one fit: same figure, one fit's contours."""
    fig = plot_posterior_contours(_fit(samples=_two_coeff_gaussians()))

    assert len(fig.axes) == 2  # the panel and the legend cell
    assert len(fig.axes[0].patches) == 3  # outline, fill and hatch layer


def test_contours_restrict_to_params_to_plot() -> None:
    """Restricting three coefficients to two leaves a single panel."""
    fits = [_fit(free=["OpA", "OpZZ", "OpC"], samples=_three_coeff_samples())]

    fig = plot_fits_posterior_contours(fits, params_to_plot=["OpC", "OpA"])

    assert len(fig.axes) == 2  # the panel and the legend cell
    assert fig.axes[0].get_xlabel()  # OpC, the first requested, is the x-axis


def test_contours_default_to_kde_for_a_quadratic_fit() -> None:
    """use_quad marks the posterior as non-Gaussian, so the contours are KDE
    iso-density lines (collections), not ellipse patches."""
    fits = [_fit(samples=_two_coeff_gaussians(), use_quad=True)]

    fig = plot_fits_posterior_contours(fits, show_sm=False)

    ax = fig.axes[0]
    assert len(ax.patches) == 0
    assert len(ax.collections) >= 2  # filled and outlined confidence region


def test_contours_kde_is_overridable_per_fit() -> None:
    """A dict keyed by fit name drives each fit separately."""
    fits = [
        _fit(fit_name="quad", samples=_two_coeff_gaussians(), use_quad=True),
        _fit(fit_name="lin", samples=_two_coeff_gaussians()),
    ]

    fig = plot_fits_posterior_contours(fits, kde={"quad": False}, show_sm=False)

    # both fits fall back to ellipses: 3 patches each, no contour collections
    assert len(fig.axes[0].patches) == 6
    assert len(fig.axes[0].collections) == 0


def test_contours_dashed_level_adds_an_ellipse() -> None:
    """A [dashed, filled] confidence_level pair adds the dashed outline to the
    outline, fill and hatch layer a single level already draws."""
    fig = plot_fits_posterior_contours(
        [_fit(samples=_two_coeff_gaussians())], confidence_level=[68, 95]
    )

    assert len(fig.axes[0].patches) == 4


def test_contours_survive_a_pair_that_only_moves_together() -> None:
    """An expr-constrained coefficient tracks a free one exactly: there is no
    2D density to estimate, and the panel gets the interval on that line
    rather than a scipy error that takes the whole report down."""
    ramp = np.linspace(-1.0, 1.0, 200)
    fit = _fit(samples={"OpA": ramp.tolist(), "OpZZ": (2.0 * ramp).tolist()})

    fig = plot_fits_posterior_contours([fit], kde=True, show_sm=False)

    ax = fig.axes[0]
    (line,) = ax.lines
    assert not ax.patches
    assert line.get_ydata() == pytest.approx(2.0 * np.array(line.get_xdata()))


def test_contours_reject_more_than_two_confidence_levels() -> None:
    """A contour panel holds one fill and one outline; a third level has
    nowhere to go, and saying so beats unpacking into a bare ValueError."""
    with pytest.raises(ValueError, match="confidence_level takes one or two"):
        plot_fits_posterior_contours(
            [_fit(samples=_two_coeff_gaussians())], confidence_level=[68, 95, 99]
        )


def test_contours_accept_a_single_level_in_a_list() -> None:
    """One level written as a list is the same figure as the bare number."""
    fig = plot_fits_posterior_contours(
        [_fit(samples=_two_coeff_gaussians())], confidence_level=[95], show_sm=False
    )

    assert len(fig.axes[0].patches) == 3  # outline, fill and hatch layer, no dashed


def test_contours_legend_names_every_fit_and_the_sm() -> None:
    """The legend is how an overlay says which contour is which; labels come
    from the fits entries, names otherwise."""
    fits = [
        _fit(fit_name="fit_a", label=r"$\mathrm{Analytic}$"),
        _fit(fit_name="fit_b"),
    ]

    fig = plot_fits_posterior_contours(fits)

    legend = fig.axes[-1].get_legend()
    texts = [t.get_text() for t in legend.get_texts()]
    assert texts == [r"$\mathrm{Analytic}$", "fit_b", r"$\mathrm{SM}$"]


def test_contours_without_sm_show_neither_marker_nor_legend_entry() -> None:
    fig = plot_fits_posterior_contours([_fit()], show_sm=False)

    assert len(fig.axes[0].collections) == 0
    texts = [t.get_text() for t in fig.axes[-1].get_legend().get_texts()]
    assert r"$\mathrm{SM}$" not in texts


def _stuck_pair() -> list:
    """Two fits over the same three coefficients, the second having frozen
    OpC at 1.5 instead of fitting it."""
    return [
        _fit(
            fit_name="fit_a",
            free=["OpA", "OpZZ", "OpC"],
            samples=_three_coeff_samples(),
        ),
        _fit(
            fit_name="fit_b",
            free=["OpA", "OpZZ"],
            samples=_two_coeff_gaussians(),
            fixed={"OpC": 1.5},
        ),
    ]


def test_contours_span_the_union_of_the_fitted_coefficients() -> None:
    """A coefficient only one fit floated keeps its panels: the other is drawn
    where it held it, which is the comparison the figure is for."""
    fig = plot_fits_posterior_contours(_stuck_pair())

    # three coefficients: three pairwise panels, plus the legend cell
    assert len(fig.axes) == 4


def test_contours_draw_a_frozen_coefficient_as_a_segment() -> None:
    """fit_b fixed OpC, so in the OpA/OpC panel it is a horizontal bar at that
    value spanning OpA's confidence interval — beside fit_a's contour."""
    fig = plot_fits_posterior_contours(_stuck_pair(), show_sm=False)

    panel = fig.axes[1]  # panels come in (OpA, OpZZ), (OpA, OpC), (OpZZ, OpC)
    assert len(panel.patches) == 3  # fit_a's contour: outline, fill and hatch
    (segment,) = panel.lines
    assert list(segment.get_ydata()) == [1.5, 1.5]

    samples = np.asarray(_two_coeff_gaussians()["OpA"])
    assert segment.get_xdata() == pytest.approx(
        tuple(np.percentile(samples, [2.5, 97.5]))
    )


def test_contours_orient_the_segment_along_the_sampled_coefficient() -> None:
    """The same fit in the OpC/OpZZ panel: OpC is on the x-axis now, so the
    bar is vertical, at the value OpC was frozen at."""
    fig = plot_fits_posterior_contours(
        _stuck_pair(), params_to_plot=["OpC", "OpZZ"], show_sm=False
    )

    (segment,) = fig.axes[0].lines
    assert list(segment.get_xdata()) == [1.5, 1.5]


def test_contours_draw_a_coefficient_a_fit_never_declared_at_its_baseline() -> None:
    """fit_b's runcard never mentions OpC: it is stuck where its SM is, which
    the fit that does declare it puts at 1.0."""
    fits = [
        _fit(
            fit_name="fit_a",
            free=["OpA", "OpZZ", "OpC"],
            samples=_three_coeff_samples(),
            baselines={"OpC": 1.0},
        ),
        _fit(fit_name="fit_b", free=["OpA", "OpZZ"], samples=_two_coeff_gaussians()),
    ]

    fig = plot_fits_posterior_contours(fits, params_to_plot=["OpA", "OpC"])

    (segment,) = fig.axes[0].lines
    assert list(segment.get_ydata()) == [1.0, 1.0]


def test_contours_draw_two_frozen_coefficients_as_a_point() -> None:
    """Neither coefficient was sampled, so the fit has a position and no
    extent: one cross, no bar."""
    fit = _fit(
        free=["OpA", "OpZZ"],
        samples=_two_coeff_gaussians(),
        fixed={"OpC": 1.5, "OpD": -2.0},
    )

    fig = plot_fits_posterior_contours(
        [fit], params_to_plot=["OpA", "OpC", "OpD"], show_sm=False
    )

    panel = fig.axes[2]  # (OpC, OpD): both frozen
    (marker,) = panel.lines
    assert list(marker.get_xdata()) == [1.5]
    assert list(marker.get_ydata()) == [-2.0]
    assert marker.get_linestyle() == "None"


def test_contours_leave_a_coefficient_no_fit_fitted_out_by_default() -> None:
    """A runcard freezing twenty operators must not add twenty panels: only a
    coefficient asked for by name is drawn when no fit floated it."""
    fit = _fit(free=["OpA", "OpZZ"], samples=_two_coeff_gaussians(), fixed={"OpC": 1.5})

    assert len(plot_fits_posterior_contours([fit]).axes) == 2  # panel + legend
    assert (
        len(plot_fits_posterior_contours([fit], params_to_plot=["OpA", "OpC"]).axes)
        == 2
    )


def test_contours_frame_a_coefficient_every_fit_froze() -> None:
    """No samples set its range, so the frame comes from the value itself."""
    fit = _fit(free=["OpA", "OpZZ"], samples=_two_coeff_gaussians(), fixed={"OpC": 1.5})

    fig = plot_fits_posterior_contours(
        [fit], params_to_plot=["OpA", "OpC"], show_sm=False
    )

    low, high = fig.axes[0].get_ylim()
    assert low < 1.5 < high


def test_contours_legend_keeps_the_richest_handle_of_every_fit() -> None:
    """fit_b is a bar in the last panel drawn and a contour in the first; the
    legend describes the fit, not the panel that happened to come last."""
    fig = plot_fits_posterior_contours(_stuck_pair())

    legend = fig.axes[-1].get_legend()
    texts = [t.get_text() for t in legend.get_texts()]
    assert texts == ["fit_a", "fit_b", r"$\mathrm{SM}$"]
    # a contour's key is a patch, a bar's is a line: fit_b keeps the patch it
    # earned in the OpA/OpZZ panel, though its last panel is a bar
    assert isinstance(legend.legend_handles[1], patches.Patch)


def test_contours_legend_key_is_a_bar_for_a_fit_that_only_ever_bars() -> None:
    """The other way round: a fit that sampled nothing jointly here has no
    contour to advertise, and its key says so."""
    fits = [
        _fit(fit_name="fit_a", samples=_two_coeff_gaussians()),
        _fit(
            fit_name="fit_b",
            free=["OpA"],
            samples={"OpA": _two_coeff_gaussians()["OpA"]},
            fixed={"OpZZ": 0.3},
        ),
    ]

    fig = plot_fits_posterior_contours(fits)

    assert isinstance(fig.axes[-1].get_legend().legend_handles[1], Line2D)


def test_contours_say_what_a_bar_means_when_one_is_drawn() -> None:
    """A bar is not a contour of anything; the legend cell says so, once."""
    caption = r"$\mathrm{Bars\:and\:crosses\:mark\:fixed\:coefficients}$"

    stuck = plot_fits_posterior_contours(_stuck_pair())
    assert caption in [t.get_text() for t in stuck.axes[-1].texts]

    plain = plot_fits_posterior_contours([_fit(samples=_two_coeff_gaussians())])
    assert caption not in [t.get_text() for t in plain.axes[-1].texts]


def test_contours_keep_the_legend_off_the_panels() -> None:
    """Two coefficients fill their only grid cell, so the legend gets a column
    of its own instead of being drawn over the contours, as it does in the
    upper-right corner a larger figure leaves free."""
    fig = plot_fits_posterior_contours([_fit()])

    panel, legend_cell = fig.axes
    assert panel.get_legend() is None
    assert legend_cell.get_legend() is not None
    assert not legend_cell.axison  # it carries the legend, not a plot
    assert legend_cell.get_position().x0 >= panel.get_position().x1


def test_contours_state_the_confidence_level() -> None:
    """The figure has to say which confidence level is filled."""
    fig = plot_fits_posterior_contours([_fit()], confidence_level=90)

    assert any("90" in t.get_text() for t in fig.axes[-1].texts)


def test_contours_reject_an_individual_fit() -> None:
    """One-at-a-time fits have no joint posterior: pairing their independent
    1D samples would draw a correlation set by the seed and nothing else."""
    with pytest.raises(ValueError, match="one coefficient at a time"):
        plot_fits_posterior_contours([_fit(action="run_individual_analytic_fits")])


def test_contours_reject_a_fit_without_samples() -> None:
    fit = Fit(
        fit_results=FitResult(
            free_parameters=["OpA", "OpZZ"],
            best_fit_point={},
            max_loglikelihood=-1.0,
            num_data=10,
            samples=None,
        ),
        fit_name="sampleless",
        fit_runcard={"actions_": ["run_analytic_fit"]},
    )

    with pytest.raises(ValueError, match="no posterior samples"):
        plot_fits_posterior_contours([fit])


def test_contours_reject_an_empty_fits_list() -> None:
    with pytest.raises(ValueError, match="No fits"):
        plot_fits_posterior_contours([])


def test_contours_reject_a_single_coefficient() -> None:
    """Contours are pairwise: one coefficient has no panel to draw."""
    with pytest.raises(ValueError, match="at least 2"):
        plot_fits_posterior_contours([_fit()], params_to_plot=["OpA"])


# ---------------------------------------------------------------------------
# plot_fits_posterior_histograms / plot_posterior_histograms
# ---------------------------------------------------------------------------


def _individual_fit(samples, fit_name="individual_fit", baselines=None):
    """A fit run one coefficient at a time: a group of single-parameter
    results, each with its own independently sampled 1D posterior."""
    return Fit(
        fit_results=FitResultGroup(
            [
                FitResult(
                    free_parameters=[name],
                    best_fit_point={name: 0.0},
                    max_loglikelihood=-1.0,
                    num_data=10,
                    samples={name: jnp.array(vals)},
                )
                for name, vals in samples.items()
            ]
        ),
        fit_name=fit_name,
        fit_runcard={
            "actions_": ["run_individual_analytic_fits"],
            **(
                {
                    "coefficients": {
                        name: {"free": True, "baseline_value": value}
                        for name, value in baselines.items()
                    }
                }
                if baselines is not None
                else {}
            ),
        },
    )


def _bimodal_samples(size=400):
    """Two well-separated modes, as quadratic corrections produce."""
    rng = np.random.default_rng(2)
    return np.concatenate(
        [rng.normal(0.0, 0.1, size=size), rng.normal(5.0, 0.1, size=size)]
    ).tolist()


def _panels(fig):
    """The coefficient panels of a histogram figure — every axes but the
    legend's, which is the last one added and carries no bars."""
    return fig.axes[:-1]


def test_histograms_draw_one_panel_per_coefficient_plus_the_legend() -> None:
    """The legend gets a cell of its own, so it is never drawn over a panel."""
    fig = plot_fits_posterior_histograms([_fit()])

    assert len(fig.axes) == 3  # OpA, OpZZ, and the legend cell
    assert not fig.axes[-1].patches  # the legend cell holds no bars
    assert fig.axes[-1].get_legend() is not None


def test_histograms_overlay_every_fit_in_every_panel() -> None:
    fig = plot_fits_posterior_histograms(
        [_fit(fit_name="fit_a"), _fit(fit_name="fit_b")], show_sm=False
    )

    colors = {tuple(bar.get_facecolor()) for bar in _panels(fig)[0].patches}
    assert len(colors) == 2  # one colour per fit, both in the same panel


def test_histograms_are_densities_not_counts() -> None:
    """Fits of different sample sizes are being compared, and only the shapes
    of their posteriors are comparable."""
    fig = plot_fits_posterior_histograms(
        [_fit(samples={"OpA": [0.0, 1.0, 2.0, 3.0] * 10, "OpZZ": [0.0, 1.0] * 20})],
        show_sm=False,
    )

    bars = _panels(fig)[0].patches
    area = sum(bar.get_height() * bar.get_width() for bar in bars)
    assert area == pytest.approx(1.0)


def test_histograms_label_each_panel_with_its_coefficient() -> None:
    """The name goes inside the axes: an axis label under every panel of a
    large grid costs a row of height each time."""
    fig = plot_fits_posterior_histograms([_fit()], show_sm=False)

    texts = [t.get_text() for ax in _panels(fig) for t in ax.texts]
    assert texts == [coeff_info_latex.get("OpA", "OpA"), "OpZZ"]


def test_histograms_hide_the_y_ticks() -> None:
    """The y-axis is a normalisation, not a quantity anybody reads off."""
    fig = plot_fits_posterior_histograms([_fit()], show_sm=False)

    ax = _panels(fig)[0]
    assert ax.get_yticks().size  # the ticks themselves are still there
    assert not [t.get_text() for t in ax.get_yticklabels() if t.get_text()]


def test_histograms_share_the_x_range_across_fits() -> None:
    """A coefficient spans the same range for every fit drawn of it, or their
    posteriors cannot be compared by eye."""
    fig = plot_fits_posterior_histograms(
        [
            _fit(fit_name="narrow", samples={"OpA": [0.0, 0.1], "OpZZ": [0.0, 0.1]}),
            _fit(fit_name="wide", samples={"OpA": [-5.0, 5.0], "OpZZ": [0.0, 0.1]}),
        ],
        show_sm=False,
    )

    low, high = _panels(fig)[0].get_xlim()
    assert low < -5.0
    assert high > 5.0


def test_histograms_mark_the_sm_with_a_dashed_line() -> None:
    fig = plot_fits_posterior_histograms([_fit(baselines={"OpA": 1.5, "OpZZ": 0.0})])

    sm_line = _panels(fig)[0].lines[0]
    assert sm_line.get_xdata()[0] == pytest.approx(1.5)
    assert sm_line.get_linestyle() == "--"


def test_histograms_without_sm_draw_no_line() -> None:
    fig = plot_fits_posterior_histograms([_fit()], show_sm=False)

    assert not _panels(fig)[0].lines


def test_histograms_keep_a_non_zero_sm_point_inside_the_frame() -> None:
    """A coefficient sampled away from its baseline would otherwise have its
    SM marker fall outside the panel."""
    fig = plot_fits_posterior_histograms([_fit(baselines={"OpA": 9.0, "OpZZ": 0.0})])

    assert _panels(fig)[0].get_xlim()[1] > 9.0


def test_histograms_legend_names_every_fit_and_the_sm() -> None:
    fig = plot_fits_posterior_histograms(
        [
            _fit(fit_name="fit_a", label=r"$\mathrm{Analytic}$"),
            _fit(fit_name="fit_b"),
        ]
    )

    legend = fig.axes[-1].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == [
        r"$\mathrm{Analytic}$",
        "fit_b",
        r"$\mathrm{SM}$",
    ]


def test_histograms_legend_drops_the_sm_entry_with_the_marker() -> None:
    fig = plot_fits_posterior_histograms([_fit()], show_sm=False)

    legend = fig.axes[-1].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["my_fit"]


def test_histograms_bin_by_freedman_diaconis_by_default() -> None:
    """The rule follows the bulk of the posterior rather than its tails."""
    samples = np.random.default_rng(3).normal(size=500).tolist()
    fig = plot_fits_posterior_histograms(
        [_fit(samples={"OpA": samples, "OpZZ": samples})], show_sm=False
    )

    expected = len(np.histogram_bin_edges(samples, bins="fd")) - 1
    assert len(_panels(fig)[0].patches) == expected


def test_histograms_bins_can_be_overridden() -> None:
    fig = plot_fits_posterior_histograms([_fit()], bins=7, show_sm=False)

    assert len(_panels(fig)[0].patches) == 7


def test_histograms_bin_a_double_solution_one_branch_at_a_time() -> None:
    """The IQR of two modes together spans the empty gap between them, and
    the single bin width it yields smears each mode into a couple of bars."""
    bimodal = _bimodal_samples()
    fit = _fit(samples={"OpA": bimodal, "OpZZ": bimodal})

    merged = plot_fits_posterior_histograms(
        [fit], double_solution=["OpA"], show_sm=False
    )
    single = plot_fits_posterior_histograms([fit], show_sm=False)

    assert len(_panels(merged)[0].patches) > len(_panels(single)[0].patches)
    # the other coefficient was not declared bimodal, and is binned as before
    assert len(_panels(merged)[1].patches) == len(_panels(single)[1].patches)


def test_histograms_double_solution_keeps_every_sample() -> None:
    """Only the binning is split: the bars are drawn from the whole posterior,
    so declaring a second solution hides nothing."""
    bimodal = _bimodal_samples()
    fig = plot_fits_posterior_histograms(
        [_fit(samples={"OpA": bimodal, "OpZZ": bimodal})],
        double_solution=["OpA"],
        show_sm=False,
    )

    bars = _panels(fig)[0].patches
    area = sum(bar.get_height() * bar.get_width() for bar in bars)
    assert area == pytest.approx(1.0)


def test_histograms_double_solution_can_be_set_per_fit() -> None:
    """Two fits of the same coefficient need not both resolve a second
    solution."""
    bimodal = _bimodal_samples()
    fits = [
        _fit(fit_name="fit_a", samples={"OpA": bimodal, "OpZZ": bimodal}),
        _fit(fit_name="fit_b", samples={"OpA": bimodal, "OpZZ": bimodal}),
    ]

    fig = plot_fits_posterior_histograms(
        fits, double_solution={"fit_a": ["OpA"]}, show_sm=False
    )

    per_fit_bars = [
        len([b for b in _panels(fig)[0].patches if tuple(b.get_facecolor()) == color])
        for color in dict.fromkeys(
            tuple(b.get_facecolor()) for b in _panels(fig)[0].patches
        )
    ]
    assert per_fit_bars[0] > per_fit_bars[1]


def test_histograms_restrict_to_params_to_plot() -> None:
    fig = plot_fits_posterior_histograms([_fit()], params_to_plot=["OpZZ"])

    assert len(_panels(fig)) == 1


def test_histograms_accept_a_single_coefficient() -> None:
    """Unlike a contour, a histogram needs no pair."""
    fig = plot_fits_posterior_histograms([_fit()], params_to_plot=["OpA"])

    assert len(_panels(fig)) == 1


def test_histograms_accept_an_individual_fit() -> None:
    """This is how the individual counterpart of a marginalised figure is
    produced: a fits: entry pointing at an individual_fits output, not an
    option here."""
    fig = plot_fits_posterior_histograms(
        [_individual_fit({"OpA": [0.0, 1.0, 2.0, 3.0], "OpZZ": [0.0, 1.0, 2.0]})],
        show_sm=False,
    )

    assert len(_panels(fig)) == 2
    assert _panels(fig)[0].patches


def test_histograms_overlay_an_individual_fit_on_a_joint_one() -> None:
    """The comparison the notebook's figure pairs are about."""
    fig = plot_fits_posterior_histograms(
        [
            _fit(fit_name="joint"),
            _individual_fit({"OpA": [0.0, 1.0], "OpZZ": [0.0, 1.0]}),
        ],
        show_sm=False,
    )

    colors = {tuple(bar.get_facecolor()) for bar in _panels(fig)[0].patches}
    assert len(colors) == 2


def test_histograms_per_fit_action_draws_a_single_fit() -> None:
    fig = plot_posterior_histograms(_fit(), show_sm=False)

    legend = fig.axes[-1].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["my_fit"]


def test_histograms_grid_stays_near_square() -> None:
    """Five coefficients and the legend fill a 3x2 grid, not a 6x1 strip."""
    samples = {f"Op{i}": [0.0, 1.0, 2.0, 3.0] for i in range(5)}
    fig = plot_fits_posterior_histograms(
        [_fit(free=tuple(samples), samples=samples)], show_sm=False
    )

    geometry = _panels(fig)[0].get_subplotspec().get_gridspec().get_geometry()
    assert geometry == (2, 3)


def test_histograms_reject_an_empty_fits_list() -> None:
    with pytest.raises(ValueError, match="No fits"):
        plot_fits_posterior_histograms([])


def test_histograms_reject_a_fit_without_samples() -> None:
    fit = Fit(
        fit_results=FitResult(
            free_parameters=["OpA", "OpZZ"],
            best_fit_point={},
            max_loglikelihood=-1.0,
            num_data=10,
            samples=None,
        ),
        fit_name="sampleless",
        fit_runcard={"actions_": ["run_analytic_fit"]},
    )

    with pytest.raises(ValueError, match="no posterior samples"):
        plot_fits_posterior_histograms([fit])


# ---------------------------------------------------------------------------
# plot_fits_coefficient_bounds / plot_coefficient_bounds
# ---------------------------------------------------------------------------


def _intervals(ax):
    """The error bars drawn, as (x_low, x_high, y) per bar.

    An errorbar container leaves its bar as a LineCollection; reading the
    segments back is what says where an interval was actually drawn.
    """
    bars = []
    for collection in ax.collections:
        for segment in collection.get_segments():
            bars.append((segment[0][0], segment[-1][0], segment[0][1]))
    return bars


def _uniform_fit(fit_name="my_fit", **kwargs):
    """A fit whose percentiles are exact: 0..1000 puts the p-th percentile at
    10p, so a 68% interval is [160, 840] and a 95% one [25, 975]."""
    ramp = np.arange(0.0, 1001.0).tolist()
    return _fit(samples={"OpA": ramp, "OpZZ": ramp}, fit_name=fit_name, **kwargs)


def test_bounds_draw_one_interval_per_coefficient() -> None:
    fig = plot_fits_coefficient_bounds([_uniform_fit()])

    assert len(_intervals(fig.axes[0])) == 2  # OpA and OpZZ, one level each


def test_bounds_span_the_confidence_interval() -> None:
    fig = plot_fits_coefficient_bounds([_uniform_fit()], confidence_level=68)

    low, high, _ = _intervals(fig.axes[0])[0]
    assert (low, high) == pytest.approx((160.0, 840.0))


def test_bounds_marker_sits_at_the_central_value() -> None:
    """The mean of the posterior, which is what the interval is quoted
    around."""
    fig = plot_fits_coefficient_bounds([_uniform_fit()], confidence_level=68)

    markers = [line for line in fig.axes[0].lines if line.get_marker() == "."]
    assert len(markers) == 2  # one per coefficient
    assert [m.get_xdata()[0] for m in markers] == pytest.approx([500.0, 500.0])


def test_bounds_two_levels_draw_a_thin_bar_under_a_thick_one() -> None:
    """The wider interval is the thin one, whichever way round the levels are
    written."""
    fig = plot_fits_coefficient_bounds([_uniform_fit()], confidence_level=[95, 68])

    spans = sorted(_intervals(fig.axes[0]), key=lambda bar: bar[1] - bar[0])
    inner, outer = spans[0], spans[-1]
    assert (inner[0], inner[1]) == pytest.approx((160.0, 840.0))
    assert (outer[0], outer[1]) == pytest.approx((25.0, 975.0))


def test_bounds_draw_the_sm_line_at_the_coefficient_baseline() -> None:
    """The reference line marks the SM, which a non-zero baseline_value moves
    off the origin — the histograms and contours of the same fit put it
    there, and the bounds have to agree."""
    fig = plot_fits_coefficient_bounds(
        [_uniform_fit(baselines={"OpA": 3.0, "OpZZ": 3.0})]
    )

    (reference,) = (line for line in fig.axes[0].lines if line.get_linestyle() == "--")
    assert list(reference.get_xdata()) == [3.0, 3.0]


def test_bounds_draw_one_sm_line_per_row_when_the_baselines_differ() -> None:
    """One line across the figure would have to be at two places at once."""
    fig = plot_fits_coefficient_bounds(
        [_uniform_fit(baselines={"OpA": 3.0, "OpZZ": -1.0})]
    )

    dashed = [line for line in fig.axes[0].lines if line.get_linestyle() == "--"]
    assert sorted(line.get_xdata()[0] for line in dashed) == [-1.0, 3.0]


def test_bounds_reject_more_than_two_confidence_levels() -> None:
    """A row holds a thin bar under a thick one: a third level was silently
    dropped before, which is worse than being told."""
    with pytest.raises(ValueError, match="confidence_level takes one or two"):
        plot_fits_coefficient_bounds([_uniform_fit()], confidence_level=[68, 95, 99])


def test_bounds_level_order_does_not_matter() -> None:
    ascending = plot_fits_coefficient_bounds(
        [_uniform_fit()], confidence_level=[68, 95]
    )
    descending = plot_fits_coefficient_bounds(
        [_uniform_fit()], confidence_level=[95, 68]
    )

    assert sorted(_intervals(ascending.axes[0])) == pytest.approx(
        sorted(_intervals(descending.axes[0]))
    )


def test_bounds_rows_run_top_to_bottom_in_the_requested_order() -> None:
    """params_to_plot is a layout as much as a filter: the first coefficient
    asked for is the first one read."""
    fig = plot_fits_coefficient_bounds([_uniform_fit()], params_to_plot=["OpZZ", "OpA"])

    ax = fig.axes[0]
    ticks = sorted(zip(ax.get_yticks(), [t.get_text() for t in ax.get_yticklabels()]))
    assert [label for _, label in ticks] == [
        "OpA",
        coeff_info_latex.get("OpZZ", "OpZZ"),
    ]


def test_bounds_offset_the_fits_within_a_row() -> None:
    """Two fits of the same coefficient would otherwise draw over each
    other."""
    fig = plot_fits_coefficient_bounds(
        [_uniform_fit(fit_name="fit_a"), _uniform_fit(fit_name="fit_b")]
    )

    ys = {round(bar[2], 6) for bar in _intervals(fig.axes[0])}
    assert len(ys) == 4  # two coefficients x two fits, none coincident


def test_bounds_of_a_single_fit_sit_on_the_row_itself() -> None:
    fig = plot_fits_coefficient_bounds([_uniform_fit()])

    ax = fig.axes[0]
    assert sorted(round(bar[2], 6) for bar in _intervals(ax)) == sorted(ax.get_yticks())


def test_bounds_colour_each_fit_differently() -> None:
    fig = plot_fits_coefficient_bounds(
        [_uniform_fit(fit_name="fit_a"), _uniform_fit(fit_name="fit_b")]
    )

    colors = {tuple(np.ravel(c.get_color())) for c in fig.axes[0].collections}
    assert len(colors) == 2


def test_bounds_draw_both_branches_of_a_double_solution() -> None:
    """The second solution goes on the same row, so a bimodal coefficient
    shows two intervals rather than one spanning the gap."""
    bimodal = _bimodal_samples()
    fig = plot_fits_coefficient_bounds(
        [_fit(samples={"OpA": bimodal, "OpZZ": bimodal})],
        params_to_plot=["OpA"],
        double_solution=["OpA"],
    )

    drawn = sorted(_intervals(fig.axes[0]))
    assert len(drawn) == 2
    assert drawn[0][1] < 1.0  # the SM-like branch stops well before the gap
    assert drawn[1][0] > 4.0


def test_bounds_without_double_solution_span_the_gap() -> None:
    """What the split is there to avoid."""
    bimodal = _bimodal_samples()
    fig = plot_fits_coefficient_bounds(
        [_fit(samples={"OpA": bimodal, "OpZZ": bimodal})], params_to_plot=["OpA"]
    )

    ((low, high, _),) = _intervals(fig.axes[0])
    assert low < 1.0 < high


def test_bounds_double_solution_can_be_set_per_fit() -> None:
    bimodal = _bimodal_samples()
    fits = [
        _fit(fit_name="fit_a", samples={"OpA": bimodal, "OpZZ": bimodal}),
        _fit(fit_name="fit_b", samples={"OpA": bimodal, "OpZZ": bimodal}),
    ]

    fig = plot_fits_coefficient_bounds(
        fits, params_to_plot=["OpA"], double_solution={"fit_a": ["OpA"]}
    )

    assert len(_intervals(fig.axes[0])) == 3  # two branches for fit_a, one for fit_b


def test_bounds_mark_the_origin() -> None:
    """Whether an interval covers zero is the question the plot is read for."""
    fig = plot_fits_coefficient_bounds([_uniform_fit()])

    zero_line = [
        line
        for line in fig.axes[0].lines
        if line.get_linestyle() == "--" and line.get_xdata()[0] == 0.0
    ]
    assert zero_line


def test_bounds_log_scale_is_symmetric_about_zero() -> None:
    """Bounds run either side of zero, so a plain log scale cannot show
    them."""
    fig = plot_fits_coefficient_bounds([_uniform_fit()], x_log=True, lin_thr=0.5)

    ax = fig.axes[0]
    assert ax.get_xscale() == "symlog"
    assert ax.xaxis.get_transform().linthresh == pytest.approx(0.5)


def test_bounds_log_scale_ticks_skip_the_linear_region() -> None:
    """Decade ticks crowd together inside the linear region, where the scale
    no longer separates them."""
    fig = plot_fits_coefficient_bounds([_uniform_fit()], x_log=True, lin_thr=1.0)

    minor = fig.axes[0].get_xticks(minor=True)
    assert minor.size
    assert np.all(np.abs(minor) > 0.1)


def test_bounds_linear_by_default() -> None:
    assert (
        plot_fits_coefficient_bounds([_uniform_fit()]).axes[0].get_xscale() == "linear"
    )


def test_bounds_axis_limits_are_settable() -> None:
    fig = plot_fits_coefficient_bounds([_uniform_fit()], x_min=-10, x_max=10)

    assert fig.axes[0].get_xlim() == pytest.approx((-10.0, 10.0))


def test_bounds_legend_names_every_fit_and_the_levels() -> None:
    fig = plot_fits_coefficient_bounds(
        [
            _uniform_fit(fit_name="fit_a", label=r"$\mathrm{Analytic}$"),
            _uniform_fit(fit_name="fit_b"),
        ],
        confidence_level=[68, 95],
    )

    legend = fig.axes[0].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == [
        r"$\mathrm{Analytic}$",
        "fit_b",
    ]
    assert "68" in legend.get_title().get_text()
    assert "95" in legend.get_title().get_text()


def test_bounds_accept_an_individual_fit() -> None:
    ramp = np.arange(0.0, 1001.0).tolist()
    fig = plot_fits_coefficient_bounds(
        [_individual_fit({"OpA": ramp, "OpZZ": ramp})], confidence_level=68
    )

    low, high, _ = _intervals(fig.axes[0])[0]
    assert (low, high) == pytest.approx((160.0, 840.0))


def test_bounds_accept_a_single_coefficient() -> None:
    fig = plot_fits_coefficient_bounds([_uniform_fit()], params_to_plot=["OpA"])

    assert len(fig.axes[0].get_yticks()) == 1


def test_bounds_per_fit_action_draws_a_single_fit() -> None:
    fig = plot_coefficient_bounds(_uniform_fit())

    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ["my_fit"]


def test_bounds_reject_an_empty_fits_list() -> None:
    with pytest.raises(ValueError, match="No fits"):
        plot_fits_coefficient_bounds([])


def test_bounds_reject_a_fit_without_samples() -> None:
    fit = Fit(
        fit_results=FitResult(
            free_parameters=["OpA", "OpZZ"],
            best_fit_point={},
            max_loglikelihood=-1.0,
            num_data=10,
            samples=None,
        ),
        fit_name="sampleless",
        fit_runcard={"actions_": ["run_analytic_fit"]},
    )

    with pytest.raises(ValueError, match="no posterior samples"):
        plot_fits_coefficient_bounds([fit])


def test_bounds_group_the_fits_of_one_coefficient_together() -> None:
    """The gap between two coefficients has to stay a fixed factor wider than
    the gap between two fits of the same one, or the grouping reads backwards
    — which a fixed spread does as soon as enough fits are added."""
    for n_fits in (2, 3, 5):
        fig = plot_fits_coefficient_bounds(
            [_uniform_fit(fit_name=f"fit_{i}") for i in range(n_fits)]
        )

        ys = sorted({round(bar[2], 9) for bar in _intervals(fig.axes[0])})
        within = ys[1] - ys[0]  # two fits of the last coefficient
        between = ys[n_fits] - ys[n_fits - 1]  # across the coefficient boundary
        assert between == pytest.approx(_ROW_GAP_RATIO * within)


# ---------------------------------------------------------------------------
# plot_fits_mass_reach / plot_mass_reach
# ---------------------------------------------------------------------------


def _bars(ax):
    """The bars drawn, as (x_centre, height) — NaN heights included, which is
    how an unconstrained coefficient shows up."""
    return [(bar.get_x() + bar.get_width() / 2, bar.get_height()) for bar in ax.patches]


def test_mass_reach_is_the_inverse_root_of_the_semi_interval() -> None:
    """The 95% interval of the 0..1000 ramp is [25, 975], so the bound is
    475 and the reach 1/sqrt(475)."""
    fig = plot_fits_mass_reach([_uniform_fit()])

    heights = [height for _, height in _bars(fig.axes[0])]
    assert heights == pytest.approx([1 / np.sqrt(475.0)] * 2)


def test_mass_reach_can_use_the_whole_interval() -> None:
    fig = plot_fits_mass_reach([_uniform_fit()], full_interval=True)

    assert _bars(fig.axes[0])[0][1] == pytest.approx(1 / np.sqrt(950.0))


def test_mass_reach_follows_the_confidence_level() -> None:
    """A narrower interval is a tighter bound, so a further reach."""
    at68 = plot_fits_mass_reach([_uniform_fit()], confidence_level=68)
    at95 = plot_fits_mass_reach([_uniform_fit()], confidence_level=95)

    assert _bars(at68.axes[0])[0][1] > _bars(at95.axes[0])[0][1]


def test_mass_reach_draws_one_bar_per_fit_and_coefficient() -> None:
    fig = plot_fits_mass_reach(
        [_uniform_fit(fit_name="fit_a"), _uniform_fit(fit_name="fit_b")]
    )

    assert len(_bars(fig.axes[0])) == 4


def test_mass_reach_groups_the_bars_by_coefficient() -> None:
    """Two fits of one coefficient touch; the next coefficient's group starts
    a clear gap away."""
    fig = plot_fits_mass_reach(
        [_uniform_fit(fit_name="fit_a"), _uniform_fit(fit_name="fit_b")]
    )

    xs = sorted(x for x, _ in _bars(fig.axes[0]))
    assert xs[1] - xs[0] < xs[2] - xs[1]


def test_mass_reach_labels_the_groups_with_the_coefficients() -> None:
    fig = plot_fits_mass_reach([_uniform_fit()])

    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        coeff_info_latex.get("OpA", "OpA"),
        coeff_info_latex.get("OpZZ", "OpZZ"),
    ]


def test_mass_reach_names_the_axis_it_is_read_on() -> None:
    fig = plot_fits_mass_reach([_uniform_fit()])

    assert "Lambda" in fig.axes[0].get_ylabel()
    assert "TeV" in fig.axes[0].get_ylabel()


def test_mass_reach_colours_each_fit_differently() -> None:
    fig = plot_fits_mass_reach(
        [_uniform_fit(fit_name="fit_a"), _uniform_fit(fit_name="fit_b")]
    )

    assert len({bar.get_facecolor() for bar in fig.axes[0].patches}) == 2


def test_mass_reach_legend_names_every_fit_and_the_level() -> None:
    fig = plot_fits_mass_reach(
        [
            _uniform_fit(fit_name="fit_a", label=r"$\mathrm{Analytic}$"),
            _uniform_fit(fit_name="fit_b"),
        ],
        confidence_level=68,
    )

    legend = fig.axes[0].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == [
        r"$\mathrm{Analytic}$",
        "fit_b",
    ]
    assert "68" in legend.get_title().get_text()


def test_mass_reach_leaves_a_gap_for_an_unconstrained_coefficient() -> None:
    """A flat posterior has no width to invert: a bar of infinite height
    would read as an infinitely good bound."""
    fig = plot_fits_mass_reach(
        [_fit(samples={"OpA": [2.0] * 10, "OpZZ": [0.0, 1.0, 2.0, 3.0]})]
    )

    heights = [height for _, height in _bars(fig.axes[0])]
    assert np.isnan(heights[0])
    assert np.isfinite(heights[1])


def test_mass_reach_log_scale() -> None:
    fig = plot_fits_mass_reach([_uniform_fit()], y_log=True)

    assert fig.axes[0].get_yscale() == "log"


def test_mass_reach_is_linear_by_default() -> None:
    assert plot_fits_mass_reach([_uniform_fit()]).axes[0].get_yscale() == "linear"


def test_mass_reach_restricts_to_params_to_plot() -> None:
    fig = plot_fits_mass_reach([_uniform_fit()], params_to_plot=["OpZZ"])

    assert len(_bars(fig.axes[0])) == 1


def test_mass_reach_accepts_an_individual_fit() -> None:
    """The individual reach figure is this action pointed at an
    individual_fits output, not a mode of it."""
    ramp = np.arange(0.0, 1001.0).tolist()
    fig = plot_fits_mass_reach([_individual_fit({"OpA": ramp, "OpZZ": ramp})])

    assert _bars(fig.axes[0])[0][1] == pytest.approx(1 / np.sqrt(475.0))


def test_mass_reach_per_fit_action_draws_a_single_fit() -> None:
    fig = plot_mass_reach(_uniform_fit())

    assert len(_bars(fig.axes[0])) == 2  # one bar per coefficient
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ["my_fit"]


def test_mass_reach_rejects_an_empty_fits_list() -> None:
    with pytest.raises(ValueError, match="No fits"):
        plot_fits_mass_reach([])


def test_mass_reach_rejects_a_fit_without_samples() -> None:
    fit = Fit(
        fit_results=FitResult(
            free_parameters=["OpA", "OpZZ"],
            best_fit_point={},
            max_loglikelihood=-1.0,
            num_data=10,
            samples=None,
        ),
        fit_name="sampleless",
        fit_runcard={"actions_": ["run_analytic_fit"]},
    )

    with pytest.raises(ValueError, match="no posterior samples"):
        plot_fits_mass_reach([fit])


def test_bounds_leave_a_gap_for_a_coefficient_a_fit_never_sampled() -> None:
    """A one-at-a-time fit can hold a coefficient whose own fit stored no
    samples: it is still a coefficient the fit fitted, so it keeps its row,
    and that row is simply empty for this fit."""
    fit = Fit(
        fit_results=FitResultGroup(
            [
                FitResult(
                    free_parameters=["OpA"],
                    best_fit_point={"OpA": 0.0},
                    max_loglikelihood=-1.0,
                    num_data=10,
                    samples={"OpA": jnp.array([0.0, 1.0, 2.0, 3.0])},
                ),
                FitResult(
                    free_parameters=["OpZZ"],
                    best_fit_point={"OpZZ": 0.0},
                    max_loglikelihood=-1.0,
                    num_data=10,
                    samples=None,
                ),
            ]
        ),
        fit_name="half_sampled",
        fit_runcard={"actions_": ["run_individual_analytic_fits"]},
    )

    fig = plot_fits_coefficient_bounds([fit])

    ax = fig.axes[0]
    assert len(ax.get_yticks()) == 2  # both coefficients keep their row
    assert len(_intervals(ax)) == 1  # only the sampled one is drawn


def test_contours_keep_small_tick_labels_readable(draw) -> None:
    """A coefficient of 1e-4 prints five leading zeros per label under the
    default formatter, and the labels of one panel run into each other."""
    rng = np.random.default_rng(7)
    tiny = {
        name: rng.normal(0.0, 1.5e-4, size=200).tolist() for name in ("OpA", "OpZZ")
    }

    fig = plot_fits_posterior_contours([_fit(samples=tiny)])
    draw(fig)

    ax = fig.axes[0]
    digits = [
        re.sub(r"[^0-9.]", "", t.get_text())
        for t in ax.get_xticklabels()
        if t.get_text()
    ]
    assert not any("0.00" in label for label in digits), digits
    assert "10" in ax.xaxis.get_offset_text().get_text()


def test_contours_write_the_power_on_the_outer_panels_only() -> None:
    """matplotlib draws it whatever `labelbottom` says, so an inner panel
    would carry a stray power of its own next to a panel it does not label."""
    fig = plot_fits_posterior_contours(
        [_fit(free=("OpA", "OpZZ", "OpC"), samples=_three_coeff_samples())]
    )

    panels = {
        (ax.get_subplotspec().colspan.start, ax.get_subplotspec().rowspan.start): ax
        for ax in fig.axes
    }
    bottom_left = panels[(0, 1)]  # bottom row, first column: labels both ways
    inner = panels[(0, 0)]  # top row, first column: y labels only

    assert bottom_left.xaxis.offsetText.get_visible()
    assert bottom_left.yaxis.offsetText.get_visible()
    assert not inner.xaxis.offsetText.get_visible()
    assert inner.yaxis.offsetText.get_visible()


def _tick_digits(ticklabels) -> list[str]:
    """The digits of a set of tick labels, which come out as mathtext."""
    return [re.sub(r"[^0-9.]", "", t.get_text()) for t in ticklabels if t.get_text()]


def test_histograms_keep_small_tick_labels_readable(draw) -> None:
    """Each panel is a different coefficient on its own range, so each carries
    its own power — unlike a contour grid, where a column shares one."""
    rng = np.random.default_rng(8)
    tiny = {
        name: rng.normal(0.0, 1.5e-4, size=300).tolist() for name in ("OpA", "OpZZ")
    }

    fig = plot_fits_posterior_histograms([_fit(samples=tiny)], show_sm=False)
    draw(fig)

    for ax in _panels(fig):
        digits = _tick_digits(ax.get_xticklabels())
        assert not any("0.00" in label for label in digits), digits
        assert ax.xaxis.offsetText.get_visible()


def test_bounds_keep_small_tick_labels_readable(draw) -> None:
    rng = np.random.default_rng(9)
    tiny = {
        name: rng.normal(0.0, 1.5e-4, size=300).tolist() for name in ("OpA", "OpZZ")
    }

    fig = plot_fits_coefficient_bounds([_fit(samples=tiny)])
    draw(fig)

    ax = fig.axes[0]
    digits = _tick_digits(ax.get_xticklabels())
    assert not any("0.00" in label for label in digits), digits
    assert "10" in ax.xaxis.get_offset_text().get_text()


def test_bounds_keep_the_coefficient_names_on_the_rows(draw) -> None:
    """The y axis is not a numeric scale: reformatting it would label the rows
    with the positions they sit at instead of the coefficients."""
    fig = plot_fits_coefficient_bounds([_uniform_fit()])
    draw(fig)

    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert coeff_info_latex.get("OpA", "OpA") in labels
    assert "OpZZ" in labels


def test_bounds_log_scale_keeps_its_own_formatter() -> None:
    """A symlog axis already writes its powers; a linear formatter over it
    would label decade ticks with plain numbers."""
    fig = plot_fits_coefficient_bounds([_uniform_fit()], x_log=True)

    assert not isinstance(fig.axes[0].xaxis.get_major_formatter(), ScalarFormatter)


def test_mass_reach_keeps_the_coefficient_names_on_the_groups(draw) -> None:
    """Same as the bounds plot, the other way round: here it is x that names
    coefficients and y that is numeric."""
    fig = plot_fits_mass_reach([_uniform_fit()])
    draw(fig)

    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert coeff_info_latex.get("OpA", "OpA") in labels
    assert isinstance(ax.yaxis.get_major_formatter(), ScalarFormatter)


def test_mass_reach_log_scale_keeps_its_own_formatter() -> None:
    fig = plot_fits_mass_reach([_uniform_fit()], y_log=True)

    assert not isinstance(fig.axes[0].yaxis.get_major_formatter(), ScalarFormatter)
