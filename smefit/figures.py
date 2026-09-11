"""
smefit.figures.py

Functions for creating figures and plots.
"""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, rc
from matplotlib.lines import Line2D
from reportengine.figure import figure

from smefit.bounds_1d import coeff_bounds, mass_reach, split_solution
from smefit.contours_2d import (
    ellipse_half_axis,
    fit_colors,
    fit_hatches,
    plot_contours,
    plot_stuck_point,
    plot_stuck_segment,
    plot_uncorrelated_contours,
)
from smefit.fit_result import FitResult
from smefit.op_to_latex import coeff_info_latex, group_info_latex
from smefit.plot_utils import (
    baseline_point,
    best_fit_pair,
    coeff_limits,
    common_free_coefficients,
    compact_tick_labels,
    contour_coefficients,
    marker_points,
    per_fit_option,
    select_params,
    stuck_values,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.figure import Figure

    from smefit.core import ReferencePoint
    from smefit.fit_result import Fit

log = logging.getLogger(__name__)


def _plot_heatmap(
    matrix,
    coeff_names,
    source_names,
    vmin=None,
    vmax=None,
    cmap="Blues",
    mask_zeros=True,
    value_fmt="{:.1f}",
    aspect="auto",
    colorbar=False,
    title=None,
):
    """Render a (n_coeffs, n_sources) matrix as a heatmap and save to PDF.

    Parameters
    ----------
    matrix : array-like, shape (n_coeffs, n_sources)
    coeff_names : list of str
    source_names : list of str
    vmin, vmax : float, optional
        Colour scale limits passed to imshow.
    cmap : str, optional
        Colormap name. The default suits a matrix of one sign; a matrix that
        spans zero wants a diverging one.
    mask_zeros : bool, optional
        Whether an exactly zero cell is left blank. True where a zero means
        "this source does not enter" (the default), False where zero is a
        value like any other. Cells that are not finite are always blank —
        they are the absence of a number, whatever the matrix means.
    value_fmt : str, optional
        Format string for the per-cell annotation.
    aspect : str, optional
        imshow aspect. ``"equal"`` keeps cells square, which a square matrix
        wants and a (coeffs x sources) one does not.
    colorbar : bool, optional
        Whether to draw the colour scale alongside.
    title : str, optional
        Heading for the plot, drawn above the column labels. Passed to
        matplotlib verbatim, so it may be raw LaTeX.
    """
    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 22})
    rc("text", usetex=True)
    rc("text.latex", preamble=r"\usepackage{amssymb}")

    matrix = np.array(matrix, dtype=float)
    n_coeffs, n_sources = matrix.shape

    fig_w = max(6, n_sources * 0.7)
    fig_h = max(3, n_coeffs * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticks(range(n_sources))
    ax.set_xticklabels(source_names, rotation=90, fontsize=14)
    if title is not None:
        ax.set_xlabel(title, fontsize=18, labelpad=12)

    coeff_labels = [coeff_info_latex.get(name, name) for name in coeff_names]
    ax.set_yticks(range(n_coeffs))
    ax.set_yticklabels(coeff_labels, fontsize=14)

    masked = np.ma.masked_invalid(matrix)
    if mask_zeros:
        masked = np.ma.masked_equal(masked, 0.0)

    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad("black", 0.7)
    im = ax.imshow(masked, aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(-0.5, n_sources, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_coeffs, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    blank = np.ma.getmaskarray(masked)
    for i in range(n_coeffs):
        for j in range(n_sources):
            if blank[i, j]:
                continue
            val = matrix[i, j]
            red, green, blue, _ = im.cmap(im.norm(val))
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            color = "white" if luminance < 0.5 else "black"
            ax.text(
                j,
                i,
                value_fmt.format(val),
                ha="center",
                va="center",
                fontsize=10,
                color=color,
            )

    return fig


@figure
def plot_fisher_diagonals_heatmap(
    fisher_diagonals_normalised,
    cmap="Blues",
    value_fmt="{:.1f}",
    colorbar=False,
):
    """Plot the Fisher diagonals matrix as a heatmap.

    Parameters
    ----------
    fisher_diagonals_normalised : pd.DataFrame
        Index = coeff_names, columns = source_names. Source names found in
        ``group_info_latex`` (the ``group:`` labels) are drawn in LaTeX.
    cmap : str, optional
        Colormap. Sequential by default: the values are percentage shares, so
        they run one way from zero and a diverging map would invent a midpoint.
    value_fmt : str, optional
        Format of the per-cell annotation, in percent.
    colorbar : bool, optional
        Whether to draw the colour scale alongside. Off by default: rows sum to
        100%, so the annotations already say what a cell is worth.
    """
    fd = fisher_diagonals_normalised
    return _plot_heatmap(
        fd.values * 100,
        fd.index.tolist(),
        [group_info_latex.get(name, name) for name in fd.columns],
        vmin=0,
        vmax=100,
        cmap=cmap,
        value_fmt=value_fmt,
        colorbar=colorbar,
    )


@figure
def plot_posterior_correlations(
    fit, params_to_plot=None, cmap="RdBu_r", value_fmt="{:.2f}", colorbar=True
):
    """Plot the posterior correlations of one fit's free coefficients.

    Takes a single ``fit``, so a runcard listing several under ``fits:`` gets
    one heatmap per fit, each headed with the fit it is drawn from.

    Parameters
    ----------
    fit : smefit.fit_result.Fit
        A previously run fit, loaded from disk.
    params_to_plot : list of str, optional
        Restrict the heatmap to these coefficients, in this order.
        All of them by default.
    cmap : str, optional
        Colormap. Diverging by default, as correlations run either way about
        zero; a sequential one misreads anti-correlation as "little".
    value_fmt : str, optional
        Format of the per-cell annotation. Widen it to read small
        correlations, or pass a format that renders nothing to drop the
        numbers on a matrix too large to label.
    colorbar : bool, optional
        Whether to draw the colour scale alongside.

    Raises
    ------
    ValueError
        If the fit was run one coefficient at a time: its coefficients were
        never sampled together, so there is no joint posterior to correlate.
    """
    if fit.individual_fit:
        raise ValueError(
            f"Fit '{fit.fit_name}' was run one coefficient at a time, so its "
            "coefficients were never sampled together and there is no joint "
            "posterior to correlate."
        )

    corr = fit.fit_results.correlations
    selected = select_params(corr.index, params_to_plot, context=fit.fit_name)
    corr = corr.loc[selected, selected]
    labels = [coeff_info_latex.get(name, name) for name in selected]
    return _plot_heatmap(
        corr.values,
        labels,
        labels,
        vmin=-1,
        vmax=1,
        mask_zeros=False,
        aspect="equal",
        cmap=cmap,
        value_fmt=value_fmt,
        colorbar=colorbar,
        title=fit.plot_label,
    )


# ----------------------------------------------------------------------
# Posterior contours — pairwise 2D marginalised confidence regions
# ----------------------------------------------------------------------


def _confidence_levels(
    confidence_level: float | Sequence[float], key: str = "confidence_level"
) -> list[float]:
    """The one or two confidence levels a figure is drawn at.

    A single number or a list of one or two, since a contour panel and a bounds
    row hold one filled region and one outline. Anything else is the runcard's
    mistake and is named as such, rather than unpacking into a message that
    does not mention the key it came from.

    Raises
    ------
    ValueError
        If no level, or more than two, were given.
    """
    if isinstance(confidence_level, (list, tuple)):
        levels = [float(level) for level in confidence_level]
    else:
        levels = [float(confidence_level)]

    if not 1 <= len(levels) <= 2:
        raise ValueError(
            f"{key} takes one or two levels, got {len(levels)}: "
            f"{list(confidence_level)}."
        )
    return levels


def _contour_levels(
    confidence_level: float | Sequence[float],
) -> tuple[float | None, float]:
    """``(dashed_level, filled_level)`` of a contour figure, in that order."""
    levels = _confidence_levels(confidence_level)
    return (None, levels[0]) if len(levels) == 1 else (levels[0], levels[1])


def _posterior_contours(
    fits: Sequence[Fit],
    params_to_plot: list[str] | str | None = None,
    confidence_level: float | Sequence[float] = 95,
    subplot_size: float = 4,
    kde: bool | Mapping[str, bool] | None = None,
    show_sm: bool = True,
    show_best_fit: bool = False,
    reference_points: Sequence[ReferencePoint] | None = None,
    hatch: bool = True,
) -> Figure:
    """Draw the pairwise 2D confidence contours of *fits* in one figure.

    The shared core of the two contour actions below, which differ only in
    how many fits reportengine hands them — their docstrings carry the
    parameter documentation. The panels form the lower triangle of the
    coefficient matrix: the panel in row ``j - 1`` and column ``i`` shows
    coefficient ``i`` on the x-axis and coefficient ``j`` on the y-axis, and
    every fit is drawn in every panel — as a contour where it sampled both
    coefficients, and otherwise at the value it held one or both of them at.
    """
    if not fits:
        raise ValueError("No fits to plot.")

    # What is drawn comes from each fit's joint posterior samples; anything
    # without one is rejected here, before any figure exists to be half-drawn.
    posteriors = []
    for fit in fits:
        if fit.individual_fit:
            raise ValueError(
                f"Fit '{fit.fit_name}' was run one coefficient at a time, so "
                "its coefficients were never sampled together and there is no "
                "joint posterior to draw contours from."
            )
        results = fit.fit_results
        assert isinstance(results, FitResult)  # individual fits were rejected
        if results.samples is None:
            raise ValueError(
                f"Fit '{fit.fit_name}' stored no posterior samples, so there "
                "is nothing to draw contours from."
            )
        posteriors.append(results.samples)

    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 22})
    rc("text", usetex=True)
    rc("text.latex", preamble=r"\usepackage{amssymb}")

    coeffs = contour_coefficients(fits, params_to_plot)
    n_par = len(coeffs)

    dashed_cl, cl = _contour_levels(confidence_level)

    colors = fit_colors(len(fits))
    # texture carries what colour carries, for print and for a reader who
    # cannot separate the hues; None everywhere turns it back into flat fills
    hatches = fit_hatches(len(fits)) if hatch else [None] * len(fits)
    kdes = per_fit_option(kde, fits, [fit.use_quad for fit in fits])
    # the SM is not always the origin: a coefficient can be parametrised so
    # that its baseline_value sits elsewhere, and that is where the marker goes
    # — and where a fit that never declared a coefficient at all sits, marker
    # or no marker, so the baselines are needed whether or not one is drawn
    baselines = baseline_point(fits, coeffs)
    stuck = stuck_values(fits, coeffs, baselines)
    # the SM marker, when drawn, then the runcard's reference_points, their
    # coordinates filled in from those same baselines
    points = marker_points(fits, coeffs, reference_points, show_sm)
    point_hatches = (
        fit_hatches(len(points), offset=len(fits)) if hatch else [None] * len(points)
    )

    # a coefficient's frame must hold every point drawn in it: its samples,
    # the values the fits that did not sample it were stuck at, and every
    # marker — the whole ellipse of one with a std, or it is drawn clipped
    include_points: dict[str, list[float]] = {}
    for name in coeffs:
        values = [held[name] for held in stuck if name in held]
        for point in points:
            centre = point.values[name]
            values.append(centre)
            if name in point.std:
                half_axis = ellipse_half_axis(point.std[name], cl)
                values.extend((centre - half_axis, centre + half_axis))
        if values:
            include_points[name] = values
    limits = coeff_limits(fits, coeffs, include_points=include_points)
    coeff_labels = [coeff_info_latex.get(name, name) for name in coeffs]

    n_cells = n_par - 1  # pairwise panels: the lower triangle has one row less
    # the legend lives in a cell of its own, which the lower triangle leaves
    # free from three coefficients on; two coefficients fill their only cell,
    # so the grid gains the column the legend would otherwise be drawn over
    n_cols = max(n_cells, 2)
    fig = plt.figure(figsize=(n_cols * subplot_size, n_cells * subplot_size))
    grid = plt.GridSpec(n_cells, n_cols, hspace=0.1, wspace=0.1)

    # A fit is not drawn the same way in every panel: it contours the pairs it
    # sampled and marks the ones it held fixed. Its legend key is the richest
    # thing it was drawn as anywhere in the figure — a contour over a segment,
    # a segment over a point — so the key describes the fit, not whichever
    # panel happened to be drawn last.
    legend_handles: list[tuple[int, Any]] = [(0, None)] * len(fits)
    # likewise a point's key is its ellipse wherever it drew one, and the bare
    # marker otherwise
    point_handles: list[tuple[int, Any]] = [(0, None)] * len(points)
    for i, j in itertools.combinations(range(n_par), 2):
        c1, c2 = coeffs[i], coeffs[j]
        ax = fig.add_subplot(grid[j - 1, i])

        for idx, fit in enumerate(fits):
            x_stuck, y_stuck = stuck[idx].get(c1), stuck[idx].get(c2)
            if x_stuck is None and y_stuck is None:
                rank, handle = 3, plot_contours(
                    ax,
                    posteriors[idx],
                    coeff1=c1,
                    coeff2=c2,
                    kde=kdes[idx],
                    color=colors[idx],
                    confidence_level=cl,
                    dashed_confidence_level=dashed_cl,
                    show_best_fit=show_best_fit,
                    best_fit=best_fit_pair(fit, c1, c2),
                    hatch=hatches[idx],
                )
            elif x_stuck is None or y_stuck is None:
                sampled, fixed_value, orientation = (
                    (c1, y_stuck, "horizontal")
                    if y_stuck is not None
                    else (c2, x_stuck, "vertical")
                )
                assert fixed_value is not None  # exactly one of the two is
                rank, handle = 2, plot_stuck_segment(
                    ax,
                    posteriors[idx][sampled],
                    fixed_value,
                    orientation=orientation,
                    color=colors[idx],
                    confidence_level=cl,
                    dashed_confidence_level=dashed_cl,
                    show_best_fit=show_best_fit,
                )
            else:
                rank, handle = 1, plot_stuck_point(
                    ax, x_stuck, y_stuck, color=colors[idx]
                )

            if rank > legend_handles[idx][0]:
                legend_handles[idx] = (rank, handle)

        for index, point in enumerate(points):
            marker = ax.scatter(
                point.values[c1],
                point.values[c2],
                color=point.color,
                marker=point.marker,
                s=50,
                zorder=10,
            )
            # both coefficients of this panel have to be known for the point to
            # describe an ellipse on it; one std alone describes a band, which
            # is not what was asked for
            if c1 in point.std and c2 in point.std:
                contour = plot_uncorrelated_contours(
                    ax,
                    center=(point.values[c1], point.values[c2]),
                    std=(point.std[c1], point.std[c2]),
                    color=point.color,
                    confidence_level=cl,
                    dashed_confidence_level=dashed_cl,
                    hatch=point_hatches[index],
                )
                # a tuple handle is drawn as its artists overlaid, so the
                # legend key becomes the filled patch of the fits with this
                # point's marker in the middle. A point without a contour keeps
                # the bare marker: there is no filled region to advertise.
                rank, handle = 2, (*contour, marker)
            else:
                rank, handle = 1, marker
            if rank > point_handles[index][0]:
                point_handles[index] = (rank, handle)

        ax.set_xlim(*limits[c1])
        ax.set_ylim(*limits[c2])
        ax.locator_params(axis="x", nbins=5)
        ax.locator_params(axis="y", nbins=6)
        ax.minorticks_on()
        ax.grid(linestyle="dotted", linewidth=0.5)

        # only the outer panels carry axis labels and tick labels
        if j == n_par - 1:
            ax.set_xlabel(coeff_labels[i], fontsize=26)
        else:
            ax.tick_params(axis="x", which="both", labelbottom=False)
        if i == 0:
            ax.set_ylabel(coeff_labels[j], fontsize=26)
        else:
            ax.tick_params(axis="y", which="both", labelleft=False)

        # a coefficient of 1e-4 would otherwise print five leading zeros in
        # every label, and the labels of one panel would run into each other
        compact_tick_labels(ax, show_x_offset=j == n_par - 1, show_y_offset=i == 0)

    # the legend, and the confidence level it is read with, go in the free
    # upper-right corner — never over a panel, whatever the figure's size
    ax = fig.add_subplot(grid[0, 1:])
    ax.axis("off")

    legend_labels = [fit.plot_label for fit in fits]
    legend_labels.extend(point.label for point in points)
    handles = [handle for _, handle in legend_handles + point_handles]

    ax.legend(
        labels=legend_labels,
        handles=handles,
        loc="lower left",
        frameon=False,
        fontsize=20,
        handlelength=1,
        borderpad=0.5,
        handletextpad=1,
        title_fontsize=24,
        # matplotlib puts a single scatter key at 3/8 of the key height, which
        # reads as off-centre once the marker sits on top of a filled patch
        scatteryoffsets=[0.5],
    )
    ax.text(
        0.05,
        0.95,
        rf"$\mathrm{{Marginalised}}\:{cl}\:\%\:\mathrm{{C.I.}}$",
        fontsize=24,
        transform=ax.transAxes,
        verticalalignment="top",
    )
    if any(stuck):
        # said once, in the legend cell: a bar or a cross is not a contour of
        # anything, it is where a fit held a coefficient it did not fit
        ax.text(
            0.05,
            0.85,
            r"$\mathrm{Bars\:and\:crosses\:mark\:fixed\:coefficients}$",
            fontsize=18,
            transform=ax.transAxes,
            verticalalignment="top",
        )

    return fig


# The two actions below take no *parameter* annotations, unlike the private
# core they wrap, and document their types in the docstring instead:
# reportengine reads a provider's parameter annotations as runtime type checks
# (`isinstance(value, annotation)` in `resourcebuilder.check_types`), and this
# module's `from __future__ import annotations` makes each of them a string
# that `isinstance` rejects outright — the run dies while the graph is built.
# The return annotation is never inspected, so it stays.
@figure
def plot_fits_posterior_contours(
    fits,
    params_to_plot=None,
    confidence_level=95,
    subplot_size=4,
    kde=None,
    show_sm=True,
    show_best_fit=False,
    reference_points=None,
    hatch=True,
) -> Figure:
    """Overlay the 2D marginalised confidence contours of every fit.

    Takes the whole ``fits`` list, so it is called bare in a report template —
    one figure, every fit drawn in every panel. Its per-fit counterpart is
    :func:`plot_posterior_contours`. The panels are the lower triangle of the
    coefficient matrix, over the coefficients free in at least one fit.

    A fit that did not float a coefficient of a panel is drawn where it held
    it: a segment spanning the other coefficient's confidence interval at that
    value, or a cross when it held both. The value is the one the runcard
    fixed the coefficient to, or its baseline (the SM point) for a coefficient
    the fit never declared.

    Parameters
    ----------
    fits : list of smefit.fit_result.Fit
        The previously run fits to overlay, each legend-labelled with the
        ``label`` of its ``fits`` entry (its name otherwise).
    params_to_plot : list of str, optional
        Restrict the panels to these coefficients, in this order. Every
        coefficient at least one fit floated by default; at least two must
        remain. A coefficient *no* fit floated can be named here too, and is
        then drawn stuck in all of them.
    confidence_level : float or list of two floats, optional
        Confidence level in percent, 95 by default. A list of two values
        draws the first as a dashed outline and fills the second. A segment
        drawn for a coefficient held fixed spans the equal-tailed percentiles
        of the other one's 1D posterior at the same level, the interval the
        bounds figures and the CL table report.
    subplot_size : float, optional
        Size in inches of a single panel.
    kde : bool or dict, optional
        Estimate the contours with a kernel density estimate instead of a
        Gaussian ellipse. Defaults to each fit's ``use_quad``, since
        quadratic corrections generally make a posterior non-Gaussian. A dict
        keyed by fit name sets it per fit.
    show_sm : bool, optional
        Mark the SM point, on by default. It sits at each coefficient's
        ``baseline_value`` in the runcard the fit was run with — the origin
        unless a coefficient was parametrised around a non-zero SM value.
    show_best_fit : bool, optional
        Mark the best-fit point of every fit, off by default — one marker per
        fit, including for a bimodal posterior, which still has a single
        maximum-likelihood point. Fits that record none are marked at their
        posterior means, and a segment is marked at the mean of the
        coefficient it does span.
    reference_points : list of ReferencePoint, optional
        Further points of coefficient space to mark, from the runcard's
        ``reference_points`` key. They come in addition to the SM marker, and
        each falls back to the baselines for the coefficients its ``values``
        does not name.
    hatch : bool, optional
        Texture every filled contour, one pattern per fit and per reference
        point, so they stay distinguishable in greyscale and to a reader who
        cannot separate the colours. On by default; False fills them flat.

    Raises
    ------
    ValueError
        If there is nothing to draw: no fits, a fit without joint posterior
        samples (run one coefficient at a time, or by a routine that stores
        none), no free coefficient in any fit, or fewer than two coefficients
        left.
    """
    return _posterior_contours(
        fits,
        params_to_plot=params_to_plot,
        confidence_level=confidence_level,
        subplot_size=subplot_size,
        kde=kde,
        show_sm=show_sm,
        show_best_fit=show_best_fit,
        reference_points=reference_points,
        hatch=hatch,
    )


@figure
def plot_posterior_contours(
    fit,
    params_to_plot=None,
    confidence_level=95,
    subplot_size=4,
    kde=None,
    show_sm=True,
    show_best_fit=False,
    reference_points=None,
    hatch=True,
) -> Figure:
    """Plot the 2D marginalised confidence contours of one fit.

    Takes a single ``fit``, so a runcard listing several under ``fits:`` gets
    one figure per fit — ``{@fits plot_posterior_contours@}``, or a ``with
    fits`` block, exactly like ``plot_posterior_correlations``. To overlay
    the fits in one figure instead, use :func:`plot_fits_posterior_contours`.

    Naming a coefficient the fit held fixed in ``params_to_plot`` adds its
    panels, showing where the fit was stuck: a segment at that value spanning
    the other coefficient's confidence interval, or a cross where two fixed
    coefficients meet.

    Parameters
    ----------
    fit : smefit.fit_result.Fit
        A previously run fit, loaded from disk.
    params_to_plot : list of str, optional
        Restrict the panels to these coefficients, in this order. The fit's
        free coefficients by default; at least two must remain. A coefficient
        the fit fixed can be named here too, and is then drawn at its value.
    confidence_level : float or list of two floats, optional
        Confidence level in percent, 95 by default. A list of two values
        draws the first as a dashed outline and fills the second. A segment
        drawn for a coefficient held fixed spans the equal-tailed percentiles
        of the other one's 1D posterior at the same level.
    subplot_size : float, optional
        Size in inches of a single panel.
    kde : bool or dict, optional
        Estimate the contours with a kernel density estimate instead of a
        Gaussian ellipse. Defaults to the fit's ``use_quad``. A dict keyed by
        fit name is accepted too, so a runcard can share one key between both
        contour actions.
    show_sm : bool, optional
        Mark the SM point, on by default. It sits at each coefficient's
        ``baseline_value`` in the runcard the fit was run with — the origin
        unless a coefficient was parametrised around a non-zero SM value.
    show_best_fit : bool, optional
        Mark the fit's best-fit point, off by default — one marker, including
        for a bimodal posterior, which still has a single maximum-likelihood
        point. A fit that records none is marked at its posterior mean.
    reference_points : list of ReferencePoint, optional
        Further points of coefficient space to mark, from the runcard's
        ``reference_points`` key, in addition to the SM marker.
    hatch : bool, optional
        Texture every filled contour, on by default. False fills them flat.

    Raises
    ------
    ValueError
        If the fit has no joint posterior samples to draw, or fewer than two
        coefficients are left to pair up.
    """
    return _posterior_contours(
        [fit],
        params_to_plot=params_to_plot,
        confidence_level=confidence_level,
        subplot_size=subplot_size,
        kde=kde,
        show_sm=show_sm,
        show_best_fit=show_best_fit,
        reference_points=reference_points,
        hatch=hatch,
    )


# ----------------------------------------------------------------------
# Posterior histograms — 1D marginalised distributions
# ----------------------------------------------------------------------


def _histogram_bins(values: np.ndarray, double_solution: bool, bins: Any | None) -> Any:
    """Bin edges for one coefficient's posterior.

    Freedman–Diaconis by default, which sets the width from the interquartile
    range and so follows the bulk of the posterior rather than its tails.

    A bimodal posterior gets one FD rule *per branch*, merged and sorted: the
    IQR of the two modes together spans the empty gap between them, and the
    single rule it yields is wide enough to smear each mode into a couple of
    bars. This is the one subtle piece of the old ``plot_posteriors``, and it
    is why ``double_solution`` matters to a histogram at all — the bars
    themselves are drawn from the unsplit samples, so nothing is hidden.

    ``bins`` overrides all of it and is passed to ``hist`` untouched, so a
    runcard can ask for a count, its own edges, or another rule by name.
    """
    if bins is not None:
        return bins
    if double_solution:
        solution1, solution2 = split_solution(values)
        return np.sort(
            np.concatenate(
                [
                    np.histogram_bin_edges(solution1, bins="fd"),
                    np.histogram_bin_edges(solution2, bins="fd"),
                ]
            )
        )
    return np.histogram_bin_edges(values, bins="fd")


def _posterior_histograms(
    fits: Sequence[Fit],
    params_to_plot: list[str] | str | None = None,
    double_solution: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    show_sm: bool = True,
    bins: Any | None = None,
    subplot_size: float = 4,
) -> Figure:
    """Draw the 1D marginalised posteriors of *fits* in one figure.

    The shared core of the two histogram actions below, which differ only in
    how many fits reportengine hands them — their docstrings carry the
    parameter documentation. One panel per coefficient, every fit overlaid in
    every panel, laid out on a near-square grid whose last cell is left to the
    legend.

    Individual (one-at-a-time) fits are accepted, unlike by the contour
    actions: a histogram is about one coefficient at a time, which is exactly
    what such a fit has to offer.
    """
    if not fits:
        raise ValueError("No fits to plot.")

    posteriors = []
    for fit in fits:
        samples = fit.fit_results.samples
        if not samples:
            raise ValueError(
                f"Fit '{fit.fit_name}' stored no posterior samples, so there "
                "is nothing to draw a histogram of."
            )
        posteriors.append(samples)

    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 22})
    rc("text", usetex=True)
    rc("text.latex", preamble=r"\usepackage{amssymb}")

    # one panel is enough: a histogram needs no pair
    coeffs = common_free_coefficients(fits, params_to_plot, min_count=1)

    colors = fit_colors(len(fits))
    # bimodality is declared per fit, since two fits of the same coefficient
    # need not both resolve a second solution
    doubles = per_fit_option(double_solution, fits, [[] for _ in fits])
    baselines = baseline_point(fits, coeffs) if show_sm else None
    limits = coeff_limits(fits, coeffs, include_points=baselines)
    coeff_labels = [coeff_info_latex.get(name, name) for name in coeffs]

    # the legend gets a cell of its own, so it is never drawn over a panel;
    # the grid is laid out for it as for one more coefficient
    n_cells = len(coeffs) + 1
    n_cols = int(np.ceil(np.sqrt(n_cells)))
    n_rows = int(np.ceil(n_cells / n_cols))
    fig = plt.figure(figsize=(n_cols * subplot_size, n_rows * subplot_size))

    for idx, name in enumerate(coeffs):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)

        for fit_idx, posterior in enumerate(posteriors):
            values = np.asarray(posterior[name], dtype=float)
            ax.hist(
                values,
                bins=_histogram_bins(values, name in doubles[fit_idx], bins),
                # densities, not counts: fits of different sample sizes are
                # being compared, and only the shapes are comparable
                density=True,
                color=colors[fit_idx],
                edgecolor="black",
                alpha=0.3,
            )

        if show_sm:
            assert baselines is not None  # set together with show_sm
            ax.axvline(baselines[name], color="k", linestyle="dashed", linewidth=1.5)

        # the coefficient names the panel from the inside: an axis label under
        # every panel of a large grid costs a row of height each time
        ax.text(0.05, 0.85, coeff_labels[idx], transform=ax.transAxes, fontsize=25)
        ax.set_xlim(*limits[name])
        # every panel is a different coefficient on its own range, so each
        # carries its own power — unlike the contour grid, where a column
        # shares one. The y axis is an unlabelled normalisation.
        compact_tick_labels(ax, axis="x")
        ax.tick_params(which="both", direction="in", labelsize=22.5)
        # the y-axis is a normalisation, not a quantity anybody reads off
        ax.tick_params(labelleft=False)

    legend_ax = fig.add_subplot(n_rows, n_cols, n_cells)
    legend_ax.axis("off")
    handles: list[Any] = [
        patches.Patch(facecolor=color, edgecolor="black", alpha=0.3) for color in colors
    ]
    labels = [fit.plot_label for fit in fits]
    if show_sm:
        handles.append(Line2D([], [], color="k", linestyle="dashed", linewidth=1.5))
        labels.append(r"$\mathrm{SM}$")
    legend_ax.legend(
        handles=handles,
        labels=labels,
        loc="upper left",
        frameon=False,
        fontsize=20,
        handlelength=1,
        borderpad=0.5,
        handletextpad=1,
    )

    return fig


@figure
def plot_fits_posterior_histograms(
    fits,
    params_to_plot=None,
    double_solution=None,
    show_sm=True,
    bins=None,
    subplot_size=4,
) -> Figure:
    """Overlay the 1D marginalised posteriors of every fit.

    Takes the whole ``fits`` list, so it is called bare in a report template —
    one figure, one panel per coefficient, every fit drawn in every panel. Its
    per-fit counterpart is :func:`plot_posterior_histograms`.

    Both accept a fit run one coefficient at a time, unlike the contour
    actions: a histogram reads one coefficient's posterior at a time, which is
    what an individual fit has. That is how the individual counterpart of a
    marginalised figure is produced — a ``fits:`` entry pointing at an
    ``individual_fits`` output, not an option here.

    Parameters
    ----------
    fits : list of smefit.fit_result.Fit
        The previously run fits to overlay, each legend-labelled with the
        ``label`` of its ``fits`` entry (its name otherwise).
    params_to_plot : list of str, optional
        Restrict the panels to these coefficients, in this order. All the
        coefficients the fits share by default.
    double_solution : list of str or dict, optional
        Coefficients whose posterior has two disjoint solutions. They are
        binned one branch at a time, so that a mode is not smeared into a
        couple of bars by a bin width set across the gap between them. A dict
        keyed by fit name sets the list per fit. As in the old pipeline this
        is declared, never detected: a posterior is bimodal because of the
        physics.
    show_sm : bool, optional
        Mark the SM with a dashed vertical line, on by default. It sits at
        each coefficient's ``baseline_value`` in the runcard the fit was run
        with — the origin unless a coefficient was parametrised around a
        non-zero SM value.
    bins : int or str or list, optional
        Binning, passed to ``matplotlib.axes.Axes.hist`` as it comes.
        Freedman–Diaconis per fit and coefficient by default.
    subplot_size : float, optional
        Size in inches of a single panel.

    Raises
    ------
    ValueError
        If there is nothing to draw: no fits, a fit that stored no posterior
        samples, or no shared coefficient left.
    """
    return _posterior_histograms(
        fits,
        params_to_plot=params_to_plot,
        double_solution=double_solution,
        show_sm=show_sm,
        bins=bins,
        subplot_size=subplot_size,
    )


@figure
def plot_posterior_histograms(
    fit,
    params_to_plot=None,
    double_solution=None,
    show_sm=True,
    bins=None,
    subplot_size=4,
) -> Figure:
    """Plot the 1D marginalised posteriors of one fit.

    Takes a single ``fit``, so a runcard listing several under ``fits:`` gets
    one figure per fit — ``{@fits plot_posterior_histograms@}``, or a ``with
    fits`` block. To overlay the fits in one figure instead, use
    :func:`plot_fits_posterior_histograms`, whose docstring describes the
    parameters, all shared.

    Raises
    ------
    ValueError
        If the fit stored no posterior samples, or no coefficient is left.
    """
    return _posterior_histograms(
        [fit],
        params_to_plot=params_to_plot,
        double_solution=double_solution,
        show_sm=show_sm,
        bins=bins,
        subplot_size=subplot_size,
    )


# ----------------------------------------------------------------------
# Coefficient bounds — central value and confidence intervals
# ----------------------------------------------------------------------

# How much wider the gap between two coefficients is than the gap between two
# fits of the same one, in the bounds plot. What makes the intervals of one
# coefficient read as a group rather than as neighbours.
_ROW_GAP_RATIO = 3.0

# Decade ticks of a symlog x-axis, at 1..9 times every power of ten either
# side of zero, as the old `plot_coeffs` drew them.
_SYMLOG_DECADES = np.concatenate([-np.logspace(-4, 2, 7), np.logspace(-4, 2, 7)])


def _symlog_minor_ticks(lin_thr: float) -> np.ndarray:
    """Minor tick positions of a symlog axis with threshold *lin_thr*.

    Ticks well inside the linear region are dropped: they crowd together
    around zero, where the scale no longer separates them.
    """
    ticks = np.concatenate([decade * np.arange(1, 10) for decade in _SYMLOG_DECADES])
    return ticks[np.abs(ticks) > lin_thr / 10]


def _coefficient_bounds(
    fits: Sequence[Fit],
    params_to_plot: list[str] | str | None = None,
    confidence_level: float | Sequence[float] = 95,
    double_solution: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    x_log: bool = False,
    lin_thr: float = 1e-2,
    x_min: float | None = None,
    x_max: float | None = None,
) -> Figure:
    """Draw the confidence intervals of *fits* one coefficient per row.

    The shared core of the two bounds actions below, which differ only in how
    many fits reportengine hands them — their docstrings carry the parameter
    documentation. One row per coefficient, top to bottom in the order asked
    for, every fit on its own offset within the row.

    Individual (one-at-a-time) fits are accepted, as by the histograms: a
    bound is about one coefficient at a time.
    """
    if not fits:
        raise ValueError("No fits to plot.")

    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 22})
    rc("text", usetex=True)
    rc("text.latex", preamble=r"\usepackage{amssymb}")

    coeffs = common_free_coefficients(fits, params_to_plot, min_count=1)

    # the wider interval is drawn thin, the narrower one thick on top of it,
    # so which way round the two levels are written does not matter
    levels = _confidence_levels(confidence_level)
    outer_cl = max(levels)
    inner_cl = min(levels) if len(levels) == 2 else None
    levels = [outer_cl] if inner_cl is None else [outer_cl, inner_cl]

    colors = fit_colors(len(fits))
    doubles = per_fit_option(double_solution, fits, [[] for _ in fits])
    bounds = [
        coeff_bounds(fit, coeffs, levels, double_solution=doubles[idx])
        for idx, fit in enumerate(fits)
    ]

    # One row per coefficient, first one at the top; the fits share the row,
    # spread over it so their intervals do not sit on top of one another.
    #
    # Rows are one apart, and the spread is set so that the gap between two
    # rows is _ROW_GAP_RATIO times the gap between two fits inside one: a
    # reader has to see at a glance which intervals belong to the same
    # coefficient, and with a fixed spread the two gaps close on each other as
    # fits are added until the grouping reads backwards.
    rows = np.arange(len(coeffs))[::-1]
    spread = (len(fits) - 1) / (len(fits) - 1 + _ROW_GAP_RATIO)
    shifts = np.linspace(spread / 2, -spread / 2, len(fits))

    fig, ax = plt.subplots(
        figsize=(10, max(3.0, len(coeffs) * (0.6 + 0.25 * len(fits))))
    )

    for coeff_idx, name in enumerate(coeffs):
        for fit_idx in range(len(fits)):
            solutions = bounds[fit_idx].get(name)
            if solutions is None:  # this fit never sampled it: leave a gap
                continue
            y = rows[coeff_idx] + shifts[fit_idx]
            # every solution of the coefficient goes on the same row —
            # explicitly, one interval per branch the split produced
            for solution in solutions:
                outer = solution[outer_cl]
                ax.errorbar(
                    x=outer.mid,
                    y=y,
                    xerr=[[outer.mid - outer.low], [outer.high - outer.mid]],
                    color=colors[fit_idx],
                    elinewidth=1,
                )
                if inner_cl is not None:
                    inner = solution[inner_cl]
                    ax.errorbar(
                        x=inner.mid,
                        y=y,
                        xerr=[[inner.mid - inner.low], [inner.high - inner.mid]],
                        color=colors[fit_idx],
                        elinewidth=3,
                        fmt=".",
                    )
                else:
                    ax.plot(outer.mid, y, ".", color=colors[fit_idx])

    ax.set_ylim(rows.min() - 1, rows.max() + 1)
    ax.set_yticks(rows, [coeff_info_latex.get(name, name) for name in coeffs])

    # the SM is not always the origin: a coefficient parametrised around a
    # non-zero baseline_value has its SM elsewhere, and the reference line has
    # to agree with the histograms and contours of the same fits. One line
    # across the figure while every coefficient shares a baseline — the usual
    # case — and one tick per row otherwise, since the rows then disagree.
    baselines = baseline_point(fits, coeffs)
    if len(set(baselines.values())) == 1:
        ax.axvline(
            next(iter(baselines.values())), ls="dashed", color="black", alpha=0.7
        )
    else:
        for coeff_idx, name in enumerate(coeffs):
            ax.plot(
                [baselines[name]] * 2,
                [rows[coeff_idx] - 0.5, rows[coeff_idx] + 0.5],
                ls="dashed",
                color="black",
                alpha=0.7,
            )

    if x_log:
        ax.set_xscale("symlog", linthresh=lin_thr)
        ax.set_xticks(_symlog_minor_ticks(lin_thr), minor=True)
    else:
        # x only: the rows are named after coefficients, and a numeric
        # formatter would replace those names with the positions they sit at
        compact_tick_labels(ax, axis="x")
    ax.grid(True, which="both", ls="dashed", axis="x", lw=0.5)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(r"$c_i/\Lambda^2\ ({\rm TeV}^{-2})$", fontsize=20)

    levels_title = (
        rf"${outer_cl:g}\:\%\:\mathrm{{C.I.}}$"
        if inner_cl is None
        else rf"${inner_cl:g}\:\%\:\mathrm{{and}}\:{outer_cl:g}\:\%\:\mathrm{{C.I.}}$"
    )
    ax.legend(
        handles=[
            Line2D([], [], color=color, marker=".", linewidth=3) for color in colors
        ],
        labels=[fit.plot_label for fit in fits],
        title=levels_title,
        loc="lower center",
        bbox_to_anchor=(0, 1.02, 1.0, 0.05),
        frameon=False,
        ncol=2,
    )

    return fig


@figure
def plot_fits_coefficient_bounds(
    fits,
    params_to_plot=None,
    confidence_level=95,
    double_solution=None,
    x_log=False,
    lin_thr=1e-2,
    x_min=None,
    x_max=None,
) -> Figure:
    """Overlay the coefficient bounds of every fit — central value and C.I.

    Takes the whole ``fits`` list, so it is called bare in a report template —
    one figure, one row per coefficient, every fit on its own offset within
    the row. Its per-fit counterpart is :func:`plot_coefficient_bounds`.

    Both accept a fit run one coefficient at a time, like the histogram
    actions: a bound reads one coefficient's posterior at a time, which is
    what an individual fit has.

    Parameters
    ----------
    fits : list of smefit.fit_result.Fit
        The previously run fits to overlay, each legend-labelled with the
        ``label`` of its ``fits`` entry (its name otherwise).
    params_to_plot : list of str, optional
        Restrict the rows to these coefficients, in this order — top to
        bottom. All the coefficients the fits share by default.
    confidence_level : float or list of two floats, optional
        Confidence level in percent, 95 by default. A list of two values
        draws both: the wider one as a thin bar, the narrower one thick over
        it, so their order does not matter.
    double_solution : list of str or dict, optional
        Coefficients whose posterior has two disjoint solutions. Each branch
        gets its own interval on the same row; without this, equal-tailed
        percentiles would span the empty gap between the modes and put the
        central value where the posterior has no mass. A dict keyed by fit
        name sets the list per fit.
    x_log : bool, optional
        Draw the x-axis on a symmetric log scale, off by default. Bounds run
        either side of zero, so a plain log scale cannot show them.
    lin_thr : float, optional
        Half-width of the linear region of that scale, ``1e-2`` by default.
        Ignored unless ``x_log``.
    x_min, x_max : float, optional
        Axis limits. Chosen from the intervals drawn by default.

    Raises
    ------
    ValueError
        If there is nothing to draw: no fits, a fit that stored no posterior
        samples, or no shared coefficient left.
    """
    return _coefficient_bounds(
        fits,
        params_to_plot=params_to_plot,
        confidence_level=confidence_level,
        double_solution=double_solution,
        x_log=x_log,
        lin_thr=lin_thr,
        x_min=x_min,
        x_max=x_max,
    )


@figure
def plot_coefficient_bounds(
    fit,
    params_to_plot=None,
    confidence_level=95,
    double_solution=None,
    x_log=False,
    lin_thr=1e-2,
    x_min=None,
    x_max=None,
) -> Figure:
    """Plot the coefficient bounds of one fit — central value and C.I.

    Takes a single ``fit``, so a runcard listing several under ``fits:`` gets
    one figure per fit — ``{@fits plot_coefficient_bounds@}``, or a ``with
    fits`` block. To overlay the fits in one figure instead, use
    :func:`plot_fits_coefficient_bounds`, whose docstring describes the
    parameters, all shared.

    Raises
    ------
    ValueError
        If the fit stored no posterior samples, or no coefficient is left.
    """
    return _coefficient_bounds(
        [fit],
        params_to_plot=params_to_plot,
        confidence_level=confidence_level,
        double_solution=double_solution,
        x_log=x_log,
        lin_thr=lin_thr,
        x_min=x_min,
        x_max=x_max,
    )


# ----------------------------------------------------------------------
# Mass reach — the scale a bound probes
# ----------------------------------------------------------------------

# Fraction of the space between two coefficients that the bars of one fill.
# The rest is what separates one coefficient's group of bars from the next.
_BAR_GROUP_WIDTH = 0.8


def _mass_reach(
    fits: Sequence[Fit],
    params_to_plot: list[str] | str | None = None,
    confidence_level: float = 95,
    full_interval: bool = False,
    y_log: bool = False,
) -> Figure:
    r"""Draw the :math:`\Lambda/\sqrt{c_i}` reach of *fits* as grouped bars.

    The shared core of the two reach actions below, which differ only in how
    many fits reportengine hands them — their docstrings carry the parameter
    documentation. One group of bars per coefficient, one bar per fit inside
    it.

    Individual (one-at-a-time) fits are accepted, as by the histograms and
    the bounds plot: a reach is read off one coefficient's interval. Which of
    the two a figure is about is which fit directory its ``fits:`` entry
    points at — a joint output or an ``individual_fits`` one — never an option
    here.
    """
    if not fits:
        raise ValueError("No fits to plot.")

    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 22})
    rc("text", usetex=True)
    rc("text.latex", preamble=r"\usepackage{amssymb}")

    coeffs = common_free_coefficients(fits, params_to_plot, min_count=1)
    colors = fit_colors(len(fits))

    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(coeffs) * len(fits)), 6))

    positions = np.arange(len(coeffs), dtype=float)
    width = _BAR_GROUP_WIDTH / len(fits)
    for fit_idx, fit in enumerate(fits):
        bounds = coeff_bounds(fit, coeffs, confidence_level)
        # a coefficient this fit never sampled, or did not constrain, becomes
        # a gap in its group rather than a missing or infinite bar
        reaches = [
            (
                mass_reach(bounds[name][0][confidence_level], full_interval)
                if name in bounds
                else float("nan")
            )
            for name in coeffs
        ]
        offset = (fit_idx - (len(fits) - 1) / 2) * width
        ax.bar(
            positions + offset,
            reaches,
            width=width,
            color=colors[fit_idx],
            label=fit.plot_label,
        )

    ax.set_xticks(positions, [coeff_info_latex.get(name, name) for name in coeffs])
    ax.set_ylabel(r"$\Lambda/\sqrt{c_i}\ ({\rm TeV})$", fontsize=20)
    if y_log:
        ax.set_yscale("log")  # its own formatter already writes the powers
    else:
        # y only: the groups are named after coefficients
        compact_tick_labels(ax, axis="y")
    ax.grid(True, which="both", ls="dashed", axis="y", lw=0.5)
    ax.set_axisbelow(True)  # the bars are the figure, the grid reads under them
    ax.legend(
        title=rf"${confidence_level:g}\:\%\:\mathrm{{C.I.}}$",
        loc="lower center",
        bbox_to_anchor=(0, 1.02, 1.0, 0.05),
        frameon=False,
        ncol=2,
    )

    return fig


@figure
def plot_fits_mass_reach(
    fits,
    params_to_plot=None,
    confidence_level=95,
    full_interval=False,
    y_log=False,
) -> Figure:
    r"""Overlay the mass reach of every fit, one group of bars per coefficient.

    Takes the whole ``fits`` list, so it is called bare in a report template.
    Its per-fit counterpart is :func:`plot_mass_reach`.

    The reach is :math:`\Lambda/\sqrt{c_i}` in TeV, the inverse square root of
    the bound on the coefficient — the scale that bound probes. It is read off
    the same percentile intervals as the bounds plot and table, so the three
    cannot disagree.

    Both actions accept a fit run one coefficient at a time. The marginalised
    and the individual figure are the same action pointed at different fit
    directories (a joint output and an ``individual_fits`` one), not two
    modes of one.

    Parameters
    ----------
    fits : list of smefit.fit_result.Fit
        The previously run fits to compare, each legend-labelled with the
        ``label`` of its ``fits`` entry (its name otherwise).
    params_to_plot : list of str, optional
        Restrict the groups to these coefficients, in this order. All the
        coefficients the fits share by default.
    confidence_level : float, optional
        Confidence level in percent of the interval the reach is read off, 95
        by default. A single value: a bar has one height.
    full_interval : bool, optional
        Take the bound to be the whole interval ``high - low`` rather than
        half of it. Off by default, so that a Gaussian posterior reproduces
        ``n_sigma * std``.
    y_log : bool, optional
        Draw the reach on a log scale, off by default. Reaches are positive,
        so this one is a plain log scale.

    Raises
    ------
    ValueError
        If there is nothing to draw: no fits, a fit that stored no posterior
        samples, or no shared coefficient left.
    """
    return _mass_reach(
        fits,
        params_to_plot=params_to_plot,
        confidence_level=confidence_level,
        full_interval=full_interval,
        y_log=y_log,
    )


@figure
def plot_mass_reach(
    fit,
    params_to_plot=None,
    confidence_level=95,
    full_interval=False,
    y_log=False,
) -> Figure:
    """Plot the mass reach of one fit, one bar per coefficient.

    Takes a single ``fit``, so a runcard listing several under ``fits:`` gets
    one figure per fit — ``{@fits plot_mass_reach@}``, or a ``with fits``
    block. To compare the fits in one figure instead, use
    :func:`plot_fits_mass_reach`, whose docstring describes the parameters,
    all shared.

    Raises
    ------
    ValueError
        If the fit stored no posterior samples, or no coefficient is left.
    """
    return _mass_reach(
        [fit],
        params_to_plot=params_to_plot,
        confidence_level=confidence_level,
        full_interval=full_interval,
        y_log=y_log,
    )
