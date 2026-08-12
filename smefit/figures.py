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
from matplotlib import rc
from reportengine.figure import figure

from smefit.contours_2d import fit_colors, plot_contours
from smefit.fit_result import FitResult
from smefit.op_to_latex import coeff_info_latex
from smefit.plot_utils import (
    best_fit_pair,
    coeff_limits,
    common_free_coefficients,
    per_fit_option,
    select_params,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.figure import Figure

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
        Index = coeff_names, columns = source_names.
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
        fd.columns.tolist(),
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


def _posterior_contours(
    fits: Sequence[Fit],
    params_to_plot: list[str] | str | None = None,
    confidence_level: float | Sequence[float] = 95,
    subplot_size: float = 4,
    kde: bool | Mapping[str, bool] | None = None,
    double_solution: list[str] | Mapping[str, list[str]] | None = None,
    show_sm: bool = True,
    show_best_fit: bool = False,
) -> Figure:
    """Draw the pairwise 2D confidence contours of *fits* in one figure.

    The shared core of the two contour actions below, which differ only in
    how many fits reportengine hands them — their docstrings carry the
    parameter documentation. The panels form the lower triangle of the
    coefficient matrix: the panel in row ``j - 1`` and column ``i`` shows
    coefficient ``i`` on the x-axis and coefficient ``j`` on the y-axis, and
    every fit is drawn in every panel.
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

    coeffs = common_free_coefficients(fits, params_to_plot)
    n_par = len(coeffs)

    if isinstance(confidence_level, (list, tuple)):
        dashed_cl, cl = confidence_level
    else:
        dashed_cl, cl = None, confidence_level

    colors = fit_colors(len(fits))
    kdes = per_fit_option(kde, fits, [fit.use_quad for fit in fits])
    double_solutions = per_fit_option(double_solution, fits, [[] for _ in fits])
    limits = coeff_limits(fits, coeffs, include_sm=show_sm)
    coeff_labels = [coeff_info_latex.get(name, name) for name in coeffs]

    n_cells = n_par - 1  # pairwise panels: the lower triangle has one row less
    fig = plt.figure(figsize=(n_cells * subplot_size, n_cells * subplot_size))
    grid = plt.GridSpec(n_cells, n_cells, hspace=0.1, wspace=0.1)

    # Every panel draws the same fits in the same colours, so the handles of
    # any one panel serve as the legend's; the last panel's are kept.
    handles: list[Any] = []
    for i, j in itertools.combinations(range(n_par), 2):
        c1, c2 = coeffs[i], coeffs[j]
        ax = fig.add_subplot(grid[j - 1, i])

        handles = []
        for idx, fit in enumerate(fits):
            handles.append(
                plot_contours(
                    ax,
                    posteriors[idx],
                    coeff1=c1,
                    coeff2=c2,
                    kde=kdes[idx],
                    color=colors[idx],
                    confidence_level=cl,
                    dashed_confidence_level=dashed_cl,
                    double_solution=double_solutions[idx],
                    show_best_fit=show_best_fit,
                    best_fit=best_fit_pair(fit, c1, c2),
                )
            )
        if show_sm:
            handles.append(ax.scatter(0, 0, c="k", marker="+", s=50, zorder=10))

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

    # legend: in the free upper-right corner when there is one, else in the
    # single panel of a two-coefficient figure
    if n_par > 2:
        ax = fig.add_subplot(grid[0, 1:])
        ax.axis("off")

    legend_labels = [fit.plot_label for fit in fits]
    if show_sm:
        legend_labels.append(r"$\mathrm{SM}$")

    ax.legend(
        labels=legend_labels,
        handles=handles,
        loc="lower left" if n_par > 2 else "best",
        frameon=False,
        fontsize=20,
        handlelength=1,
        borderpad=0.5,
        handletextpad=1,
        title_fontsize=24,
    )
    ax.text(
        0.05,
        0.95,
        rf"$\mathrm{{Marginalised}}\:{cl}\:\%\:\mathrm{{C.I.}}$",
        fontsize=24,
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
    double_solution=None,
    show_sm=True,
    show_best_fit=False,
) -> Figure:
    """Overlay the 2D marginalised confidence contours of every fit.

    Takes the whole ``fits`` list, so it is called bare in a report template —
    one figure, every fit drawn in every panel. Its per-fit counterpart is
    :func:`plot_posterior_contours`. The panels are the lower triangle of the
    coefficient matrix, over the free coefficients every fit shares.

    Parameters
    ----------
    fits : list of smefit.fit_result.Fit
        The previously run fits to overlay, each legend-labelled with the
        ``label`` of its ``fits`` entry (its name otherwise).
    params_to_plot : list of str, optional
        Restrict the panels to these coefficients, in this order. All the
        coefficients the fits share by default; at least two must remain.
    confidence_level : float or list of two floats, optional
        Confidence level in percent, 95 by default. A list of two values
        draws the first as a dashed outline and fills the second.
    subplot_size : float, optional
        Size in inches of a single panel.
    kde : bool or dict, optional
        Estimate the contours with a kernel density estimate instead of a
        Gaussian ellipse. Defaults to each fit's ``use_quad``, since
        quadratic corrections generally make a posterior non-Gaussian. A dict
        keyed by fit name sets it per fit.
    double_solution : list of str or dict, optional
        Coefficients whose posterior has two disjoint modes, marked with one
        best-fit point per mode (KDE mode only). A dict keyed by fit name
        sets it per fit.
    show_sm : bool, optional
        Mark the SM point at the origin, on by default.
    show_best_fit : bool, optional
        Mark the best-fit point of every fit, off by default. Fits that do
        not record one are marked at their posterior means.

    Raises
    ------
    ValueError
        If there is nothing to draw: no fits, a fit without joint posterior
        samples (run one coefficient at a time, or by a routine that stores
        none), or fewer than two shared coefficients left.
    """
    return _posterior_contours(
        fits,
        params_to_plot=params_to_plot,
        confidence_level=confidence_level,
        subplot_size=subplot_size,
        kde=kde,
        double_solution=double_solution,
        show_sm=show_sm,
        show_best_fit=show_best_fit,
    )


@figure
def plot_posterior_contours(
    fit,
    params_to_plot=None,
    confidence_level=95,
    subplot_size=4,
    kde=None,
    double_solution=None,
    show_sm=True,
    show_best_fit=False,
) -> Figure:
    """Plot the 2D marginalised confidence contours of one fit.

    Takes a single ``fit``, so a runcard listing several under ``fits:`` gets
    one figure per fit — ``{@fits plot_posterior_contours@}``, or a ``with
    fits`` block, exactly like ``plot_posterior_correlations``. To overlay
    the fits in one figure instead, use :func:`plot_fits_posterior_contours`.

    Parameters
    ----------
    fit : smefit.fit_result.Fit
        A previously run fit, loaded from disk.
    params_to_plot : list of str, optional
        Restrict the panels to these coefficients, in this order. All the
        fit's free coefficients by default; at least two must remain.
    confidence_level : float or list of two floats, optional
        Confidence level in percent, 95 by default. A list of two values
        draws the first as a dashed outline and fills the second.
    subplot_size : float, optional
        Size in inches of a single panel.
    kde : bool or dict, optional
        Estimate the contours with a kernel density estimate instead of a
        Gaussian ellipse. Defaults to the fit's ``use_quad``. A dict keyed by
        fit name is accepted too, so a runcard can share one key between both
        contour actions.
    double_solution : list of str or dict, optional
        Coefficients whose posterior has two disjoint modes, marked with one
        best-fit point per mode (KDE mode only). Also accepts a dict keyed by
        fit name.
    show_sm : bool, optional
        Mark the SM point at the origin, on by default.
    show_best_fit : bool, optional
        Mark the fit's best-fit point, off by default. A fit that does not
        record one is marked at its posterior mean.

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
        double_solution=double_solution,
        show_sm=show_sm,
        show_best_fit=show_best_fit,
    )
