"""
smefit.figures.py

Functions for creating figures and plots.
"""

import itertools
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from reportengine.figure import figure

from smefit.contours_2d import fit_colors, plot_contours
from smefit.op_to_latex import coeff_info_latex
from smefit.plot_utils import (
    best_fit_pair,
    coeff_limits,
    common_coefficients,
    default_labels,
    per_fit_option,
    require_joint_posterior,
    require_samples,
)

log = logging.getLogger(__name__)


def _plot_heatmap(matrix, coeff_names, source_names, vmin=None, vmax=None):
    """Render a (n_coeffs, n_sources) matrix as a heatmap and save to PDF.

    Parameters
    ----------
    matrix : array-like, shape (n_coeffs, n_sources)
    coeff_names : list of str
    source_names : list of str
    save_path : pathlib.Path
    vmin, vmax : float, optional
        Colour scale limits passed to imshow.
    """
    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 22})
    rc("text", usetex=True)
    rc("text.latex", preamble=r"\usepackage{amssymb}")

    matrix = np.array(matrix)
    n_coeffs, n_sources = matrix.shape

    fig_w = max(6, n_sources * 0.7)
    fig_h = max(3, n_coeffs * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticks(range(n_sources))
    ax.set_xticklabels(source_names, rotation=90, fontsize=14)

    coeff_labels = [coeff_info_latex.get(name, name) for name in coeff_names]
    ax.set_yticks(range(n_coeffs))
    ax.set_yticklabels(coeff_labels, fontsize=14)

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")
    ax.imshow(
        np.ma.masked_equal(matrix, 0.0), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax
    )

    ax.set_xticks(np.arange(-0.5, n_sources, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_coeffs, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    color_threshold = vmax * 0.6
    for i in range(n_coeffs):
        for j in range(n_sources):
            val = matrix[i, j]
            if val == 0.0:
                continue
            color = "white" if val > color_threshold else "black"
            ax.text(
                j, i, f"{val:.1f}", ha="center", va="center", fontsize=10, color=color
            )

    return fig


@figure
def plot_fisher_diagonals_heatmap(fisher_diagonals_normalised):
    """Plot the Fisher diagonals matrix as a heatmap.

    Parameters
    ----------
    fisher_diagonals_normalised : pd.DataFrame
        Index = coeff_names, columns = source_names.
    """
    fd = fisher_diagonals_normalised
    return _plot_heatmap(
        fd.values * 100,
        fd.index.tolist(),
        fd.columns.tolist(),
        vmin=0,
        vmax=100,
    )


# ----------------------------------------------------------------------
# Corner plot — 2D marginalised confidence contours
# ----------------------------------------------------------------------


def corner_plot(
    fit_results,
    confidence_level=95,
    dofs_show=None,
    subplot_size=4,
    kde=None,
    double_solution=None,
    labels=None,
    show_sm=True,
    show_best_fit=False,
):
    """Plot the 2D marginalised confidence contours of one or more fits.

    The panels form the lower triangle of the coefficient matrix: the panel in
    row ``j - 1`` and column ``i`` shows coefficient ``i`` on the x-axis and
    coefficient ``j`` on the y-axis.

    Options that describe a single fit rather than the plot as a whole (``kde``,
    ``double_solution``, ``labels``) accept either one value applied to every
    fit, or a dict keyed by fit name — the ``name`` of a fits entry in the
    runcard, or the sampler name for a fit run in the same session.

    Parameters
    ----------
    fit_results : list of Fit
        One entry per fit to overlay.
    confidence_level : float or list of float, optional
        Confidence level in percent, 95 by default. When a list of two values
        is given the first one is drawn dashed and the second one filled.
    dofs_show : list of str, optional
        Coefficients to include. By default all coefficients fitted in every
        fit are shown.
    subplot_size : float, optional
        Size in inches of a single panel.
    kde : bool or dict, optional
        Estimate the contours with a kernel density estimate instead of a
        Gaussian ellipse. Defaults to each fit's ``use_quad``, since quadratic
        fits generally have non-Gaussian posteriors.
    double_solution : list or dict, optional
        Coefficients with a double (disjoint) solution.
    labels : str or dict, optional
        Legend label per fit, overriding the ``label`` of its ``fits`` entry.
        Defaults to that label, or to the LaTeX-escaped fit name without one.
    show_sm : bool, optional
        Mark the SM point at the origin, on by default.
    show_best_fit : bool, optional
        Mark the best-fit point of every fit, off by default. Falls back to the
        posterior mean for fits that do not record a best-fit point.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not fit_results:
        raise ValueError("No fits to plot.")
    require_samples(fit_results)
    require_joint_posterior(fit_results)

    coeffs = common_coefficients(fit_results, dofs_show)
    n_par = len(coeffs)
    if n_par < 2:
        raise ValueError(
            f"A corner plot needs at least 2 coefficients, {n_par} given: {coeffs}."
        )

    if isinstance(confidence_level, (list, tuple)):
        dashed_cl, cl = confidence_level
    else:
        dashed_cl, cl = None, confidence_level

    colors = fit_colors(len(fit_results))
    kdes = per_fit_option(kde, fit_results, [f.use_quad for f in fit_results])
    double_solutions = per_fit_option(
        double_solution, fit_results, [[] for _ in fit_results]
    )
    legend_labels = per_fit_option(labels, fit_results, default_labels(fit_results))
    limits = coeff_limits(fit_results, coeffs, include_sm=show_sm)
    coeff_labels = [coeff_info_latex.get(name, name) for name in coeffs]

    n_cells = max(n_par - 1, 1)
    fig = plt.figure(figsize=(n_cells * subplot_size, n_cells * subplot_size))
    grid = plt.GridSpec(n_cells, n_cells, hspace=0.1, wspace=0.1)

    hndls_all = []
    for i, j in itertools.combinations(range(n_par), 2):
        c1, c2 = coeffs[i], coeffs[j]
        ax = fig.add_subplot(grid[j - 1, i])

        hndls_all = []
        for clr_idx, fit_result in enumerate(fit_results):
            hndls_all.append(
                plot_contours(
                    ax,
                    fit_result.samples_frame,
                    coeff1=c1,
                    coeff2=c2,
                    kde=kdes[clr_idx],
                    color=colors[clr_idx],
                    confidence_level=cl,
                    dashed_confidence_level=dashed_cl,
                    double_solution=double_solutions[clr_idx],
                    show_best_fit=show_best_fit,
                    best_fit=best_fit_pair(fit_result, c1, c2),
                )
            )

        if show_sm:
            hndls_all.append(ax.scatter(0, 0, c="k", marker="+", s=50, zorder=10))

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
    # single panel of a two-coefficient plot
    if n_par > 2:
        ax = fig.add_subplot(grid[0, 1:])
        ax.axis("off")

    if show_sm:
        legend_labels = legend_labels + [r"$\mathrm{SM}$"]

    ax.legend(
        labels=legend_labels,
        handles=hndls_all,
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


@figure
def plot_corner(fit_results, corner_settings=None):
    """Corner plot of the fits declared in the ``fits`` runcard key.

    Parameters
    ----------
    fit_results : list of Fit
        Produced by ``produce_fit_results`` from the ``fits`` runcard key.
    corner_settings : dict, optional
        Settings parsed by ``parse_corner_settings``.
    """
    return corner_plot(fit_results, **(corner_settings or {}))


@figure
def plot_corner_ultranest(ultranest_fit, corner_settings=None):
    """Corner plot of the UltraNest fit run in this runcard."""
    return corner_plot([ultranest_fit], **(corner_settings or {}))


@figure
def plot_corner_blackjax(blackjax_fit, corner_settings=None):
    """Corner plot of the BlackJAX fit run in this runcard."""
    return corner_plot([blackjax_fit], **(corner_settings or {}))


@figure
def plot_corner_hessian(hessian_fit, corner_settings=None):
    """Corner plot of the Hessian fit run in this runcard."""
    return corner_plot([hessian_fit], **(corner_settings or {}))


@figure
def plot_corner_analytic(analytic_fit, corner_settings=None):
    """Corner plot of the analytic fit run in this runcard."""
    return corner_plot([analytic_fit], **(corner_settings or {}))
