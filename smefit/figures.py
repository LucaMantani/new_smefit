"""
smefit.figures.py

Functions for creating figures and plots.
"""

import itertools
import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
from reportengine.figure import figure, figuregen

from smefit.credible_intervals import bounds_settings
from smefit.plot_utils import select_params, set_plot_style

log = logging.getLogger(__name__)

# Resolution of the grid a 2D KDE is drawn on, how far past the outermost
# samples it extends (in bandwidths, so the tails are not truncated), and the
# sample count above which a posterior is thinned before the estimate: its cost
# grows linearly with the samples while the estimate barely moves past a few
# thousand.
_KDE_GRIDSIZE = 200
_KDE_CUT = 3
_KDE_MAX_SAMPLES = 5000

# Hatch patterns handed out alongside the colour cycle, one per fit, so a filled
# contour stays distinguishable in print and in greyscale. A hatch is always its
# own unfilled layer over the fill: the PDF backend silently drops the hatch of a
# patch that is both filled and hatched.
_HATCH_CYCLE = ("///", "\\\\\\", "---", "|||", "...", "+++", "ooo")


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
        Row and column labels, drawn as given: the callers have already
        turned names into labels.
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
    set_plot_style()

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

    ax.set_yticks(range(n_coeffs))
    ax.set_yticklabels(coeff_names, fontsize=14)

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


@figuregen
def plot_chi2_scan(individual_chi2_scans, latex_label):
    """Plot the 1D chi2 scan for each free coefficient.

    Yields one ``(figure, coeff_name)`` per scanned coefficient, with the
    coefficient value on the x axis and chi2 on the y axis.

    Parameters
    ----------
    individual_chi2_scans : list[dict]
        Each entry maps ``{coeff_name: {"points": [...], "chi2": [...]}}``.
    latex_label : Callable[[str], str]
        Writes a name in LaTeX (the ``latex_label`` resource).
    """
    set_plot_style()

    results = {k: v for d in individual_chi2_scans for k, v in d.items()}
    for name, data in results.items():
        label = latex_label(name)
        fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
        ax.plot(data["points"], data["chi2"], "-o", color="C0", markersize=4)
        ax.set_xlabel(label)
        ax.set_ylabel(r"$\chi^2$")
        ax.grid(alpha=0.3)
        yield fig, name


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
def plot_pca_components_heatmap(
    pca_components,
    cmap="RdBu_r",
    value_fmt="{:.2f}",
    colorbar=True,
):
    """Plot the principal-direction weights as a heatmap.

    Reading it: a column is one principal direction, and the coloured cells say
    which coefficients it is made of. The leftmost columns are the directions
    the data pin down; the rightmost are the flat ones, and the coefficients
    lit up there are the combinations the fit cannot resolve.

    Parameters
    ----------
    pca_components : pd.DataFrame
        Index = coeff_names, columns = PC1..PCn.
    cmap : str, optional
        Colormap. Diverging by default: eigenvector components run either way
        about zero, and the relative sign of two entries is what says whether a
        direction is a sum or a difference of them.
    value_fmt : str, optional
        Format of the per-cell annotation.
    colorbar : bool, optional
        Whether to draw the colour scale alongside.
    """
    comps = pca_components
    return _plot_heatmap(
        comps.values,
        comps.index.tolist(),
        comps.columns.tolist(),
        vmin=-1,
        vmax=1,
        mask_zeros=False,
        cmap=cmap,
        value_fmt=value_fmt,
        colorbar=colorbar,
    )


@figure
def plot_pca_spectrum(pca, latex_label):
    """Plot the eigenvalue spectrum, with the flat-direction threshold marked.

    The one picture of the whole analysis: bars falling below the dashed line
    are the directions the data do not constrain.

    Parameters
    ----------
    pca : smefit.pca.PCA
    latex_label : Callable[[str], str]
        Writes a name in LaTeX (the ``latex_label`` resource).
    """
    set_plot_style()

    ratios = np.array(pca.eigenvalue_ratios, dtype=float)
    index = np.arange(1, pca.n_components + 1)
    flat = np.array(pca.flat_mask)

    # A log axis has no zero to grow the bars from, so the bottom of the axis
    # plays that role: every bar starts on it, and its length is how many
    # decades the direction stands above it. That floor follows the spectrum
    # rather than the threshold — the threshold is far below any physical scale
    # by design, and anchoring the axis to it would spend a dozen empty decades
    # on a plot whose bars all sit at the top. Only when something is actually
    # flagged flat does the floor drop below it, so that the dashed line and the
    # bars it condemns are in view together.
    positive = ratios[ratios > 0]
    lowest = positive.min() if positive.size else pca.threshold
    if flat.any():
        lowest = min(lowest, pca.threshold)
    floor = lowest / 10
    shown = np.clip(ratios, floor, None)
    heights = shown - floor

    fig, ax = plt.subplots(figsize=(max(6, pca.n_components * 0.4), 5))
    ax.set_axisbelow(True)
    ax.bar(
        index[~flat],
        heights[~flat],
        bottom=floor,
        width=0.7,
        color="C0",
        label="constrained",
    )
    if flat.any():
        ax.bar(
            index[flat],
            heights[flat],
            bottom=floor,
            width=0.7,
            color="C3",
            label="flat",
        )
        # A direction with no curvature at all sits on the floor, leaving a bar
        # of zero length; mark it with a downward triangle just above the
        # baseline so it is still read as present and off the bottom of the
        # scale.
        off_scale = flat & (ratios <= floor)
        if off_scale.any():
            ax.scatter(
                index[off_scale],
                np.full(int(off_scale.sum()), floor * 2.0),
                s=60,
                color="C3",
                marker="v",
                zorder=3,
            )
    # Below the bottom of the view when nothing is flat, which is the right
    # reading: there is then no direction for the threshold to separate.
    ax.axhline(pca.threshold, color="black", linestyle="--", linewidth=1)

    ax.set_yscale("log")
    # exactly the floor, so the bars sit on the bottom axis rather than hanging
    # above it
    ax.set_ylim(floor, 2.0)
    ax.set_xticks(index)
    ax.set_xticklabels(
        [latex_label(name) for name in pca.component_names], rotation=90, fontsize=12
    )
    ax.set_ylabel(r"$\lambda_i / \lambda_{\rm max}$", fontsize=16)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=12)
    fig.tight_layout()
    return fig


@figure
def plot_posterior_correlations(
    fit,
    latex_label,
    params_to_plot=None,
    cmap="RdBu_r",
    value_fmt="{:.2f}",
    colorbar=True,
):
    """Plot the posterior correlations of one fit's free coefficients.

    Takes a single ``fit``, so a runcard listing several under ``fits:`` gets
    one heatmap per fit, each headed with the fit it is drawn from.

    Parameters
    ----------
    fit : smefit.fit_result.Fit
        A previously run fit, loaded from disk.
    latex_label : Callable[[str], str]
        Writes a name in LaTeX (the ``latex_label`` resource).
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
    labels = [latex_label(name) for name in selected]
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


def _coefficient_bounds(
    fits,
    latex_label,
    params_to_plot,
    bounds_levels,
    interval_types,
    x_log,
    lin_thr,
    x_min,
    x_max,
):
    """The shared core of :func:`plot_coefficient_bounds` and
    :func:`plot_coefficient_bounds_comparison`.

    One figure per interval type, yielded as ``(figure, interval_type)`` so
    each is saved under its own suffix. One row per coefficient, top to
    bottom; every fit on its own offset within the row, one colour each; every
    level as a bar of its own width, the widest interval thinnest so the
    narrower ones read on top of it.
    """
    if not fits:
        raise ValueError("fits is empty: list at least one fit under 'fits:'.")
    bounds_levels, interval_types = bounds_settings(bounds_levels, interval_types)
    # widest first: drawn first and thinnest, under the narrower ones
    levels = sorted(set(bounds_levels), reverse=True)

    # the union over the fits, in order of first appearance: a coefficient one
    # fit did not sample is a gap in its row, not an error. Chosen before any
    # bound is computed, so the coefficients left out cost nothing.
    names = list(dict.fromkeys(name for fit in fits for name in fit.bounded_parameters))
    context = ", ".join(fit.fit_name for fit in fits)
    names = select_params(names, params_to_plot, context=context)

    for interval_type in interval_types:
        bounds = [
            {
                level: fit.confidence_bounds(level, interval_type, params=names)
                for level in levels
            }
            for fit in fits
        ]

        set_plot_style()
        fig, ax = plt.subplots(
            figsize=(10, max(3.0, len(names) * (0.5 + 0.2 * len(fits))))
        )

        rows = np.arange(len(names))[::-1]
        # the fits of one row fill at most 60% of it, so the gap between two
        # coefficients stays wider than the gap between two fits of the same one
        spread = 0.6 * (len(fits) - 1) / len(fits)
        shifts = np.linspace(spread / 2, -spread / 2, len(fits))
        widths = np.linspace(1.0, 4.0, len(levels)) if len(levels) > 1 else [2.0]

        for fit_idx, per_fit in enumerate(bounds):
            color = f"C{fit_idx}"
            for row, name in zip(rows, names):
                y = row + shifts[fit_idx]
                for level, width in zip(levels, widths):
                    # every piece of the region on the same row: a multimodal
                    # posterior is drawn as the disjoint intervals it has
                    for low, high in per_fit[level].get(name, []):
                        ax.hlines(y, low, high, color=color, lw=width)

        ax.axvline(0.0, ls="dashed", color="black", alpha=0.7)
        ax.set_yticks(rows, [latex_label(name) for name in names])
        ax.set_ylim(-0.75, len(names) - 0.25)
        if x_log:
            # bounds run either side of zero, which a plain log scale cannot show
            ax.set_xscale("symlog", linthresh=lin_thr)
        ax.set_xlim(x_min, x_max)
        ax.grid(True, which="both", ls="dashed", axis="x", lw=0.5)
        ax.set_xlabel(r"$c_i/\Lambda^2\ ({\rm TeV}^{-2})$", fontsize=20)

        handles = [Line2D([], [], color=f"C{i}", lw=3) for i in range(len(fits))]
        labels = [fit.plot_label for fit in fits]
        handles.append(Line2D([], [], color="black", ls="dashed", alpha=0.7))
        labels.append(r"$\mathrm{SM}$")
        for level, width in zip(levels, widths):
            handles.append(Line2D([], [], color="grey", lw=width))
            labels.append(
                rf"${level:.10g}\%\ \mathrm{{CL}}\ (\mathrm{{{interval_type.upper()}}})$"
            )
        ax.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0, 1.02, 1.0, 0.05),
            frameon=False,
            fontsize=14,
            ncol=2,
        )
        yield fig, interval_type


@figuregen
def plot_coefficient_bounds(
    fit,
    latex_label,
    params_to_plot=None,
    bounds_levels=None,
    interval_types=None,
    x_log=False,
    lin_thr=1e-2,
    x_min=None,
    x_max=None,
):
    """Plot the confidence bounds of one fit, one coefficient per row.

    Parameters
    ----------
    fit : smefit.fit_result.Fit
        A previously run fit, loaded from disk.
    latex_label : Callable[[str], str]
        Writes a name in LaTeX (the ``latex_label`` resource).
    params_to_plot : list of str, optional
        Restrict the rows to these coefficients, in this order, top to bottom.
        All of them by default.
    bounds_levels : float or list of float, optional
        The confidence levels drawn, in percent, 68 and 95 by default — the
        same key the bounds tables read. Each is a bar of its own width, the
        widest interval thinnest.
    interval_types : str or list of str, optional
        The credible intervals drawn, among those
        :meth:`Fit.confidence_bounds` accepts: ``eti`` (the default) or
        ``hdi`` — the same key the bounds tables read, so a report quoting
        both shows the same intervals. One figure per type, each yielded
        with the type as its suffix. A region in several pieces is drawn as
        all of them.
    x_log : bool, optional
        Draw the x-axis on a symmetric log scale, off by default.
    lin_thr : float, optional
        Half-width of the linear region of that scale, ``1e-2`` by default.
        Ignored unless ``x_log``.
    x_min, x_max : float, optional
        Axis limits, chosen from the intervals drawn by default.

    """
    return _coefficient_bounds(
        [fit],
        latex_label,
        params_to_plot,
        bounds_levels,
        interval_types,
        x_log,
        lin_thr,
        x_min,
        x_max,
    )


@figuregen
def plot_coefficient_bounds_comparison(
    fits,
    latex_label,
    params_to_plot=None,
    bounds_levels=None,
    interval_types=None,
    x_log=False,
    lin_thr=1e-2,
    x_min=None,
    x_max=None,
):
    """Overlay the confidence bounds of several fits, one coefficient per row.

    One figure for the whole ``fits`` list, every fit on its own offset within
    a row and legend-labelled with the ``label`` of its ``fits`` entry (its
    name otherwise). The rows are the union of the fits' coefficients; one a
    fit did not sample is a gap for that fit. The parameters are those of
    :func:`plot_coefficient_bounds`.

    Raises
    ------
    ValueError
        If ``fits`` is empty, or for any reason :func:`plot_coefficient_bounds`
        raises.
    """
    return _coefficient_bounds(
        fits,
        latex_label,
        params_to_plot,
        bounds_levels,
        interval_types,
        x_log,
        lin_thr,
        x_min,
        x_max,
    )


def _kde_grid(x, y, bw_adjust=1.5):
    """Evaluate a 2D Gaussian KDE of the samples ``(x, y)`` on a regular grid.

    Returns ``(xx, yy, density, sample_density)``: the first three are the
    ``(_KDE_GRIDSIZE, _KDE_GRIDSIZE)`` grid and the density on it, which is what
    gets drawn; ``sample_density`` is the same KDE at the (thinned) samples,
    which is what :func:`_density_level` reads the contour level off.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # checked before thinning: a common stride can erase a small difference in
    # length (6000 and 5999 both thin to 3000), silently pairing up samples
    # from different draws
    if x.size != y.size:
        raise ValueError(f"x and y must be the same size, got {x.size} and {y.size}.")

    if x.size > _KDE_MAX_SAMPLES:
        stride = int(np.ceil(x.size / _KDE_MAX_SAMPLES))
        x, y = x[::stride], y[::stride]

    kde = scipy.stats.gaussian_kde(np.vstack([x, y]))
    kde.set_bandwidth(kde.factor * bw_adjust)

    bw_x = float(np.sqrt(kde.covariance[0, 0]))
    bw_y = float(np.sqrt(kde.covariance[1, 1]))
    x_grid = np.linspace(
        x.min() - _KDE_CUT * bw_x, x.max() + _KDE_CUT * bw_x, _KDE_GRIDSIZE
    )
    y_grid = np.linspace(
        y.min() - _KDE_CUT * bw_y, y.max() + _KDE_CUT * bw_y, _KDE_GRIDSIZE
    )
    xx, yy = np.meshgrid(x_grid, y_grid)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    sample_density = kde(np.vstack([x, y]))
    return xx, yy, density, sample_density


def _density_level(sample_density, level):
    """The iso-density value enclosing ``level`` percent of the posterior.

    It is the ``1 - level`` quantile of the density *at the samples*, so
    ``level`` percent of them lie inside the contour by construction. Reading
    it off the grid instead — sorting the grid densities and cutting where
    their cumulative sum reaches ``level`` — systematically over-covers.
    """
    return float(np.quantile(sample_density, 1.0 - level / 100.0))


def _confidence_ellipse(x, y, level):
    """The ``level`` percent ellipse of the samples ``(x, y)``, not yet drawn.

    The iso-contour of the Gaussian with the samples' covariance, centred on
    their median: each half-axis is ``sqrt(chi2_2(level) * eigenvalue)``.
    The caller styles it and adds it to an axes.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError(f"x and y must be the same size, got {x.size} and {y.size}.")

    # symmetric: eigh keeps everything real and sorts the eigenvalues ascending
    eig_val, eig_vec = np.linalg.eigh(np.cov(x, y))
    major = eig_vec[:, -1]
    angle = np.degrees(np.arctan2(major[1], major[0]))
    chi2_qnt = scipy.stats.chi2.ppf(level / 100.0, 2)
    return Ellipse(
        (float(np.median(x)), float(np.median(y))),
        width=2 * np.sqrt(chi2_qnt * eig_val[-1]),
        height=2 * np.sqrt(chi2_qnt * eig_val[-2]),
        angle=float(angle),
    )


def _draw_pair(ax, x, y, color, levels, kde, hatch):
    """Draw one fit's 2D posterior of the samples ``(x, y)`` on ``ax``.

    ``levels`` are in percent, widest first: the widest region is filled
    (``alpha=0.3``, the ``hatch`` layer over it when one is given, a solid
    outline), every narrower one a dashed outline. With ``kde`` the regions
    are iso-density contours of one KDE evaluation shared by every level,
    otherwise Gaussian ellipses. Returns the fit's legend handle: a tuple of
    patches, overlaid by the legend.
    """
    filled, *dashed = levels

    if kde:
        xx, yy, density, sample_density = _kde_grid(x, y)

        def region(level, **kwargs):
            low = _density_level(sample_density, level)
            # the level comes from the samples, the fill from the grid, which
            # need not resolve the very peak: nudge the top past the level so
            # contourf's levels always increase
            top = float(np.nextafter(max(float(density.max()), low), np.inf))
            return ax.contourf(xx, yy, density, levels=[low, top], **kwargs)

        def outline(level, **kwargs):
            low = _density_level(sample_density, level)
            return ax.contour(xx, yy, density, levels=[low], colors=[color], **kwargs)

        region(filled, colors=[color], alpha=0.3)
        if hatch:
            region(filled, colors="none", hatches=[hatch]).set_edgecolor(color)
        outline(filled, linewidths=1.5)
        for level in dashed:
            outline(level, linestyles="dashed", linewidths=1.5)
    else:

        def ellipse(level, **kwargs):
            shape = _confidence_ellipse(x, y, level)
            ax.add_patch(
                Ellipse(
                    shape.center,
                    shape.width,
                    shape.height,
                    angle=shape.angle,
                    **kwargs,
                )
            )

        ellipse(filled, facecolor=color, edgecolor="none", alpha=0.3)
        if hatch:
            ellipse(filled, facecolor="none", edgecolor=color, hatch=hatch, lw=0)
        ellipse(filled, fill=False, edgecolor=color, lw=1.5)
        for level in dashed:
            ellipse(level, fill=False, edgecolor=color, ls="dashed", lw=1.5)

    return (
        Patch(facecolor=color, edgecolor="none", alpha=0.3),
        *(
            [Patch(facecolor="none", edgecolor=color, hatch=hatch, lw=0)]
            if hatch
            else []
        ),
        Patch(fill=False, edgecolor=color, lw=1.5),
    )


def _posterior_contours(
    fits, latex_label, params_to_plot, bounds_levels, kde, show_sm, hatch
):
    """The shared core of :func:`plot_posterior_contours` and
    :func:`plot_posterior_contours_comparison`.

    The lower triangle of the coefficient matrix: the panel in row ``j - 1``,
    column ``i`` has coefficient ``i`` on the x-axis and ``j`` on the y-axis.
    Every fit is drawn in every panel whose two coefficients it sampled; one it
    did not sample leaves that fit out of the panel. The legend and title go in
    the free upper-right cell, or inside the axes for a single panel.
    """
    if not fits:
        raise ValueError("fits is empty: list at least one fit under 'fits:'.")
    # rejected before any figure exists to be half-drawn
    for fit in fits:
        if fit.individual_fit:
            raise ValueError(
                f"Fit '{fit.fit_name}' was run one coefficient at a time, so "
                "there is no joint posterior to draw contours from."
            )
        if not fit.fit_results.samples:
            raise ValueError(
                f"Fit '{fit.fit_name}' stored no posterior samples, so there is "
                "nothing to draw contours from."
            )

    bounds_levels, _ = bounds_settings(bounds_levels, None)
    # widest first: filled, the narrower ones dashed on top of it
    levels = sorted(set(bounds_levels), reverse=True)

    # the union over the fits, in order of first appearance, as for the bounds
    names = list(dict.fromkeys(name for fit in fits for name in fit.bounded_parameters))
    context = ", ".join(fit.fit_name for fit in fits)
    names = select_params(names, params_to_plot, context=context)
    if len(names) < 2:
        raise ValueError(
            f"Contours need at least two coefficients, {context} leave "
            f"{names} to plot."
        )

    samples = [
        {
            name: np.asarray(values, dtype=float)
            for name, values in fit.fit_results.samples.items()
        }
        for fit in fits
    ]
    kdes = [fit.use_quad if kde is None else bool(kde) for fit in fits]
    hatches = [
        _HATCH_CYCLE[idx % len(_HATCH_CYCLE)] if hatch else None
        for idx in range(len(fits))
    ]

    # one range per coefficient, wherever it appears: the pooled samples of
    # every fit, and the SM point when it is marked, padded by 10%
    limits = {}
    for name in names:
        values = np.concatenate([s[name] for s in samples if name in s])
        if show_sm:
            values = np.append(values, 0.0)
        low, high = float(values.min()), float(values.max())
        pad = 0.1 * (high - low) if high > low else abs(low) or 1.0
        limits[name] = (low - pad, high + pad)

    set_plot_style()
    n_par = len(names)
    n_cells = n_par - 1
    single_panel = n_par == 2
    subplot_size = 6 if single_panel else 4
    label_size, tick_size = (22, 14) if single_panel else (26, 22)
    legend_size, title_size = (12, 20) if single_panel else (20, 24)
    # from three coefficients the lower triangle leaves the upper-right cell
    # free for the legend; two columns at least, so it has room
    n_cols = n_cells if single_panel else max(n_cells, 2)
    fig = plt.figure(figsize=(n_cols * subplot_size, n_cells * subplot_size))
    grid = plt.GridSpec(n_cells, n_cols, hspace=0.1, wspace=0.1)

    handles = [None] * len(fits)
    for i, j in itertools.combinations(range(n_par), 2):
        x_name, y_name = names[i], names[j]
        ax = fig.add_subplot(grid[j - 1, i])

        for idx, fit_samples in enumerate(samples):
            if x_name not in fit_samples or y_name not in fit_samples:
                continue
            handles[idx] = _draw_pair(
                ax,
                fit_samples[x_name],
                fit_samples[y_name],
                f"C{idx}",
                levels,
                kdes[idx],
                hatches[idx],
            )
        if show_sm:
            ax.scatter(0.0, 0.0, color="black", marker="+", s=100, zorder=10)

        ax.set_xlim(*limits[x_name])
        ax.set_ylim(*limits[y_name])
        ax.locator_params(axis="x", nbins=5)
        ax.locator_params(axis="y", nbins=6)
        ax.minorticks_on()
        ax.grid(linestyle="dotted", linewidth=0.5)
        ax.tick_params(axis="both", which="major", labelsize=tick_size)
        # only the outer panels carry axis and tick labels
        if j == n_par - 1:
            ax.set_xlabel(latex_label(x_name), fontsize=label_size)
        else:
            ax.tick_params(axis="x", which="both", labelbottom=False)
        if i == 0:
            ax.set_ylabel(latex_label(y_name), fontsize=label_size)
        else:
            ax.tick_params(axis="y", which="both", labelleft=False)

    # a fit every panel left out (it sampled no pair of the selection) has
    # nothing to stand for in the legend
    drawn = [idx for idx, handle in enumerate(handles) if handle is not None]
    legend_handles = [handles[idx] for idx in drawn]
    legend_labels = [fits[idx].plot_label for idx in drawn]
    if show_sm:
        legend_handles.append(
            Line2D([], [], color="black", marker="+", ls="none", ms=10)
        )
        legend_labels.append(r"$\mathrm{SM}$")

    level_text = r",\ ".join(f"{level:.10g}" for level in sorted(levels))
    title = rf"$\mathrm{{Marginalised}}\ {level_text}\%\ \mathrm{{CL}}$"
    if single_panel:
        # ``ax`` is the one panel: an ordinary plot's legend and title
        ax.legend(
            legend_handles,
            legend_labels,
            loc="best",
            fontsize=legend_size,
            handlelength=1,
            framealpha=0.9,
        )
        fig.suptitle(title, fontsize=title_size)
    else:
        legend_ax = fig.add_subplot(grid[0, 1:])
        legend_ax.axis("off")
        legend_ax.legend(
            legend_handles,
            legend_labels,
            loc="lower left",
            frameon=False,
            fontsize=legend_size,
            handlelength=1,
            handletextpad=1,
        )
        legend_ax.text(
            0.05,
            0.95,
            title,
            fontsize=title_size,
            transform=legend_ax.transAxes,
            verticalalignment="top",
        )
    return fig


@figure
def plot_posterior_contours(
    fit,
    latex_label,
    params_to_plot=None,
    bounds_levels=None,
    kde=None,
    show_sm=True,
    hatch=True,
):
    """Plot the pairwise 2D posterior contours of one fit, as a corner plot.

    Parameters
    ----------
    fit : smefit.fit_result.Fit
        A previously run joint fit with posterior samples, loaded from disk.
    latex_label : Callable[[str], str]
        Writes a name in LaTeX (the ``latex_label`` resource).
    params_to_plot : list of str, optional
        Restrict the panels to these coefficients, in this order. All of them
        by default.
    bounds_levels : float or list of float, optional
        The confidence levels drawn, in percent, 68 and 95 by default — the
        same key the bounds tables and plots read. The widest region is
        filled, every narrower one a dashed outline.
    kde : bool, optional
        Draw kernel density contours (True) or Gaussian ellipses (False). By
        default a fit with quadratic corrections gets a KDE, a linear one —
        whose posterior is Gaussian — ellipses.
    show_sm : bool, optional
        Mark the SM point, the origin, in every panel. On by default.
    hatch : bool, optional
        Hatch the filled region, one pattern per fit, so it reads in print
        and greyscale. On by default.
    """
    return _posterior_contours(
        [fit], latex_label, params_to_plot, bounds_levels, kde, show_sm, hatch
    )


@figure
def plot_posterior_contours_comparison(
    fits,
    latex_label,
    params_to_plot=None,
    bounds_levels=None,
    kde=None,
    show_sm=True,
    hatch=True,
):
    """Overlay the pairwise 2D posterior contours of several fits.

    One figure for the whole ``fits`` list, one colour (and hatch) per fit,
    legend-labelled with the ``label`` of its ``fits`` entry (its name
    otherwise). The coefficients are the union of the fits'; a fit is left
    out of every panel showing a coefficient it did not sample. ``kde``
    defaults per fit. The parameters are those of
    :func:`plot_posterior_contours`.
    """
    return _posterior_contours(
        fits, latex_label, params_to_plot, bounds_levels, kde, show_sm, hatch
    )


@figure
def plot_global_information_heatmap(
    global_information_table,
    cmap="Blues",
    value_fmt="{:.1f}",
    colorbar=False,
):
    """Plot the global information matrix as a heatmap.

    Parameters
    ----------
    global_information_table : pd.DataFrame
        Index = coeff_names, columns = source_names.
    cmap : str, optional
        Colormap. Sequential by default, as for
        :func:`plot_fisher_diagonals_heatmap`: the values are percentage shares.
    value_fmt : str, optional
        Format of the per-cell annotation, in percent.
    colorbar : bool, optional
        Whether to draw the colour scale alongside. Off by default: rows sum to
        100%, so the annotations already say what a cell is worth.
    """
    shares = global_information_table.values * 100
    # 0-100% unless a share falls outside it (see global_information_matrix):
    # clipping would hide exactly the cells the warning is about
    return _plot_heatmap(
        shares,
        global_information_table.index.tolist(),
        global_information_table.columns.tolist(),
        vmin=min(0.0, shares.min()),
        vmax=max(100.0, shares.max()),
        cmap=cmap,
        value_fmt=value_fmt,
        colorbar=colorbar,
    )
