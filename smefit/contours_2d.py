"""
smefit.contours_2d.py

Low-level primitives for 2D marginalised confidence contours.

Two contour styles are supported, following the old smefit report code:

* **ellipses** — Gaussian approximation, appropriate for linear (purely
  interference) fits whose posteriors are close to elliptical.
* **KDE contours** — kernel density estimate of the 2D marginal, needed when
  quadratic corrections make the posterior non-Gaussian (possibly with two
  disjoint solutions).

The KDE is computed with ``scipy.stats.gaussian_kde`` and the contour is drawn
at the iso-density level enclosing ``confidence_level`` percent of the
posterior mass, calibrated on the samples themselves — see
:func:`density_level`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib import patches, transforms
from matplotlib.patches import Ellipse

from smefit.bounds_1d import confidence_bounds

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.contour import QuadContourSet
    from matplotlib.lines import Line2D
    from matplotlib.typing import ColorType
    from numpy.typing import ArrayLike

# A KDE evaluated on a regular grid *and* at the samples it was estimated from:
# the ``(xx, yy, density, sample_density)`` of `kde_grid`. The grid is what gets
# drawn, the sample densities are what the contour level is read off — see
# `density_level`.
KDEGrid = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

# Hatch patterns handed out alongside the colour cycle. They carry the same
# information as the colour does, so a filled contour stays distinguishable in
# print, in greyscale, and to a reader who cannot separate the hues.
#
# A hatch is always drawn as its own unfilled layer over the fill, never as the
# fill's own `hatch`: the PDF backend silently drops the hatch of a patch that
# is both filled and hatched, and PDF is what a report is made of.
_HATCH_CYCLE = ("///", "\\\\\\", "xxx", "...", "+++", "ooo")

# Density grid resolution and support padding (in bandwidth units) of the KDE,
# and the sample cap above which posteriors are thinned before estimating it.
_KDE_GRIDSIZE = 200
_KDE_CUT = 3
_KDE_MAX_SAMPLES = 5000

# Widths and opacity of the interval a coefficient stuck at a value is drawn
# with, and the size of its end caps. A second confidence level is drawn on the
# same line as the first, so the wider one is thinned and faded to the alpha of
# a contour fill and the narrower one keeps the width of a contour outline.
_SEGMENT_LINEWIDTH = 4
_SEGMENT_OUTER_LINEWIDTH = 2
_SEGMENT_OUTER_ALPHA = 0.4
_SEGMENT_CAP_SIZE = 10

# Size of the cross marking a panel where the fit sampled neither coefficient.
_STUCK_POINT_SIZE = 9


def confidence_ellipse(
    coeff1: ArrayLike,
    coeff2: ArrayLike,
    ax: Axes,
    facecolor: ColorType = "none",
    confidence_level: float = 95,
    **kwargs: Any,
) -> Ellipse:
    """Draw the confidence-level ellipse of the samples ``coeff1``, ``coeff2``.

    The ellipse is the iso-contour of the Gaussian with the same covariance as
    the samples, centred on their median.

    Parameters
    ----------
    coeff1 : array_like
        ``(N,)`` posterior samples of the coefficient on the x-axis.
    coeff2 : array_like
        ``(N,)`` posterior samples of the coefficient on the y-axis.
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    facecolor : ColorType, optional
        Fill colour of the ellipse, ``"none"`` by default.
    confidence_level : float, optional
        Confidence level in percent, 95 by default.
    **kwargs
        Additional settings passed to ``matplotlib.patches.Ellipse``.

    Returns
    -------
    matplotlib.patches.Ellipse
        The ellipse added to ``ax``.
    """
    coeff1 = np.asarray(coeff1)
    coeff2 = np.asarray(coeff2)

    if coeff1.size != coeff2.size:
        raise ValueError("coeff1 and coeff2 must be the same size")

    # construct covariance matrix of coefficients
    cov = np.cov(coeff1, coeff2)

    # diagonalise; the covariance is symmetric, so eigh keeps eigenvalues and
    # eigenvectors real (eig lets numerical noise make them complex) and
    # returns the eigenvalues sorted ascending
    eig_val, eig_vec = np.linalg.eigh(cov)

    # eigenvector with largest eigenvalue
    eig_vec_max = eig_vec[:, -1]

    # angle of eigenvector with largest eigenvalue with the horizontal axis
    cos_th = eig_vec_max[0] / np.linalg.norm(eig_vec_max)
    if eig_vec_max[1] > 0:
        inclination = np.arccos(cos_th)
    else:
        # pay attention to range of arccos (extend to [0, -\pi] domain)
        inclination = -np.arccos(cos_th)

    chi2_qnt = scipy.stats.chi2.ppf(confidence_level / 100.0, 2)

    ell_radius_x = np.sqrt(chi2_qnt * eig_val[-1])
    ell_radius_y = np.sqrt(chi2_qnt * eig_val[-2])

    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    transf = (
        transforms.Affine2D()
        .rotate(inclination)
        .translate(np.median(coeff1), np.median(coeff2))
    )
    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)


def kde_grid(
    x_values: ArrayLike,
    y_values: ArrayLike,
    bw_adjust: float = 1.5,
    gridsize: int = _KDE_GRIDSIZE,
    max_samples: int | None = _KDE_MAX_SAMPLES,
) -> KDEGrid:
    """Evaluate a 2D Gaussian KDE of the samples on a regular grid.

    Parameters
    ----------
    x_values, y_values : array_like
        ``(N,)`` posterior samples.
    bw_adjust : float, optional
        Multiplicative factor applied to Scott's bandwidth, 1.5 by default.
        Larger values smooth the density further, at the risk of washing out
        genuine structure such as a shallow second mode.
    gridsize : int, optional
        Number of grid points per axis.
    max_samples : int, optional
        Cap on the number of samples entering the estimate. The cost of a KDE
        grows linearly with the sample size while the estimate itself barely
        moves past a few thousand samples, so larger posteriors are thinned by
        an even stride (deterministic, and the samples are already unordered).

    Returns
    -------
    tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        ``(xx, yy, density, sample_density)``. The first three have shape
        ``(gridsize, gridsize)``; ``sample_density`` is the same KDE evaluated
        at the (thinned) samples, shape ``(N,)``, and is what
        :func:`density_level` calibrates the contour level on.

    Raises
    ------
    ValueError
        If the two samples are not the same size, so they cannot be the
        coordinates of one set of posterior draws.
    """
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    # checked before thinning: a common stride can erase a small difference in
    # length (6000 and 5999 samples both thin to 3000), and the estimate would
    # then silently pair up samples from different posterior draws
    if x_values.size != y_values.size:
        raise ValueError("x_values and y_values must be the same size")

    if max_samples is not None and x_values.size > max_samples:
        stride = int(np.ceil(x_values.size / max_samples))
        x_values = x_values[::stride]
        y_values = y_values[::stride]

    kde = scipy.stats.gaussian_kde(np.vstack([x_values, y_values]))
    kde.set_bandwidth(kde.factor * bw_adjust)

    # pad the support by a few bandwidths so the tails are not truncated
    bw_x = np.sqrt(kde.covariance[0, 0])
    bw_y = np.sqrt(kde.covariance[1, 1])
    x_grid = np.linspace(
        x_values.min() - _KDE_CUT * bw_x, x_values.max() + _KDE_CUT * bw_x, gridsize
    )
    y_grid = np.linspace(
        y_values.min() - _KDE_CUT * bw_y, y_values.max() + _KDE_CUT * bw_y, gridsize
    )

    xx, yy = np.meshgrid(x_grid, y_grid)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    sample_density = kde(np.vstack([x_values, y_values]))

    return xx, yy, density, sample_density


def density_level(sample_density: np.ndarray, confidence_level: float) -> float:
    """Iso-density level enclosing ``confidence_level`` percent of the mass.

    The level is the ``1 - confidence_level`` quantile of the density *at the
    posterior samples*, so that exactly ``confidence_level`` percent of them
    lie inside the contour by construction.

    Reading the level off the grid instead — sorting the grid densities and
    cutting where their cumulative sum reaches the confidence level, the
    obvious alternative — systematically over-covers.

    Parameters
    ----------
    sample_density : np.ndarray
        ``(N,)`` density evaluated at the samples, from :func:`kde_grid`.
    confidence_level : float
        Confidence level in percent.

    Returns
    -------
    float
        Density value such that ``confidence_level`` percent of the samples
        sit where the density exceeds it.
    """
    return float(np.quantile(sample_density, 1.0 - confidence_level / 100.0))


def kde_contour(
    x_values: ArrayLike,
    y_values: ArrayLike,
    ax: Axes,
    color: ColorType,
    confidence_level: float = 95,
    fill: bool = False,
    bw_adjust: float = 1.5,
    grid: KDEGrid | None = None,
    **kwargs: Any,
) -> QuadContourSet:
    """Draw the KDE confidence contour of the samples on ``ax``.

    Parameters
    ----------
    x_values, y_values : array_like
        ``(N,)`` posterior samples.
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    color : str
        Contour (or fill) colour.
    confidence_level : float, optional
        Confidence level in percent, 95 by default.
    fill : bool, optional
        If True the enclosed region is filled instead of outlined.
    bw_adjust : float, optional
        Bandwidth adjustment factor passed to :func:`kde_grid`, 1.5 by
        default, as there. Ignored when ``grid`` is given.
    grid : tuple, optional
        Pre-computed ``(xx, yy, density, sample_density)`` from
        :func:`kde_grid`. Evaluating the KDE is by far the most expensive
        step, so pass it whenever several contours are drawn from the same
        samples.
    **kwargs
        Additional settings passed to ``contour``/``contourf``.

    Returns
    -------
    matplotlib.contour.QuadContourSet
    """
    xx, yy, density, sample_density = (
        grid if grid is not None else kde_grid(x_values, y_values, bw_adjust=bw_adjust)
    )
    level = density_level(sample_density, confidence_level)

    if fill:
        # the level comes from the samples, the fill from the grid, and the
        # grid need not resolve the very peak the densest sample sits on — so
        # the upper bound is nudged past the level rather than taken from the
        # grid alone, which contourf rejects unless the levels increase
        top = float(np.nextafter(max(float(density.max()), level), np.inf))
        return ax.contourf(
            xx, yy, density, levels=[level, top], colors=[color], **kwargs
        )
    return ax.contour(xx, yy, density, levels=[level], colors=[color], **kwargs)


def plot_contours(
    ax: Axes,
    posterior: Mapping[str, ArrayLike] | pd.DataFrame,
    coeff1: str,
    coeff2: str,
    kde: bool,
    color: ColorType,
    confidence_level: float = 95,
    dashed_confidence_level: float | None = None,
    show_best_fit: bool = False,
    best_fit: tuple[float, float] | None = None,
    hatch: str | None = None,
) -> tuple[patches.Patch, patches.Patch]:
    """Plot the 2D marginalised contour of a pair of coefficients.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    posterior : mapping of str to array_like
        Posterior samples, keyed by coefficient name —
        :attr:`FitResult.samples` as it comes. A ``pandas.DataFrame`` works
        too, since it maps a column name to its values the same way.
    coeff1 : str
        Name of the coefficient on the x-axis.
    coeff2 : str
        Name of the coefficient on the y-axis.
    kde : bool
        If True use a kernel density estimate (needed when quadratic
        corrections are included), otherwise draw Gaussian ellipses.
    color : str
        Colour associated to the fit this posterior belongs to.
    confidence_level : float, optional
        Confidence level in percent, 95 by default. Drawn filled.
    dashed_confidence_level : float, optional
        Secondary confidence level, drawn as a dashed outline only. Shares the
        density estimate of the main contour.
    show_best_fit : bool, optional
        If True mark the best-fit point of this fit. Off by default. One
        marker, wherever the posterior's modes are: a bimodal posterior still
        has a single maximum-likelihood point, and the contour already shows
        both modes.
    best_fit : tuple(float, float), optional
        Best-fit ``(coeff1, coeff2)`` values. Defaults to the posterior means,
        for a fit that recorded no best-fit point.
    hatch : str, optional
        Matplotlib hatch pattern for the filled contour, from
        :func:`fit_hatches`. None fills it flat.

    Returns
    -------
    tuple
        Handles (Patch objects) to be used in the figure legend.
    """
    x_values = np.asarray(posterior[coeff1], dtype=float)
    y_values = np.asarray(posterior[coeff2], dtype=float)

    if show_best_fit and best_fit is None:
        best_fit = (float(np.mean(x_values)), float(np.mean(y_values)))

    if kde:
        # the KDE dominates the cost of the figure: evaluate it once and draw
        # every contour of this fit and panel from the same density grid
        grid = kde_grid(x_values, y_values)

        if dashed_confidence_level is not None:
            kde_contour(
                x_values,
                y_values,
                ax,
                color=color,
                confidence_level=dashed_confidence_level,
                grid=grid,
                linestyles="dashed",
                linewidths=2,
            )
        kde_contour(
            x_values,
            y_values,
            ax,
            color=color,
            confidence_level=confidence_level,
            fill=True,
            grid=grid,
            alpha=0.3,
        )
        if hatch:
            hatched = kde_contour(
                x_values,
                y_values,
                ax,
                color="none",
                confidence_level=confidence_level,
                fill=True,
                grid=grid,
                hatches=[hatch],
            )
            hatched.set_edgecolor(color)
        kde_contour(
            x_values,
            y_values,
            ax,
            color=color,
            confidence_level=confidence_level,
            fill=False,
            grid=grid,
            alpha=1,
        )

        if show_best_fit:
            ax.scatter(*best_fit, color=color, s=50, marker="o")

        hndls = (
            patches.Patch(ec=color, fc=color, fill=True, alpha=0.3),
            *([patches.Patch(ec=color, fill=False, hatch=hatch)] if hatch else []),
            patches.Patch(ec=color, fc=color, fill=False, alpha=1.0),
        )

    else:  # draw ellipses for linear EFT fit
        if dashed_confidence_level is not None:
            confidence_ellipse(
                x_values,
                y_values,
                ax,
                edgecolor=color,
                confidence_level=dashed_confidence_level,
                linestyle="dashed",
                linewidth=2,
            )
        p1 = confidence_ellipse(
            x_values,
            y_values,
            ax,
            alpha=1,
            edgecolor=color,
            confidence_level=confidence_level,
        )
        p2 = confidence_ellipse(
            x_values,
            y_values,
            ax,
            alpha=0.3,
            facecolor=color,
            edgecolor=None,
            confidence_level=confidence_level,
        )
        hatch_layer = ()
        if hatch:
            hatch_layer = (
                confidence_ellipse(
                    x_values,
                    y_values,
                    ax,
                    facecolor="none",
                    edgecolor=color,
                    hatch=hatch,
                    linewidth=0,
                    confidence_level=confidence_level,
                ),
            )
        if show_best_fit:
            ax.scatter(*best_fit, color=color, s=50, marker="o")

        hndls = (p1, p2, *hatch_layer)

    ax.tick_params(which="both", direction="in", labelsize=22)

    return hndls


def plot_stuck_segment(
    ax: Axes,
    values: ArrayLike,
    fixed_value: float,
    orientation: str,
    color: ColorType,
    confidence_level: float = 95,
    dashed_confidence_level: float | None = None,
    show_best_fit: bool = False,
) -> tuple[Line2D, ...]:
    """Draw one coefficient's interval at the value the other is stuck at.

    Half of a contour: the fit sampled one of the pair and held the other
    fixed, so what it knows about the panel is a segment — the sampled
    coefficient's 1D confidence interval, drawn on the line where the fixed
    one sits.

    The interval is the equal-tailed one of :func:`smefit.bounds_1d.confidence_bounds`,
    the same numbers the 1D bounds figures and the CL table report, rather than
    the width of the 2D contour a joint fit would have drawn at that level.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    values : array_like
        ``(N,)`` posterior samples of the coefficient that *was* sampled.
    fixed_value : float
        Where the other coefficient was held, on the other axis.
    orientation : {"horizontal", "vertical"}
        Which axis the sampled coefficient is on: ``"horizontal"`` runs the
        segment along x at ``y = fixed_value``, ``"vertical"`` the other way.
    color : ColorType
        Colour associated to the fit these samples belong to.
    confidence_level : float, optional
        Confidence level in percent, 95 by default.
    dashed_confidence_level : float, optional
        Secondary confidence level, drawn on the same line. The two are told
        apart by width rather than by which argument they arrived in: the
        wider interval is drawn thin and faded and the narrower one thick and
        opaque, the usual double error bar.
    show_best_fit : bool, optional
        Mark the central value (the posterior mean) along the segment, off by
        default — the counterpart of the best-fit marker of a contour.

    Returns
    -------
    tuple of matplotlib.lines.Line2D
        Handles to be used in the figure legend.

    Raises
    ------
    ValueError
        If ``orientation`` is neither of the two names.
    """
    if orientation not in ("horizontal", "vertical"):
        raise ValueError(
            f"orientation is 'horizontal' or 'vertical', got {orientation!r}."
        )

    levels = [confidence_level]
    if dashed_confidence_level is not None:
        levels.append(dashed_confidence_level)
    bounds = [confidence_bounds(values, level) for level in levels]
    # widest first, so a nested interval stays visible on top of the one
    # containing it whichever order the levels were given in
    bounds.sort(key=lambda b: b.high - b.low, reverse=True)

    styles = (
        [(_SEGMENT_LINEWIDTH, 1.0)]
        if len(bounds) == 1
        else [
            (_SEGMENT_OUTER_LINEWIDTH, _SEGMENT_OUTER_ALPHA),
            (_SEGMENT_LINEWIDTH, 1.0),
        ]
    )

    handles = []
    for interval, (linewidth, alpha) in zip(bounds, styles):
        span = (interval.low, interval.high)
        fixed = (fixed_value, fixed_value)
        x_values, y_values = (
            (span, fixed) if orientation == "horizontal" else (fixed, span)
        )
        (line,) = ax.plot(
            x_values,
            y_values,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            solid_capstyle="butt",
            # end caps across the segment, so an interval reads as an interval
            # and not as a line that happens to stop there
            marker="|" if orientation == "horizontal" else "_",
            markersize=_SEGMENT_CAP_SIZE,
            markeredgewidth=linewidth,
        )
        handles.append(line)

    if show_best_fit:
        mid = bounds[0].mid  # the mean, the same at either level
        point = (
            (mid, fixed_value) if orientation == "horizontal" else (fixed_value, mid)
        )
        ax.scatter(*point, color=color, s=50, marker="o")

    ax.tick_params(which="both", direction="in", labelsize=22)

    return tuple(handles)


def plot_stuck_point(
    ax: Axes, x_value: float, y_value: float, color: ColorType
) -> tuple[Line2D]:
    """Mark where a fit held *both* coefficients of a panel.

    Nothing was sampled in this plane, so the fit has one point to contribute
    and no extent at all. Drawn as a cross in the fit's colour, which the
    black ``+`` of the SM marker cannot be confused with.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    x_value, y_value : float
        Where the two coefficients were held.
    color : ColorType
        Colour associated to the fit.

    Returns
    -------
    tuple of matplotlib.lines.Line2D
        A handle to be used in the figure legend.
    """
    (line,) = ax.plot(
        [x_value],
        [y_value],
        color=color,
        linestyle="none",
        marker="x",
        markersize=_STUCK_POINT_SIZE,
        markeredgewidth=2,
    )
    ax.tick_params(which="both", direction="in", labelsize=22)

    return (line,)


def fit_colors(n_fits: int) -> list[ColorType]:
    """Return ``n_fits`` colours from the current matplotlib colour cycle."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return [colors[i % len(colors)] for i in range(n_fits)]


def fit_hatches(n_fits: int) -> list[str]:
    """Return ``n_fits`` hatch patterns, the counterpart of :func:`fit_colors`."""
    return [_HATCH_CYCLE[i % len(_HATCH_CYCLE)] for i in range(n_fits)]
