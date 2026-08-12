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
posterior mass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib import patches, transforms
from matplotlib.patches import Ellipse

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.contour import QuadContourSet
    from matplotlib.typing import ColorType
    from numpy.typing import ArrayLike

# A KDE evaluated on a regular grid: the ``(xx, yy, density)`` of `kde_grid`.
KDEGrid = tuple[np.ndarray, np.ndarray, np.ndarray]

# Density grid resolution and support padding (in bandwidth units) of the KDE,
# and the sample cap above which posteriors are thinned before estimating it.
_KDE_GRIDSIZE = 200
_KDE_CUT = 3
_KDE_MAX_SAMPLES = 5000


def split_solution(full_solution: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Split a posterior into two disjoint solutions.

    The samples are split at the midpoint between the smallest and the largest
    sample. The solution closer to zero is returned first.

    Parameters
    ----------
    full_solution : array_like
        ``(N,)`` posterior samples of a single coefficient.

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        The two solutions, the one closer to zero first.
    """
    full_solution = np.asarray(full_solution)

    min_val = full_solution.min()
    max_val = full_solution.max()
    mid = np.mean([max_val, min_val])

    solution1 = full_solution[full_solution <= mid]
    solution2 = full_solution[full_solution > mid]

    if solution1.size == 0 or solution2.size == 0:
        return full_solution, full_solution

    # solution 1 should be closer to 0
    if np.min(np.abs(solution2)) < np.min(np.abs(solution1)):
        solution1, solution2 = solution2, solution1

    return solution1, solution2


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


def ellipse_half_axis(std: float, confidence_level: float = 95) -> float:
    """Half-axis of the confidence ellipse of an uncorrelated Gaussian.

    The same scaling :func:`confidence_ellipse` applies to the square root of a
    covariance eigenvalue — here the standard deviation is given directly.

    Parameters
    ----------
    std : float
        Standard deviation along the axis.
    confidence_level : float, optional
        Confidence level in percent, 95 by default.

    Returns
    -------
    float
        Half the extent of the ellipse along that axis.
    """
    return float(np.sqrt(scipy.stats.chi2.ppf(confidence_level / 100.0, 2)) * std)


def uncorrelated_ellipse(
    center: tuple[float, float],
    std: tuple[float, float],
    ax: Axes,
    facecolor: ColorType = "none",
    confidence_level: float = 95,
    **kwargs: Any,
) -> Ellipse:
    """Draw the confidence ellipse of an uncorrelated Gaussian on ``ax``.

    The counterpart of :func:`confidence_ellipse` for a point known by a
    central value and a standard deviation per coefficient rather than by
    samples: with no correlation to orient it, the ellipse is axis-aligned.

    Parameters
    ----------
    center : tuple of two floats
        Central value on each axis.
    std : tuple of two floats
        Standard deviation on each axis.
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
    ellipse = Ellipse(
        center,
        width=2 * ellipse_half_axis(std[0], confidence_level),
        height=2 * ellipse_half_axis(std[1], confidence_level),
        facecolor=facecolor,
        **kwargs,
    )
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
    tuple(np.ndarray, np.ndarray, np.ndarray)
        ``(xx, yy, density)``, each of shape ``(gridsize, gridsize)``.

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

    return xx, yy, density


def density_level(density: np.ndarray, confidence_level: float) -> float:
    """Iso-density level enclosing ``confidence_level`` percent of the mass.

    Parameters
    ----------
    density : np.ndarray
        Density evaluated on a regular grid.
    confidence_level : float
        Confidence level in percent.

    Returns
    -------
    float
        Density value such that the region where ``density`` exceeds it
        contains ``confidence_level`` percent of the total mass.
    """
    sorted_values = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(sorted_values) / sorted_values.sum()
    idx = np.searchsorted(cumulative, confidence_level / 100.0)
    return float(np.take(sorted_values, idx, mode="clip"))


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
        Pre-computed ``(xx, yy, density)`` from :func:`kde_grid`. Evaluating
        the KDE is by far the most expensive step, so pass it whenever several
        contours are drawn from the same samples.
    **kwargs
        Additional settings passed to ``contour``/``contourf``.

    Returns
    -------
    matplotlib.contour.QuadContourSet
    """
    xx, yy, density = (
        grid if grid is not None else kde_grid(x_values, y_values, bw_adjust=bw_adjust)
    )
    level = density_level(density, confidence_level)

    if fill:
        return ax.contourf(
            xx, yy, density, levels=[level, density.max()], colors=[color], **kwargs
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
    double_solution: list[str] | None = None,
    show_best_fit: bool = False,
    best_fit: tuple[float, float] | None = None,
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
    double_solution : list, optional
        Coefficients of this fit that have a double (disjoint) solution. Only
        used in KDE mode, to place one best-fit marker per solution.
    show_best_fit : bool, optional
        If True mark the best-fit point of this fit. Off by default.
    best_fit : tuple(float, float), optional
        Best-fit ``(coeff1, coeff2)`` values. Defaults to the posterior means,
        which is also what is used for a coefficient with a double solution
        (a single stored best-fit value cannot represent both modes).

    Returns
    -------
    tuple
        Handles (Patch objects) to be used in the figure legend.
    """
    double_solution = double_solution or []

    x_values = np.asarray(posterior[coeff1], dtype=float)
    y_values = np.asarray(posterior[coeff2], dtype=float)

    if kde:
        solution1x = solution2x = x_values
        if coeff1 in double_solution:
            solution1x, solution2x = split_solution(x_values)

        solution1y = solution2y = y_values
        if coeff2 in double_solution:
            solution1y, solution2y = split_solution(y_values)

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
            split = coeff1 in double_solution or coeff2 in double_solution
            if best_fit is not None and not split:
                ax.scatter(*best_fit, color=color, s=50, marker="o")
            else:
                # one marker per solution: a single best-fit value cannot
                # represent two disjoint modes
                ax.scatter(
                    np.mean(solution1x),
                    np.mean(solution1y),
                    color=color,
                    s=50,
                    marker="o",
                )
                ax.scatter(
                    np.mean(solution2x),
                    np.mean(solution2y),
                    color=color,
                    s=50,
                    marker="o",
                )

        hndls = (
            patches.Patch(ec=color, fc=color, fill=True, alpha=0.3),
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
        if show_best_fit:
            point = (
                best_fit
                if best_fit is not None
                else (np.mean(x_values), np.mean(y_values))
            )
            ax.scatter(*point, color=color, s=50, marker="o")

        hndls = (p1, p2)

    ax.tick_params(which="both", direction="in", labelsize=22)

    return hndls


def plot_uncorrelated_contours(
    ax: Axes,
    center: tuple[float, float],
    std: tuple[float, float],
    color: ColorType,
    confidence_level: float = 95,
    dashed_confidence_level: float | None = None,
) -> tuple[Ellipse, Ellipse]:
    """Draw a point known by a central value and a std as a filled contour.

    The three layers :func:`plot_contours` gives a linear fit — a dashed
    outline at the outer level, then a solid outline and a translucent fill at
    the inner one — so a reference point is read exactly like the posteriors
    beside it. What differs is only what the ellipse comes from: a standard
    deviation per coefficient, which says nothing about how the two covary, so
    this one is axis-aligned where a fit's follows its sample covariance.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    center : tuple of two floats
        Central value on each axis.
    std : tuple of two floats
        Standard deviation on each axis.
    color : ColorType
        Colour of the outline and of the fill.
    confidence_level : float, optional
        Confidence level in percent of the filled contour, 95 by default.
    dashed_confidence_level : float, optional
        Confidence level in percent of an additional dashed outline.

    Returns
    -------
    tuple(matplotlib.patches.Ellipse, matplotlib.patches.Ellipse)
        The outline and the fill, in the order :func:`plot_contours` returns
        its handles.
    """
    if dashed_confidence_level is not None:
        uncorrelated_ellipse(
            center,
            std,
            ax,
            edgecolor=color,
            confidence_level=dashed_confidence_level,
            linestyle="dashed",
            linewidth=2,
        )
    outline = uncorrelated_ellipse(
        center,
        std,
        ax,
        alpha=1,
        edgecolor=color,
        confidence_level=confidence_level,
    )
    fill = uncorrelated_ellipse(
        center,
        std,
        ax,
        alpha=0.3,
        facecolor=color,
        edgecolor=None,
        confidence_level=confidence_level,
    )
    return outline, fill


def fit_colors(n_fits: int) -> list[ColorType]:
    """Return ``n_fits`` colours from the current matplotlib colour cycle."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return [colors[i % len(colors)] for i in range(n_fits)]
