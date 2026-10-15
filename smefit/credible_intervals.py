"""
Credible intervals of one coefficient's posterior samples.

The interval types :data:`_INTERVAL_TYPES` registers: ``eti`` (equal-tailed)
and ``hdi`` (highest-density). The registry owns the calling convention: the
samples, a level in percent, and the hard ``(low, high)`` bounds of the
coefficient's support, or None. Each returns the ``(low, high)`` pieces of the
region, as a list so that a multimodal posterior can be given several.

The HDI relies in two different backends: getdist and arviz. getdist estimates the density: its
boundary-corrected KDE pins a hard bound (a sign-definite coefficient's 0)
exactly, but always returns a single interval. arviz's multimodal HDI counts
the modes first, and with one the region is getdist's interval. getdist is
given each distinct draw once, weighted by its repeats, so that resampled
draws do not shrink its bandwidth.

The count uses Silverman's rule, with the effective number of draws in place
of their number. Nested-sampling posteriors are resampled with replacement,
and a narrower bandwidth (arviz's default, Improved Sheather-Jones, or
``experimental``, the mean of ISJ and Silverman's rule) reads their repeated
draws as modes. Silverman's rule sizes the bandwidth by the spread of the
whole posterior. Its cost: a small mode separated from its neighbour only by a
shallow dip is counted with it, and the region is reported as one interval
covering both, erring on the wide side.

With several modes, the draws are split at the midpoint of each gap between
arviz's pieces, and getdist estimates each mode's density from its own draws,
so that its bandwidth follows the mode's width rather than the distance
between modes. Each cut is a hard edge of the densities on either side, and
each density is weighted by its mode's share of the draws. The threshold is
the ``(100 - level)`` percentile of the weighted densities at the draws, so
the region holds ``level`` percent of them. Two pieces still above the
threshold at the cut between them are one piece: the cut went through a mode,
not a dip.
"""

from typing import Callable, Dict, List, Optional, Tuple

import arviz as az
import numpy as np
from getdist import MCSamples
from getdist import chains as getdist_chains

# Silences getdist's per-call "Removed no burn in" print.
getdist_chains.print_load_details = False

Bounds = Optional[Tuple[Optional[float], Optional[float]]]


def equal_tailed_interval(
    values: np.ndarray, level: float, bounds: Bounds = None
) -> List[Tuple[float, float]]:
    """The ``[tail, 100 - tail]`` percentiles: equal posterior mass cut from
    each side. NaNs are ignored. Always a single interval. ``bounds`` is
    unused: percentiles of the samples already lie within them."""
    tail = (100.0 - level) / 2.0
    low, high = np.nanpercentile(values, [tail, 100.0 - tail])
    return [(float(low), float(high))]


def highest_density_interval(
    values: np.ndarray, level: float, bounds: Bounds = None
) -> List[Tuple[float, float]]:
    """The narrowest region holding ``level`` percent of the posterior.

    arviz counts the modes. With one, the region is getdist's interval. With
    several, it is where getdist's per-mode densities, weighted by each
    mode's share of the draws, exceed one common threshold. NaNs are ignored.

    Parameters
    ----------
    values : np.ndarray
        1D posterior draws of one coefficient.
    level : float
        In percent: 95, not 0.95.
    bounds : (float or None, float or None), optional
        Hard bounds of the coefficient's support, either side ``None`` when
        open. They are hard edges of the density next to them.
    """
    values = values[~np.isnan(values)]
    prob = level / 100.0

    # Silverman's rule scales as n^(-1/5); bw_fct swaps the number of draws
    # for their effective number, as getdist's weights do for its own.
    bw_fct = (values.size / _effective_size(values)) ** 0.2
    modes = np.atleast_2d(
        az.hdi(values, prob=prob, method="multimodal", bw="silverman", bw_fct=bw_fct)
    )
    if len(modes) == 1:
        return [_getdist_interval(values, prob, bounds)]

    # Sort by the left edge.
    modes = modes[np.argsort(modes[:, 0])]
    # The cut between each pair of modes is the midpoint between their edges.
    cuts = (modes[:-1, 1] + modes[1:, 0]) / 2
    low, high = bounds if bounds is not None else (None, None)
    edges = [low, *cuts, high]
    owner = np.searchsorted(cuts, values)

    densities, at_draws = [], []
    for k in range(len(modes)):
        draws = values[owner == k]
        if draws.min() == draws.max():
            # This is for the case of a single point, drawn many times.
            # getdist's KDE cannot resolve its density, hence we assign it a infinite one without width.
            x, density = draws[:1], np.array([np.inf])
        else:
            # Normal case.
            # KDE for each mode.
            x, density = _getdist_density(draws, (edges[k], edges[k + 1]))
            # Weight by the mode's share of the draws.
            density *= draws.size / values.size
        densities.append((x, density))
        # evaluates the weighted density at each draw of the mode.
        at_draws.append(np.interp(draws, x, density))
    threshold = np.quantile(np.concatenate(at_draws), 1.0 - prob, method="inverted_cdf")

    pieces: List[Tuple[float, float]] = []
    for x, density in densities:
        for piece in _above(x, density, threshold):
            # Only two pieces reaching the cut between them can touch, but one can't enter into the other's territory.
            if pieces and pieces[-1][1] >= piece[0]:
                pieces[-1] = (pieces[-1][0], piece[1])
            else:
                pieces.append(piece)
    return pieces


def _effective_size(values: np.ndarray) -> float:
    """Kish's effective number of draws, ``n^2 / sum(c^2)`` over the repeat
    counts ``c`` of the distinct draws: ``n`` when none repeats."""
    # Kish, L. (1992). "Weighting for unequal Pᵢ". Journal of Official Statistics 8(2), 183–200
    _, counts = np.unique(values, return_counts=True)
    return values.size**2 / np.sum(counts.astype(float) ** 2)


def _mcsamples(values: np.ndarray, bounds: Bounds) -> MCSamples:
    """getdist's samples of ``values``: each distinct draw once, weighted by
    how often it repeats.

    Nested sampling stores its posterior resampled with replacement. Passed
    as they are, the repeats count as independent draws, and getdist's
    bandwidth, sized by that count, resolves them into spikes. As weights,
    they lower its effective sample size instead. The draws are treated as
    uncorrelated, since ``np.unique`` sorts them and their order is no longer
    a chain's."""
    unique, counts = np.unique(values, return_counts=True)
    ranges = {"x": list(bounds)} if bounds is not None else None
    return MCSamples(
        samples=unique,
        weights=counts.astype(float),
        names=["x"],
        ranges=ranges,
        sampler="uncorrelated",
    )


def _getdist_interval(
    values: np.ndarray, prob: float, bounds: Bounds
) -> Tuple[float, float]:
    density = _mcsamples(values, bounds).get1DDensity("x")
    low, high, _has_low, _has_high = density.getLimits(prob)
    return float(low), float(high)


def _getdist_density(
    values: np.ndarray, bounds: Tuple[Optional[float], Optional[float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """getdist's density of ``values`` on its grid, cut to ``bounds`` and
    normalised to unit integral there.

    getdist treats a bound as a hard edge, and starts its grid on it, only
    when the draws come close to it. Otherwise its grid runs past the last
    draw and may cross the bound, into the neighbouring mode."""
    density = _mcsamples(values, bounds).get1DDensity("x")
    x, p = density.x, density.P
    low = x[0] if bounds[0] is None else max(bounds[0], x[0])
    high = x[-1] if bounds[1] is None else min(bounds[1], x[-1])
    inner = x[(x > low) & (x < high)]
    grid = np.concatenate([[low], inner, [high]])
    p = np.interp(grid, x, p)
    return grid, p / np.trapezoid(p, grid)


def _above(
    x: np.ndarray, density: np.ndarray, threshold: float
) -> List[Tuple[float, float]]:
    """The stretches of ``x`` where ``density`` is at least ``threshold``, with
    each edge linearly interpolated between the grid points around it; a
    stretch reaching an end of the grid ends on it."""
    above = np.concatenate([[False], density >= threshold, [False]])
    starts = np.flatnonzero(~above[:-1] & above[1:])
    stops = np.flatnonzero(above[:-1] & ~above[1:]) - 1

    def crossing(below: int, over: int) -> float:
        # density[below] < threshold <= density[over]: np.interp needs its
        # sample points increasing.
        return np.interp(threshold, density[[below, over]], x[[below, over]])

    return [
        (
            float(x[start] if start == 0 else crossing(start - 1, start)),
            float(x[stop] if stop == x.size - 1 else crossing(stop + 1, stop)),
        )
        for start, stop in zip(starts, stops)
    ]


# Types of credible intervals that :meth:`smefit.fit_result.Fit.confidence_bounds` can
# compute. The ``interval_type`` argument (and the
# ``interval_types`` runcard key) defines the algorithm used to compute them. Each maps one coefficient's samples, a
# level in percent and the hard ``(low, high)`` bounds of its support (or None)
# to the ``(low, high)`` pieces of the credible region. The list format is to support multimodal posteriors.
_INTERVAL_TYPES: Dict[
    str, Callable[[np.ndarray, float, Bounds], List[Tuple[float, float]]]
] = {
    "eti": equal_tailed_interval,
    "hdi": highest_density_interval,
}
