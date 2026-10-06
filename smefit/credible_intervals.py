"""
Credible intervals of one coefficient's posterior samples.

The interval types :data:`smefit.fit_result._INTERVAL_TYPES` registers:
``eti`` (equal-tailed) and ``hdi`` (highest-density). The registry owns the
calling convention: the samples, a level in percent, and the hard
``(low, high)`` bounds of the coefficient's support, or None. Each returns the
``(low, high)`` pieces of the region, as a list so that a multimodal posterior
can be given several.

The HDI combines two estimators, each for what it does well. arviz's
multimodal HDI, a KDE on a grid spanning the samples, splits a multimodal
posterior into disjoint pieces, but knows no bound beyond the sample range;
getdist's boundary-corrected KDE pins a hard bound (a sign-definite
coefficient's 0) exactly, but always returns a single interval.

arviz's KDE uses its ``experimental`` bandwidth, the mean of Silverman's rule
and Improved Sheather-Jones, not its ISJ default. Nested-sampling posteriors
are resampled with replacement, and ISJ undersmooths their repeated draws into
spurious modes, while Silverman's rule alone merges a small narrow mode into
its neighbour.
"""

from typing import List, Optional, Tuple

import numpy as np

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

    arviz decides the number of modes: when it finds several, its disjoint
    pieces are the answer. Otherwise the single interval is recomputed with
    getdist, so that it is smooth and respects ``bounds``. NaNs are ignored.

    Parameters
    ----------
    values : np.ndarray
        1D posterior draws of one coefficient.
    level : float
        In percent: 95, not 0.95.
    bounds : (float or None, float or None), optional
        Hard bounds of the coefficient's support, either side ``None`` when
        open. Only the getdist interval uses them.
    """
    # Imported here, not at module level: only an hdi report needs it.
    import arviz as az

    values = values[~np.isnan(values)]
    prob = level / 100.0

    pieces = np.atleast_2d(
        az.hdi(values, prob=prob, method="multimodal", bw="experimental")
    )
    if len(pieces) > 1:
        return [(float(low), float(high)) for low, high in pieces]
    return [_getdist_interval(values, prob, bounds)]


def _getdist_interval(
    values: np.ndarray, prob: float, bounds: Bounds
) -> Tuple[float, float]:
    from getdist import MCSamples
    from getdist import chains as getdist_chains

    # Silences getdist's per-call "Removed no burn in" print.
    getdist_chains.print_load_details = False

    ranges = {"x": list(bounds)} if bounds is not None else None
    density = MCSamples(samples=values, names=["x"], ranges=ranges).get1DDensity("x")
    low, high, _has_low, _has_high = density.getLimits(prob)
    return float(low), float(high)
