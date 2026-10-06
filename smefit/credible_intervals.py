"""
Highest-density credible intervals from a posterior samples.

Registered as the ``hdi`` interval type in
:data:`smefit.fit_result._INTERVAL_TYPES`, which owns the calling convention:
samples, a level in percent, the coefficient's hard ``(low, high)`` bounds;
returns the ``(low, high)`` pieces of the region.

Two estimators are combined, each for what it does well. arviz's sample-window
HDI splits a multimodal posterior into disjoint pieces, but is noisy at modest
sample sizes; getdist's boundary-corrected KDE is smooth and pins a hard bound
(a sign-definite coefficient's 0) exactly, but always returns a single interval.
"""

from typing import List, Optional, Tuple

import numpy as np

Bounds = Optional[Tuple[Optional[float], Optional[float]]]


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

    pieces = np.atleast_2d(az.hdi(values, prob=prob, method="multimodal"))
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
