"""
smefit.credible_intervals — ETI / HDI credible interval calculators.

Numerical core only: takes and returns plain arrays/tuples,
unit-testable on raw sample arrays. Dispatch: az.hdi(samples, prob=level, method="multimodal") first.
If it reports >1 segment, that's the answer. Otherwise discard it and use
getdist's Density1D.getLimits on the same samples instead, passing
`bounds` through as getdist's `ranges` so a hard prior edge (e.g.
positivity, low=0) is pinned exactly rather than left to leak.
This addresses the known high-sensitivity to statistical fluctuations of
arviz's implementation while keeping its capability to handle multimodal posteriors.

Both functions take ``level`` as a percentage in ``[1, 100)`` — 95, not 0.95
— to match :meth:`smefit.fit_result.Fit.confidence_bounds`.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


def _check_level(level: float) -> None:
    if not 1.0 <= level < 100.0:
        raise ValueError(
            f"level is a percentage between 1 and 100, got {level}. "
            "Write 95, not 0.95."
        )


def equal_tailed_interval(samples, level: float) -> Tuple[float, float]:
    """Equal-tailed credible interval: the ``[tail, 100 - tail]`` percentiles.

    ``level`` is a percentage between 1 and 100. This is the
    ``np.percentile`` logic, so ETI and HDI share one entry point.
    """
    _check_level(level)
    values = np.asarray(samples, dtype=float)
    tail = (100.0 - level) / 2.0
    low, high = np.nanpercentile(values, [tail, 100.0 - tail])
    return float(low), float(high)


def highest_density_interval(
    samples,
    level: float,
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> List[Tuple[float, float]]:
    """Highest-density credible interval(s), boundary-corrected.

    Always returns a list of disjoint ``(low, high)`` segments: one element
    for a unimodal (or boundary-touching) posterior, two or more if the
    posterior is genuinely bimodal.

    Dispatch:

    1. Run ``arviz.hdi(..., method="multimodal")`` — if it reports more than one segment, that is the answer.
    2. Otherwise the posterior is unimodal; re-derive that single segment
       from getdist's boundary-corrected KDE (``Density1D.getLimits``)
       instead, passing ``bounds`` through as getdist's ``ranges`` so a hard
       prior edge is pinned exactly.

    Parameters
    ----------
    samples
        1D array of posterior draws for one coefficient.
    level : float
        In percent: 95, not 0.95.
    bounds : tuple of (float or None, float or None), optional
        Hard physical bounds on the coefficient, ``(low, high)``, either side
        ``None`` when unbounded. Used only by the getdist fallback.

    Raises
    ------
    ValueError
        If ``level`` is outside ``[1, 100)``.
    """
    _check_level(level)
    import arviz as az

    values = np.asarray(samples, dtype=float)
    hdi_prob = level / 100.0

    segments = np.atleast_2d(az.hdi(values, prob=hdi_prob, method="multimodal"))
    if len(segments) > 1:
        return [(float(low), float(high)) for low, high in segments]

    return [_getdist_interval(values, hdi_prob, bounds)]


def _getdist_interval(
    values: np.ndarray,
    hdi_prob: float,
    bounds: Optional[Tuple[Optional[float], Optional[float]]],
) -> Tuple[float, float]:
    """The single boundary-corrected HDI segment, via getdist."""
    from getdist import MCSamples
    from getdist import chains as getdist_chains

    # Silences getdist's per-call "Removed no burn in" stdout print.
    getdist_chains.print_load_details = False

    ranges = {"x": list(bounds)} if bounds is not None else None
    mc_samples = MCSamples(samples=values, names=["x"], ranges=ranges)
    density = mc_samples.get1DDensity("x")
    low, high, _has_min, _has_top = density.getLimits(hdi_prob)
    return float(low), float(high)
