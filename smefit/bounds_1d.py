"""
smefit.bounds_1d.py

Low-level primitives for 1D marginalised confidence bounds.

Every posterior-bounds report routine — the 1D histograms, the central-value
plot, the mass-reach bars, the bounds table — is about the same three numbers
per coefficient and confidence level: ``(low, mid, high)``. They are computed
here, once, so that a figure and the table beside it cannot disagree.

Bounds are **equal-tailed percentiles** with the sample **mean** as the central
value (68% → 16/84, 95% → 2.5/97.5), matching the old report pipeline. No
highest-density intervals: they need a dependency (``arviz``) that only the
deferred spider plot ever used.

Deliberately not a provider module (it is absent from ``smefit_providers`` in
``smefit/app.py``), mirroring :mod:`smefit.contours_2d`: it holds the numerics
the report routines call, not nodes reportengine resolves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from numpy.typing import ArrayLike

    from smefit.fit_result import Fit


class Bounds(NamedTuple):
    """A confidence interval and the central value inside it.

    Attributes
    ----------
    low, high : float
        The equal-tailed percentiles bracketing the confidence level.
    mid : float
        The sample mean — the central value every consumer marks.
    """

    low: float
    mid: float
    high: float


def confidence_bounds(samples: ArrayLike, confidence_level: float) -> Bounds:
    """The ``confidence_level`` percent bounds of a 1D posterior.

    Parameters
    ----------
    samples : array_like
        ``(N,)`` posterior samples of one coefficient.
    confidence_level : float
        Confidence level in percent, so 68 or 95 rather than 0.68 or 0.95.

    Returns
    -------
    Bounds
        ``(low, mid, high)``: the equal-tailed percentiles of the samples
        around the level, and their mean.

    Raises
    ------
    ValueError
        If the level is not a percentage inside ``[1, 100)``, or if there are
        no samples to read a percentile off. The lower end is 1 rather than 0
        to catch ``confidence_level: 0.95``, a plausible way to write it that
        would otherwise quietly draw a 0.95% interval; a genuine sub-percent
        confidence level is not something a report is ever about.
    """
    if not 1.0 <= confidence_level < 100.0:
        raise ValueError(
            f"confidence_level is a percentage between 1 and 100, got "
            f"{confidence_level}. Write 95, not 0.95."
        )

    values = np.asarray(samples, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot compute bounds of an empty sample.")

    tail = (100.0 - confidence_level) / 2.0
    low, high = np.nanpercentile(values, [tail, 100.0 - tail])
    return Bounds(float(low), float(np.nanmean(values)), float(high))


def split_solution(samples: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Split a bimodal posterior into its two branches, nearest zero first.

    The old midpoint-of-range split: samples are cut at the middle of their
    range, which separates two well-isolated modes — the case quadratic EFT
    corrections produce. The branch whose closest sample to the origin is
    closest wins the first slot, so that "solution 1" is the SM-like one
    wherever this is used.

    Equal-tailed percentiles of the *unsplit* samples would span the empty gap
    between the modes and put the mean where the posterior has no mass, which
    is why the 1D bounds need this even though the 2D contours (whose KDE
    level is calibrated on the samples themselves) do not.

    Parameters
    ----------
    samples : array_like
        ``(N,)`` posterior samples of one coefficient.

    Returns
    -------
    tuple of np.ndarray
        The two branches. A sample sitting exactly at the midpoint goes to the
        upper branch, so between them they hold every sample.

    Raises
    ------
    ValueError
        If either branch comes out empty: the coefficient was declared to have
        two solutions but its posterior only has one, and silently returning
        an empty branch would put NaN bounds in a plot.
    """
    values = np.asarray(samples, dtype=float)
    mid = 0.5 * (values.min() + values.max())

    solution1 = values[values < mid]
    solution2 = values[values >= mid]
    if solution1.size == 0 or solution2.size == 0:
        raise ValueError(
            "Cannot split this posterior in two: every sample falls on one "
            "side of the midpoint of its range. Remove the coefficient from "
            "double_solution — its posterior has a single solution."
        )

    # solution 1 is the one closer to the SM
    if np.abs(solution2).min() < np.abs(solution1).min():
        solution1, solution2 = solution2, solution1

    return solution1, solution2


def coeff_bounds(
    fit: Fit,
    coeffs: Sequence[str],
    confidence_levels: float | Sequence[float],
    double_solution: Iterable[str] | str | None = None,
) -> dict[str, list[dict[float, Bounds]]]:
    """Confidence bounds of *fit*, per coefficient, solution and level.

    Works for a joint fit and for an individual (one-at-a-time) one alike:
    both hold a 1D posterior per coefficient — a :class:`FitResultGroup`
    exposes its own under the same ``samples`` name — which is all a bound is
    about. That is how "individual bounds" reach the report: a ``fits:`` entry
    pointing at an ``individual_fits`` output. So, unlike the contour
    routines, no consumer of this needs to reject :attr:`Fit.individual_fit`.

    Parameters
    ----------
    fit : Fit
        The fit to read the posteriors of.
    coeffs : sequence of str
        The coefficients to compute bounds for. Ones this fit has no samples
        for are left out of the result rather than filled with NaN: a fit that
        never sampled a coefficient has nothing to say about it, and consumers
        show a gap.
    confidence_levels : float or sequence of float
        One or more levels in percent.
    double_solution : iterable of str or str, optional
        Coefficients whose posterior is to be split in two by
        :func:`split_solution` before the bounds are taken. As in the old
        pipeline this is a manual per-fit list, not an automatic detection: a
        posterior is bimodal because of the physics, not because a criterion
        fired on this particular set of samples.

    Returns
    -------
    dict of str to list of dict of float to Bounds
        Per coefficient, one entry per solution — a single one unless the
        coefficient is in ``double_solution``, in which case solution 0 is the
        branch nearest the SM — mapping each requested level to its
        :class:`Bounds`.

    Raises
    ------
    ValueError
        If the fit stored no posterior samples at all, naming it: there is
        nothing to take a percentile of, and that is a fit run with the wrong
        routine rather than a coefficient missing from one.
    """
    if isinstance(confidence_levels, (int, float)):
        confidence_levels = [float(confidence_levels)]
    levels = [float(level) for level in confidence_levels]

    if double_solution is None:
        double_solution = []
    elif isinstance(double_solution, str):
        double_solution = [double_solution]
    disjoint = set(double_solution)

    samples = fit.fit_results.samples
    if not samples:
        raise ValueError(
            f"The fit '{fit.fit_name}' stored no posterior samples, so it has "
            "no bounds to report."
        )

    bounds: dict[str, list[dict[float, Bounds]]] = {}
    for name in coeffs:
        if name not in samples:
            continue
        values = np.asarray(samples[name], dtype=float)
        solutions = split_solution(values) if name in disjoint else (values,)
        bounds[name] = [
            {level: confidence_bounds(solution, level) for level in levels}
            for solution in solutions
        ]
    return bounds
