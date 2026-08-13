"""
smefit.plot_utils.py

Helpers shared by the report figures and tables.

Deliberately not a provider module (it is absent from ``smefit_providers`` in
``smefit/app.py``): it holds helpers the report routines call, not nodes
reportengine resolves.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from smefit.fit_result import FitResult

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from smefit.fit_result import Fit

log = logging.getLogger(__name__)


def _joint_results(fit: Fit) -> FitResult:
    """The joint :class:`FitResult` of *fit*.

    The contour helpers below read fields only a joint fit has — samples
    drawn together, one best-fit point. The actions reject anything else
    (individual fits, fits without samples) with the user-facing error before
    calling them; the assert restates that contract for the reader and the
    type checker, it is not the error path.
    """
    assert isinstance(fit.fit_results, FitResult)
    return fit.fit_results


def select_params(
    names: Iterable[str],
    params_to_plot: list[str] | str | None,
    context: str | None = None,
) -> list[str]:
    """Return the subset of *names* to show, in ``params_to_plot`` order.

    Parameters
    ----------
    names : iterable of str
        The coefficients the caller has, in its own order.
    params_to_plot : list of str or str or None
        The coefficients the runcard asked for. None selects everything, which
        leaves *names* untouched. A bare string is taken as a one-element list,
        so ``params_to_plot: OtG`` works in a runcard.
    context : str, optional
        What is being selected from — a fit name, say. Only used to say which
        plot a message is about.

    Returns
    -------
    list of str
        The requested names that *names* has, in requested order, deduplicated.
        Names it does not have are dropped: one global ``params_to_plot`` is
        meant to serve several fits, which need not have fitted the same
        coefficients, so each shows the largest subset it can.

    Raises
    ------
    ValueError
        If nothing at all was selected. An empty plot is a misspelt list rather
        than a subset anybody asked for.
    """
    names = list(names)
    if params_to_plot is None:
        return names

    if isinstance(params_to_plot, str):
        params_to_plot = [params_to_plot]
    requested = list(dict.fromkeys(params_to_plot))

    available = set(names)
    selected = [p for p in requested if p in available]
    dropped = [p for p in requested if p not in available]

    where = f" of '{context}'" if context else ""
    if dropped:
        log.info(
            "params_to_plot: %s not among the coefficients%s, skipping.",
            ", ".join(dropped),
            where,
        )
    if not selected:
        raise ValueError(
            f"params_to_plot selected none of the coefficients{where}: "
            f"asked for {requested}, available are {names}."
        )
    return selected


def common_free_coefficients(
    fits: Sequence[Fit],
    params_to_plot: list[str] | str | None = None,
    min_count: int = 2,
) -> list[str]:
    """The coefficients an overlaid plot of *fits* is about.

    The free coefficients every fit shares, in the order of the first one: a
    panel of a coefficient some fit never sampled would have nothing to show
    for that fit. This is stricter than :func:`select_params` alone, which
    lets each figure keep whatever subset it has — panels of one overlaid
    figure cannot, so the intersection is taken first and coefficients not
    free in every fit are dropped with a warning. ``params_to_plot`` then
    restricts and orders the intersection, through :func:`select_params` like
    every other report routine.

    Parameters
    ----------
    fits : sequence of Fit
        The fits to overlay. An individual (one-at-a-time) fit reports the
        coefficients it fitted one by one, which is what a routine reading 1D
        posteriors needs; rejecting one that cannot be read that way — a
        contour — is its consumer's job.
    params_to_plot : list of str or str or None
        The coefficients the runcard asked for, all common ones by default.
    min_count : int, optional
        How many coefficients the caller needs left. Two by default, for the
        pairwise routines: contours and correlations have no panel to draw of
        a single coefficient. The 1D bounds routines pass 1.

    Returns
    -------
    list of str
        At least ``min_count`` coefficient names, free in every fit.

    Raises
    ------
    ValueError
        If the fits have no free coefficient in common, or fewer than
        ``min_count`` remain after ``params_to_plot``.
    """
    per_fit = [list(fit.fit_results.free_parameters) for fit in fits]
    common = [name for name in per_fit[0] if all(name in rest for rest in per_fit[1:])]
    if not common:
        raise ValueError(
            f"The fits given ({', '.join(str(fit) for fit in fits)}) have no "
            "free coefficient in common."
        )

    dropped = sorted({name for coeffs in per_fit for name in coeffs} - set(common))
    if dropped:
        log.warning(
            "Coefficients %s are not free in every fit and are not shown. "
            "Fit the same coefficients everywhere to compare them.",
            ", ".join(dropped),
        )

    selected = select_params(
        common, params_to_plot, context=", ".join(str(fit) for fit in fits)
    )
    if len(selected) < min_count:
        raise ValueError(
            f"This plot needs at least {min_count} coefficients, "
            f"{len(selected)} left: {selected}."
        )
    return selected


def coeff_limits(
    fits: Sequence[Fit],
    coeffs: Sequence[str],
    padding: float = 0.1,
    include_points: Mapping[str, float] | None = None,
) -> dict[str, tuple[float, float]]:
    """Axis limits per coefficient, shared by every fit and every panel.

    A coefficient spans the same range wherever it appears — on the x-axis of
    one panel and the y-axis of another — so the limits are computed once per
    coefficient over the pooled samples of every fit, not per panel.

    Parameters
    ----------
    fits : sequence of Fit
        The fits whose samples set the range.
    coeffs : sequence of str
        The coefficients to compute limits for. Every fit must have samples
        for each of them — :func:`common_free_coefficients` guarantees that.
    padding : float, optional
        Fraction of the sample range added on each side, so a contour does
        not touch the frame. A range of zero width is padded by 1 instead:
        a fraction of nothing would leave nothing to draw in.
    include_points : mapping of str to float, optional
        A point per coefficient the range must contain, whatever the samples
        do — the SM marker, which :func:`baseline_point` locates, would
        otherwise fall outside the frame of a coefficient sampled away from
        it. Coefficients absent from the mapping keep their sample range.

    Returns
    -------
    dict of str to tuple of (float, float)
        ``(low, high)`` axis limits per coefficient.
    """
    per_fit_samples = []
    for fit in fits:
        samples = _joint_results(fit).samples
        assert samples is not None  # the actions reject sample-less fits first
        per_fit_samples.append(samples)

    limits: dict[str, tuple[float, float]] = {}
    for name in coeffs:
        values = np.concatenate(
            [np.asarray(samples[name], dtype=float) for samples in per_fit_samples]
        )
        low, high = float(values.min()), float(values.max())
        if include_points is not None and name in include_points:
            point = float(include_points[name])
            low, high = min(low, point), max(high, point)
        pad = padding * (high - low) if high > low else 1.0
        limits[name] = (low - pad, high + pad)
    return limits


def per_fit_option(
    option: Any, fits: Sequence[Fit], defaults: Sequence[Any]
) -> list[Any]:
    """Resolve a plot option that may be set globally or per fit.

    Parameters
    ----------
    option : None, dict or any
        ``None`` falls back to *defaults*; a dict keyed by fit name sets the
        option for the named fits only, the others keeping their default; any
        other value applies to every fit.
    fits : sequence of Fit
        The fits being plotted, whose names key the dict form.
    defaults : sequence
        Per-fit default, one entry per fit.

    Returns
    -------
    list
        One resolved value per fit, in *fits* order.
    """
    if option is None:
        return list(defaults)
    if isinstance(option, dict):
        names = [fit.fit_name for fit in fits]
        unknown = [key for key in option if key not in names]
        if unknown:
            log.warning(
                "Plot option given for %s, which are not among the fits "
                "plotted (%s) — ignored.",
                ", ".join(unknown),
                ", ".join(names),
            )
        return [option.get(name, default) for name, default in zip(names, defaults)]
    return [option for _ in fits]


def _baseline_value(fit: Fit, coeff: str) -> float:
    """``baseline_value`` of one coefficient in the runcard *fit* was run with.

    Zero when the runcard leaves it out, which is both the runcard default and
    what a coefficient means when the SM sits at the origin of its own
    parametrisation.
    """
    entry = fit.setting("coefficients", {}).get(coeff) or {}
    return float(entry.get("baseline_value", 0.0))


def baseline_point(fits: Sequence[Fit], coeffs: Sequence[str]) -> dict[str, float]:
    """Where the SM sits for each of *coeffs*, as the fits were configured.

    A coefficient's ``baseline_value`` is the SM point of its parametrisation,
    zero unless the runcard moved it. It is read back from the runcard the fit
    was run with (``input/runcard.yaml``, kept as ``Fit.fit_runcard``), since
    ``fit_results.json`` does not carry it.

    Parameters
    ----------
    fits : sequence of Fit
        The fits being plotted. The first one fixes the point: the panels
        overlay every fit against a single SM marker, so a fit that declares a
        different baseline for the same coefficient is warned about, then
        ignored.
    coeffs : sequence of str
        The coefficients to look up.

    Returns
    -------
    dict of str to float
        The SM value per coefficient, 0.0 where no baseline was set.
    """
    per_fit = [{name: _baseline_value(fit, name) for name in coeffs} for fit in fits]
    baselines = per_fit[0]

    for name, value in baselines.items():
        disagreeing = [
            fit.fit_name
            for fit, values in zip(fits[1:], per_fit[1:])
            if values[name] != value
        ]
        if disagreeing:
            log.warning(
                "Fits disagree on the SM point of %s: %s puts it at %g, %s "
                "elsewhere. Drawing the first.",
                name,
                fits[0].fit_name,
                value,
                ", ".join(disagreeing),
            )

    return baselines


def best_fit_pair(fit: Fit, coeff1: str, coeff2: str) -> tuple[float, float] | None:
    """Best-fit ``(coeff1, coeff2)`` of a fit, or None if either is missing.

    None rather than an error: the consumer falls back to the posterior means
    then, which every fit with samples can offer.
    """
    best_fit = _joint_results(fit).best_fit_point or {}
    if coeff1 in best_fit and coeff2 in best_fit:
        return float(best_fit[coeff1]), float(best_fit[coeff2])
    return None
