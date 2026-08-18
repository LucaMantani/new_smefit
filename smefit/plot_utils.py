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
from matplotlib.ticker import ScalarFormatter

from smefit.fit_result import FitResult

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from matplotlib.axes import Axes

    from smefit.fit_result import Fit

log = logging.getLogger(__name__)

# Orders of magnitude a tick label is written out in full between; outside
# them it is written as a mantissa, with the power factored out to the end of
# the axis. Tighter than matplotlib's default (-5, 6), which spells out
# 0.00015 in a panel that has room for about four characters — the labels then
# overlap into an unreadable smear. A Wilson coefficient in TeV^-2 is routinely
# that small, so this is the common case, not the corner one.
_SCI_POWER_LIMITS = (-2, 3)


def compact_tick_labels(
    ax: Axes,
    axis: str = "both",
    show_x_offset: bool = True,
    show_y_offset: bool = True,
    fontsize: float = 18,
) -> None:
    """Keep the tick labels of *ax* short enough to read.

    Small values switch to a mantissa with the shared power written once, at
    the end of the axis, instead of every label carrying its own leading
    zeros.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to reformat.
    axis : {"both", "x", "y"}, optional
        Which of them to reformat. Never an axis that is **not a numeric
        scale**: the bounds plot names its rows after coefficients and the
        reach plot names its columns after them, and a numeric formatter would
        replace those names with the positions they happen to sit at. Nor a
        **log or symlog** axis, whose own formatter already writes powers.
    show_x_offset, show_y_offset : bool, optional
        Whether to show the power at the end of that axis. A grid of panels
        sharing a range labels only its outer ones, and the power belongs with
        the labels: matplotlib draws it regardless of ``labelbottom``, so an
        inner panel would otherwise carry a stray ``x10^-3`` of its own.
    fontsize : float, optional
        Size of that power, which does not follow the tick label size.

    Raises
    ------
    ValueError
        If ``axis`` is not one of the three names.
    """
    if axis not in ("both", "x", "y"):
        raise ValueError(f"axis is 'both', 'x' or 'y', got {axis!r}.")

    wanted = []
    if axis in ("both", "x"):
        wanted.append((ax.xaxis, show_x_offset))
    if axis in ("both", "y"):
        wanted.append((ax.yaxis, show_y_offset))

    for target, show in wanted:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits(_SCI_POWER_LIMITS)
        target.set_major_formatter(formatter)
        target.offsetText.set_visible(show)
        target.offsetText.set_fontsize(fontsize)


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


def contour_coefficients(
    fits: Sequence[Fit],
    params_to_plot: list[str] | str | None = None,
    min_count: int = 2,
) -> list[str]:
    """The coefficients a contour figure of *fits* is about.

    Every coefficient free in **at least one** fit, in the order of the first
    fit followed by each later fit's newcomers. The counterpart of
    :func:`common_free_coefficients`, which the routines reading one posterior
    per panel keep using: a contour panel has something to draw for a fit that
    did not float a coefficient — the value it was stuck at, which
    :func:`stuck_values` locates — so the intersection would hide exactly the
    comparison the panel is for.

    Parameters
    ----------
    fits : sequence of Fit
        The fits to overlay.
    params_to_plot : list of str or str or None
        The coefficients the runcard asked for, the union above by default.
        Named coefficients are looked up among **every** coefficient the fits
        know — the free ones, the fixed and derived ones their samples carry,
        and whatever their runcards declare — so a coefficient no fit floated
        can be asked for by name, and is then drawn stuck in every fit.
    min_count : int, optional
        How many coefficients the caller needs left, two by default: contours
        are pairwise, and a single coefficient has no panel to draw.

    Returns
    -------
    list of str
        At least ``min_count`` coefficient names.

    Raises
    ------
    ValueError
        If no fit has a free coefficient, or fewer than ``min_count`` remain
        after ``params_to_plot``.
    """
    free_union = list(
        dict.fromkeys(name for fit in fits for name in fit.fit_results.free_parameters)
    )
    if not free_union:
        raise ValueError(
            f"None of the fits given ({', '.join(str(fit) for fit in fits)}) "
            "has a free coefficient to draw."
        )

    if params_to_plot is None:
        selected = free_union
    else:
        # a coefficient no fit floated is a legitimate request — it is drawn
        # stuck everywhere — so the pool widens past the free ones, keeping
        # them first so the default order survives
        pool = dict.fromkeys(free_union)
        for fit in fits:
            for name in _known_coefficients(fit):
                pool.setdefault(name)
        selected = select_params(
            list(pool), params_to_plot, context=", ".join(str(fit) for fit in fits)
        )

    if len(selected) < min_count:
        raise ValueError(
            f"This plot needs at least {min_count} coefficients, "
            f"{len(selected)} left: {selected}."
        )
    return selected


def _known_coefficients(fit: Fit) -> list[str]:
    """Every coefficient name *fit* knows anything about.

    The free ones, those its samples carry (a fixed coefficient is resolved
    into the posterior as a constant, a constrained one as a function of the
    free ones) and those its runcard declares — which is the only place a
    coefficient a sampler never saw can appear.
    """
    results = fit.fit_results
    names = list(results.free_parameters)
    if results.samples is not None:
        names += list(results.samples)
    names += list(fit.setting("coefficients", {}) or {})
    return list(dict.fromkeys(names))


def stuck_values(
    fits: Sequence[Fit],
    coeffs: Sequence[str],
    baselines: Mapping[str, float] | None = None,
) -> list[dict[str, float]]:
    """Which of *coeffs* each fit held fixed, and at what value.

    A fit's posterior carries every coefficient of its runcard, not only the
    free ones: :meth:`CoefficientGroup.resolve` fills the fixed ones in, so a
    coefficient frozen with ``free: False, value: v`` reaches
    ``fit_results.json`` as ``v`` repeated once per sample. So a coefficient is
    read as stuck when its samples never move, and as sampled when they do —
    which keeps an ``expr``-constrained coefficient that follows the free ones
    a posterior to contour, as it is, rather than a point.

    A coefficient a fit never declared at all has no samples to read and is
    stuck at its baseline: where that fit's SM sits, which is where the SM
    marker is drawn.

    Parameters
    ----------
    fits : sequence of Fit
        The fits being plotted, each with joint samples.
    coeffs : sequence of str
        The coefficients the figure is about.
    baselines : mapping of str to float, optional
        The SM point per coefficient, from :func:`baseline_point`, which is
        computed by default. Where a coefficient a fit never declared is
        placed.

    Returns
    -------
    list of dict
        One mapping per fit, in *fits* order, holding only the coefficients
        that fit is stuck on. A coefficient absent from a mapping was sampled.
    """
    if baselines is None:
        baselines = baseline_point(fits, coeffs)

    per_fit: list[dict[str, float]] = []
    for fit in fits:
        samples = fit.fit_results.samples
        free = set(fit.fit_results.free_parameters)
        stuck: dict[str, float] = {}
        for name in coeffs:
            values = None if samples is None else samples.get(name)
            array = np.asarray(values, dtype=float) if values is not None else None
            if array is None or array.size == 0:
                stuck[name] = float(baselines.get(name, 0.0))
                continue
            if np.ptp(array) != 0.0:
                continue  # a posterior that moves is one to draw
            if name in free:
                # not a fixed coefficient but an unexplored one: there is no
                # density to estimate, and a KDE would raise on the singular
                # covariance rather than produce a figure
                log.warning(
                    "Coefficient %s is free in fit '%s' but its samples never "
                    "move; drawing it as fixed at %g.",
                    name,
                    fit.fit_name,
                    float(array.flat[0]),
                )
            stuck[name] = float(array.flat[0])
        per_fit.append(stuck)
    return per_fit


def coeff_limits(
    fits: Sequence[Fit],
    coeffs: Sequence[str],
    padding: float = 0.1,
    include_points: Mapping[str, float | Sequence[float]] | None = None,
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
        The coefficients to compute limits for. A fit whose samples do not
        carry one of them is skipped for that coefficient — the contours draw
        such a fit at the value it was stuck at, which reaches the range
        through ``include_points`` instead.
    padding : float, optional
        Fraction of the sample range added on each side, so a contour does
        not touch the frame. A range of zero width — a coefficient every fit
        held fixed — is padded by its own distance from the origin instead,
        and by 1 at the origin itself: a fraction of nothing would leave
        nothing to draw in, and a flat 1 would swamp a coefficient of 1e-3.
    include_points : mapping of str to float or sequence of float, optional
        One point, or several, per coefficient that the range must contain
        whatever the samples do — the SM marker, which :func:`baseline_point`
        locates, and the values a fit held a coefficient at, from
        :func:`stuck_values`, would otherwise fall outside the frame.
        Coefficients absent from the mapping keep their sample range.

    Returns
    -------
    dict of str to tuple of (float, float)
        ``(low, high)`` axis limits per coefficient.

    Raises
    ------
    ValueError
        If a coefficient has neither samples in any fit nor an included point,
        so there is nothing to place it by.
    """
    per_fit_samples = []
    for fit in fits:
        # a coefficient's range is about that coefficient alone, so an
        # individual fit's independently sampled posteriors are read here too
        samples = fit.fit_results.samples
        assert samples is not None  # the actions reject sample-less fits first
        per_fit_samples.append(samples)

    limits: dict[str, tuple[float, float]] = {}
    for name in coeffs:
        pools = [
            np.asarray(samples[name], dtype=float)
            for samples in per_fit_samples
            if name in samples
        ]
        if include_points is not None and name in include_points:
            pools.append(np.atleast_1d(np.asarray(include_points[name], dtype=float)))
        if not pools:
            raise ValueError(
                f"No fit has samples for {name} and no point was given to "
                "place it by, so it has no range to be drawn in."
            )

        values = np.concatenate(pools)
        low, high = float(values.min()), float(values.max())
        if high > low:
            pad = padding * (high - low)
        else:
            pad = abs(low) or 1.0
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


def _baseline_value(fit: Fit, coeff: str) -> float | None:
    """``baseline_value`` of one coefficient in the runcard *fit* was run with.

    Zero when the runcard declares the coefficient without one, which is both
    the runcard default and what a coefficient means when the SM sits at the
    origin of its own parametrisation. None when the runcard does not declare
    the coefficient at all: that fit has no baseline to offer, rather than one
    at the origin, and a fit that did declare it should be asked instead.
    """
    entry = (fit.setting("coefficients", {}) or {}).get(coeff)
    if entry is None:
        return None
    return float((entry or {}).get("baseline_value", 0.0))


def baseline_point(fits: Sequence[Fit], coeffs: Sequence[str]) -> dict[str, float]:
    """Where the SM sits for each of *coeffs*, as the fits were configured.

    A coefficient's ``baseline_value`` is the SM point of its parametrisation,
    zero unless the runcard moved it. It is read back from the runcard the fit
    was run with (``input/runcard.yaml``, kept as ``Fit.fit_runcard``), since
    ``fit_results.json`` does not carry it.

    Parameters
    ----------
    fits : sequence of Fit
        The fits being plotted. The first one to *declare* a coefficient fixes
        its point: the panels overlay every fit against a single SM marker, so
        another fit declaring a different baseline for it is warned about, then
        ignored, while a fit that never declared it has no say.
    coeffs : sequence of str
        The coefficients to look up.

    Returns
    -------
    dict of str to float
        The SM value per coefficient, 0.0 where no baseline was set and where
        no fit declares the coefficient at all.
    """
    per_fit = [{name: _baseline_value(fit, name) for name in coeffs} for fit in fits]

    baselines: dict[str, float] = {}
    for name in coeffs:
        declaring = [
            (fit, values[name])
            for fit, values in zip(fits, per_fit)
            if values[name] is not None
        ]
        if not declaring:
            baselines[name] = 0.0
            continue

        source, value = declaring[0]
        assert value is not None  # filtered above
        baselines[name] = value

        disagreeing = [fit.fit_name for fit, other in declaring[1:] if other != value]
        if disagreeing:
            log.warning(
                "Fits disagree on the SM point of %s: %s puts it at %g, %s "
                "elsewhere. Drawing the first.",
                name,
                source.fit_name,
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
