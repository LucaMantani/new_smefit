"""
smefit.plot_utils.py

Helpers shared by the plotting routines in :mod:`smefit.figures`.

Everything here operates on :class:`~smefit.fit_result.Fit` objects,
whether they were produced by a fit in this run or loaded from a fit directory
with ``Fit.from_json``. Loading fits is that class' job; deciding what to
draw is this module's.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


def latex_label(name):
    """Turn a plain fit name into a LaTeX-safe label.

    Figures are rendered with ``usetex``, so underscores in raw fit names have
    to be escaped.
    """
    escaped = str(name).replace("_", r"\_")
    return rf"$\mathrm{{{escaped}}}$"


def default_labels(fit_results) -> List[str]:
    """Legend label of every fit, before any per-plot override.

    The ``label`` of the fit's runcard entry when it has one — taken verbatim,
    so it may be raw LaTeX — and its escaped name otherwise.
    """
    return [
        fit_result.label if fit_result.label else latex_label(fit_result.fit_name)
        for fit_result in fit_results
    ]


def plot_coefficients(fit_result) -> List[str]:
    """Coefficients of a fit that are plotted by default.

    The free ones, or every sampled one when the fit does not record which
    parameters were free. Coefficients that are fixed or derived are kept in the
    samples but are not plotted.
    """
    columns = list(fit_result.samples_frame.columns)
    if fit_result.free_parameters:
        return [name for name in fit_result.free_parameters if name in columns]
    return columns


def require_samples(fit_results) -> None:
    """Raise unless every fit carries posterior samples to plot."""
    for fit_result in fit_results:
        if fit_result.samples is None or fit_result.samples_frame.empty:
            raise ValueError(
                f"Fit '{fit_result.fit_name}' has no posterior samples, nothing to "
                "plot. Use a sampler that returns samples (e.g. ultranest, "
                "blackjax, hessian or analytic)."
            )


def require_joint_posterior(fit_results) -> None:
    """Raise on fits whose coefficients were not sampled jointly.

    Individual (one-at-a-time) fits vary a single coefficient with the others
    held at their baseline. Their merged samples are independent 1D posteriors,
    so pairing them up produces a correlation set by the random seed and
    nothing else — any 2D contour drawn from them would be an artefact.
    """
    for fit_result in fit_results:
        if fit_result.individual_fit:
            raise ValueError(
                f"Fit '{fit_result.fit_name}' is a set of individual (one-at-a-time) "
                "fits: each coefficient was fitted with the others held at their "
                "baseline, so there is no joint posterior and 2D contours would "
                "show a correlation that is an artefact of the sampling. Plot one "
                "of its individual_fits/<coefficient> subdirectories instead."
            )


def common_coefficients(fit_results, dofs_show: Optional[List[str]] = None):
    """Coefficients to show for a set of fits.

    Parameters
    ----------
    fit_results : list of Fit
    dofs_show : list of str, optional
        Explicit selection. When given, every coefficient must be present in
        every fit.

    Returns
    -------
    list of str
        Coefficients present in all fits, in the order of the first one (or in
        the requested order when ``dofs_show`` is given).
    """
    if dofs_show is not None:
        for fit_result in fit_results:
            missing = [
                c for c in dofs_show if c not in fit_result.samples_frame.columns
            ]
            if missing:
                raise ValueError(
                    f"Coefficients {missing} requested in dofs_show are not "
                    f"sampled in fit '{fit_result.fit_name}'."
                )
        return list(dofs_show)

    per_fit = [plot_coefficients(fit_result) for fit_result in fit_results]
    common = [name for name in per_fit[0] if all(name in c for c in per_fit[1:])]
    if not common:
        raise ValueError("The fits given have no free coefficient in common.")

    dropped = {name for coeffs in per_fit for name in coeffs if name not in common}
    if dropped:
        log.warning(
            "Coefficients %s are not fitted in all fits and are not shown. "
            "Use 'dofs_show' to select explicitly.",
            sorted(dropped),
        )
    return common


def coeff_limits(fit_results, coeffs, padding=0.1, include_sm=True):
    """Axis limits per coefficient, common to every fit and every panel."""
    limits = {}
    for name in coeffs:
        values = np.concatenate(
            [
                np.asarray(fit_result.samples_frame[name].values, dtype=float)
                for fit_result in fit_results
            ]
        )
        low, high = values.min(), values.max()
        if include_sm:  # keep the SM marker inside the frame
            low, high = min(low, 0.0), max(high, 0.0)
        pad = padding * (high - low) if high > low else 1.0
        limits[name] = (low - pad, high + pad)
    return limits


def per_fit_option(option, fit_results, defaults):
    """Resolve a plot option that may be set globally or per fit.

    Parameters
    ----------
    option : None, dict or any
        ``None`` falls back to *defaults*; a dict keyed by fit name sets the
        option for the named fits only; any other value applies to every fit.
    fit_results : list of Fit
    defaults : list
        Per-fit default, one entry per fit.

    Returns
    -------
    list
        One resolved value per fit.
    """
    if option is None:
        return list(defaults)
    if isinstance(option, dict):
        names = [fit_result.fit_name for fit_result in fit_results]
        unknown = [key for key in option if key not in names]
        if unknown:
            log.warning(
                "Plot option given for %s, which are not among the fits plotted "
                "(%s) — ignored.",
                unknown,
                names,
            )
        return [option.get(name, default) for name, default in zip(names, defaults)]
    return [option for _ in fit_results]


def best_fit_pair(fit_result, coeff1, coeff2) -> Optional[tuple]:
    """Best-fit ``(coeff1, coeff2)`` pair, or None if either is missing."""
    best_fit: Dict[str, float] = fit_result.best_fit_point or {}
    if coeff1 in best_fit and coeff2 in best_fit:
        return (float(best_fit[coeff1]), float(best_fit[coeff2]))
    return None
