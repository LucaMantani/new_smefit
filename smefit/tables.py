"""
smefit.tables.py

This module contains functions for producing tables for reports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from reportengine.table import table

from smefit.op_to_latex import coeff_info_latex
from smefit.plot_utils import select_params

if TYPE_CHECKING:
    from collections.abc import Sequence

    from smefit.fit_result import Fit


@table
def chi2_scan_table(individual_chi2_scans):
    """Per-coefficient 1D chi2 scan results as a table.

    Parameters
    ----------
    individual_chi2_scans : list[dict]
        Each entry maps ``{coeff_name: {"points": [...], "chi2": [...]}}``.

    Returns
    -------
    pd.DataFrame
        Rows indexed by scan-point number; MultiIndex columns
        ``(coeff_latex, {"value", "chi2"})`` where ``value`` holds the scan
        points (the coefficient values).
    """
    results = {k: v for d in individual_chi2_scans for k, v in d.items()}
    frames = {
        coeff_info_latex.get(name, name): pd.DataFrame(
            {"value": data["points"], "chi2": data["chi2"]}
        )
        for name, data in results.items()
    }
    return pd.concat(frames, axis=1)


@table
def mass_scan_table(coefficients, individual_mass_scales, individual_mass_scan_points):
    """Mass scan results as a table.

    Returns
    -------
    pd.DataFrame
        Columns ``<mass_name>`` (mass scale, the scan points) and ``chi2``.
    """
    mass_name = coefficients.free_names[0]
    latex = coeff_info_latex.get(mass_name, mass_name)
    return pd.DataFrame(
        {
            latex: [float(s) for s in individual_mass_scales],
            "chi2": [float(c) for c in individual_mass_scan_points],
        }
    )


@table
def fisher_diagonals_normalised(
    aggregate_fisher_information_matrices, params_to_plot=None
):
    """Extract row-normalised diagonals of per-source Fisher matrices.

    Parameters
    ----------
    aggregate_fisher_information_matrices : dict[str, pd.DataFrame]
    params_to_plot : list of str, optional
        Restrict the rows to these coefficients, in this order. All of them by
        default. Each row is normalised on its own, so a row says the same
        thing whichever others are kept alongside it.

    Returns
    -------
    pd.DataFrame
        Index = coeff_names, columns = source_names. Rows sum to 1.
    """
    fim = aggregate_fisher_information_matrices
    coeff_names = next(iter(fim.values())).index.tolist()
    raw = pd.DataFrame(
        {name: np.diag(df.values) for name, df in fim.items()},
        index=coeff_names,
    )
    raw = raw.loc[select_params(coeff_names, params_to_plot, context="Fisher")]
    raw.index = [coeff_info_latex.get(name, name) for name in raw.index]
    return raw.div(raw.sum(axis=1), axis=0)


def _number(value: float, round_val: int) -> str:
    """One number as a table cell, to ``round_val`` decimals.

    A string rather than a float: reportengine formats float columns to
    scientific notation (``-1.000E-2``), which reads nothing like the
    ``[low, high]`` intervals beside it.
    """
    text = f"{value:.{round_val}f}"
    # "-0.000" reads as a defect rather than as a number below the precision
    return text.lstrip("-") if float(text) == 0.0 else text


def _coefficient_bounds_table(
    fit: Fit,
    params_to_plot: list[str] | str | None,
    round_val: int,
    bounds_levels: float | Sequence[float] | None,
    interval_types: str | Sequence[str] | None,
) -> pd.DataFrame:
    """The typed core of :func:`coefficient_bounds_table`."""
    if bounds_levels is None:
        bounds_levels = [68.0, 95.0]
    elif isinstance(bounds_levels, (int, float)):
        bounds_levels = [bounds_levels]
    if not bounds_levels:
        raise ValueError("bounds_levels is empty: give at least one level.")
    if interval_types is None:
        interval_types = ["eti"]
    elif isinstance(interval_types, str):
        interval_types = [interval_types]
    if not interval_types:
        raise ValueError("interval_types is empty: give at least one type.")

    # level outer, type inner: the columns of one level sit side by side,
    # which is the comparison a reader makes
    bounds = {
        (float(level), interval_type): fit.confidence_bounds(level, interval_type)
        for level in bounds_levels
        for interval_type in interval_types
    }

    # the mean is the same at every level and type, so any entry's will do
    first = next(iter(bounds.values()))
    names = select_params(first, params_to_plot, context=fit.fit_name)

    rows: dict[str, dict[str, str]] = {}
    for name in names:
        cells = {"mean": _number(first[name]["mean"], round_val)}
        for (level, interval_type), per_coeff in bounds.items():
            # .10g: plain g would round 99.99994 (5 sigma) to 99.9999
            cells[f"{level:.10g}% CL ({interval_type})"] = " ∪ ".join(
                f"[{_number(low, round_val)}, {_number(high, round_val)}]"
                for low, high in per_coeff[name]["intervals"]
            )
        rows[coeff_info_latex.get(name, name)] = cells
    return pd.DataFrame.from_dict(rows, orient="index")


@table
def coefficient_bounds_table(
    fit, params_to_plot=None, round_val=3, bounds_levels=None, interval_types=None
) -> pd.DataFrame:
    """Tabulate the posterior mean and the confidence bounds of one fit.

    Takes a single ``fit``, so a runcard listing several under ``fits:`` gets
    one table per fit. The bounds are those of :meth:`Fit.confidence_bounds`,
    of the credible interval(s) ``interval_types`` names.

    Parameters
    ----------
    fit : smefit.fit_result.Fit
        A previously run fit, loaded from disk.
    params_to_plot : list of str, optional
        Restrict the rows to these coefficients, in this order. All of them by
        default.
    round_val : int, optional
        Number of decimals every cell is written with.
    bounds_levels : float or list of float, optional
        The confidence levels quoted, in percent, one column each in this
        order. 68 and 95 by default. A list has to be a top-level runcard
        key: the template parser splits action arguments on commas.
    interval_types : str or list of str, optional
        The credible intervals quoted, among those
        :meth:`Fit.confidence_bounds` accepts: one column per type under each
        level, in this order, so several types sit side by side for
        comparison. ``eti`` alone by default. Like ``bounds_levels``, a list
        has to be a top-level runcard key.

    Returns
    -------
    pd.DataFrame
        Index = LaTeX coefficient labels, columns = ``mean`` then one
        ``<level>% CL (<interval_type>)`` per level and type, grouped by
        level, each cell the pieces of the region as ``[low, high]``, joined
        by ``∪`` when there are several. Coefficients without samples are
        left out.

    Raises
    ------
    ValueError
        If ``bounds_levels`` or ``interval_types`` is empty, or holds a value
        :meth:`Fit.confidence_bounds` rejects.
    """
    # parameters unannotated: reportengine isinstance-checks every annotated
    # one, which fails on the string annotations of `from __future__`
    return _coefficient_bounds_table(
        fit, params_to_plot, round_val, bounds_levels, interval_types
    )


@table
def pca_components(pca):
    """Weight of each coefficient in each principal direction.

    Parameters
    ----------
    pca : smefit.pca.PCA

    Returns
    -------
    pd.DataFrame
        Index = coeff_names, columns = PC1..PCn.
    """
    frame = pca.as_frame()
    frame.index = [coeff_info_latex.get(name, name) for name in frame.index]
    return frame


@table
def pca_spectrum(pca):
    """One row per principal direction, strongest first.

    Parameters
    ----------
    pca : smefit.pca.PCA

    Returns
    -------
    pd.DataFrame
        Index = PC1..PCn. ``Sigma`` is the width the data allow along the
        direction, ``Ratio`` its eigenvalue relative to the largest, and
        ``Cumulative`` the share of the total eigenvalue sum reached by that
        row — how much of the constraint the leading directions carry.
    """
    eigenvalues = pca.eigenvalues
    return pd.DataFrame(
        {
            "Eigenvalue": eigenvalues,
            "Sigma": pca.constraints,
            "Ratio": pca.eigenvalue_ratios,
            "Cumulative": np.cumsum(eigenvalues) / eigenvalues.sum(),
            "Flat": pca.flat_mask,
            "Direction": [pca.describe(i) for i in range(pca.n_components)],
        },
        index=pca.component_names,
    )
