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
) -> pd.DataFrame:
    """The typed core of :func:`coefficient_bounds_table`."""
    if bounds_levels is None:
        bounds = fit.bounds
    else:
        if isinstance(bounds_levels, (int, float)):
            bounds_levels = [bounds_levels]
        if not bounds_levels:
            raise ValueError("bounds_levels is empty: give at least one level.")
        bounds = {float(level): fit.confidence_bounds(level) for level in bounds_levels}

    # the mean is the same at every level, so any level's will do
    first = next(iter(bounds.values()))
    names = select_params(first, params_to_plot, context=fit.fit_name)

    rows: dict[str, dict[str, str]] = {}
    for name in names:
        cells = {"mean": _number(first[name][1], round_val)}
        for level, per_coeff in bounds.items():
            low, _, high = per_coeff[name]
            # .10g: plain g would round 99.99994 (5 sigma) to 99.9999
            cells[f"{level:.10g}% CL"] = (
                f"[{_number(low, round_val)}, {_number(high, round_val)}]"
            )
        rows[coeff_info_latex.get(name, name)] = cells
    return pd.DataFrame.from_dict(rows, orient="index")


@table
def coefficient_bounds_table(
    fit, params_to_plot=None, round_val=3, bounds_levels=None
) -> pd.DataFrame:
    """Tabulate the posterior mean and the confidence bounds of one fit.

    Takes a single ``fit``, so a runcard listing several under ``fits:`` gets
    one table per fit. The bounds are those of :meth:`Fit.confidence_bounds`.

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
        order. The 68% and 95% of :attr:`Fit.bounds` by default. A list has
        to be a top-level runcard key: the template parser splits action
        arguments on commas.

    Returns
    -------
    pd.DataFrame
        Index = LaTeX coefficient labels, columns = ``mean`` then one
        ``<level>% CL`` per level. Coefficients without samples are left out.

    Raises
    ------
    ValueError
        If ``bounds_levels`` is empty, or holds a level
        :meth:`Fit.confidence_bounds` rejects.
    """
    # parameters unannotated: reportengine isinstance-checks every annotated
    # one, which fails on the string annotations of `from __future__`
    return _coefficient_bounds_table(fit, params_to_plot, round_val, bounds_levels)
