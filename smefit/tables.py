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
    fit: Fit, params_to_plot: list[str] | str | None, round_val: int
) -> pd.DataFrame:
    """The typed core of :func:`coefficient_bounds_table`."""
    bounds = fit.bounds
    names = select_params(bounds[68.0], params_to_plot, context=fit.fit_name)

    rows: dict[str, dict[str, str]] = {}
    for name in names:
        cells = {"mean": _number(bounds[68.0][name][1], round_val)}
        for level, per_coeff in bounds.items():
            low, _, high = per_coeff[name]
            cells[f"{level:g}% CL"] = (
                f"[{_number(low, round_val)}, {_number(high, round_val)}]"
            )
        rows[coeff_info_latex.get(name, name)] = cells
    return pd.DataFrame.from_dict(rows, orient="index")


@table
def coefficient_bounds_table(fit, params_to_plot=None, round_val=3) -> pd.DataFrame:
    """Tabulate the posterior mean and the 68% and 95% CL bounds of one fit.

    Takes a single ``fit``, so a runcard listing several under ``fits:`` gets
    one table per fit. The bounds are those of :attr:`Fit.bounds`.

    Parameters
    ----------
    fit : smefit.fit_result.Fit
        A previously run fit, loaded from disk.
    params_to_plot : list of str, optional
        Restrict the rows to these coefficients, in this order. All of them by
        default.
    round_val : int, optional
        Number of decimals every cell is written with.

    Returns
    -------
    pd.DataFrame
        Index = LaTeX coefficient labels, columns = ``mean``, ``68% CL``,
        ``95% CL``. Coefficients without samples are left out.
    """
    # parameters unannotated: reportengine isinstance-checks every annotated
    # one, which fails on the string annotations of `from __future__`
    return _coefficient_bounds_table(fit, params_to_plot, round_val)
