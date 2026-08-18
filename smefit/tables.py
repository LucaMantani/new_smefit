"""
smefit.tables.py

This module contains functions for producing tables for reports.
"""

import numpy as np
import pandas as pd
from reportengine.table import table

from smefit.bounds_1d import coeff_bounds, mass_reach
from smefit.latex_table import latex_table
from smefit.op_to_latex import coeff_info_latex
from smefit.plot_utils import common_free_coefficients, per_fit_option, select_params

# The confidence level the bounds columns quote unless the runcard says
# otherwise. One level, so that a table is readable as it comes; the old
# `write_cl_table` quoted 68% and 95% side by side, which `bounds_levels` still
# does on request.
_BOUNDS_LEVELS = (95.0,)

# Header of the reach column. Written with an explicit power rather than
# \sqrt: a report is rendered to HTML by pandoc, whose plain-HTML math writer
# has no square root and falls back to printing the raw TeX source at the
# reader. Every other label in the table is LaTeX too, so it stays LaTeX.
_REACH_COLUMN = r"$\Lambda/c_i^{1/2}$ [TeV]"

# What a cell holds when there is no number to put in it: a coefficient this
# fit never sampled, a second solution it does not have, or a bound too wide to
# invert. Blank would read as a zero.
_MISSING = "—"


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


def _number(value, round_val):
    """One number as a table cell.

    A string, not a float, and so is every other numeric cell: reportengine
    formats a float column to four significant figures (``-1.000E-2``, ``-0``),
    which reads nothing like the ``[low, high]`` interval beside it. Fixed
    decimals instead, so a column of numbers lines up and says how precise it
    is.
    """
    text = f"{value:.{round_val}f}"
    # a small negative value prints as "-0.00", which reads as a defect rather
    # than as a number too small for the precision asked for
    return text.lstrip("-") if float(text) == 0.0 else text


def _interval(bounds, round_val):
    """One confidence interval as the ``[low, high]`` a table quotes."""
    return f"[{_number(bounds.low, round_val)}, {_number(bounds.high, round_val)}]"


def _caption(bounds_levels, show_bounds, show_reach, confidence_level):
    """What the LaTeX table says about itself when the runcard does not.

    Written from the columns the table actually has, so a caption cannot
    promise a column that was switched off.
    """
    said = []
    if show_bounds:
        levels = " and ".join(rf"{level:.10g}\%" for level in bounds_levels)
        said.append(rf"best-fit values and {levels} CL intervals")
    if show_reach:
        said.append(
            rf"the scale $\Lambda/\sqrt{{c_i}}$ probed at "
            rf"{float(confidence_level):.10g}\% CL"
        )
    return "Coefficient bounds per fit: " + ", and ".join(said) + "."


def _level_column(level):
    r"""Header of the bounds columns of one confidence level.

    ``.10g`` rather than ``g``: a 5-sigma level is 99.99994%, and the default
    six significant figures would round its name to ``100\% CL``.
    """
    return rf"{level:.10g}\% CL"


@latex_table
def coefficient_bounds_table(
    fits,
    params_to_plot=None,
    double_solution=None,
    round_val=3,
    show_bounds=True,
    bounds_levels=None,
    show_reach=False,
    confidence_level=95,
    full_interval=False,
    latex_caption=None,
    latex_label="tab:coefficient_bounds",
):
    r"""Tabulate the coefficient bounds of every fit.

    The numbers of the bounds plot and the mass-reach plot, read off the same
    percentile intervals, so a report cannot quote one thing and draw another.
    Takes the whole ``fits`` list, so it is called bare in a report template.

    Which columns a fit gets is chosen in the template, per call —
    ``{@coefficient_bounds_table(show_reach=True)@}`` — since both switches are
    plain booleans and scalars survive the template argument parser. As with
    every action option, a top-level runcard key works too and applies to every
    call.

    Accepts a fit run one coefficient at a time, like the bounds figures: a
    bound is about one coefficient at a time. Which of the two a table is about
    is which fit directory its ``fits:`` entry points at.

    Parameters
    ----------
    fits : list of smefit.fit_result.Fit
        The previously run fits to tabulate, each column group headed with the
        ``label`` of its ``fits`` entry (its name otherwise).
    params_to_plot : list of str, optional
        Restrict the rows to these coefficients, in this order. All the
        coefficients the fits share by default.
    double_solution : list of str or dict, optional
        Coefficients whose posterior has two disjoint solutions. Each branch
        gets a row of its own, the second marked ``(2)``; without this,
        equal-tailed percentiles would span the empty gap between the modes. A
        dict keyed by fit name sets the list per fit.
    round_val : int, optional
        Decimals every number is written to. Cells are strings at fixed
        decimals, so a column lines up and says how precise it is.
    show_bounds : bool, optional
        Contribute the ``best`` column and one ``<level>% CL`` column per
        entry of ``bounds_levels``, on by default. Intervals are written
        ``[low, high]``.
    bounds_levels : float or list of float, optional
        The confidence levels the bounds columns quote, in percent and in the
        order they should be read. ``95`` alone by default; a list gives a
        column each — ``[68, 95]``, as the old report quoted them — and any
        level works, ``99.99994`` for 5 sigma. Each column is named after its
        own level, so a table always says what it quotes.
    show_reach : bool, optional
        Contribute the :math:`\Lambda/\sqrt{c_i}` column, off by default. It
        is the reach of the mass-reach plot, at ``confidence_level``. Its
        header is written with an explicit power, see ``_REACH_COLUMN``.
    confidence_level : float, optional
        Confidence level in percent of the interval **the reach** is read off,
        95 by default — the reach is one column, so it is one level. The
        bounds columns have their own, ``bounds_levels``, and this does not
        touch them.
    full_interval : bool, optional
        Take the reach bound to be the whole interval rather than half of it,
        as in the mass-reach plot.
    latex_caption : str, optional
        Caption of the LaTeX version of the table — the one the report's "Copy
        LaTeX" button copies and ``tables/<name>.tex`` holds. Written from the
        columns the table has when not given.
    latex_label : str, optional
        ``\\label`` of that same table. Pass None to leave it out.

    Returns
    -------
    pd.DataFrame
        Index = LaTeX coefficient labels, a second row per coefficient with a
        second solution. Columns = a ``(fit label, column)`` MultiIndex.

    Raises
    ------
    ValueError
        If there is nothing to tabulate: no fits, both switches off, a fit that
        stored no posterior samples, or no shared coefficient left.
    """
    if not fits:
        raise ValueError("No fits to tabulate.")
    if not show_bounds and not show_reach:
        raise ValueError(
            "coefficient_bounds_table has nothing to show: turn on "
            "show_bounds, show_reach, or both."
        )

    coeffs = common_free_coefficients(fits, params_to_plot, min_count=1)
    doubles = per_fit_option(double_solution, fits, [[] for _ in fits])

    if bounds_levels is None:
        bounds_levels = _BOUNDS_LEVELS
    elif isinstance(bounds_levels, (int, float)):
        bounds_levels = [bounds_levels]
    # the order they were asked for is the order the columns read in
    bounds_levels = list(dict.fromkeys(float(level) for level in bounds_levels))

    levels = list(bounds_levels) if show_bounds else []
    if show_reach and float(confidence_level) not in levels:
        levels.append(float(confidence_level))

    per_fit = [
        coeff_bounds(fit, coeffs, levels, double_solution=doubles[idx])
        for idx, fit in enumerate(fits)
    ]

    # a coefficient split in one fit only still needs the row its second
    # solution goes in, so the row count is the most any fit resolved
    n_solutions = {
        name: max((len(bounds.get(name, [])) for bounds in per_fit), default=1)
        for name in coeffs
    }

    column_names = [
        *(
            ["best", *(_level_column(level) for level in bounds_levels)]
            if show_bounds
            else []
        ),
        *([_REACH_COLUMN] if show_reach else []),
    ]

    columns, data = [], []
    for fit, bounds in zip(fits, per_fit):
        cells_per_column = [[] for _ in column_names]

        for name in coeffs:
            solutions = bounds.get(name, [])
            for index in range(n_solutions[name]):
                # this fit may have fewer solutions than another one did
                solution = solutions[index] if index < len(solutions) else None
                row = []
                if show_bounds:
                    row.extend(
                        [_MISSING] * (1 + len(bounds_levels))
                        if solution is None
                        else [
                            # the central value is one number however many
                            # intervals are quoted around it
                            _number(solution[bounds_levels[0]].mid, round_val),
                            *(
                                _interval(solution[level], round_val)
                                for level in bounds_levels
                            ),
                        ]
                    )
                if show_reach:
                    reach = (
                        float("nan")
                        if solution is None
                        else mass_reach(
                            solution[float(confidence_level)], full_interval
                        )
                    )
                    row.append(
                        _MISSING if np.isnan(reach) else _number(reach, round_val)
                    )
                for cells, cell in zip(cells_per_column, row):
                    cells.append(cell)

        columns.extend((fit.plot_label, column) for column in column_names)
        data.extend(cells_per_column)

    index = []
    for name in coeffs:
        label = coeff_info_latex.get(name, name)
        # the second branch is marked rather than repeated: two rows under one
        # label would read as two readings of the same thing
        index.extend([label] + [f"{label} (2)"] * (n_solutions[name] - 1))

    frame = pd.DataFrame(
        np.array(data, dtype=object).T,
        index=index,
        columns=pd.MultiIndex.from_tuples(columns),
    )
    # how the LaTeX version of this table captions itself; reportengine hands
    # `attrs` to the final action that renders it
    frame.attrs["latex_caption"] = (
        _caption(bounds_levels, show_bounds, show_reach, confidence_level)
        if latex_caption is None
        else latex_caption
    )
    frame.attrs["latex_label"] = latex_label
    return frame
