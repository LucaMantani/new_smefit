"""
smefit.tables.py

This module contains functions for producing tables for reports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from reportengine.table import table

from smefit.credible_intervals import bounds_settings
from smefit.dataset_chi2 import UNGROUPED
from smefit.plot_utils import select_params

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from smefit.fit_result import Fit


@table
def chi2_scan_table(individual_chi2_scans, latex_label):
    """Per-coefficient 1D chi2 scan results as a table.

    Parameters
    ----------
    individual_chi2_scans : list[dict]
        Each entry maps ``{coeff_name: {"points": [...], "chi2": [...]}}``.
    latex_label : Callable[[str], str]
        Writes a name in LaTeX (the ``latex_label`` resource).

    Returns
    -------
    pd.DataFrame
        Rows indexed by scan-point number; MultiIndex columns
        ``(coeff_latex, {"value", "chi2"})`` where ``value`` holds the scan
        points (the coefficient values).
    """
    results = {k: v for d in individual_chi2_scans for k, v in d.items()}
    frames = {
        name: pd.DataFrame({"value": data["points"], "chi2": data["chi2"]})
        for name, data in results.items()
    }
    table = pd.concat(frames, axis=1)
    # relabel only now: as dict keys, two coefficients sharing a label would
    # silently lose one of the scans
    table.columns = pd.MultiIndex.from_tuples(
        [(latex_label(name), column) for name, column in table.columns]
    )
    return table


@table
def mass_scan_table(
    coefficients, individual_mass_scales, individual_mass_scan_points, latex_label
):
    """Mass scan results as a table.

    Parameters
    ----------
    latex_label : Callable[[str], str]
        Writes a name in LaTeX (the ``latex_label`` resource).

    Returns
    -------
    pd.DataFrame
        Columns ``<mass_name>`` (mass scale, the scan points) and ``chi2``.
    """
    mass_name = coefficients.free_names[0]
    latex = latex_label(mass_name)
    return pd.DataFrame(
        {
            latex: [float(s) for s in individual_mass_scales],
            "chi2": [float(c) for c in individual_mass_scan_points],
        }
    )


@table
def fisher_diagonals_normalised(
    aggregate_fisher_information_matrices, latex_label, params_to_plot=None
):
    """Extract row-normalised diagonals of per-source Fisher matrices.

    Parameters
    ----------
    aggregate_fisher_information_matrices : dict[str, pd.DataFrame]
    latex_label : Callable[[str], str]
        Writes a name in LaTeX (the ``latex_label`` resource).
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
    raw.index = [latex_label(name) for name in raw.index]
    raw.columns = [latex_label(name) for name in raw.columns]
    return raw.div(raw.sum(axis=1), axis=0)


@table
def global_information_table(
    global_information_matrix, latex_label, params_to_plot=None
):
    """Share of each coefficient's variance contributed by each source.

    Parameters
    ----------
    global_information_matrix : pd.DataFrame
    latex_label : Callable[[str], str]
        Writes a name in LaTeX (the ``latex_label`` resource).
    params_to_plot : list of str, optional
        Restrict the rows to these coefficients, in this order. All of them by
        default. The shares of a row are taken of that coefficient's variance
        in the full fit, so a row says the same thing whichever others are kept
        alongside it. The full fit leaves out systematics shared across
        datasets: see :func:`smefit.fisher.global_information_matrix`.

    Returns
    -------
    pd.DataFrame
        Index = coefficient labels, columns = source labels. Fractions,
        like :func:`fisher_diagonals_normalised`; rows sum to 1.
    """
    gim = global_information_matrix
    gim = gim.loc[
        select_params(gim.index, params_to_plot, context="global information")
    ]
    gim.index = [latex_label(name) for name in gim.index]
    gim.columns = [latex_label(name) for name in gim.columns]
    return gim


def _number(value: float, round_val: int) -> str:
    """One number as a table cell, to ``round_val`` decimals.

    A string rather than a float: reportengine formats float columns to
    scientific notation (``-1.000E-2``), which reads nothing like the
    ``[low, high]`` intervals beside it.
    """
    text = f"{value:.{round_val}f}"
    # "-0.000" reads as a defect rather than as a number below the precision
    return text.lstrip("-") if float(text) == 0.0 else text


def _bounds_by_coefficient(
    fit: Fit,
    names: list[str],
    round_val: int,
    bounds_levels: float | Sequence[float] | None,
    interval_types: str | Sequence[str] | None,
) -> pd.DataFrame:
    """The bounds of ``fit`` for ``names`` as table cells, indexed by
    coefficient name, in the fit's order.

    Only ``names`` is bounded, so the rows a table leaves out cost nothing;
    those the fit does not have are left out. Choosing ``names``, ordering the
    rows and relabelling them is left to the caller: the comparison table does
    all three on the union of its fits, not fit by fit. The
    columns are there even with no rows, so a fit without samples for any of
    its coefficients still has its columns in a comparison.
    """
    bounds_levels, interval_types = bounds_settings(bounds_levels, interval_types)

    # level outer, type inner: the columns of one level sit side by side,
    # which is the comparison a reader makes
    bounds = {
        (level, interval_type): fit.confidence_bounds(
            level, interval_type, params=names
        )
        for level in bounds_levels
        for interval_type in interval_types
    }
    # .10g: plain g would round 99.99994 (5 sigma) to 99.9999; the type is an
    # acronym, so it is shown upper case: "(ETI)"
    columns = [
        f"{level:.10g}% CL ({interval_type.upper()})" for level, interval_type in bounds
    ]

    # every entry covers the same coefficients, so any one will do
    names = next(iter(bounds.values()))
    rows = {
        name: [
            " ∪ ".join(
                f"[{_number(low, round_val)}, {_number(high, round_val)}]"
                for low, high in per_coeff[name]
            )
            for per_coeff in bounds.values()
        ]
        for name in names
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns)


def _coefficient_bounds_table(
    fit: Fit,
    latex_label: Callable[[str], str],
    params_to_plot: list[str] | str | None,
    round_val: int,
    bounds_levels: float | Sequence[float] | None,
    interval_types: str | Sequence[str] | None,
) -> pd.DataFrame:
    """The typed core of :func:`coefficient_bounds_table`."""
    names = select_params(fit.bounded_parameters, params_to_plot, context=fit.fit_name)
    table = _bounds_by_coefficient(fit, names, round_val, bounds_levels, interval_types)
    table = table.loc[names]
    # relabel only now: as dict keys, two coefficients sharing a label would
    # silently lose one of the rows
    table.index = [latex_label(name) for name in table.index]
    return table


@table
def coefficient_bounds_table(
    fit,
    latex_label,
    params_to_plot=None,
    round_val=3,
    bounds_levels=None,
    interval_types=None,
) -> pd.DataFrame:
    """Tabulate the confidence bounds of one fit.

    Parameters
    ----------
    fit : smefit.fit_result.Fit
        A previously run fit, loaded from disk.
    latex_label : Callable[[str], str]
        Writes a name in LaTeX (the ``latex_label`` resource).
    params_to_plot : list of str, optional
        Restrict the rows to these coefficients, in this order. All of them by
        default.
    round_val : int, optional
        Number of decimals every cell is written with.
    bounds_levels : float or list of float, optional
        The confidence levels quoted, in percent, one column each in this
        order. 68 and 95 by default.
    interval_types : str or list of str, optional
        The credible intervals quoted, among those
        :meth:`Fit.confidence_bounds` accepts: one column per type under each
        level. ``eti`` alone by default.

    Returns
    -------
    pd.DataFrame
        Index = LaTeX coefficient labels, columns = one
        ``<level>% CL (<INTERVAL_TYPE>)`` per level and type, grouped by
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
        fit, latex_label, params_to_plot, round_val, bounds_levels, interval_types
    )


def _fit_column_labels(fits: Sequence[Fit]) -> list[str]:
    """The column label of each of ``fits`` in a comparison table.

    Raises
    ------
    ValueError
        If ``fits`` is empty or two fits share a label: their columns would
        merge.
    """
    if not fits:
        raise ValueError("fits is empty: list at least one fit under 'fits:'.")
    labels = [fit.plot_label for fit in fits]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(
            "Two fits share the column label(s) "
            + ", ".join(repr(label) for label in duplicates)
            + ": give each a distinct 'label' in its fits entry."
        )
    return labels


def _coefficient_bounds_comparison_table(
    fits: Sequence[Fit],
    latex_label: Callable[[str], str],
    params_to_plot: list[str] | str | None,
    round_val: int,
    bounds_levels: float | Sequence[float] | None,
    interval_types: str | Sequence[str] | None,
) -> pd.DataFrame:
    """The typed core of :func:`coefficient_bounds_comparison_table`."""
    labels = _fit_column_labels(fits)
    # selected once, on the union and before any bound is computed:
    # ``params_to_plot`` is global, so a fit having none of the coefficients is
    # an empty column rather than an error, and the order is the requested one
    # whichever fit has which
    names = list(dict.fromkeys(name for fit in fits for name in fit.bounded_parameters))
    context = ", ".join(fit.fit_name for fit in fits)
    names = select_params(names, params_to_plot, context=context)
    # outer join on the rows: a coefficient one fit did not sample is a
    # missing cell, not an error
    comparison = pd.concat(
        [
            _bounds_by_coefficient(fit, names, round_val, bounds_levels, interval_types)
            for fit in fits
        ],
        axis=1,
        keys=labels,
        sort=False,
    )
    comparison = comparison.loc[names]
    comparison.index = [latex_label(name) for name in comparison.index]
    return comparison


@table
def coefficient_bounds_comparison_table(
    fits,
    latex_label,
    params_to_plot=None,
    round_val=3,
    bounds_levels=None,
    interval_types=None,
) -> pd.DataFrame:
    """Tabulate the confidence bounds of several fits side by side.

    Parameters
    ----------
    fits : list of smefit.fit_result.Fit
    latex_label : Callable[[str], str]
        Writes a name in LaTeX (the ``latex_label`` resource).
    params_to_plot : list of str, optional
        Restrict the rows to these coefficients, in this order, keeping those
        at least one of the fits has. All of them by default.
    round_val : int, optional
        Number of decimals every cell is written with.
    bounds_levels : float or list of float, optional
        The confidence levels quoted, in percent, one column each in this
        order. 68 and 95 by default.
    interval_types : str or list of str, optional
        The credible intervals quoted, among those
        :meth:`Fit.confidence_bounds` accepts: one column per type under each
        level. ``eti`` alone by default.

    Returns
    -------
    pd.DataFrame
        Index = LaTeX coefficient labels, the union over the fits, in
        ``params_to_plot`` order if given and in order of first appearance
        otherwise; two-level columns ``(<fit label>, <level>% CL
        (<INTERVAL_TYPE>))``.

    Raises
    ------
    ValueError
        If ``fits`` is empty, two fits share a label, ``params_to_plot`` selects
        nothing from any fit, or ``bounds_levels`` or ``interval_types`` is
        empty or holds a value :meth:`Fit.confidence_bounds` rejects.
    """
    # parameters unannotated: reportengine isinstance-checks every annotated
    # one, which fails on the string annotations of `from __future__`
    return _coefficient_bounds_comparison_table(
        fits, latex_label, params_to_plot, round_val, bounds_levels, interval_types
    )


_NDAT = r"$N_{\mathrm{dat}}$"
_CHI2_PER_POINT = r"$\chi^2/N_{\mathrm{dat}}$"
_TOTAL = "Total"
_ALL = "All"


@table
def datasets_chi2_comparison_table(
    fits, fits_best_fit_datasets_chi2, latex_label, round_val=3
):
    r"""Compare the chi2 per data point of each dataset across fits, in blocks
    by data group.

    Each fit's chi2 is rebuilt from its own setup and evaluated at its stored
    best-fit point (:func:`smefit.dataset_chi2.best_fit_datasets_chi2`): a
    runcard needs only ``fits:``. Each dataset is evaluated on its own block of
    the fit's covariance matrix, so correlations between datasets are left out,
    of the totals too. Datasets without a ``group:`` are tabulated under
    ``Other``. One table rather than one per group, so that every group is
    drawn at the same column widths.

    Parameters
    ----------
    fits : list of smefit.fit_result.Fit
    fits_best_fit_datasets_chi2 : list of pd.DataFrame
        :func:`smefit.dataset_chi2.best_fit_datasets_chi2` of each fit, in the
        order of ``fits``.
    latex_label : Callable[[str], str]
        Writes a name in LaTeX (the ``latex_label`` resource).
    round_val : int, optional
        Number of decimals of each chi2 per data point.

    Returns
    -------
    pd.DataFrame
        Index = (group label, dataset label): a block per group, in order of
        first appearance and ``Other`` last, listing the datasets any fit has
        in it and ending in a ``Total`` row, blocks separated by a blank row
        ``("", "")``; then ``(All, Total)`` over every dataset. Columns
        ``(N_dat, "")`` and ``(chi2/N_dat, <fit label>)``. A ``Total`` is the
        summed chi2 over the summed points of the datasets the fit has; its
        ``N_dat`` counts every dataset listed, so a fit missing one (an empty
        cell) is divided by fewer.

    Raises
    ------
    ValueError
        If ``fits`` is empty, two fits share a label, or two fits disagree on
        the number of points of a dataset.
    """
    labels = _fit_column_labels(fits)
    # one row per (fit, dataset)
    rows = pd.concat(fits_best_fit_datasets_chi2, keys=labels, names=["fit", "dataset"])
    rows = rows.reset_index()

    ndat = rows.groupby("dataset", sort=False)["ndat"].unique()
    differing = ndat[ndat.map(len) > 1]
    if len(differing):
        raise ValueError(
            "The fits disagree on the number of data points of "
            + ", ".join(
                f"'{name}' ({', '.join(map(str, n))})" for name, n in differing.items()
            )
            + ": the chi2 comparison quotes one N_dat per dataset."
        )
    ndat = ndat.map(lambda n: int(n[0]))

    def line(selected, n):
        """``N_dat``, then each fit's summed chi2 over its summed points; empty
        for a fit with none of ``selected`` (or no point to divide by)."""
        by_fit = selected.groupby("fit")[["chi2", "ndat"]].sum()
        return [str(n)] + [
            (
                _number(by_fit.at[label, "chi2"] / by_fit.at[label, "ndat"], round_val)
                if label in by_fit.index and by_fit.at[label, "ndat"] > 0
                else ""
            )
            for label in labels
        ]

    groups = list(dict.fromkeys(rows["group"]))
    if UNGROUPED in groups:
        groups.append(groups.pop(groups.index(UNGROUPED)))

    # a list, not a dict: a dataset or group may be called "Total" or "All"
    index, cells = [], []
    for group in groups:
        in_group = rows[rows["group"] == group]
        # a dataset two fits group differently is listed under both groups
        names = list(dict.fromkeys(in_group["dataset"]))
        for name in names:
            index.append((latex_label(group), latex_label(name)))
            cells.append(line(in_group[in_group["dataset"] == name], ndat[name]))
        index.append((latex_label(group), _TOTAL))
        cells.append(line(in_group, ndat[names].sum()))
        index.append(("", ""))
        cells.append([""] * (1 + len(labels)))
    # ...but counts once in the overall total
    index.append((_ALL, _TOTAL))
    cells.append(line(rows, ndat.sum()))

    return pd.DataFrame(
        cells,
        index=pd.MultiIndex.from_tuples(index),
        columns=pd.MultiIndex.from_tuples(
            [(_NDAT, "")] + [(_CHI2_PER_POINT, label) for label in labels]
        ),
    )


@table
def dataset_table(dataset_info):
    """The datasets of the runcard, as :func:`write_dataset_table` lists them.

    The report counterpart of the LaTeX table: the same rows in the same
    order, and the same total, with the references as links labelled with
    their identifiers, so it needs no INSPIRE lookup.
    The links are markdown.

    Parameters
    ----------
    dataset_info : list of smefit.dataset_info.DatasetInfo

    Returns
    -------
    pd.DataFrame
        Index = dataset names, in the order of ``dataset_info`` (a group's
        datasets together), then ``Total``, whose only cell is the sum of
        ``n_dat``. Columns ``Group``, ``Description``, ``n_dat``,
        ``Luminosity [fb^-1]`` (``--`` when the commondata gives none) and
        ``References``.
    """
    table = pd.DataFrame(
        {
            "Group": [info.group or "" for info in dataset_info],
            "Description": [info.markdown_description() for info in dataset_info],
            "n_dat": [info.num_data for info in dataset_info],
            "Luminosity [fb^-1]": [
                ", ".join(f"{value:g}" for value in info.luminosity) or "--"
                for info in dataset_info
            ],
            "References": [info.markdown_references() for info in dataset_info],
        },
        index=[info.name for info in dataset_info],
    )
    table.loc["Total"] = ["", "", table["n_dat"].sum(), "", ""]
    return table


@table
def pca_components(pca, latex_label):
    """Weight of each coefficient in each principal direction.

    Parameters
    ----------
    pca : smefit.pca.PCA
    latex_label : Callable[[str], str]
        Writes a name in LaTeX (the ``latex_label`` resource).

    Returns
    -------
    pd.DataFrame
        Index = coefficient labels, columns = labels of PC1..PCn.
    """
    frame = pca.as_frame()
    frame.index = [latex_label(name) for name in frame.index]
    frame.columns = [latex_label(name) for name in frame.columns]
    return frame


@table
def pca_spectrum(pca, latex_label):
    """One row per principal direction, strongest first.

    Parameters
    ----------
    pca : smefit.pca.PCA
    latex_label : Callable[[str], str]
        Writes a name in LaTeX (the ``latex_label`` resource).

    Returns
    -------
    pd.DataFrame
        Index = labels of PC1..PCn. ``Sigma`` is the width the data allow along the
        direction, ``Ratio`` its eigenvalue relative to the largest, and
        ``Cumulative`` the share of the total eigenvalue sum reached by that
        row — how much of the constraint the leading directions carry.
        ``Direction`` writes each direction as a combination of the
        coefficient labels.
    """
    eigenvalues = pca.eigenvalues
    return pd.DataFrame(
        {
            "Eigenvalue": eigenvalues,
            "Sigma": pca.constraints,
            "Ratio": pca.eigenvalue_ratios,
            "Cumulative": np.cumsum(eigenvalues) / eigenvalues.sum(),
            "Flat": pca.flat_mask,
            "Direction": [
                pca.describe(i, latex_label) for i in range(pca.n_components)
            ],
        },
        index=[latex_label(name) for name in pca.component_names],
    )
