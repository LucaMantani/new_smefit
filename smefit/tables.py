"""
smefit.tables.py

This module contains functions for producing tables for reports.
"""

import numpy as np
import pandas as pd
from reportengine.table import table

from smefit.op_to_latex import coeff_info_latex


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
def fisher_diagonals_normalised(aggregate_fisher_information_matrices):
    """Extract row-normalised diagonals of per-source Fisher matrices.

    Parameters
    ----------
    aggregate_fisher_information_matrices : dict[str, pd.DataFrame]

    Returns
    -------
    pd.DataFrame
        Index = coeff_names, columns = source_names. Rows sum to 1.
    """
    fim = aggregate_fisher_information_matrices
    coeff_names = next(iter(fim.values())).index.tolist()
    raw = pd.DataFrame(
        {name: np.diag(df.values) for name, df in fim.items()},
        index=[coeff_info_latex.get(name, name) for name in coeff_names],
    )
    return raw.div(raw.sum(axis=1), axis=0)
