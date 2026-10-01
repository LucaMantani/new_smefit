"""
smefit.tables.py

This module contains functions for producing tables for reports.
"""

import numpy as np
import pandas as pd
from reportengine.table import table

from smefit.latex_labels import latex_label
from smefit.plot_utils import select_params


@table
def chi2_scan_table(individual_chi2_scans, latex_labels=None):
    """Per-coefficient 1D chi2 scan results as a table.

    Parameters
    ----------
    individual_chi2_scans : list[dict]
        Each entry maps ``{coeff_name: {"points": [...], "chi2": [...]}}``.
    latex_labels : dict[str, str], optional
        Runcard overrides of the coefficient labels.

    Returns
    -------
    pd.DataFrame
        Rows indexed by scan-point number; MultiIndex columns
        ``(coeff_latex, {"value", "chi2"})`` where ``value`` holds the scan
        points (the coefficient values).
    """
    results = {k: v for d in individual_chi2_scans for k, v in d.items()}
    frames = {
        latex_label(name, latex_labels): pd.DataFrame(
            {"value": data["points"], "chi2": data["chi2"]}
        )
        for name, data in results.items()
    }
    return pd.concat(frames, axis=1)


@table
def mass_scan_table(
    coefficients, individual_mass_scales, individual_mass_scan_points, latex_labels=None
):
    """Mass scan results as a table.

    Parameters
    ----------
    latex_labels : dict[str, str], optional
        Runcard overrides of the coefficient labels.

    Returns
    -------
    pd.DataFrame
        Columns ``<mass_name>`` (mass scale, the scan points) and ``chi2``.
    """
    mass_name = coefficients.free_names[0]
    latex = latex_label(mass_name, latex_labels)
    return pd.DataFrame(
        {
            latex: [float(s) for s in individual_mass_scales],
            "chi2": [float(c) for c in individual_mass_scan_points],
        }
    )


@table
def fisher_diagonals_normalised(
    aggregate_fisher_information_matrices, params_to_plot=None, latex_labels=None
):
    """Extract row-normalised diagonals of per-source Fisher matrices.

    Parameters
    ----------
    aggregate_fisher_information_matrices : dict[str, pd.DataFrame]
    params_to_plot : list of str, optional
        Restrict the rows to these coefficients, in this order. All of them by
        default. Each row is normalised on its own, so a row says the same
        thing whichever others are kept alongside it.
    latex_labels : dict[str, str], optional
        Runcard overrides of the labels, for the coefficients and the sources
        (data groups) alike.

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
    raw.index = [latex_label(name, latex_labels) for name in raw.index]
    raw.columns = [latex_label(name, latex_labels) for name in raw.columns]
    return raw.div(raw.sum(axis=1), axis=0)


@table
def pca_components(pca, latex_labels=None):
    """Weight of each coefficient in each principal direction.

    Parameters
    ----------
    pca : smefit.pca.PCA
    latex_labels : dict[str, str], optional
        Runcard overrides of the coefficient labels.

    Returns
    -------
    pd.DataFrame
        Index = coeff_names, columns = PC1..PCn.
    """
    frame = pca.as_frame()
    frame.index = [latex_label(name, latex_labels) for name in frame.index]
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
