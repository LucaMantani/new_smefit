"""
smefit.tables.py

This module contains functions for producing tables for reports.
"""

import numpy as np
import pandas as pd
from reportengine.table import table

from smefit.op_to_latex import coeff_info_latex
from smefit.plot_utils import select_params


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
