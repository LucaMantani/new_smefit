"""
smefit.tables.py

This module contains functions for producing tables for reports.
"""

import numpy as np
import pandas as pd
from reportengine.table import table

from smefit.op_to_latex import coeff_info_latex


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


@table
def posterior_correlations(fit):
    """Posterior correlations of one fit's free coefficients.

    Takes a single ``fit``, so a runcard listing several under ``fits:`` gets
    one table per fit.

    Parameters
    ----------
    fit : smefit.fit_result.Fit
        A previously run fit, loaded from disk.

    Returns
    -------
    pd.DataFrame
        Square, index and columns both the free coefficients, labelled in
        LaTeX where the operator is known.
    """
    if fit.individual_fit:
        raise ValueError(
            f"Fit '{fit.fit_name}' was run one coefficient at a time, so its "
            "coefficients were never sampled together and there is no joint "
            "posterior to correlate."
        )

    corr = fit.fit_results.correlations
    labels = [coeff_info_latex.get(name, name) for name in corr.index]
    return corr.set_axis(labels, axis=0).set_axis(labels, axis=1)
