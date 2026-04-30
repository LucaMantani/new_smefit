"""
smefit.fisher.py

This module contains functions for computing Fisher information matrices.
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def fisher_information_matrices(datasets_chi2):
    """Compute per-dataset Fisher information matrices at the SM point (c=0).

    For each source (data-based dataset or external chi2), the Fisher matrix is
    F = 0.5 * H where H is the Hessian of the chi2 at c=0.

    Parameters
    ----------
    datasets_chi2 : list of Chi2
        Per-dataset chi2 objects (including any external chi2 contributions).

    Returns
    -------
    dict[str, pd.DataFrame]
        Maps each source name to its Fisher matrix as a DataFrame with
        coeff_names as both index and columns.
    """
    coeff_names = datasets_chi2[0].param_names
    n_free = len(coeff_names)
    c0 = jnp.zeros(n_free)
    log.info(
        "Computing Fisher information matrices for %d sources, %d free coefficients.",
        len(datasets_chi2),
        n_free,
    )

    result = {}
    for chi2 in datasets_chi2:
        F = np.array(0.5 * jax.hessian(chi2)(c0))
        result[chi2.name] = pd.DataFrame(F, index=coeff_names, columns=coeff_names)

    return result


def _resolve_groups(source_names, data_groups):
    """Map source_names to grouped (name, indices) pairs according to data_groups.

    Declared groups come first (in data_groups order), followed by ungrouped
    sources as individual entries. Groups that match no source are skipped with
    a warning.

    Parameters
    ----------
    source_names : list of str
    data_groups : dict[str, list[str]]

    Returns
    -------
    list of (str, list[int])
    """
    assigned: set = set()
    groups = []
    for group_name, members in data_groups.items():
        indices = [i for i, s in enumerate(source_names) if s in members]
        if not indices:
            log.warning("data_groups: group '%s' matched no sources.", group_name)
            continue
        groups.append((group_name, indices))
        assigned.update(indices)
    for i, name in enumerate(source_names):
        if i not in assigned:
            groups.append((name, [i]))
    return groups


def aggregate_fisher_information_matrices(
    fisher_information_matrices, data_groups=None
):
    """Aggregate Fisher matrices by summing within each group.

    Parameters
    ----------
    fisher_information_matrices : dict[str, pd.DataFrame]
    data_groups : dict[str, list[str]], optional
        Maps group label to source names to merge. Sources not listed in any
        group are kept as individual entries. If None, the original dict is
        returned unchanged.

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    if data_groups is None:
        return fisher_information_matrices

    fim = fisher_information_matrices
    source_names = list(fim.keys())
    groups = _resolve_groups(source_names, data_groups)
    return {
        group_name: sum(fim[source_names[i]] for i in indices)
        for group_name, indices in groups
    }


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
        index=coeff_names,
    )
    return raw.div(raw.sum(axis=1), axis=0)
