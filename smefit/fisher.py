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


def fisher_information_matrices(datasets_chi2, gd_best_fit):
    """Compute per-dataset Fisher information matrices at the best-fit point.

    For each source (data-based dataset or external chi2), the Fisher matrix is
    F = 0.5 * H where H is the Hessian of the chi2 evaluated at ``gd_best_fit``.
    To evaluate at the coefficient baseline values, set ``sm_solution: true`` in the
    ``gradient_descent_settings`` runcard block.

    Parameters
    ----------
    datasets_chi2 : list of Chi2
        Per-dataset chi2 objects (including any external chi2 contributions).
    gd_best_fit : jnp.ndarray
        Best-fit coefficient vector of shape ``(n_free,)`` produced by the
        ``gd_best_fit`` node.

    Returns
    -------
    dict[str, pd.DataFrame]
        Maps each source name to its Fisher matrix as a DataFrame with
        coeff_names as both index and columns.
    """
    coeff_names = datasets_chi2[0].param_names
    c0 = gd_best_fit
    log.info(
        "Computing Fisher information matrices for %d sources, %d free coefficients.",
        len(datasets_chi2),
        len(coeff_names),
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
