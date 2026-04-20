"""
smefit.fisher.py

Fisher information matrices, one per dataset, evaluated at the SM point.

All public functions return plain pandas objects:
  - fisher_information_matrices  → dict[str, pd.DataFrame]  (source → n_free x n_free)
  - constraining_power_matrix    → pd.DataFrame  (index=coeff, columns=source)
  - fisher_diagonals_normalised  → pd.DataFrame  (index=coeff, columns=source)
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def fisher_information_matrices(eft_model, data, fit_covmat, ext_chi2_func=None):
    """Compute per-dataset Fisher information matrices at the SM point (c=0).

    For data-based datasets: F_d = J_d^T @ inv_cov_d @ J_d, where J_d is the
    Jacobian of forward_map for dataset d and inv_cov_d is the corresponding
    diagonal block of the inverted fit covariance matrix.

    For external chi2 components (assumed Gaussian): F_ext = 0.5 * H, where H
    is the Hessian of the external chi2 at c=0.

    Parameters
    ----------
    eft_model : EFTModel
    data : DataGroup
    fit_covmat : jnp.ndarray, shape (n_data, n_data)
    ext_chi2_func : list of Chi2, optional

    Returns
    -------
    dict[str, pd.DataFrame]
        Maps each source name to its Fisher matrix as a DataFrame with
        coeff_names as both index and columns.
    """
    coeff_names = eft_model.coefficients.free_names
    n_free = len(coeff_names)
    n_ext = len(ext_chi2_func) if ext_chi2_func is not None else 0
    log.info(
        "Computing Fisher information matrices for %d datasets, %d external chi2, "
        "%d free coefficients.",
        len(data.names),
        n_ext,
        n_free,
    )

    c0 = jnp.zeros(n_free)
    jac = jax.jacobian(eft_model.forward_map)(c0)  # (n_data, n_free)
    inv_covmat = jnp.linalg.inv(fit_covmat)  # (n_data, n_data)

    result = {}
    offset = 0
    for name, n_d in zip(data.names, data.ndata_list):
        jac_d = jac[offset : offset + n_d, :]
        inv_cov_d = inv_covmat[offset : offset + n_d, offset : offset + n_d]
        F_d = np.array(jac_d.T @ inv_cov_d @ jac_d)
        log.debug("  %s: n_d=%d, trace=%.4f", name, n_d, float(np.trace(F_d)))
        result[name] = pd.DataFrame(F_d, index=coeff_names, columns=coeff_names)
        offset += n_d

    if ext_chi2_func is not None:
        for i, ext in enumerate(ext_chi2_func):
            ext_name = ext.name if ext.name is not None else f"ext_chi2_{i}"
            F_ext = np.array(0.5 * jax.hessian(ext)(c0))
            log.debug("  %s: trace=%.4f", ext_name, float(np.trace(F_ext)))
            result[ext_name] = pd.DataFrame(
                F_ext, index=coeff_names, columns=coeff_names
            )

    return result


def constraining_power_matrix(fisher_information_matrices):
    """Compute the constraining power of each source on each coefficient.

    Given the per-source Fisher matrices F_k, forms the total Fisher matrix
    F = sum_k F_k and its inverse Sigma = F^{-1}. The constraining power is::

        alpha[i, k] = (Sigma @ F_k @ Sigma)[i, i] / Sigma[i, i]

    so that sum_k alpha[i, k] = 1 for every coefficient i.

    Parameters
    ----------
    fisher_information_matrices : dict[str, pd.DataFrame]

    Returns
    -------
    pd.DataFrame
        Index = coeff_names, columns = source_names. Rows sum to 1.
    """
    fim = fisher_information_matrices
    coeff_names = next(iter(fim.values())).index.tolist()
    source_names = list(fim.keys())
    matrices = [df.values for df in fim.values()]

    F_total = sum(matrices)
    Sigma = np.linalg.inv(F_total)
    diag_Sigma = np.diag(Sigma)

    alpha = np.stack(
        [np.diag(Sigma @ F_k @ Sigma) / diag_Sigma for F_k in matrices],
        axis=1,
    )  # (n_ops, n_sources)

    return pd.DataFrame(alpha, index=coeff_names, columns=source_names)


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


def aggregate_constraining_power_matrix(constraining_power_matrix, data_groups=None):
    """Aggregate constraining power matrix columns according to data_groups.

    Parameters
    ----------
    constraining_power_matrix : pd.DataFrame
        Index = coeff_names, columns = source_names.
    data_groups : dict[str, list[str]], optional
        Maps group label to source names to merge. Sources not listed in any
        group are kept as individual columns. If None, the original DataFrame is
        returned unchanged.

    Returns
    -------
    pd.DataFrame
        Aggregated matrix whose columns correspond to declared groups followed
        by any ungrouped sources.
    """
    if data_groups is None:
        return constraining_power_matrix

    cpm = constraining_power_matrix
    source_names = cpm.columns.tolist()
    groups = _resolve_groups(source_names, data_groups)
    return pd.DataFrame(
        {
            group_name: cpm.iloc[:, indices].sum(axis=1)
            for group_name, indices in groups
        },
        index=cpm.index,
    )


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
