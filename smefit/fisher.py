"""
smefit.fisher.py

Fisher information matrices, one per dataset, evaluated at the SM point.
"""

import logging
from dataclasses import dataclass
from typing import List

import jax
import jax.numpy as jnp

log = logging.getLogger(__name__)


@dataclass
class FisherInformationMatrices:
    """Fisher information matrices with coefficient labelling.

    Attributes
    ----------
    coeff_names : list of str
        Names of the free coefficients. Row/column i of every matrix
        corresponds to ``coeff_names[i]``.
    source_names : list of str
        Names of the sources (datasets and external chi2 components).
    matrices : list of jnp.ndarray
        Per-source Fisher matrices, one per entry in ``source_names``.
        Each array has shape ``(n_free, n_free)``.
    """

    coeff_names: List[str]
    source_names: List[str]
    matrices: List[jnp.ndarray]


@dataclass
class ConstrainingPowerMatrix:
    """Constraining power of each source on each coefficient.

    Attributes
    ----------
    coeff_names : list of str
        Names of the free coefficients (rows of ``alpha``).
    source_names : list of str
        Names of the sources — datasets and external chi2 (columns of ``alpha``).
    alpha : jnp.ndarray, shape (n_ops, n_sources)
        ``alpha[i, k]`` is the fraction of the marginal variance of coefficient
        ``i`` attributable to source ``k``. Rows sum to 1.
    """

    coeff_names: List[str]
    source_names: List[str]
    alpha: jnp.ndarray


@dataclass
class FisherDiagonals:
    """Diagonal entries of per-source Fisher matrices.

    Attributes
    ----------
    coeff_names : list of str
        Names of the free coefficients (rows).
    source_names : list of str
        Names of the sources (columns).
    diagonals : jnp.ndarray, shape (n_free, n_sources)
        ``diagonals[i, k]`` is the row-normalised diagonal entry for coefficient
        ``i`` and source ``k``. Rows sum to 1.
    """

    coeff_names: List[str]
    source_names: List[str]
    diagonals: jnp.ndarray


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
    FisherInformationMatrices
        Object carrying ``coeff_names`` and ``matrices`` (one entry per dataset
        then one per external chi2 component).
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

    source_names = []
    matrices = []
    offset = 0
    for name, n_d in zip(data.names, data.ndata_list):
        jac_d = jac[offset : offset + n_d, :]
        inv_cov_d = inv_covmat[offset : offset + n_d, offset : offset + n_d]
        F_d = jac_d.T @ inv_cov_d @ jac_d
        log.debug("  %s: n_d=%d, trace=%.4f", name, n_d, float(jnp.trace(F_d)))
        source_names.append(name)
        matrices.append(F_d)
        offset += n_d

    if ext_chi2_func is not None:
        for i, ext in enumerate(ext_chi2_func):
            ext_name = ext.name if ext.name is not None else f"ext_chi2_{i}"
            F_ext = 0.5 * jax.hessian(ext)(c0)
            log.debug("  %s: trace=%.4f", ext_name, float(jnp.trace(F_ext)))
            source_names.append(ext_name)
            matrices.append(F_ext)

    return FisherInformationMatrices(
        coeff_names=coeff_names, source_names=source_names, matrices=matrices
    )


def constraining_power_matrix(fisher_information_matrices):
    """Compute the constraining power of each source on each coefficient.

    Given the per-source Fisher matrices F_k, forms the total Fisher matrix
    F = sum_k F_k and its inverse Sigma = F^{-1}. The constraining power is::

        alpha[i, k] = (Sigma @ F_k @ Sigma)[i, i] / Sigma[i, i]

    so that sum_k alpha[i, k] = 1 for every coefficient i.

    Parameters
    ----------
    fisher_information_matrices : FisherInformationMatrices

    Returns
    -------
    ConstrainingPowerMatrix
    """
    coeff_names = fisher_information_matrices.coeff_names
    source_names = fisher_information_matrices.source_names
    fs = fisher_information_matrices.matrices

    F_total = sum(fs)
    Sigma = jnp.linalg.inv(F_total)
    diag_Sigma = jnp.diag(Sigma)

    alpha = jnp.stack(
        [jnp.diag(Sigma @ F_k @ Sigma) / diag_Sigma for F_k in fs],
        axis=1,
    )

    return ConstrainingPowerMatrix(
        coeff_names=coeff_names,
        source_names=source_names,
        alpha=alpha,
    )


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
    """Aggregate FisherInformationMatrices by summing matrices within each group.

    Parameters
    ----------
    fisher_information_matrices : FisherInformationMatrices
    data_groups : dict[str, list[str]], optional
        Maps group label to source names to merge. Sources not listed in any
        group are kept as individual entries. If None, the original object is
        returned unchanged.

    Returns
    -------
    FisherInformationMatrices
    """
    if data_groups is None:
        return fisher_information_matrices

    fim = fisher_information_matrices
    groups = _resolve_groups(fim.source_names, data_groups)
    return FisherInformationMatrices(
        coeff_names=fim.coeff_names,
        source_names=[name for name, _ in groups],
        matrices=[sum(fim.matrices[i] for i in indices) for _, indices in groups],
    )


def aggregate_constraining_power_matrix(constraining_power_matrix, data_groups=None):
    """Aggregate ConstrainingPowerMatrix columns according to data_groups.

    Parameters
    ----------
    constraining_power_matrix : ConstrainingPowerMatrix
    data_groups : dict[str, list[str]], optional
        Maps group label to source names to merge. Sources not listed in any
        group are kept as individual columns. If None, the original object is
        returned unchanged.

    Returns
    -------
    ConstrainingPowerMatrix
        Aggregated matrix whose columns correspond to the declared groups
        followed by any ungrouped sources.
    """
    if data_groups is None:
        return constraining_power_matrix

    cpm = constraining_power_matrix
    groups = _resolve_groups(cpm.source_names, data_groups)
    return ConstrainingPowerMatrix(
        coeff_names=cpm.coeff_names,
        source_names=[name for name, _ in groups],
        alpha=jnp.stack(
            [cpm.alpha[:, jnp.array(indices)].sum(axis=1) for _, indices in groups],
            axis=1,
        ),
    )


def fisher_diagonals_normalised(aggregate_fisher_information_matrices):
    """Extract diagonals of per-source Fisher matrices.

    Parameters
    ----------
    aggregate_fisher_information_matrices : FisherInformationMatrices

    Returns
    -------
    FisherDiagonals
        Array of shape ``(n_free, n_sources)`` where each column is the
        diagonal of the corresponding per-source Fisher matrix.
    """
    fim = aggregate_fisher_information_matrices
    raw = jnp.stack([jnp.diag(F) for F in fim.matrices], axis=1)  # (n_free, n_sources)
    diagonals = raw / raw.sum(axis=1, keepdims=True)
    return FisherDiagonals(
        coeff_names=fim.coeff_names,
        source_names=fim.source_names,
        diagonals=diagonals,
    )
