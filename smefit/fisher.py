"""
smefit.fisher.py

This module contains functions for computing Fisher information matrices.
"""

import logging

import jax
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
        F = np.array(0.5 * jax.jit(jax.hessian(chi2))(c0))
        result[chi2.name] = pd.DataFrame(F, index=coeff_names, columns=coeff_names)

    return result


def total_fisher_information_matrix(chi2, gd_best_fit):
    """Fisher information matrix of the full likelihood at the best-fit point.

    ``F = 0.5 * H``, with ``H`` the Hessian of the total chi2.

    To evaluate at the coefficient baseline values instead of a fitted minimum,
    set ``sm_solution: true`` in the ``gradient_descent_settings`` block.

    Parameters
    ----------
    chi2 : Chi2
        The total chi2 (data and any external contributions).
    gd_best_fit : jnp.ndarray
        Best-fit coefficient vector of shape ``(n_free,)``.

    Returns
    -------
    pd.DataFrame
        Square, with coeff_names as both index and columns.
    """
    F = np.array(0.5 * jax.hessian(chi2)(gd_best_fit))
    return pd.DataFrame(F, index=chi2.param_names, columns=chi2.param_names)


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


############################
# Global Information Matrix
############################

# Same criterion, and default, as ``pca_settings.threshold``: an eigenvalue below
# this fraction of the largest is a flat direction. Not read from pca_settings,
# which would make every report runcard carry that block.
_FLAT_THRESHOLD = 1e-12
# Coefficients with at least this weight in a flat direction are named in the error.
_FLAT_NAME_WEIGHT = 0.1


# Below this a share is negative, not rounding: shares are O(1) fractions and
# rounding leaves them off by O(machine_eps).
_NEGATIVE_SHARE_TOL = 1e-6


def _warn_on_negative_shares(gim):
    """Log the entries of ``gim`` that are negative beyond rounding.

    Parameters
    ----------
    gim : pd.DataFrame
        Index = coeff_names, columns = source_names.
    """
    negative = gim.stack()
    negative = negative[negative < -_NEGATIVE_SHARE_TOL]
    if negative.empty:
        return
    log.warning(
        "Global information matrix: %d negative share(s), from sources whose "
        "Hessian has negative curvature at the evaluation point:\n%s",
        len(negative),
        "\n".join(
            f"  {coeff} <- {source}: {value:.1%}"
            for (coeff, source), value in negative.items()
        ),
    )


def _check_invertible(fisher, coeff_names):
    """Raise if ``fisher`` has a flat or negative-curvature direction.

    Along such a direction the total Fisher matrix is no covariance's inverse,
    so the variance a share is taken of is infinite or negative, and
    ``np.linalg.inv`` returns large numbers rather than an error.

    Parameters
    ----------
    fisher : np.ndarray
        The total Fisher matrix, square.
    coeff_names : list of str

    Raises
    ------
    ValueError
        Naming the coefficients that make up each offending direction.
    """
    # symmetrise: eigh reads one triangle only, see smefit.pca
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (fisher + fisher.T))
    lam_max = eigenvalues.max()
    bad = np.flatnonzero(eigenvalues <= _FLAT_THRESHOLD * max(lam_max, 0.0))
    if bad.size == 0:
        return

    directions = []
    for i in bad:
        v = eigenvectors[:, i]
        names = [
            coeff_names[j]
            for j in np.argsort(-np.abs(v))
            if abs(v[j]) >= _FLAT_NAME_WEIGHT
        ]
        directions.append(f"  lambda = {eigenvalues[i]:.3e}: {', '.join(names)}")
    raise ValueError(
        f"Global information matrix: the total Fisher matrix has {bad.size} "
        f"flat or negative-curvature direction(s), so the variances it would "
        f"split between sources are not defined (lambda_max = {lam_max:.3e}):\n"
        + "\n".join(directions)
        + "\nFix or remove these coefficients, or add data constraining them; "
        "pca_spectrum shows the full spectrum."
    )


def global_information_matrix(aggregate_fisher_information_matrices):
    """Compute the constraining power of each source on each coefficient.

    Given the per-source Fisher matrices F_k, forms the total Fisher matrix
    F = sum_k F_k and its inverse Sigma = F^{-1}. The constraining power is::

        alpha[i, k] = (Sigma @ F_k @ Sigma)[i, i] / Sigma[i, i]

    so that sum_k alpha[i, k] = 1 for every coefficient i.

    A share is non-negative when F_k is positive semi-definite, as it is for a
    chi2 linear in the coefficients. With quadratic corrections, or away from
    the SM point, one source's Hessian can have negative curvature, and its
    share of some coefficient comes out negative: that source pulls the
    variance up rather than down, which the others make up for. Such entries
    are kept and logged with a warning.

    The sources are those of ``aggregate_fisher_information_matrices``, so
    ``data_groups`` are merged before the attribution, not after. Here the two
    agree, ``alpha`` being linear in F_k, but an attribution that is not (one
    evaluated numerically on a non-Gaussian posterior, say) has to see each
    group as one source.

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
    source_names = list(fim.keys())
    matrices = [df.values for df in fim.values()]

    F_total = sum(matrices)
    _check_invertible(F_total, coeff_names)
    Sigma = np.linalg.inv(F_total)
    diag_Sigma = np.diag(Sigma)

    alpha = np.stack(
        [np.diag(Sigma @ F_k @ Sigma) / diag_Sigma for F_k in matrices],
        axis=1,
    )  # (n_ops, n_sources)

    gim = pd.DataFrame(alpha, index=coeff_names, columns=source_names)
    _warn_on_negative_shares(gim)
    return gim
