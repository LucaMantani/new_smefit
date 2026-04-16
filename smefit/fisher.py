"""
smefit.fisher.py

Fisher information matrices, one per dataset, evaluated at the SM point.
"""

import logging

import jax
import jax.numpy as jnp

log = logging.getLogger(__name__)


def fisher_information_matrices(eft_model, data, fit_covmat, ext_chi2_func=None):
    """Compute per-dataset Fisher information matrices at the SM point (c=0).

    Returns a list of (name, matrix) tuples — one per dataset and one per
    external chi2 component (if any). Matrices have shape (n_free, n_free).

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
    list of (str, jnp.ndarray)
        Tuples of (name, Fisher_matrix), one per dataset then one per external
        chi2 component. Matrices have shape (n_free, n_free).
    """
    n_free = len(eft_model.coefficients.free_names)
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

    fisher_matrices = []
    offset = 0
    for name, n_d in zip(data.names, data.ndata_list):
        jac_d = jac[offset : offset + n_d, :]
        inv_cov_d = inv_covmat[offset : offset + n_d, offset : offset + n_d]
        F_d = jac_d.T @ inv_cov_d @ jac_d
        log.debug("  %s: n_d=%d, trace=%.4f", name, n_d, float(jnp.trace(F_d)))
        fisher_matrices.append((name, F_d))
        offset += n_d

    if ext_chi2_func is not None:
        for i, ext in enumerate(ext_chi2_func):
            ext_name = ext.name if ext.name is not None else f"ext_chi2_{i}"
            F_ext = 0.5 * jax.hessian(ext)(c0)
            log.debug("  %s: trace=%.4f", ext_name, float(jnp.trace(F_ext)))
            fisher_matrices.append((ext_name, F_ext))

    return fisher_matrices
