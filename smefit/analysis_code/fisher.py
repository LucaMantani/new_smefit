"""
smefit.fisher.py

Fisher Information Matrix computation and plotting module for smefit.
"""


def FIM(fit_covmat, linear_matrix_correction):
    J = linear_matrix_correction
    CJ = jnp.linalg.solve(fit_covmat, J)  # C^{-1} J
    fim = J.T @ CJ
    return 0.5 * (fim + fim.T)  # numeric symmetry cleanup
