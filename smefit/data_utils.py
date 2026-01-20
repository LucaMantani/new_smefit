"""
smefit.data_utils.py

Data utilities module of smefit, containing functions for data processing.
"""

import jax.numpy as jnp
import jax.scipy.linalg as la
import pandas as pd

INTRA_DATASET_SYS_NAME = ("UNCORR", "CORR", "THEORYUNCORR", "THEORYCORR")


def construct_covmat(stat_errors: jnp.ndarray, sys_errors: pd.DataFrame):
    """
    Basic function to construct a covariance matrix (covmat), given the
    statistical error and a dataframe of systematics.
    Errors with name UNCORR or THEORYUNCORR are added in quadrature with
    the statistical error to the diagonal of the covmat.
    Other systematics are treated as correlated; their covmat contribution is
    found by multiplying them by their transpose.

    Parameters
    ----------
        stat_errors: jnp.ndarray
            a 1-D array of statistical uncertainties
        sys_errors: pandas.DataFrame
            a dataframe with shape (N_data * N_sys) and systematic name as the
            column headers. The uncertainties should be in the same units as the
            data.

    Returns
    -------
        cov_mat: jnp.ndarray
            Covariance matrix

    Notes
    -----
    This function doesn't contain any logic to ignore certain contributions to
    the covmat, if you wanted to not include a particular systematic/set of
    systematics i.e all uncertainties with MULT errors, then filter those out
    of ``sys_errors`` before passing that to this function.

    """
    diagonal = stat_errors**2

    is_uncorr = sys_errors.columns.isin(("UNCORR", "THEORYUNCORR"))
    diagonal += jnp.asarray((sys_errors.loc[:, is_uncorr].to_numpy() ** 2).sum(axis=1))

    corr_sys_mat = jnp.asarray(sys_errors.loc[:, ~is_uncorr].to_numpy())
    return jnp.diag(diagonal) + corr_sys_mat @ corr_sys_mat.T


def covmat_from_systematics(stat_errors: list, sys_errors: list):
    """Given two lists containing the statistic and systematic errors,
    construct the full covariance matrix.

    This is similar to :py:meth:`construct_covmat`
    except that special corr systematics are concatenated across all datasets
    before being multiplied by their transpose to give off block-diagonal
    contributions. The other systematics contribute to the block diagonal in the
    same way as :py:meth:`construct_covmat`.

    Parameters
    ----------
    stat_errors : list[(stat_errors: jnp.ndarray)]
        list of stat_errors for each dataset.

    sys_errors : list[(sys_errors: pd.DataFrame)]
        list of sys_errors for each dataset.

    Returns
    -------
    cov_mat : jnp.ndarray
        JAX array which is N_dat x N_dat (where N_dat is the number of data points)
        containing uncertainty and correlation information.
    """
    special_corrs = []
    block_diags = []

    for dataset_stat_errors, dataset_sys_errors in zip(stat_errors, sys_errors):
        # separate out the special uncertainties which can be correlated across
        # datasets
        is_intra_dataset_error = dataset_sys_errors.columns.isin(INTRA_DATASET_SYS_NAME)
        block_diags.append(
            construct_covmat(
                dataset_stat_errors, dataset_sys_errors.loc[:, is_intra_dataset_error]
            )
        )
        special_corrs.append(dataset_sys_errors.loc[:, ~is_intra_dataset_error])

    # concat systematics across datasets
    special_sys = pd.concat(special_corrs, axis=0, sort=False)
    # non-overlapping systematics are set to NaN by concat, fill with 0 instead.
    special_sys.fillna(0, inplace=True)

    diag = la.block_diag(*block_diags)

    special_sys_arr = jnp.asarray(special_sys.to_numpy())
    covmat = diag + special_sys_arr @ special_sys_arr.T
    return covmat
