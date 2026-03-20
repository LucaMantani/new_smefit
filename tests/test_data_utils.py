"""Unit tests for smefit.data_utils — construct_covmat and covmat_from_systematics."""

import jax.numpy as jnp
import pandas as pd
import pytest

from smefit.data_utils import construct_covmat, covmat_from_systematics

# ---------------------------------------------------------------------------
# construct_covmat
# ---------------------------------------------------------------------------


def test_construct_covmat_stat_only():
    stat = jnp.array([1.0, 2.0, 3.0])
    sys = pd.DataFrame(index=range(3))  # no columns → no systematics
    result = construct_covmat(stat, sys)
    expected = jnp.diag(stat**2)
    assert jnp.allclose(result, expected)


def test_construct_covmat_uncorr_sys():
    stat = jnp.array([1.0, 2.0, 3.0])
    sys = pd.DataFrame({"UNCORR": [0.5, 0.5, 0.5]})
    result = construct_covmat(stat, sys)
    # diagonal = stat^2 + UNCORR^2 = [1+0.25, 4+0.25, 9+0.25]
    expected = jnp.diag(jnp.array([1.25, 4.25, 9.25]))
    assert jnp.allclose(result, expected)


def test_construct_covmat_corr_sys():
    stat = jnp.array([1.0, 2.0])
    sys = pd.DataFrame({"MY_CORR": [1.0, 1.0]})
    result = construct_covmat(stat, sys)
    # diagonal from stat: [1, 4]; corr contribution: [[1,1],[1,1]]
    expected = jnp.array([[2.0, 1.0], [1.0, 5.0]])
    assert jnp.allclose(result, expected)


def test_construct_covmat_symmetry():
    stat = jnp.array([1.0, 2.0, 3.0])
    sys = pd.DataFrame(
        {
            "MY_CORR": [1.0, 0.5, 0.3],
            "UNCORR": [0.1, 0.2, 0.3],
        }
    )
    result = construct_covmat(stat, sys)
    assert jnp.allclose(result, result.T)


# ---------------------------------------------------------------------------
# covmat_from_systematics
# ---------------------------------------------------------------------------


def test_covmat_from_systematics_block_diag():
    """Two datasets with only intra-dataset systematics → strictly block-diagonal."""
    stat1 = jnp.array([1.0, 2.0])
    sys1 = pd.DataFrame({"UNCORR": [0.1, 0.2]})
    stat2 = jnp.array([3.0, 4.0])
    sys2 = pd.DataFrame({"UNCORR": [0.3, 0.4]})
    result = covmat_from_systematics([stat1, stat2], [sys1, sys2])
    assert jnp.allclose(result[0:2, 2:4], jnp.zeros((2, 2)))
    assert jnp.allclose(result[2:4, 0:2], jnp.zeros((2, 2)))


def test_covmat_from_systematics_cross_dataset():
    """Shared cross-dataset systematic name → non-zero off-diagonal block."""
    stat1 = jnp.array([1.0, 2.0])
    sys1 = pd.DataFrame({"CROSS_SYS": [1.0, 0.5]})
    stat2 = jnp.array([3.0, 4.0])
    sys2 = pd.DataFrame({"CROSS_SYS": [0.3, 0.4]})
    result = covmat_from_systematics([stat1, stat2], [sys1, sys2])
    # Off-diagonal block must be non-zero
    assert not jnp.allclose(result[0:2, 2:4], jnp.zeros((2, 2)))


def test_covmat_from_systematics_total_size():
    stat1 = jnp.array([1.0, 2.0, 3.0])
    sys1 = pd.DataFrame({"UNCORR": [0.1, 0.2, 0.3]})
    stat2 = jnp.array([4.0, 5.0])
    sys2 = pd.DataFrame({"UNCORR": [0.4, 0.5]})
    result = covmat_from_systematics([stat1, stat2], [sys1, sys2])
    assert result.shape == (5, 5)
