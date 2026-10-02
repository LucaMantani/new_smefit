"""Shared exact-Gaussian test problem for the slow sampler integration tests.

Kept out of conftest.py deliberately: this is a physics problem definition, not
generic fixture plumbing, and only the sampler accuracy tests need it.

Setup
-----
  N correlated parameters.
  D1 : chi2_1(θ) = θᵀ Σ1⁻¹ θ  →  posterior N(0, Σ1)
  D2 : chi2_2(θ) = θᵀ Σ2⁻¹ θ  →  posterior N(0, Σ2)
  Combined       : N(0, Σ_exact),  Σ_exact = (Σ1⁻¹ + Σ2⁻¹)⁻¹

chi2_full = chi2_D1 + chi2_D2 by construction (block-diagonal covmat assumption).

Used by tests/test_bayes_update_blackjax.py and tests/test_blackjax_nuts.py.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from smefit.chi2 import Chi2
from smefit.whitening import WhitenTransform

N_PARAMS = 10
SIGMA_PRIOR = 5.0
N_SAMPLES = 4000
TOL_MARGINAL = 0.05  # 5% relative error on each marginal σ_i
TOL_FROBENIUS = 0.10  # 10% relative Frobenius error on the full covariance matrix
SEED = 42


# ---------------------------------------------------------------------------
# Minimal coefficient stubs (no reportengine dependency)
# ---------------------------------------------------------------------------


class _Coeffs:
    def __init__(self, names):
        self.names = list(names)
        self.free_names = list(names)

    def resolve(self, x):
        return x

    def whitened(self, transform):
        return _CoeffsWhitened(self.names, transform)


class _CoeffsWhitened:
    def __init__(self, names, transform):
        self.names = list(names)
        self.free_names = list(names)
        self._transform = transform

    def resolve(self, c_w):
        return self._transform.to_physical(c_w)

    def whitened(self, transform):
        composed = WhitenTransform(
            matrix=self._transform.matrix @ transform.matrix,
            shift=self._transform.matrix @ transform.shift + self._transform.shift,
        )
        return _CoeffsWhitened(self.names, composed)


# ---------------------------------------------------------------------------
# Module-level fixture: shared problem definition
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def correlated_problem():
    """correlated Gaussian problem built from a fixed random seed."""
    rng = np.random.default_rng(0)
    n = N_PARAMS

    def _random_spd(scale):
        A = rng.normal(size=(n + 5, n)) * scale
        return A.T @ A / (n + 5) + 0.05 * np.eye(n)

    Sigma1 = _random_spd(scale=1.0)
    Sigma2 = _random_spd(scale=0.7)
    Sigma_exact = np.linalg.inv(np.linalg.inv(Sigma1) + np.linalg.inv(Sigma2))

    names = [f"c{i:02d}" for i in range(n)]
    sigma1_marg = np.sqrt(np.diag(Sigma1))

    inv_S1 = jnp.array(np.linalg.inv(Sigma1))
    inv_S2 = jnp.array(np.linalg.inv(Sigma2))

    chi2_D1 = Chi2(lambda t: t @ inv_S1 @ t, names, num_data=n)
    chi2_D2 = Chi2(lambda t: t @ inv_S2 @ t, names, num_data=n)

    return dict(
        names=names,
        Sigma1=Sigma1,
        Sigma_exact=Sigma_exact,
        sigma1_marg=sigma1_marg,
        chi2_D1=chi2_D1,
        chi2_D2=chi2_D2,
    )


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def assert_posterior(fr, names, Sigma_ref):
    """Assert per-parameter marginal σ and full covariance matrix against Sigma_ref."""
    sigma_ref = np.sqrt(np.diag(Sigma_ref))

    for i, name in enumerate(names):
        s_est = float(jnp.std(jnp.array(fr.samples[name])))
        s_ref = float(sigma_ref[i])
        rel_err = abs(s_est - s_ref) / s_ref
        assert rel_err < TOL_MARGINAL, (
            f"{name}: σ_est={s_est:.4f}  σ_ref={s_ref:.4f}  "
            f"({rel_err * 100:.1f}% > {TOL_MARGINAL * 100:.0f}%)"
        )

    samples = np.stack([np.array(fr.samples[n]) for n in names], axis=1)
    Sigma_est = np.cov(samples.T)
    frob_rel = np.linalg.norm(Sigma_est - Sigma_ref) / np.linalg.norm(Sigma_ref)
    assert (
        frob_rel < TOL_FROBENIUS
    ), f"Covariance Frobenius error {frob_rel:.3f} > {TOL_FROBENIUS:.2f}"


def hessian_whitening(chi2, n):
    """The whitening transform smefit would build at the baseline point."""
    import jax

    H = jax.hessian(lambda x: chi2(x))(jnp.zeros(n)) + 1e-8 * jnp.eye(n)
    L = jnp.linalg.cholesky(H)
    W = jnp.linalg.solve(L.T, jnp.eye(n))
    return WhitenTransform(matrix=W, shift=jnp.zeros(n))
