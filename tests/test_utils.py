"""Unit tests for pure functions in smefit.utils."""

import jax
import jax.numpy as jnp
import pytest

from smefit.core import Coefficient, CoefficientGroup
from smefit.utils import apply_whitening, ensure_list, resolve_posterior

_PRIOR = {"dist": "uniform", "low": -5.0, "high": 5.0}


def _free(name):
    return Coefficient(name=name, free=True, prior=_PRIOR)


# ---------------------------------------------------------------------------
# ensure_list
# ---------------------------------------------------------------------------


def test_ensure_list_passthrough():
    lst = [1, 2, 3]
    assert ensure_list(lst) is lst


def test_ensure_list_wraps_scalar():
    assert ensure_list(42) == [42]


def test_ensure_list_wraps_string():
    assert ensure_list("abc") == ["abc"]


# ---------------------------------------------------------------------------
# apply_whitening
# ---------------------------------------------------------------------------


def test_apply_whitening_chi2_equivalent():
    """Whitened chi2 evaluated at c_w equals original chi2 at W @ c_w."""
    W = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    chi2 = lambda c: jnp.sum(c**2)
    cg = CoefficientGroup([_free("OpA"), _free("OpB")])

    whitened_chi2, _ = apply_whitening(chi2, cg, W)

    c_w = jnp.array([1.0, 1.0])
    assert float(whitened_chi2(c_w)) == pytest.approx(float(chi2(W @ c_w)))


def test_apply_whitening_coeff_group():
    """Returned CoefficientGroup should have _W set (whitening active)."""
    W = jnp.eye(2)
    chi2 = lambda c: jnp.sum(c**2)
    cg = CoefficientGroup([_free("OpA"), _free("OpB")])

    _, whitened_cg = apply_whitening(chi2, cg, W)

    assert whitened_cg._W is not None


# ---------------------------------------------------------------------------
# resolve_posterior
# ---------------------------------------------------------------------------


def test_resolve_posterior_shapes():
    """samples dict has shape (n_samples,) per name; best_fit_point is a dict."""
    cg = CoefficientGroup([_free("OpA"), _free("OpB")])

    n_samples = 20
    posterior_free = jax.random.normal(jax.random.PRNGKey(0), (n_samples, 2))
    best_free = jnp.array([0.5, -0.3])

    samples, best_fit_point = resolve_posterior(cg, posterior_free, best_free)

    for name in cg.names:
        assert name in samples
        assert samples[name].shape == (n_samples,)

    assert isinstance(best_fit_point, dict)
    for name in cg.names:
        assert name in best_fit_point
        assert isinstance(best_fit_point[name], float)
