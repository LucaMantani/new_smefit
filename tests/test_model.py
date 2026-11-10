"""Unit tests for smefit.model — EFTModel.forward_map and posterior_predictive."""

import logging
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from smefit.core import Coefficient, CoefficientGroup, TheoryGroup
from smefit.model import EFTModel, gaussian_draws

_PRIOR = {"dist": "uniform", "low": -5.0, "high": 5.0}


def _free(name):
    return Coefficient(name=name, free=True, prior=_PRIOR)


# ---------------------------------------------------------------------------
# forward_map
# ---------------------------------------------------------------------------


def test_forward_map_zero_coeffs_is_sm(theory_a):
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory_a, cg, use_quad=False)
    result = model.forward_map(jnp.array([0.0]))
    assert jnp.allclose(result, theory_a.sm_pred)


def test_forward_map_linear(theory_a):
    # Only OpA declared; lin_corr[:, OpA] = [1, 2, 3]
    # coeffs = [2.0] → predictions = [10+2, 20+4, 30+6] = [12, 24, 36]
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory_a, cg, use_quad=False)
    result = model.forward_map(jnp.array([2.0]))
    assert jnp.allclose(result, jnp.array([12.0, 24.0, 36.0]))


def test_forward_map_quadratic(theory_a):
    # Only OpA declared; OpA*OpA = [0.1, 0.2, 0.3]
    # coeffs = [2.0]:
    #   linear  = sm + [1,2,3]*2 = [12, 24, 36]
    #   quad    = [0.1, 0.2, 0.3]*2^2 = [0.4, 0.8, 1.2]
    #   total   = [12.4, 24.8, 37.2]
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory_a, cg, use_quad=True)
    result = model.forward_map(jnp.array([2.0]))
    assert jnp.allclose(result, jnp.array([12.4, 24.8, 37.2]))


def test_forward_map_no_match_warns(theory_a, caplog):
    group = TheoryGroup([theory_a])
    cg = CoefficientGroup([_free("OpX")])  # OpX not in theory_a
    with caplog.at_level(logging.WARNING, logger="smefit.model"):
        model = EFTModel(group, cg, use_quad=False)
    assert "None of the declared coefficients enter the predictions" in caplog.text
    assert "DS_A" in caplog.text
    # the "inactive coefficients" warning would be redundant here
    assert "not present in any theory dataset" not in caplog.text
    # the coefficient simply does not enter: predictions stay at the SM
    assert jnp.allclose(model.forward_map(jnp.array([2.0])), group.sm_pred)


def test_forward_map_subset_operators(theory_a):
    # theory_a has OpA, OpB; only OpA declared → OpB contribution silently ignored
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory_a, cg, use_quad=False)
    result = model.forward_map(jnp.array([0.0]))
    assert jnp.allclose(result, theory_a.sm_pred)


# ---------------------------------------------------------------------------
# posterior_predictive
# ---------------------------------------------------------------------------


def test_posterior_predictive_is_forward_map_at_every_sample(theory_a):
    """Row i is forward_map of sample i, quadratic corrections included."""
    cg = CoefficientGroup([_free("OpA"), _free("OpB")])
    model = EFTModel(theory_a, cg, use_quad=True)
    posterior = jnp.array([[0.0, 0.0], [2.0, 0.0], [-1.0, 0.5], [0.3, -2.0]])

    result = model.posterior_predictive(posterior)

    assert result.shape == (4, 3)
    for sample, row in zip(posterior, result):
        assert jnp.allclose(row, model.forward_map(sample))


def test_posterior_predictive_does_not_depend_on_the_batch_size(theory_a):
    """A batch size that does not divide the number of samples changes nothing."""
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory_a, cg, use_quad=True)
    posterior = jnp.linspace(-2.0, 2.0, 7)[:, None]

    assert jnp.allclose(
        model.posterior_predictive(posterior, batch_size=3),
        model.posterior_predictive(posterior),
    )


def test_posterior_predictive_samples_the_theory_error_when_it_is_in_the_model(
    theory_a,
):
    """With use_theory_covmat, a posterior pinned at the SM spreads by the
    theory covmat; without it, it stays at the SM."""
    cg = CoefficientGroup([_free("OpA")])
    posterior = jnp.zeros((50_000, 1))
    covmat = jnp.diag(jnp.array([0.01, 0.04, 0.09]))  # the fixture's

    plain = EFTModel(theory_a, cg).posterior_predictive(posterior)
    assert jnp.allclose(plain, theory_a.sm_pred)

    model = EFTModel(theory_a, cg, use_theory_covmat=True)
    predictions = np.asarray(model.posterior_predictive(posterior))
    np.testing.assert_allclose(predictions.mean(axis=0), theory_a.sm_pred, atol=0.01)
    np.testing.assert_allclose(np.cov(predictions.T), covmat, atol=2e-3)


def test_posterior_predictive_is_reproducible_and_seeded(theory_a):
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory_a, cg, use_theory_covmat=True)
    posterior = jnp.zeros((10, 1))

    first = model.posterior_predictive(posterior)
    assert jnp.array_equal(first, model.posterior_predictive(posterior))
    assert not jnp.array_equal(first, model.posterior_predictive(posterior, seed=1))


# ---------------------------------------------------------------------------
# gaussian_draws
# ---------------------------------------------------------------------------


def test_gaussian_draws_follow_the_covariance():
    covmat = np.array([[4.0, 1.2], [1.2, 1.0]])
    draws = np.asarray(gaussian_draws(covmat, 200_000, jax.random.PRNGKey(1)))
    assert draws.shape == (200_000, 2)
    np.testing.assert_allclose(np.cov(draws.T), covmat, rtol=0.02)


def test_gaussian_draws_treat_a_slightly_negative_eigenvalue_as_zero():
    """A covmat a rounding error off positive-semidefinite is sampled as its
    nearest PSD matrix, without a warning or a nan."""
    # perfectly correlated, with the correlation rounded just above 1
    covmat = np.array([[1.0, 1.0001], [1.0001, 1.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        draws = np.asarray(gaussian_draws(covmat, 100_000, jax.random.PRNGKey(1)))
    assert np.isfinite(draws).all()
    np.testing.assert_allclose(np.var(draws, axis=0), [1.0, 1.0], rtol=0.02)
    np.testing.assert_allclose(np.corrcoef(draws.T)[0, 1], 1.0, rtol=1e-6)
