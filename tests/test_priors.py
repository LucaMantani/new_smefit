"""Unit tests for smefit.priors — _UniformDist, _GaussianDist, _build_dist, Prior."""

import math

import jax
import jax.numpy as jnp
import pytest

from smefit.priors import (
    _DIST_REGISTRY,
    ExactPosteriorPrior,
    Prior,
    _build_dist,
    _GaussianDist,
    _UniformDist,
    _WhitenedToPhysicalPrior,
)
from smefit.whitening import WhitenTransform

# ---------------------------------------------------------------------------
# _UniformDist
# ---------------------------------------------------------------------------


def test_uniform_ppf():
    d = _UniformDist(low=-1.0, high=3.0)
    assert float(d.ppf(0)) == pytest.approx(-1.0)
    assert float(d.ppf(1)) == pytest.approx(3.0)
    assert float(d.ppf(0.5)) == pytest.approx(1.0)


def test_uniform_log_prob_in_support():
    d = _UniformDist(low=0.0, high=4.0)
    assert float(d.log_prob(2.0)) == pytest.approx(-math.log(4.0))


def test_uniform_log_prob_out_of_support():
    d = _UniformDist(low=0.0, high=1.0)
    assert float(d.log_prob(-0.1)) == float(-jnp.inf)
    assert float(d.log_prob(1.1)) == float(-jnp.inf)


# ---------------------------------------------------------------------------
# _GaussianDist
# ---------------------------------------------------------------------------


def test_gaussian_ppf_at_median():
    d = _GaussianDist(mean=3.0, std=1.0)
    assert float(d.ppf(0.5)) == pytest.approx(3.0, abs=1e-5)


def test_gaussian_log_prob_at_mean():
    d = _GaussianDist(mean=0.0, std=2.0)
    expected = -math.log(2.0) - 0.5 * math.log(2 * math.pi)
    assert float(d.log_prob(0.0)) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _build_dist
# ---------------------------------------------------------------------------


def test_build_dist_uniform():
    d = _build_dist({"dist": "uniform", "low": -1.0, "high": 1.0})
    assert isinstance(d, _UniformDist)
    assert d.low == pytest.approx(-1.0)
    assert d.high == pytest.approx(1.0)


def test_build_dist_gaussian():
    for name in ("gaussian", "normal"):
        d = _build_dist({"dist": name, "mean": 0.0, "std": 1.0})
        assert isinstance(d, _GaussianDist)


def test_build_dist_unknown():
    with pytest.raises(ValueError, match="Unknown prior"):
        _build_dist({"dist": "laplace"})


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------


@pytest.fixture
def two_uniform_prior():
    d1 = _UniformDist(low=-1.0, high=1.0)
    d2 = _UniformDist(low=0.0, high=2.0)
    return Prior([d1, d2], ["a", "b"])


def test_prior_transform_uniform(two_uniform_prior):
    result = two_uniform_prior.prior_transform(jnp.array([0.5, 0.5]))
    assert float(result[0]) == pytest.approx(0.0)
    assert float(result[1]) == pytest.approx(1.0)


def test_prior_transform_maps_boundaries(two_uniform_prior):
    result = two_uniform_prior.prior_transform(jnp.array([0.0, 1.0]))
    assert float(result[0]) == pytest.approx(-1.0)
    assert float(result[1]) == pytest.approx(2.0)


def test_prior_log_prob_in_support():
    d1 = _UniformDist(low=-1.0, high=1.0)  # log_prob = -log(2)
    d2 = _UniformDist(low=0.0, high=4.0)  # log_prob = -log(4)
    prior = Prior([d1, d2], ["a", "b"])
    expected = -math.log(2.0) + (-math.log(4.0))
    assert float(prior.log_prob(jnp.array([0.0, 2.0]))) == pytest.approx(expected)


def test_prior_log_prob_out_of_support():
    d1 = _UniformDist(low=-1.0, high=1.0)
    d2 = _UniformDist(low=0.0, high=1.0)
    prior = Prior([d1, d2], ["a", "b"])
    # a=2.0 is outside d1's support
    result = float(prior.log_prob(jnp.array([2.0, 0.5])))
    assert result == float(-jnp.inf)


def test_prior_sample_shape(two_uniform_prior):
    key = jax.random.PRNGKey(0)
    samples = two_uniform_prior.sample(key, n_samples=100)
    assert samples.shape == (100, 2)


# ---------------------------------------------------------------------------
# Bijectors to the unconstrained space (used by gradient-based samplers)
# ---------------------------------------------------------------------------


# Tolerances are set for JAX's default float32: the tests must not silently
# require x64. Values of |u| are kept moderate for the same reason — the
# saturation regime is covered by its own tests below.

# --- Contract obeyed by EVERY registered distribution ---------------------
#
# The bijector methods are abstract on _Distribution rather than defaulting to
# the identity, precisely so that a bounded distribution cannot silently
# inherit a wrong one. These tests are the other half of that guard: they run
# against everything in _DIST_REGISTRY, so a new distribution is checked
# automatically instead of only when someone remembers to write a test.

#: Representative parameters per registered distribution. Adding an entry to
#: _DIST_REGISTRY must add one here too — enforced by the test below.
_CONTRACT_SPECS = {
    "uniform": {"dist": "uniform", "low": -3.0, "high": 7.0},
    "gaussian": {"dist": "gaussian", "mean": 1.0, "std": 2.0},
    "normal": {"dist": "normal", "mean": 0.0, "std": 0.5},
}


def test_contract_specs_cover_the_registry():
    assert set(_CONTRACT_SPECS) == set(_DIST_REGISTRY), (
        "every prior distribution needs an entry in _CONTRACT_SPECS so that the "
        "bijector contract tests below cover it"
    )


@pytest.mark.parametrize("name", sorted(_CONTRACT_SPECS))
@pytest.mark.parametrize("u", [-5.0, -3.0, -0.5, 0.0, 0.5, 3.0, 5.0])
def test_bijector_round_trip_contract(name, u):
    d = _build_dist(_CONTRACT_SPECS[name])
    assert float(d.to_unconstrained(d.from_unconstrained(u))) == pytest.approx(
        u, abs=1e-3
    )


@pytest.mark.parametrize("name", sorted(_CONTRACT_SPECS))
@pytest.mark.parametrize("u", [-5.0, -2.0, 0.0, 2.0, 5.0])
def test_bijector_log_det_matches_autodiff_contract(name, u):
    d = _build_dist(_CONTRACT_SPECS[name])
    expected = jnp.log(jnp.abs(jax.grad(d.from_unconstrained)(u)))
    assert float(d.log_det_jacobian(u)) == pytest.approx(
        float(expected), rel=1e-5, abs=1e-5
    )


@pytest.mark.parametrize("name", sorted(_CONTRACT_SPECS))
@pytest.mark.parametrize("u", [-30.0, -5.0, 0.0, 5.0, 30.0])
def test_bijector_maps_into_support_contract(name, u):
    """from_unconstrained must land inside the support.

    This is the test that catches a bounded distribution which copied or
    inherited an identity bijector: the mapped point would fall outside the
    support, log_prob would be -inf, and a gradient sampler would stall at the
    wall and return a biased posterior without ever failing.

    |u| stops at 30 on purpose. A distribution whose support is *open* at the
    boundary (a log-normal, say: x = exp(u) on (0, inf)) underflows to exactly
    the excluded endpoint for |u| ~ 1e3, so a correct implementation would fail
    a stricter range. `_UniformDist` tolerates the saturation regime only
    because its log_prob uses inclusive bounds — which is why the extreme case
    is checked in test_uniform_bijector_stays_in_bounds instead of here.
    """
    d = _build_dist(_CONTRACT_SPECS[name])
    x = d.from_unconstrained(u)
    assert math.isfinite(
        float(d.log_prob(x))
    ), f"{name}: u={u} maps to x={float(x)}, outside the support"


@pytest.mark.parametrize("name", sorted(_CONTRACT_SPECS))
@pytest.mark.parametrize("u", [-1e3, 0.0, 1e3])
def test_bijector_log_det_gradient_finite_contract(name, u):
    """A NaN or inf here poisons the whole NUTS trajectory."""
    d = _build_dist(_CONTRACT_SPECS[name])
    assert math.isfinite(float(d.log_det_jacobian(u)))
    assert math.isfinite(float(jax.grad(d.log_det_jacobian)(u)))


# --- Distribution-specific properties --------------------------------------


@pytest.mark.parametrize("x", [-2.9, -1.0, 0.0, 3.0, 6.9])
def test_uniform_bijector_round_trip_from_x(x):
    d = _UniformDist(-3.0, 7.0)
    assert float(d.from_unconstrained(d.to_unconstrained(x))) == pytest.approx(
        x, abs=1e-3
    )


def test_uniform_log_det_finite_in_far_tail():
    """The log_sigmoid form stays exact where log(s) + log(1 - s) underflows.

    At u = 60 the naive form gives log(0) = -inf (and a NaN gradient), and so
    does differentiating from_unconstrained; this closed form gives log(w) - u.
    """
    d = _UniformDist(-1.0, 1.0)
    value = float(d.log_det_jacobian(60.0))
    assert math.isfinite(value)
    assert value == pytest.approx(math.log(2.0) - 60.0, rel=1e-5)


def test_uniform_bijector_stays_in_bounds():
    d = _UniformDist(-2.0, 5.0)
    for u in (-1e3, 1e3):
        x = d.from_unconstrained(u)
        assert -2.0 <= float(x) <= 5.0
        assert math.isfinite(float(d.log_prob(x)))  # inclusive bounds, so not -inf


def test_uniform_to_unconstrained_at_bound_is_finite():
    """The dtype-aware clip is what keeps this finite; a hardcoded 1e-12 epsilon
    rounds away in float32 and lets log(0) through."""
    d = _UniformDist(-2.0, 5.0)
    assert math.isfinite(float(d.to_unconstrained(-2.0)))
    assert math.isfinite(float(d.to_unconstrained(5.0)))


def test_gaussian_bijector_is_identity():
    d = _GaussianDist(1.0, 2.0)
    assert float(d.from_unconstrained(3.5)) == 3.5
    assert float(d.to_unconstrained(3.5)) == 3.5
    assert float(d.log_det_jacobian(3.5)) == 0.0


# ---------------------------------------------------------------------------
# Prior: unconstrained reparametrisation
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_prior():
    return Prior(
        [_UniformDist(-3.0, 7.0), _GaussianDist(1.0, 2.0)],
        ["a", "b"],
        specs={
            "a": {"dist": "uniform", "low": -3.0, "high": 7.0},
            "b": {"dist": "gaussian", "mean": 1.0, "std": 2.0},
        },
    )


def test_unconstrained_round_trip(mixed_prior):
    u = jnp.array([0.7, -1.3])
    assert jnp.allclose(
        mixed_prior.to_unconstrained(mixed_prior.from_unconstrained(u)), u, atol=1e-4
    )


def test_unconstrained_log_det_matches_jacobian(mixed_prior):
    u = jnp.array([0.7, -1.3])
    jac = jax.jacobian(mixed_prior.from_unconstrained)(u)
    expected = jnp.log(jnp.abs(jnp.linalg.det(jac)))
    assert float(mixed_prior.log_det_jacobian(u)) == pytest.approx(
        float(expected), rel=1e-5
    )


def test_unconstrained_log_prob_matches_closed_form():
    """For a uniform, the two log-width terms cancel exactly."""
    prior = Prior([_UniformDist(-3.0, 7.0)], ["a"])
    u = jnp.array([0.6])
    expected = float(jax.nn.log_sigmoid(0.6) + jax.nn.log_sigmoid(-0.6))
    assert float(prior.log_prob_unconstrained(u)) == pytest.approx(expected)


def test_unconstrained_log_prob_finite_far_from_origin(mixed_prior):
    """The whole point of the bijector: no -inf and no NaN gradient at the walls."""
    u = jnp.array([1e3, 5.0])
    assert math.isfinite(float(mixed_prior.log_prob_unconstrained(u)))
    grad = jax.grad(mixed_prior.log_prob_unconstrained)(u)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_sample_unconstrained_shape(mixed_prior):
    samples = mixed_prior.sample_unconstrained(jax.random.PRNGKey(0), 17)
    assert samples.shape == (17, 2)
    assert bool(jnp.all(jnp.isfinite(samples)))


@pytest.mark.parametrize(
    "joint_only_prior",
    [
        ExactPosteriorPrior(
            base_prior=Prior([_UniformDist(-1.0, 1.0)], ["a"]),
            log_likelihood_1=lambda x: 0.0,
            samples_dict={"a": jnp.zeros(4)},
            param_names=["a"],
        ),
        _WhitenedToPhysicalPrior(
            Prior([_UniformDist(-1.0, 1.0)], ["a"]),
            WhitenTransform(matrix=jnp.eye(1) * 2.0, shift=jnp.zeros(1)),
        ),
    ],
    ids=["exact_posterior", "whitened_to_physical"],
)
def test_joint_only_priors_have_no_unconstrained_interface(joint_only_prior):
    """These know a joint log_prob and nothing more, so they cannot be
    reparametrised per parameter. `_run_nuts` rejects them on that basis."""
    assert not isinstance(joint_only_prior, Prior)
    assert not hasattr(joint_only_prior, "log_prob_unconstrained")
