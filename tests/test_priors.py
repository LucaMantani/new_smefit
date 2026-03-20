"""Unit tests for smefit.priors — _UniformDist, _GaussianDist, _build_dist, Prior."""

import math

import jax
import jax.numpy as jnp
import pytest

from smefit.priors import Prior, _build_dist, _GaussianDist, _UniformDist

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
