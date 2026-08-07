"""
Accuracy tests for the BlackJAX NUTS algorithm (blackjax_settings.algorithm: nuts).

Marked @pytest.mark.slow — these run the real sampler and are skipped in fast CI.
Run explicitly with:  pytest -m slow tests/test_blackjax_nuts.py

The problem definition and posterior assertions live in tests/gaussian_problem.py,
shared with tests/test_bayes_update_blackjax.py. Fast, mocked coverage of the
same code path is in tests/test_blackjax_samplers.py.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from smefit.blackjax_fit import blackjax_fit
from smefit.priors import Prior, _GaussianDist, _UniformDist
from tests.gaussian_problem import correlated_problem  # noqa: F401  (pytest fixture)
from tests.gaussian_problem import (
    N_SAMPLES,
    SEED,
    SIGMA_PRIOR,
    _Coeffs,
    assert_posterior,
    hessian_whitening,
)

NUM_CHAINS = 4
NUM_WARMUP = 1000
NUM_DRAWS = 2500


def _nuts_cfg(log_dir, **overrides):
    cfg = {
        "algorithm": "nuts",
        "seed": SEED,
        "log_dir": str(log_dir),
        "num_chains": NUM_CHAINS,
        "num_warmup": NUM_WARMUP,
        "num_samples": NUM_DRAWS,
        "target_acceptance_rate": 0.8,
        "max_num_doublings": 10,
        "init": "prior",
    }
    cfg.update(overrides)
    return cfg


def _assert_converged(log_dir):
    diag = json.loads((log_dir / "nuts_diagnostics.json").read_text())
    assert diag["max_rhat"] < 1.01, f"max R-hat = {diag['max_rhat']}"
    assert diag["divergences"] == 0, f"{diag['divergences']} divergent transitions"
    assert diag["min_ess"] > 100 * NUM_CHAINS
    return diag


@pytest.mark.slow
def test_nuts_recovers_correlated_gaussian(correlated_problem, tmp_path):
    """Uniform priors, no whitening — the logit bijector path."""
    names = correlated_problem["names"]
    chi2 = correlated_problem["chi2_D1"]
    Sigma = correlated_problem["Sigma1"]
    sigma_marg = correlated_problem["sigma1_marg"]

    prior = Prior([_UniformDist(-5 * s, 5 * s) for s in sigma_marg], names)
    log_dir = tmp_path / "nuts"

    fr = blackjax_fit(
        prior, chi2, _Coeffs(names), _nuts_cfg(log_dir), n_samples=N_SAMPLES
    )

    assert_posterior(fr, names, Sigma)
    assert fr.logz is None, "NUTS provides no evidence estimate"
    # The chi2 minimum is 0 and a typical posterior draw sits at ~n_params = 10;
    # the best of the draws must be well down in that tail. This also catches a
    # max_loglikelihood that accidentally includes the prior term.
    assert fr.chi2_val < 5.0, "best-fit chi2 should be near the minimum of 0"
    assert len(fr.samples[names[0]]) == N_SAMPLES
    _assert_converged(log_dir)
    assert (log_dir / "nuts_samples.csv").exists()


@pytest.mark.slow
def test_nuts_with_whitening_recovers_gaussian(correlated_problem, tmp_path):
    """Whitening forces a uniform prior on [-sigma_prior, sigma_prior], so this
    exercises the logit bijector in whitened coordinates."""
    names = correlated_problem["names"]
    chi2 = correlated_problem["chi2_D1"]
    Sigma = correlated_problem["Sigma1"]
    n = len(names)

    transform = hessian_whitening(chi2, n)
    prior_w = Prior([_UniformDist(-SIGMA_PRIOR, SIGMA_PRIOR)] * n, names)
    log_dir = tmp_path / "nuts_w"

    fr = blackjax_fit(
        prior_w,
        chi2,
        _Coeffs(names),
        _nuts_cfg(log_dir),
        whitening_transformation=transform,
        n_samples=N_SAMPLES,
    )

    assert_posterior(fr, names, Sigma)
    assert fr.whitening_active is True
    _assert_converged(log_dir)


@pytest.mark.slow
def test_nuts_with_whitening_and_baseline_init(correlated_problem, tmp_path):
    """init: baseline must start from the whitened baseline point, not the raw
    physical one — otherwise every chain starts away from the mode."""
    names = correlated_problem["names"]
    chi2 = correlated_problem["chi2_D1"]
    Sigma = correlated_problem["Sigma1"]
    n = len(names)

    transform = hessian_whitening(chi2, n)
    prior_w = Prior([_UniformDist(-SIGMA_PRIOR, SIGMA_PRIOR)] * n, names)
    log_dir = tmp_path / "nuts_baseline"

    fr = blackjax_fit(
        prior_w,
        chi2,
        _Coeffs(names),
        _nuts_cfg(log_dir, init="baseline"),
        whitening_transformation=transform,
        n_samples=N_SAMPLES,
    )

    assert_posterior(fr, names, Sigma)
    _assert_converged(log_dir)


@pytest.mark.slow
def test_nuts_matches_exact_posterior_with_gaussian_prior(correlated_problem, tmp_path):
    """Gaussian priors take the identity bijector. This is the only test that
    pins the prior normalisation: a wrong sign or factor in
    log_prob_unconstrained shifts the posterior width."""
    names = correlated_problem["names"]
    chi2 = correlated_problem["chi2_D1"]
    Sigma = correlated_problem["Sigma1"]
    n = len(names)
    s_prior = 3.0

    # posterior precision = Sigma^-1 + I / s_prior^2
    Sigma_post = np.linalg.inv(np.linalg.inv(Sigma) + np.eye(n) / s_prior**2)

    prior = Prior([_GaussianDist(0.0, s_prior)] * n, names)
    log_dir = tmp_path / "nuts_gauss"

    fr = blackjax_fit(
        prior, chi2, _Coeffs(names), _nuts_cfg(log_dir), n_samples=N_SAMPLES
    )

    assert_posterior(fr, names, Sigma_post)
    _assert_converged(log_dir)


@pytest.mark.slow
def test_nuts_agrees_with_nested_sampling(correlated_problem, tmp_path):
    """Cross-check the two algorithms against each other on the same problem."""
    names = correlated_problem["names"]
    chi2 = correlated_problem["chi2_D1"]
    sigma_marg = correlated_problem["sigma1_marg"]

    prior = Prior([_UniformDist(-5 * s, 5 * s) for s in sigma_marg], names)
    coeffs = _Coeffs(names)

    fr_nuts = blackjax_fit(
        prior, chi2, coeffs, _nuts_cfg(tmp_path / "nuts"), n_samples=N_SAMPLES
    )

    ns_dir = tmp_path / "ns"
    ns_dir.mkdir()
    fr_ns = blackjax_fit(
        prior,
        chi2,
        coeffs,
        {
            "algorithm": "nested_sampling",
            "seed": SEED,
            "n_live": 4000,
            "delete_fraction": 0.1,
            "repeats": 5,
            "log_precision": -4,
            "log_dir": str(ns_dir),
        },
        n_samples=N_SAMPLES,
    )

    for name in names:
        s_nuts = float(jnp.std(fr_nuts.samples[name]))
        s_ns = float(jnp.std(fr_ns.samples[name]))
        assert (
            abs(s_nuts - s_ns) / s_ns < 0.10
        ), f"{name}: NUTS σ={s_nuts:.4f} vs nested sampling σ={s_ns:.4f}"
