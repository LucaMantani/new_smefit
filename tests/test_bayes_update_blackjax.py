"""
Integration tests for the Bayesian sequential update via blackjax_fit.

Marked @pytest.mark.slow — these run the real sampler and are skipped in fast CI.
Run explicitly with:  pytest -m slow tests/test_bayes_update_blackjax.py

The problem definition, coefficient stubs and posterior assertions live in
tests/gaussian_problem.py, shared with tests/test_blackjax_nuts.py.

The ``_bj_cfg`` settings dicts deliberately carry no ``algorithm`` key: they
are the regression guard that runcards and code predating the key still run
nested sampling. ``_nuts_cfg`` covers the same update under gradient MCMC.
"""

import json
import os

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from smefit.blackjax_fit import blackjax_fit
from smefit.priors import (
    ExactPosteriorPrior,
    Prior,
    WhitenedToPhysicalPrior,
    _UniformDist,
)
from tests.gaussian_problem import correlated_problem  # noqa: F401  (pytest fixture)
from tests.gaussian_problem import (
    N_SAMPLES,
    SEED,
    SIGMA_PRIOR,
    _Coeffs,
    assert_posterior,
    hessian_whitening,
)

N_LIVE = 4000
NUM_CHAINS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bj_cfg(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    return {
        "seed": SEED,
        "n_live": N_LIVE,
        "delete_fraction": 0.1,
        "repeats": 5,
        "log_precision": -4,
        "log_dir": str(log_dir),
    }


def _nuts_cfg(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    return {
        "algorithm": "nuts",
        "seed": SEED,
        "num_chains": NUM_CHAINS,
        "num_warmup": 1000,
        "num_samples": 2500,
        "target_acceptance_rate": 0.8,
        "max_num_doublings": 10,
        "log_dir": str(log_dir),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_bayes_update_no_whitening(correlated_problem, tmp_path):
    """Update fit (no whitening) reproduces the full posterior N(0, Σ_exact)."""
    names = correlated_problem["names"]
    chi2_D1 = correlated_problem["chi2_D1"]
    chi2_D2 = correlated_problem["chi2_D2"]
    Sigma1 = correlated_problem["Sigma1"]
    Sigma_exact = correlated_problem["Sigma_exact"]
    sigma1_marg = correlated_problem["sigma1_marg"]

    coeffs = _Coeffs(names)
    prior = Prior([_UniformDist(-5 * s, 5 * s) for s in sigma1_marg], names)

    fr_fit1 = blackjax_fit(
        prior, chi2_D1, coeffs, _bj_cfg(tmp_path / "fit1"), n_samples=N_SAMPLES
    )
    assert_posterior(fr_fit1, names, Sigma1)

    prior_upd = ExactPosteriorPrior(
        base_prior=prior,
        log_likelihood_1=jax.jit(lambda t: -chi2_D1(t) / 2.0),
        samples_dict=fr_fit1.samples,
        param_names=names,
    )
    fr_upd = blackjax_fit(
        prior_upd, chi2_D2, coeffs, _bj_cfg(tmp_path / "update"), n_samples=N_SAMPLES
    )
    assert_posterior(fr_upd, names, Sigma_exact)


@pytest.mark.slow
def test_bayes_update_with_whitening(correlated_problem, tmp_path):
    """Update fit with whitened fit1 reproduces the full posterior N(0, Σ_exact)."""
    names = correlated_problem["names"]
    chi2_D1 = correlated_problem["chi2_D1"]
    chi2_D2 = correlated_problem["chi2_D2"]
    Sigma1 = correlated_problem["Sigma1"]
    Sigma_exact = correlated_problem["Sigma_exact"]
    n = len(names)

    coeffs = _Coeffs(names)
    transform = hessian_whitening(chi2_D1, n)

    prior_w = Prior([_UniformDist(-SIGMA_PRIOR, SIGMA_PRIOR)] * n, names)

    fr_fit1_w = blackjax_fit(
        prior_w,
        chi2_D1,
        coeffs,
        _bj_cfg(tmp_path / "fit1_w"),
        whitening_transformation=transform,
        n_samples=N_SAMPLES,
    )
    assert_posterior(fr_fit1_w, names, Sigma1)

    prior_1_phys = WhitenedToPhysicalPrior(prior_w, transform)
    prior_upd = ExactPosteriorPrior(
        base_prior=prior_1_phys,
        log_likelihood_1=jax.jit(lambda t: -chi2_D1(t) / 2.0),
        samples_dict=fr_fit1_w.samples,
        param_names=names,
    )
    fr_upd = blackjax_fit(
        prior_upd, chi2_D2, coeffs, _bj_cfg(tmp_path / "update_w"), n_samples=N_SAMPLES
    )
    assert_posterior(fr_upd, names, Sigma_exact)


@pytest.mark.slow
def test_nested_sampling_still_default_without_algorithm_key(
    correlated_problem, tmp_path
):
    """No `algorithm` key at all → nested sampling, with a real log evidence."""
    names = correlated_problem["names"]
    chi2_D1 = correlated_problem["chi2_D1"]
    sigma1_marg = correlated_problem["sigma1_marg"]

    prior = Prior([_UniformDist(-5 * s, 5 * s) for s in sigma1_marg], names)
    cfg = _bj_cfg(tmp_path / "ns")
    assert "algorithm" not in cfg

    fr = blackjax_fit(prior, chi2_D1, _Coeffs(names), cfg, n_samples=N_SAMPLES)

    assert fr.logz is not None
    assert (tmp_path / "ns" / "nested_samples.csv").exists()


@pytest.mark.slow
def test_bayes_update_nuts(correlated_problem, tmp_path):
    """The same update under gradient MCMC: NUTS reparametrises the
    ExactPosteriorPrior through the bijector it delegates to its base prior,
    and starts its chains on fit1's posterior draws."""
    names = correlated_problem["names"]
    chi2_D1 = correlated_problem["chi2_D1"]
    chi2_D2 = correlated_problem["chi2_D2"]
    Sigma1 = correlated_problem["Sigma1"]
    Sigma_exact = correlated_problem["Sigma_exact"]
    sigma1_marg = correlated_problem["sigma1_marg"]

    coeffs = _Coeffs(names)
    prior = Prior([_UniformDist(-5 * s, 5 * s) for s in sigma1_marg], names)

    fr_fit1 = blackjax_fit(
        prior, chi2_D1, coeffs, _nuts_cfg(tmp_path / "fit1"), n_samples=N_SAMPLES
    )
    assert_posterior(fr_fit1, names, Sigma1)

    prior_upd = ExactPosteriorPrior(
        base_prior=prior,
        log_likelihood_1=jax.jit(lambda t: -chi2_D1(t) / 2.0),
        samples_dict=fr_fit1.samples,
        param_names=names,
    )
    log_dir = tmp_path / "update"
    fr_upd = blackjax_fit(
        prior_upd, chi2_D2, coeffs, _nuts_cfg(log_dir), n_samples=N_SAMPLES
    )
    assert_posterior(fr_upd, names, Sigma_exact)

    # NUTS estimates no evidence, whichever prior it is given.
    assert fr_upd.logz is None
    diag = json.loads((log_dir / "nuts_diagnostics.json").read_text())
    assert diag["max_rhat"] < 1.01, f"max R-hat = {diag['max_rhat']}"
    assert diag["divergences"] == 0, f"{diag['divergences']} divergent transitions"
