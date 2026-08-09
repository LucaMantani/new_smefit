"""Unit tests for smefit/blackjax_fit.py — blackjax and anesthetic are mocked."""

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest

from smefit.blackjax_fit import blackjax_fit
from smefit.blackjax_samplers import _SAMPLER_REGISTRY, SamplerOutput
from smefit.fit_result import FitResult
from smefit.whitening import WhitenTransform

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_final_states():
    """Return a mock final_states (NSInfo-like) object with real JAX arrays for indexing."""
    fs = MagicMock()
    fs.particles.loglikelihood = jnp.array([-1.5, -2.0])
    fs.particles.loglikelihood_birth = jnp.array([-3.0, -4.0])
    fs.particles.position = jnp.zeros((2, 1))
    return fs


def _blackjax_settings(log_dir, **overrides):
    settings = {
        "algorithm": "nested_sampling",
        "n_live": 10,
        "repeats": 1,
        "delete_fraction": 0.5,
        "log_precision": -2,
        "seed": 0,
        "log_dir": str(log_dir),
    }
    settings.update(overrides)
    return settings


_MOCK_SAMPLES = {"OpA": jnp.zeros(2), "OpB": jnp.full(2, 2.0), "OpC": jnp.zeros(2)}
_MOCK_BEST = {"OpA": 0.0, "OpB": 2.0, "OpC": 0.0}


# ---------------------------------------------------------------------------
# Shared patching context
# ---------------------------------------------------------------------------


def _run_blackjax_fit(
    prior, chi2, coeff_group, settings, whitening_transformation=None, n_samples=10000
):
    """Helper that patches external dependencies and calls blackjax_fit."""
    final_states = _make_final_states()
    mock_algo = MagicMock()

    # state whose logZ_live - logZ satisfies termination immediately:
    # 0.0 - 5.0 = -5.0 < log_precision=-2 → while condition is False on first check
    mock_state = MagicMock()
    mock_state.integrator.logZ_live = 0.0
    mock_state.integrator.logZ = 5.0
    mock_algo.init.return_value = mock_state

    mock_nested = MagicMock()

    mock_sample_result = MagicMock()
    mock_sample_result.position = jnp.zeros((2, 1))

    with (
        patch("smefit.blackjax_samplers.blackjax.nss", return_value=mock_algo),
        patch("smefit.blackjax_samplers.finalise", return_value=final_states),
        patch("smefit.blackjax_samplers.ess", return_value=2),
        patch("smefit.blackjax_samplers.log_weights", return_value=jnp.zeros(3)),
        patch("smefit.blackjax_samplers.sample", return_value=mock_sample_result),
        patch(
            "smefit.blackjax_samplers.anesthetic.NestedSamples",
            return_value=mock_nested,
        ),
        patch(
            "smefit.blackjax_fit.resolve_posterior",
            return_value=(_MOCK_SAMPLES, _MOCK_BEST),
        ),
    ):
        result = blackjax_fit(
            prior=prior,
            chi2=chi2,
            coefficients=coeff_group,
            blackjax_settings=settings,
            whitening_transformation=whitening_transformation,
            n_samples=n_samples,
        )
    return result, mock_nested


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_blackjax_fit_happy_path(minimal_prior, minimal_chi2, coeff_group, tmp_path):
    """Non-whitening path: FitResult fields are populated correctly."""
    settings = _blackjax_settings(tmp_path / "bj_logs")
    result, _ = _run_blackjax_fit(minimal_prior, minimal_chi2, coeff_group, settings)

    assert isinstance(result, FitResult)
    assert result.num_data == 10
    assert result.whitening_active is False
    assert result.samples is _MOCK_SAMPLES
    assert result.best_fit_point is _MOCK_BEST


def test_blackjax_fit_whitening_active(
    minimal_prior, minimal_chi2, coeff_group, tmp_path
):
    """whitening_transformation is not None → whitening_active=True in result."""
    settings = _blackjax_settings(tmp_path / "bj_logs")
    transform = WhitenTransform(matrix=jnp.eye(1), shift=jnp.zeros(1))
    result, _ = _run_blackjax_fit(
        minimal_prior, minimal_chi2, coeff_group, settings, transform
    )

    assert result.whitening_active is True


def test_blackjax_fit_csv_written(minimal_prior, minimal_chi2, coeff_group, tmp_path):
    """nested_samples.to_csv should be called with the correct path."""
    log_dir = tmp_path / "bj_logs"
    settings = _blackjax_settings(log_dir)
    _, mock_nested = _run_blackjax_fit(
        minimal_prior, minimal_chi2, coeff_group, settings
    )

    mock_nested.to_csv.assert_called_once_with(str(log_dir) + "/nested_samples.csv")


def test_blackjax_fit_posterior_truncation_warning(
    minimal_prior, minimal_chi2, coeff_group, tmp_path, caplog
):
    """When n_samples > available posterior samples, a warning is logged."""
    settings = _blackjax_settings(tmp_path / "bj_logs")

    with caplog.at_level("WARNING"):
        _run_blackjax_fit(
            minimal_prior,
            minimal_chi2,
            coeff_group,
            settings,
            n_samples=10000,  # far more than the 2 mock samples available
        )

    assert any("posterior samples" in msg.lower() for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Algorithm dispatch
# ---------------------------------------------------------------------------


def _patched_runner(monkeypatch, algorithm, output=None):
    """Swap one registry entry for a recording stub; returns the calls list."""
    calls = []

    def _stub(rng_key, prior, log_likelihood, n_samples, settings):
        calls.append(
            {
                "prior": prior,
                "log_likelihood": log_likelihood,
                "n_samples": n_samples,
                "settings": settings,
            }
        )
        return output or SamplerOutput(
            samples=jnp.zeros((2, 1)),
            best_point=jnp.zeros(1),
            max_loglikelihood=-1.0,
            logz=None,
        )

    monkeypatch.setitem(_SAMPLER_REGISTRY, algorithm, _stub)
    return calls


def test_blackjax_fit_dispatches_to_nuts(
    minimal_prior, minimal_chi2, coeff_group, tmp_path, monkeypatch
):
    """algorithm: nuts routes to the NUTS runner and yields logz=None."""
    calls = _patched_runner(monkeypatch, "nuts")
    settings = _blackjax_settings(tmp_path / "bj_logs", algorithm="nuts")

    result = blackjax_fit(
        prior=minimal_prior,
        chi2=minimal_chi2,
        coefficients=coeff_group,
        blackjax_settings=settings,
        n_samples=123,
    )

    assert len(calls) == 1
    assert calls[0]["n_samples"] == 123
    assert result.logz is None
    # log_likelihood must be -chi2/2 in sampler space
    x = jnp.array([2.0])
    assert float(calls[0]["log_likelihood"](x)) == pytest.approx(
        -float(minimal_chi2(x)) / 2.0
    )


def test_blackjax_fit_defaults_to_nested_sampling_when_key_absent(
    minimal_prior, minimal_chi2, coeff_group, tmp_path, monkeypatch
):
    """Settings dicts built before `algorithm` existed still run nested sampling."""
    calls = _patched_runner(monkeypatch, "nested_sampling")
    settings = _blackjax_settings(tmp_path / "bj_logs")
    del settings["algorithm"]

    blackjax_fit(
        prior=minimal_prior,
        chi2=minimal_chi2,
        coefficients=coeff_group,
        blackjax_settings=settings,
    )

    assert len(calls) == 1


def test_blackjax_fit_unknown_algorithm_raises(
    minimal_prior, minimal_chi2, coeff_group, tmp_path
):
    settings = _blackjax_settings(tmp_path / "bj_logs", algorithm="metropolis")

    with pytest.raises(ValueError, match="Unknown BlackJAX algorithm"):
        blackjax_fit(
            prior=minimal_prior,
            chi2=minimal_chi2,
            coefficients=coeff_group,
            blackjax_settings=settings,
        )
