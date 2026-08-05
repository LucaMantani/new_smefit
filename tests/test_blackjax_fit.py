"""Unit tests for smefit/blackjax_fit.py — blackjax and anesthetic are mocked."""

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest

from smefit.blackjax_fit import blackjax_fit
from smefit.fit_result import Fit
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


def _blackjax_settings(log_dir):
    return {
        "n_posterior_samples": 1000,
        "n_live": 10,
        "repeats": 1,
        "delete_fraction": 0.5,
        "log_precision": -2,
        "seed": 0,
        "posterior_resampling_seed": 0,
        "log_dir": str(log_dir),
    }


_MOCK_SAMPLES = {"OpA": jnp.zeros(2), "OpB": jnp.full(2, 2.0), "OpC": jnp.zeros(2)}
_MOCK_BEST = {"OpA": 0.0, "OpB": 2.0, "OpC": 0.0}


# ---------------------------------------------------------------------------
# Shared patching context
# ---------------------------------------------------------------------------


def _run_blackjax_fit(
    prior, chi2, coeff_group, settings, whitening_transformation=None
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
        patch("smefit.blackjax_fit.blackjax.nss", return_value=mock_algo),
        patch("smefit.blackjax_fit.finalise", return_value=final_states),
        patch("smefit.blackjax_fit.ess", return_value=2),
        patch("smefit.blackjax_fit.log_weights", return_value=jnp.zeros(3)),
        patch("smefit.blackjax_fit.sample", return_value=mock_sample_result),
        patch("smefit.blackjax_fit.anesthetic.NestedSamples", return_value=mock_nested),
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
        )
    return result, mock_nested


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_blackjax_fit_happy_path(minimal_prior, minimal_chi2, coeff_group, tmp_path):
    """Non-whitening path: Fit fields are populated correctly."""
    settings = _blackjax_settings(tmp_path / "bj_logs")
    result, _ = _run_blackjax_fit(minimal_prior, minimal_chi2, coeff_group, settings)

    assert isinstance(result, Fit)
    assert result.fit_type == "blackjax_NS"
    assert result.fit_name == "blackjax_NS"
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
    settings["n_posterior_samples"] = 10000  # far more than 2 available

    with caplog.at_level("WARNING"):
        _run_blackjax_fit(minimal_prior, minimal_chi2, coeff_group, settings)

    assert any("posterior samples" in msg.lower() for msg in caplog.messages)
