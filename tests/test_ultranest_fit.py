"""Unit tests for smefit/ultranest_fit.py — sampler is mocked."""

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from smefit.fit_result import Fit
from smefit.ultranest_fit import ultranest_fit
from smefit.whitening import WhitenTransform

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_RESULT = {
    "logz": -3.0,
    "maximum_likelihood": {"logl": -1.5, "point": [0.1]},
    "samples": np.zeros((20, 1)),
}

_ULTRANEST_SETTINGS = {
    "ultranest_seed": 42,
    "sampler_plot": False,
    "ReactiveNS_settings": {
        "log_dir": "/tmp/test_ultranest",
        "resume": "overwrite",
        "vectorized": False,
    },
    "Run_settings": {},
    "SliceSampler_settings": {},
}

_MOCK_SAMPLES = {"OpA": jnp.zeros(20), "OpB": jnp.full(20, 2.0), "OpC": jnp.zeros(20)}
_MOCK_BEST = {"OpA": 0.1, "OpB": 2.0, "OpC": 0.01}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ultranest_fit_happy_path(minimal_prior, minimal_chi2, coeff_group):
    """Non-whitening path: verify Fit fields are populated correctly."""
    mock_sampler = MagicMock()
    mock_sampler.run.return_value = _FAKE_RESULT

    with (
        patch(
            "smefit.ultranest_fit.ultranest.ReactiveNestedSampler",
            return_value=mock_sampler,
        ),
        patch(
            "smefit.ultranest_fit.resolve_posterior",
            return_value=(_MOCK_SAMPLES, _MOCK_BEST),
        ),
    ):
        result = ultranest_fit(
            prior=minimal_prior,
            chi2=minimal_chi2,
            coefficients=coeff_group,
            ultranest_settings=_ULTRANEST_SETTINGS,
            whitening_transformation=None,
        )

    assert isinstance(result, Fit)
    assert result.fit_type == "ultranest"
    assert result.fit_name == "ultranest"
    assert result.logz == pytest.approx(-3.0)
    assert result.max_loglikelihood == pytest.approx(-1.5)
    assert result.num_data == 10
    assert result.whitening_active is False
    assert result.samples is _MOCK_SAMPLES
    assert result.best_fit_point is _MOCK_BEST


def test_ultranest_fit_whitening_active(minimal_prior, minimal_chi2, coeff_group):
    """When a whitening_transformation is provided, whitening_active should be True."""
    mock_sampler = MagicMock()
    mock_sampler.run.return_value = _FAKE_RESULT
    transform = WhitenTransform(matrix=jnp.eye(1), shift=jnp.zeros(1))

    with (
        patch(
            "smefit.ultranest_fit.ultranest.ReactiveNestedSampler",
            return_value=mock_sampler,
        ),
        patch(
            "smefit.ultranest_fit.resolve_posterior",
            return_value=(_MOCK_SAMPLES, _MOCK_BEST),
        ),
    ):
        result = ultranest_fit(
            prior=minimal_prior,
            chi2=minimal_chi2,
            coefficients=coeff_group,
            ultranest_settings=_ULTRANEST_SETTINGS,
            whitening_transformation=transform,
        )

    assert result.whitening_active is True


def test_ultranest_fit_vectorized_mode(minimal_prior, minimal_chi2, coeff_group):
    """vectorized=True selects the vmap log_likelihood path; fit still completes."""
    mock_sampler = MagicMock()
    mock_sampler.run.return_value = _FAKE_RESULT

    settings = {**_ULTRANEST_SETTINGS}
    settings["ReactiveNS_settings"] = {
        **settings["ReactiveNS_settings"],
        "vectorized": True,
    }

    with (
        patch(
            "smefit.ultranest_fit.ultranest.ReactiveNestedSampler",
            return_value=mock_sampler,
        ),
        patch(
            "smefit.ultranest_fit.resolve_posterior",
            return_value=(_MOCK_SAMPLES, _MOCK_BEST),
        ),
    ):
        result = ultranest_fit(
            prior=minimal_prior,
            chi2=minimal_chi2,
            coefficients=coeff_group,
            ultranest_settings=settings,
        )

    assert isinstance(result, Fit)


def test_ultranest_fit_slice_sampler(minimal_prior, minimal_chi2, coeff_group):
    """SliceSampler_settings non-empty → sampler.stepsampler is set."""
    mock_sampler = MagicMock()
    mock_sampler.run.return_value = _FAKE_RESULT
    mock_slice = MagicMock()

    settings = {**_ULTRANEST_SETTINGS}
    settings["SliceSampler_settings"] = {"nsteps": 5}

    with (
        patch(
            "smefit.ultranest_fit.ultranest.ReactiveNestedSampler",
            return_value=mock_sampler,
        ),
        patch(
            "smefit.ultranest_fit.ustepsampler.SliceSampler", return_value=mock_slice
        ),
        patch(
            "smefit.ultranest_fit.resolve_posterior",
            return_value=(_MOCK_SAMPLES, _MOCK_BEST),
        ),
    ):
        ultranest_fit(
            prior=minimal_prior,
            chi2=minimal_chi2,
            coefficients=coeff_group,
            ultranest_settings=settings,
        )

    # stepsampler attribute should have been set on the mock sampler
    assert mock_sampler.stepsampler == mock_slice


def test_ultranest_fit_posterior_truncation_warning(
    minimal_prior, minimal_chi2, coeff_group, caplog
):
    """When n_samples exceeds available posterior samples, a warning is issued."""
    few_samples_result = {
        "logz": -3.0,
        "maximum_likelihood": {"logl": -1.5, "point": [0.1]},
        "samples": np.zeros((5, 1)),  # only 5 samples available
    }
    mock_sampler = MagicMock()
    mock_sampler.run.return_value = few_samples_result

    with (
        patch(
            "smefit.ultranest_fit.ultranest.ReactiveNestedSampler",
            return_value=mock_sampler,
        ),
        patch(
            "smefit.ultranest_fit.resolve_posterior",
            return_value=(_MOCK_SAMPLES, _MOCK_BEST),
        ),
        caplog.at_level("WARNING"),
    ):
        ultranest_fit(
            prior=minimal_prior,
            chi2=minimal_chi2,
            coefficients=coeff_group,
            ultranest_settings=_ULTRANEST_SETTINGS,
            n_samples=10000,  # much larger than 5
        )

    assert any("posterior samples" in msg.lower() for msg in caplog.messages)
