"""Unit tests for smefit.individual_fit."""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# individual_ultranest_fit
# ---------------------------------------------------------------------------


def test_individual_ultranest_fit_log_dir(coeff_group):
    settings = {
        "ReactiveNS_settings": {
            "log_dir": "/base",
            "resume": "resume",
        }
    }

    with patch("smefit.individual_fit.ultranest_fit") as mock_fit:
        from smefit.individual_fit import individual_ultranest_fit

        individual_ultranest_fit(
            individual_prior=MagicMock(),
            individual_chi2=MagicMock(),
            individual_coefficients=coeff_group,
            ultranest_settings=settings,
            individual_fit_coefficient="OpA",
        )

    called_settings = mock_fit.call_args[0][3]
    assert called_settings["ReactiveNS_settings"]["log_dir"] == "/base/OpA"


def test_individual_ultranest_fit_resume_overwrite(coeff_group):
    settings = {
        "ReactiveNS_settings": {
            "log_dir": "/base",
            "resume": "resume",
        }
    }

    with patch("smefit.individual_fit.ultranest_fit"):
        from smefit.individual_fit import individual_ultranest_fit

        individual_ultranest_fit(
            individual_prior=MagicMock(),
            individual_chi2=MagicMock(),
            individual_coefficients=coeff_group,
            ultranest_settings=settings,
            individual_fit_coefficient="OpA",
        )

    # original settings must not be mutated (deepcopy)
    assert settings["ReactiveNS_settings"]["resume"] == "resume"
    assert settings["ReactiveNS_settings"]["log_dir"] == "/base"


def test_individual_ultranest_fit_delegates_args(coeff_group):
    settings = {"ReactiveNS_settings": {"log_dir": "/x", "resume": "resume"}}
    mock_prior = MagicMock()
    mock_chi2 = MagicMock()

    with patch("smefit.individual_fit.ultranest_fit") as mock_fit:
        from smefit.individual_fit import individual_ultranest_fit

        individual_ultranest_fit(
            individual_prior=mock_prior,
            individual_chi2=mock_chi2,
            individual_coefficients=coeff_group,
            ultranest_settings=settings,
            individual_fit_coefficient="OpB",
        )

    args = mock_fit.call_args[0]
    assert args[0] is mock_prior
    assert args[1] is mock_chi2
    assert args[2] is coeff_group


# ---------------------------------------------------------------------------
# individual_blackjax_fit
# ---------------------------------------------------------------------------


def test_individual_blackjax_fit_log_dir(coeff_group):
    settings = {"log_dir": "/logs"}

    with patch("smefit.individual_fit.blackjax_fit"):
        from smefit.individual_fit import individual_blackjax_fit

        individual_blackjax_fit(
            individual_prior=MagicMock(),
            individual_chi2=MagicMock(),
            individual_coefficients=coeff_group,
            blackjax_settings=settings,
            individual_fit_coefficient="OpC",
        )

    # original not mutated
    assert settings["log_dir"] == "/logs"


def test_individual_blackjax_fit_log_dir_per_coeff(coeff_group):
    settings = {"log_dir": "/logs"}

    with patch("smefit.individual_fit.blackjax_fit") as mock_fit:
        from smefit.individual_fit import individual_blackjax_fit

        individual_blackjax_fit(
            individual_prior=MagicMock(),
            individual_chi2=MagicMock(),
            individual_coefficients=coeff_group,
            blackjax_settings=settings,
            individual_fit_coefficient="OpC",
        )

    called_settings = mock_fit.call_args[0][3]
    assert called_settings["log_dir"] == "/logs/OpC"


# ---------------------------------------------------------------------------
# individual_analytic_fit
# ---------------------------------------------------------------------------


def test_individual_analytic_fit_delegates(coeff_group):
    mock_model = MagicMock()
    mock_data = MagicMock()
    mock_covmat = MagicMock()
    mock_chi2 = MagicMock()
    mock_result = MagicMock()

    with patch(
        "smefit.individual_fit.analytic_fit", return_value=mock_result
    ) as mock_fn:
        from smefit.individual_fit import individual_analytic_fit

        result = individual_analytic_fit(
            mock_model, mock_data, mock_covmat, mock_chi2, "OpA"
        )

    mock_fn.assert_called_once_with(
        mock_model, mock_data, mock_covmat, mock_chi2, 10000, 42
    )
    assert result is mock_result
    # named after the subdirectory it is written to
    assert result.fit_name == "OpA"


# ---------------------------------------------------------------------------
# individual_hessian_fit
# ---------------------------------------------------------------------------


def test_individual_hessian_fit_delegates():
    mock_model = MagicMock()
    mock_chi2 = MagicMock()
    mock_optimizer = MagicMock()
    mock_settings = {
        "sm_solution": True,
        "n_steps": 100,
        "tol": 1e-8,
        "n_samples": 50,
        "seed": 0,
    }
    mock_result = MagicMock()

    with patch(
        "smefit.individual_fit.hessian_fit", return_value=mock_result
    ) as mock_fn:
        from smefit.individual_fit import individual_hessian_fit

        result = individual_hessian_fit(
            mock_model, mock_chi2, mock_optimizer, mock_settings, "OpA"
        )

    mock_fn.assert_called_once_with(
        mock_model, mock_chi2, mock_optimizer, mock_settings
    )
    assert result is mock_result
    assert result.fit_name == "OpA"
