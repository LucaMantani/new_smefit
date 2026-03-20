"""Unit tests for smefit.fit_actions."""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from smefit.fit_actions import (
    run_analytic_fit,
    run_blackjax_fit,
    run_individual_analytic_fits,
    run_individual_blackjax_fits,
    run_individual_ultranest_fits,
    run_ultranest_fit,
)
from smefit.fit_result import FitResult


def _make_mock_result(name):
    """Return a MagicMock that looks enough like FitResult for fit_actions."""
    result = MagicMock(spec=FitResult)
    result.free_parameters = [name]
    return result


# ---------------------------------------------------------------------------
# Single-fit actions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action, fixture_name",
    [
        (run_analytic_fit, "analytic_fit"),
        (run_ultranest_fit, "ultranest_fit"),
        (run_blackjax_fit, "blackjax_fit"),
    ],
)
def test_single_fit_action(action, fixture_name):
    result = _make_mock_result("OpA")
    out = Path("/output")
    action(result, out)
    result.print_summary.assert_called_once()
    result.write.assert_called_once_with(out)


# ---------------------------------------------------------------------------
# Individual-fits group actions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action",
    [
        run_individual_analytic_fits,
        run_individual_ultranest_fits,
        run_individual_blackjax_fits,
    ],
)
def test_individual_fits_action(action, tmp_path):
    results = [_make_mock_result("OpA"), _make_mock_result("OpB")]

    with patch("smefit.fit_actions.FitResultGroup") as MockGroup:
        mock_group = MagicMock()
        MockGroup.return_value = mock_group

        action(results, tmp_path)

    MockGroup.assert_called_once_with(results)
    mock_group.print_summary.assert_called_once()
    mock_group.write_results.assert_called_once_with(tmp_path)
