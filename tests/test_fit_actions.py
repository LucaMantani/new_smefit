"""Unit tests for smefit.fit_actions."""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from smefit.fit_actions import (
    run_analytic_fit,
    run_blackjax_fit,
    run_hessian_fit,
    run_individual_analytic_fits,
    run_individual_blackjax_fits,
    run_individual_hessian_fits,
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
        (run_hessian_fit, "hessian_fit"),
    ],
)
def test_single_fit_action(action, fixture_name):
    result = _make_mock_result("OpA")
    out = Path("/output")
    with patch("smefit.fit_actions._write_input_manifest") as write_manifest:
        action(result, out)
    result.print_summary.assert_called_once()
    result.write.assert_called_once_with(out)
    write_manifest.assert_called_once_with(out)


# ---------------------------------------------------------------------------
# Individual-fits group actions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action",
    [
        run_individual_analytic_fits,
        run_individual_ultranest_fits,
        run_individual_blackjax_fits,
        run_individual_hessian_fits,
    ],
)
def test_individual_fits_action(action, tmp_path):
    results = [_make_mock_result("OpA"), _make_mock_result("OpB")]

    with patch("smefit.fit_actions.FitResultGroup") as MockGroup, patch(
        "smefit.fit_actions._write_input_manifest"
    ) as write_manifest:
        mock_group = MagicMock()
        MockGroup.return_value = mock_group

        action(results, tmp_path)

    MockGroup.assert_called_once_with(results)
    mock_group.print_summary.assert_called_once()
    mock_group.write_results.assert_called_once_with(tmp_path)
    # one manifest for the whole group, at the top of the fit directory
    write_manifest.assert_called_once_with(tmp_path)


# ---------------------------------------------------------------------------
# The input manifest every fit action writes
# ---------------------------------------------------------------------------


def test_write_input_manifest_records_the_copied_runcard(
    tmp_path, fixtures_data_path, fixtures_theory_path
):
    """Built from the runcard reportengine copied into the fit directory."""
    import yaml

    from smefit.fit_actions import _write_input_manifest
    from smefit.input_manifest import InputManifest

    runcard = {
        "data_path": str(fixtures_data_path),
        "theory_path": str(fixtures_theory_path),
        "datasets": [{"name": "TESTDATA", "order": "NLO"}],
    }
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "runcard.yaml").write_text(yaml.dump(runcard))

    _write_input_manifest(tmp_path)

    assert InputManifest.load(tmp_path) == InputManifest.from_runcard(runcard)
