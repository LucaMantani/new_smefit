"""Unit tests for smefit.utils_actions."""

from pathlib import Path
from unittest.mock import MagicMock

from smefit.pca import PCA
from smefit.utils_actions import run_pca


def test_run_pca_delegates_to_the_pca_object():
    """The action only prints and writes; the analysis itself is the pca node."""
    result = MagicMock(spec=PCA)
    out = Path("/output")

    run_pca(result, out)

    result.print_summary.assert_called_once_with()
    result.write.assert_called_once_with(out)


def test_rge_matrix_graph_resolves(fixtures_data_path, fixtures_theory_path, tmp_path):
    """The action resolves through reportengine on a fit-free runcard.

    Resolving (without executing) runs check_types and proves no fit settings
    block is needed to reach rge_matrix.
    """
    from reportengine.resourcebuilder import FuzzyTarget, ResourceBuilder

    from smefit.api import smefitAPI

    prior = {"dist": "uniform", "low": -1.0, "high": 1.0}
    runcard = {
        "data_path": str(fixtures_data_path),
        "theory_path": str(fixtures_theory_path),
        "datasets": [{"name": "TESTDATA", "order": "NLO"}],
        "coefficients": {
            "OpA": {"free": True, "prior": prior},
            "OpB": {"free": True, "prior": prior},
        },
        "rge": {"init_scale": 1000.0},
    }

    config = smefitAPI.config_class(runcard, environment=smefitAPI.loadedenv)
    builder = ResourceBuilder(
        config,
        smefitAPI.provider_loaded,
        [FuzzyTarget("rge_matrix", (), (), ())],
        perform_final=False,
    )
    builder.rootns.update(smefitAPI.loadedenv.ns_dump())
    builder.rootns["output_path"] = tmp_path
    builder.resolve_fuzzytargets()
