"""Unit tests for smefit.utils_actions."""

from pathlib import Path
from unittest.mock import MagicMock

from smefit.dataset_info import Bibliography
from smefit.pca import PCA
from smefit.utils_actions import run_pca, write_dataset_table


def test_write_dataset_table_writes_the_table_and_its_bibtex(tmp_path):
    write_dataset_table("TABLE", Bibliography(bibtex="@article{A,}\n"), tmp_path)

    assert (tmp_path / "dataset_table.tex").read_text(encoding="utf-8") == "TABLE"
    assert (tmp_path / "dataset_table.bib").read_text(encoding="utf-8") == (
        "@article{A,}\n"
    )


def test_write_dataset_table_without_bibliography_writes_no_bib(tmp_path):
    """Nothing looked up: the table links to its references and stands alone."""
    write_dataset_table("TABLE", Bibliography(), tmp_path)

    assert (tmp_path / "dataset_table.tex").exists()
    assert not (tmp_path / "dataset_table.bib").exists()


def test_write_dataset_table_runs_on_a_fit_free_runcard(fixtures_data_path, tmp_path):
    """datasets and data_path are all it needs: no theory, no coefficients."""
    from reportengine.resourcebuilder import FuzzyTarget, ResourceBuilder

    from smefit.api import smefitAPI

    runcard = {
        "data_path": str(fixtures_data_path),
        "datasets": [{"name": "TESTDATA", "order": "NLO", "group": "Test"}],
        "fetch_bibtex": False,
    }

    config = smefitAPI.config_class(runcard, environment=smefitAPI.loadedenv)
    builder = ResourceBuilder(
        config,
        smefitAPI.provider_loaded,
        [FuzzyTarget("write_dataset_table", (), (), ())],
    )
    builder.rootns.update(smefitAPI.loadedenv.ns_dump())
    builder.rootns["output_path"] = tmp_path
    builder.resolve_fuzzytargets()
    builder.execute_sequential()

    tex = (tmp_path / "dataset_table.tex").read_text(encoding="utf-8")
    assert r"\texttt{TESTDATA} &  & 3 & 1.5 &  \\" in tex
    assert not (tmp_path / "dataset_table.bib").exists()


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
