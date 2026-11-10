"""Unit tests for smefit.utils_actions."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from smefit import utils_actions as ua
from smefit.dataset_info import Bibliography, DatasetInfo, dataset_table_tex
from smefit.latex_preview import preview_tex
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


def _files(folder):
    return {path.name for path in Path(folder).iterdir()}


def _record_compile(monkeypatch):
    """Stand in for compile_preview, recording what it was asked to compile.

    How the preview compiles is tested in test_latex_preview.py; here only
    which files the action hands it.
    """
    calls = []
    monkeypatch.setattr(ua, "compile_preview", lambda *args: calls.append(args))
    return calls


def test_write_dataset_table_compiles_only_when_asked(monkeypatch, tmp_path):
    calls = _record_compile(monkeypatch)

    write_dataset_table("TABLE", Bibliography(), tmp_path)

    assert calls == []
    assert _files(tmp_path) == {"dataset_table.tex"}


def test_compile_pdf_previews_the_table_with_its_bibliography(monkeypatch, tmp_path):
    calls = _record_compile(monkeypatch)

    write_dataset_table(
        "TABLE", Bibliography(bibtex="@article{A,}\n"), tmp_path, compile_pdf=True
    )

    preview = (tmp_path / "dataset_table_preview.tex").read_text(encoding="utf-8")
    assert preview == preview_tex("dataset_table", "dataset_table")
    assert calls == [
        (tmp_path, "dataset_table_preview", "dataset_table", "dataset_table")
    ]


def test_compile_pdf_without_bibliography_previews_the_table_alone(
    monkeypatch, tmp_path
):
    calls = _record_compile(monkeypatch)

    write_dataset_table("TABLE", Bibliography(), tmp_path, compile_pdf=True)

    preview = (tmp_path / "dataset_table_preview.tex").read_text(encoding="utf-8")
    assert preview == preview_tex("dataset_table")
    assert calls == [(tmp_path, "dataset_table_preview", "dataset_table", None)]


@pytest.mark.skipif(
    shutil.which("pdflatex") is None or shutil.which("bibtex") is None,
    reason="needs pdflatex and bibtex",
)
def test_compile_pdf_typesets_the_table_and_its_bibliography(tmp_path):
    info = DatasetInfo(
        "ATLAS_tt_8TeV", 7, luminosity=(20.3,), references=("arxiv:1511.04716",)
    )
    bibliography = Bibliography(
        texkeys={"arxiv:1511.04716": "ATLAS:2015lsn"},
        bibtex=(
            '@article{ATLAS:2015lsn,\n    author = "Aad, Georges",\n'
            '    title = "{A measurement}",\n    journal = "Eur. Phys. J. C",\n'
            '    doi = "10.1140/epjc/s10052-016-3910-7",\n    year = "2016"\n}\n'
        ),
    )

    write_dataset_table(
        dataset_table_tex([info], bibliography),
        bibliography,
        tmp_path,
        compile_pdf=True,
    )

    assert _files(tmp_path) == {
        "dataset_table.tex",
        "dataset_table.bib",
        "dataset_table_preview.tex",
        "dataset_table_preview.pdf",
    }


def test_write_dataset_table_runs_on_a_fit_free_runcard(fixtures_data_path, tmp_path):
    """datasets and data_path are all it needs: no theory, no coefficients."""
    from reportengine.resourcebuilder import FuzzyTarget, ResourceBuilder

    from smefit.api import smefitAPI

    runcard = {
        "data_path": str(fixtures_data_path),
        "datasets": [{"name": "TESTDATA", "order": "NLO", "group": "Test"}],
        "fetch_bibtex": False,
        "dataset_table_caption": "The data of this fit.",
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
    # a runcard key, resolved by name like any provider argument
    assert r"\caption{The data of this fit.}" in tex
    assert not (tmp_path / "dataset_table.bib").exists()


def test_run_pca_delegates_to_the_pca_object():
    """The action only prints and writes; the analysis itself is the pca node."""
    result = MagicMock(spec=PCA)
    out = Path("/output")

    run_pca(result, out)

    result.print_summary.assert_called_once_with()
    result.write.assert_called_once_with(out)


def test_run_pca_per_fit_gives_each_fit_its_own_folder():
    """Under ``- fits run_pca`` they would otherwise share one pca.json."""
    result = MagicMock(spec=PCA)
    fit = MagicMock(fit_name="my_fit")

    run_pca(result, Path("/output"), fit=fit)

    result.write.assert_called_once_with(Path("/output/my_fit"))


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
