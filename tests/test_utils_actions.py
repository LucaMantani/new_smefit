"""Unit tests for smefit.utils_actions."""

import logging
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from smefit import dataset_info as di
from smefit.dataset_info import Bibliography, DatasetInfo, dataset_table_tex
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


def _fake_latex(monkeypatch, logs, fail=None):
    """Stand in for pdflatex and bibtex, recording which ran.

    Each pdflatex run leaves the next of *logs* as its log, an auxiliary file
    and a PDF; the tool named *fail* exits with a LaTeX error instead.
    """
    calls = []
    logs = iter(logs)

    def run(command, cwd, **kwargs):
        tool = command[0]
        calls.append(tool)
        if tool == "pdflatex":
            (Path(cwd) / "dataset_table_preview.log").write_text(next(logs))
            (Path(cwd) / "dataset_table_preview.aux").write_text("")
            (Path(cwd) / "dataset_table_preview.pdf").write_bytes(b"%PDF")
        if tool == fail:
            return subprocess.CompletedProcess(
                command, 1, stdout=b"! LaTeX Error: File `xltabular.sty' not found."
            )
        return subprocess.CompletedProcess(command, 0, stdout=b"")

    monkeypatch.setattr(di.subprocess, "run", run)
    monkeypatch.setattr(di.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    return calls


def test_write_dataset_table_compiles_only_when_asked(monkeypatch, tmp_path):
    calls = _fake_latex(monkeypatch, [])

    write_dataset_table("TABLE", Bibliography(), tmp_path)

    assert calls == []
    assert _files(tmp_path) == {"dataset_table.tex"}


def test_compile_pdf_reruns_latex_until_it_settles(monkeypatch, tmp_path):
    """After bibtex, and for as long as the log asks: longtable settles its
    column widths over runs, and stopping one short overfills the line."""
    calls = _fake_latex(
        monkeypatch,
        [
            "",
            "Package natbib Warning: Citation(s) may have changed.\n"
            "(natbib) Rerun to get citations correct.",
            "Package longtable Warning: Table widths have changed. Rerun LaTeX.",
            "",
        ],
    )

    write_dataset_table(
        "TABLE", Bibliography(bibtex="@article{A,}\n"), tmp_path, compile_pdf=True
    )

    assert calls == ["pdflatex", "bibtex", "pdflatex", "pdflatex", "pdflatex"]
    # the auxiliary files stay in the scratch folder
    assert _files(tmp_path) == {
        "dataset_table.tex",
        "dataset_table.bib",
        "dataset_table_preview.tex",
        "dataset_table_preview.pdf",
    }


def test_compile_pdf_without_bibliography_runs_no_bibtex(monkeypatch, tmp_path):
    calls = _fake_latex(monkeypatch, ["Table widths have changed. Rerun LaTeX.", ""])

    write_dataset_table("TABLE", Bibliography(), tmp_path, compile_pdf=True)

    assert calls == ["pdflatex", "pdflatex"]
    assert "dataset_table_preview.pdf" in _files(tmp_path)


def test_compile_pdf_gives_up_on_a_log_that_never_settles(
    monkeypatch, tmp_path, caplog
):
    calls = _fake_latex(monkeypatch, ["Rerun LaTeX."] * 10)

    with caplog.at_level(logging.WARNING, logger="smefit.dataset_info"):
        write_dataset_table("TABLE", Bibliography(), tmp_path, compile_pdf=True)

    assert calls == ["pdflatex"] * (1 + di._MAX_LATEX_RERUNS)
    assert "still asks for a rerun" in caplog.text
    assert "dataset_table_preview.pdf" in _files(tmp_path)


def test_compile_pdf_without_pdflatex_writes_the_table_and_warns(
    monkeypatch, tmp_path, caplog
):
    calls = _fake_latex(monkeypatch, [])
    monkeypatch.setattr(di.shutil, "which", lambda tool: None)

    with caplog.at_level(logging.WARNING, logger="smefit.dataset_info"):
        write_dataset_table("TABLE", Bibliography(), tmp_path, compile_pdf=True)

    assert calls == []
    assert "pdflatex not found" in caplog.text
    assert _files(tmp_path) == {"dataset_table.tex", "dataset_table_preview.tex"}


def test_compile_pdf_gives_up_on_a_tool_that_hangs(monkeypatch, tmp_path, caplog):
    def hang(command, cwd, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(di.subprocess, "run", hang)
    monkeypatch.setattr(di.shutil, "which", lambda tool: f"/usr/bin/{tool}")

    with caplog.at_level(logging.WARNING, logger="smefit.dataset_info"):
        write_dataset_table("TABLE", Bibliography(), tmp_path, compile_pdf=True)

    assert "pdflatex timed out" in caplog.text
    # it never got as far as writing a log
    assert _files(tmp_path) == {"dataset_table.tex", "dataset_table_preview.tex"}


def test_compile_pdf_latex_error_warns_and_keeps_the_log(monkeypatch, tmp_path, caplog):
    _fake_latex(monkeypatch, [""], fail="pdflatex")

    with caplog.at_level(logging.WARNING, logger="smefit.dataset_info"):
        write_dataset_table("TABLE", Bibliography(), tmp_path, compile_pdf=True)

    assert "xltabular.sty" in caplog.text
    assert _files(tmp_path) == {
        "dataset_table.tex",
        "dataset_table_preview.tex",
        "dataset_table_preview.log",
    }


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
