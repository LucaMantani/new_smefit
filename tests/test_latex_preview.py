"""Unit tests for smefit.latex_preview."""

import logging
import subprocess
from pathlib import Path

from smefit import latex_preview as lp
from smefit.latex_preview import compile_preview, preview_tex


def test_preview_typesets_the_source_at_paper_width():
    tex = preview_tex("dataset_table")

    assert r"\usepackage[textwidth=483pt]{geometry}" in tex
    assert r"\input{dataset_table}" in tex
    assert "natbib" not in tex
    assert r"\bibliography" not in tex


def test_preview_with_bibliography_links_its_dois():
    tex = preview_tex("dataset_table", "dataset_table")

    assert r"\usepackage[numbers]{natbib}" in tex
    assert r"\newcommand{\doi}[1]{\href{https://doi.org/#1}{doi:#1}}" in tex
    assert tex.endswith(
        "\\bibliographystyle{unsrtnat}\n\\bibliography{dataset_table}\n"
        "\\end{document}\n"
    )


def test_preview_reads_the_files_it_is_given():
    tex = preview_tex("bounds_table", "references")

    assert r"\input{bounds_table}" in tex
    assert r"\bibliography{references}" in tex
    assert "dataset_table" not in tex


def _files(folder):
    return {path.name for path in Path(folder).iterdir()}


def _write_sources(folder, bibliography=None):
    """A preview of ``table.tex`` written to *folder*, as an action leaves it."""
    (folder / "table.tex").write_text("TABLE")
    (folder / "table_preview.tex").write_text(preview_tex("table", bibliography))
    if bibliography is not None:
        (folder / f"{bibliography}.bib").write_text("@article{A,}\n")


def _fake_latex(monkeypatch, logs, fail=None):
    """Stand in for pdflatex and bibtex, recording which ran.

    Each pdflatex run leaves the next of *logs* as the log of the document it
    compiles, an auxiliary file and a PDF; the tool named *fail* exits with a
    LaTeX error instead.
    """
    calls = []
    logs = iter(logs)

    def run(command, cwd, **kwargs):
        tool = command[0]
        calls.append(tool)
        if tool == "pdflatex":
            document = Path(cwd) / Path(command[-1]).stem
            document.with_suffix(".log").write_text(next(logs))
            document.with_suffix(".aux").write_text("")
            document.with_suffix(".pdf").write_bytes(b"%PDF")
        if tool == fail:
            return subprocess.CompletedProcess(
                command, 1, stdout=b"! LaTeX Error: File `xltabular.sty' not found."
            )
        return subprocess.CompletedProcess(command, 0, stdout=b"")

    monkeypatch.setattr(lp.subprocess, "run", run)
    monkeypatch.setattr(lp.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    return calls


def test_compile_reruns_latex_until_it_settles(monkeypatch, tmp_path):
    """After bibtex, and for as long as the log asks: longtable settles its
    column widths over runs, and stopping one short overfills the line."""
    _write_sources(tmp_path, bibliography="refs")
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

    compile_preview(tmp_path, "table_preview", "table", "refs")

    assert calls == ["pdflatex", "bibtex", "pdflatex", "pdflatex", "pdflatex"]
    # the auxiliary files stay in the scratch folder
    assert _files(tmp_path) == {
        "table.tex",
        "refs.bib",
        "table_preview.tex",
        "table_preview.pdf",
    }


def test_compile_without_bibliography_runs_no_bibtex(monkeypatch, tmp_path):
    _write_sources(tmp_path)
    calls = _fake_latex(monkeypatch, ["Table widths have changed. Rerun LaTeX.", ""])

    compile_preview(tmp_path, "table_preview", "table")

    assert calls == ["pdflatex", "pdflatex"]
    assert "table_preview.pdf" in _files(tmp_path)


def test_compile_gives_up_on_a_log_that_never_settles(monkeypatch, tmp_path, caplog):
    _write_sources(tmp_path)
    calls = _fake_latex(monkeypatch, ["Rerun LaTeX."] * 10)

    with caplog.at_level(logging.WARNING, logger="smefit.latex_preview"):
        compile_preview(tmp_path, "table_preview", "table")

    assert calls == ["pdflatex"] * (1 + lp._MAX_LATEX_RERUNS)
    assert "table_preview.pdf may not be final" in caplog.text
    assert "table_preview.pdf" in _files(tmp_path)


def test_compile_without_pdflatex_warns(monkeypatch, tmp_path, caplog):
    _write_sources(tmp_path)
    calls = _fake_latex(monkeypatch, [])
    monkeypatch.setattr(lp.shutil, "which", lambda tool: None)

    with caplog.at_level(logging.WARNING, logger="smefit.latex_preview"):
        compile_preview(tmp_path, "table_preview", "table")

    assert calls == []
    assert "pdflatex not found: table_preview.pdf is not compiled" in caplog.text
    assert _files(tmp_path) == {"table.tex", "table_preview.tex"}


def test_compile_gives_up_on_a_tool_that_hangs(monkeypatch, tmp_path, caplog):
    _write_sources(tmp_path)

    def hang(command, cwd, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(lp.subprocess, "run", hang)
    monkeypatch.setattr(lp.shutil, "which", lambda tool: f"/usr/bin/{tool}")

    with caplog.at_level(logging.WARNING, logger="smefit.latex_preview"):
        compile_preview(tmp_path, "table_preview", "table")

    assert "pdflatex timed out" in caplog.text
    # it never got as far as writing a log
    assert _files(tmp_path) == {"table.tex", "table_preview.tex"}


def test_compile_latex_error_warns_and_keeps_the_log(monkeypatch, tmp_path, caplog):
    _write_sources(tmp_path)
    _fake_latex(monkeypatch, [""], fail="pdflatex")

    with caplog.at_level(logging.WARNING, logger="smefit.latex_preview"):
        compile_preview(tmp_path, "table_preview", "table")

    assert "xltabular.sty" in caplog.text
    assert "see table_preview.log" in caplog.text
    assert _files(tmp_path) == {
        "table.tex",
        "table_preview.tex",
        "table_preview.log",
    }
