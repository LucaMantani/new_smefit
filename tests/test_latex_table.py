"""Unit tests for smefit.latex_table — the LaTeX a report table copies out."""

from __future__ import annotations

import re
import shutil
import subprocess

import pandas as pd
import pytest

from smefit.latex_table import savelatextable, to_latex
from smefit.op_to_latex import coeff_info_latex

# ---------------------------------------------------------------------------
# to_latex
# ---------------------------------------------------------------------------


@pytest.fixture
def grouped_frame() -> pd.DataFrame:
    """A table shaped like the bounds one: two fits, two columns each."""
    return pd.DataFrame(
        [["0.10", "[0.0, 0.2]", "0.20", "[0.1, 0.3]"]],
        index=[r"$c_{\varphi D}$"],
        columns=pd.MultiIndex.from_tuples(
            [
                ("fit_a", "best"),
                ("fit_a", r"95\% CL"),
                ("fit_b", "best"),
                ("fit_b", r"95\% CL"),
            ]
        ),
    )


def test_to_latex_wraps_the_table_in_a_float(grouped_frame: pd.DataFrame) -> None:
    latex = to_latex(grouped_frame)

    assert latex.startswith("% \\usepackage{tabularx}")  # what it needs, said
    assert r"\begin{table}" in latex
    assert r"\end{table}" in latex
    assert r"\centering" in latex
    assert r"\footnotesize" in latex


def test_to_latex_column_spec_stretches_only_the_first_column(
    grouped_frame: pd.DataFrame,
) -> None:
    """The coefficient column takes up the slack, the numbers stay tight."""
    assert r"\begin{tabularx}{\linewidth}{|X|c|c|c|c|}" in to_latex(grouped_frame)


def test_to_latex_heads_each_fit_with_a_spanning_cell(
    grouped_frame: pd.DataFrame,
) -> None:
    latex = to_latex(grouped_frame)

    assert r"\multicolumn{2}{c|}{fit_a} & \multicolumn{2}{c|}{fit_b}" in latex
    assert r"\multirow{2}{*}{Coefficient}" in latex
    # the column names go on the second header row, under their fit
    assert r" & best & 95\% CL & best & 95\% CL \\" in latex


def test_to_latex_rules_the_header_off_twice(grouped_frame: pd.DataFrame) -> None:
    """The house style separates the header from the body with a double
    rule."""
    assert "\\hline\n\\hline" in to_latex(grouped_frame)


def test_to_latex_writes_the_rows(grouped_frame: pd.DataFrame) -> None:
    assert r"$c_{\varphi D}$ & 0.10 & [0.0, 0.2] & 0.20 & [0.1, 0.3] \\" in to_latex(
        grouped_frame
    )


def test_to_latex_takes_a_flat_column_index() -> None:
    """Not every table groups its columns by fit — the Fisher one does not."""
    frame = pd.DataFrame([[1, 2]], index=["OpA"], columns=["DS_A", "DS_B"])

    latex = to_latex(frame)

    assert r"Coefficient & DS_A & DS_B \\" in latex
    assert r"\multicolumn" not in latex


def test_to_latex_index_header_is_settable() -> None:
    frame = pd.DataFrame([[1]], index=["OpA"], columns=["DS_A"])

    assert r"Operator & DS_A \\" in to_latex(frame, index_header="Operator")


def test_to_latex_captions_and_labels_when_asked(grouped_frame: pd.DataFrame) -> None:
    latex = to_latex(grouped_frame, caption="The bounds.", label="tab:bounds")

    assert r"\caption{The bounds.}" in latex
    assert r"\label{tab:bounds}" in latex
    # the caption goes under the table, as in the house style
    assert latex.index(r"\end{tabularx}") < latex.index(r"\caption")


def test_to_latex_leaves_out_a_caption_it_was_not_given(
    grouped_frame: pd.DataFrame,
) -> None:
    latex = to_latex(grouped_frame)

    assert r"\caption" not in latex
    assert r"\label" not in latex


def test_to_latex_spans_the_two_solutions_of_one_coefficient() -> None:
    """A coefficient with two solutions is named once, over both its rows —
    the same \\multirow idiom the rest of the style uses."""
    frame = pd.DataFrame(
        [["0.1"], ["5.0"], ["0.3"]],
        index=[r"$c_1$", r"$c_1$ (2)", r"$c_2$"],
        columns=["best"],
    )

    latex = to_latex(frame)

    assert r"\multirow{2}{*}{$c_1$} & 0.1 \\" in latex
    assert r" & 5.0 \\" in latex  # the continuation row leaves the name blank
    assert r"$c_2$ & 0.3 \\" in latex


def test_to_latex_rules_between_coefficients_not_between_solutions() -> None:
    """A rule inside a \\multirow would cut the name it spans in half."""
    frame = pd.DataFrame(
        [["0.1"], ["5.0"], ["0.3"]],
        index=[r"$c_1$", r"$c_1$ (2)", r"$c_2$"],
        columns=["best"],
    )

    body = to_latex(frame).split("\\hline\n\\hline\n")[1]

    assert body.splitlines()[:4] == [
        r"\multirow{2}{*}{$c_1$} & 0.1 \\",
        r" & 5.0 \\",
        r"\hline",
        r"$c_2$ & 0.3 \\",
    ]


def test_to_latex_restores_what_the_html_could_not_render() -> None:
    """The reach header avoids \\sqrt only because pandoc cannot render it;
    a LaTeX document has no such problem."""
    frame = pd.DataFrame(
        [["—"]], index=[r"$c_1$"], columns=[r"$\Lambda/c_i^{1/2}$ [TeV]"]
    )

    latex = to_latex(frame)

    assert r"$\Lambda/\sqrt{c_i}$ [TeV]" in latex
    assert r"\textemdash" in latex  # and the em dash becomes a LaTeX one
    assert "—" not in latex


# ---------------------------------------------------------------------------
# savelatextable — the final action, and what it puts in the report
# ---------------------------------------------------------------------------


@pytest.fixture
def saved(tmp_path, grouped_frame: pd.DataFrame):
    grouped_frame.attrs["latex_caption"] = "The bounds."
    grouped_frame.attrs["latex_label"] = "tab:bounds"
    return savelatextable(grouped_frame, tmp_path / "bounds.csv"), tmp_path


def test_savelatextable_writes_the_csv_and_the_tex(saved) -> None:
    """The .tex is there whatever the browser thinks of the clipboard."""
    _, tmp_path = saved

    assert (tmp_path / "bounds.csv").exists()
    assert r"\begin{table}" in (tmp_path / "bounds.tex").read_text()


def test_savelatextable_reads_the_caption_off_the_frame(saved) -> None:
    """How a provider passes it: reportengine hands the object the provider
    returned straight to the final action, so `attrs` survives."""
    _, tmp_path = saved

    latex = (tmp_path / "bounds.tex").read_text()
    assert r"\caption{The bounds.}" in latex
    assert r"\label{tab:bounds}" in latex


def test_saved_table_still_renders_as_html(saved) -> None:
    """The report table itself is unchanged; this only adds to it."""
    table, _ = saved

    assert 'class="dataframe"' in table.as_markdown
    assert "<tbody>" in table.as_markdown


def test_saved_table_offers_a_button(saved) -> None:
    table, _ = saved

    assert 'class="copy-latex"' in table.as_markdown
    assert "smefitCopyLatex(this, 'latex-bounds')" in table.as_markdown


def test_saved_table_button_attribute_is_quoted_once(saved) -> None:
    """A double quote inside a double-quoted attribute ends it early, and
    pandoc then fails to read the tag and escapes it into the page as text."""
    button = [line for line in table_lines(saved) if "copy-latex" in line][0]

    assert button.count('"') % 2 == 0
    assert 'onclick="smefitCopyLatex' in button


def test_saved_table_carries_the_latex_verbatim(saved) -> None:
    """A <script> is a raw-text element: HTML-escaping its content would put
    the escapes themselves on the clipboard."""
    table, tmp_path = saved

    payload = table.as_markdown.split(
        '<script type="application/x-latex" id="latex-bounds">'
    )[1].split("</script>")[0]

    assert payload == (tmp_path / "bounds.tex").read_text().rstrip("\n")
    assert "&amp;" not in payload
    assert r"\multicolumn{2}{c|}{fit_a}" in payload


def test_saved_table_defines_the_copy_helper_once_per_page(saved) -> None:
    """Two tables in one report each carry the helper; the second must not
    replace the first's under a reader's feet."""
    table, _ = saved

    assert "window.smefitCopyLatex = window.smefitCopyLatex ||" in table.as_markdown


def test_saved_table_falls_back_for_a_page_opened_from_disk(saved) -> None:
    """navigator.clipboard needs a secure context, which file:// is not."""
    table, _ = saved

    assert "isSecureContext" in table.as_markdown
    assert 'document.execCommand("copy")' in table.as_markdown


def table_lines(saved) -> list[str]:
    table, _ = saved
    return table.as_markdown.splitlines()


def test_to_latex_rules_the_fit_names_off_partially(
    grouped_frame: pd.DataFrame,
) -> None:
    """A full \\hline there would cut through the corner cell the coefficient
    header spans, so the rule covers the data columns only."""
    latex = to_latex(grouped_frame)

    header = latex.split("\\hline\n", 1)[1].splitlines()
    assert r"\multirow{2}{*}{Coefficient}" in header[0]  # the fit names
    assert header[1] == r"\cline{2-5}"  # four data columns, numbered from 2
    assert header[2].startswith(" & best")  # then the column names


def test_to_latex_partial_rule_covers_every_data_column() -> None:
    frame = pd.DataFrame(
        [["1", "2"]],
        index=["OpA"],
        columns=pd.MultiIndex.from_tuples([("f", "a"), ("f", "b")]),
    )

    assert r"\cline{2-3}" in to_latex(frame)


def test_to_latex_flat_header_has_no_partial_rule() -> None:
    """One header row has nothing to separate from the row above it."""
    frame = pd.DataFrame([[1]], index=["OpA"], columns=["DS_A"])

    assert r"\cline" not in to_latex(frame)


def test_to_latex_names_the_packages_it_needs(grouped_frame: pd.DataFrame) -> None:
    """Including amssymb, which nothing in the table's own markup needs but
    the \\Box in some operator names does."""
    preamble = to_latex(grouped_frame).split(r"\begin{table}")[0]

    assert r"\usepackage{tabularx}" in preamble
    assert r"\usepackage{multirow}" in preamble
    assert r"\usepackage{amssymb}" in preamble


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no pdflatex")
def test_to_latex_compiles_with_the_packages_it_names(tmp_path) -> None:
    """The emitted LaTeX is only useful if it builds — against exactly the
    packages the preamble comment lists, over every operator label there is,
    since those are what a real table's rows are made of."""
    labels = list(dict.fromkeys(str(name) for name in coeff_info_latex.values()))
    frame = pd.DataFrame(
        [["0.10", "[0.0, 0.2]", "—"] for _ in labels],
        index=labels,
        columns=pd.MultiIndex.from_tuples(
            [
                (r"$\mathrm{Fit}$", "best"),
                (r"$\mathrm{Fit}$", r"99.99994\% CL"),
                (r"$\mathrm{Fit}$", r"$\Lambda/c_i^{1/2}$ [TeV]"),
            ]
        ),
    )
    latex = to_latex(frame, caption="Every operator label.", label="tab:all")
    # the packages are read back out of the emitted comment, so that adding a
    # macro without declaring it fails here rather than in someone's document
    packages = re.findall(r"\\usepackage\{(\w+)\}", latex.split(r"\begin{table}")[0])

    (tmp_path / "table.tex").write_text(latex)
    (tmp_path / "doc.tex").write_text(
        "\\documentclass{article}\n"
        + "".join(rf"\usepackage{{{name}}}" + "\n" for name in packages)
        + "\\begin{document}\n\\input{table}\n\\end{document}\n"
    )
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout[-3000:]
    assert (tmp_path / "doc.pdf").exists()
