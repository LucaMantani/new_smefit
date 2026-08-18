"""
smefit.latex_table.py

A report table that also offers itself as LaTeX.

``@latex_table`` is ``reportengine.table.table`` plus two things: the table is
written as a ``.tex`` file beside the ``.csv``, and the report page gets a
button that copies that LaTeX to the clipboard. The HTML table itself is
unchanged — this only adds to it.

The LaTeX follows the house style: a ``table`` float around a ``tabularx`` of
the full line width, whose first column stretches and whose others are
centred, ``\\hline`` rules with a double one under the header, ``\\multirow``
for a coefficient spanning two rows, and a caption and label at the bottom. It
needs ``\\usepackage{tabularx}`` and ``\\usepackage{multirow}``, which the
emitted comment says.

Nothing here is specific to one table: any ``@table`` provider becomes
copyable by swapping its decorator. Caption and label travel from the provider
in ``DataFrame.attrs``, which reportengine hands straight to the final action.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from reportengine.table import Table, prepare_path
from reportengine.utils import add_highlight

if TYPE_CHECKING:
    import pandas as pd

log = logging.getLogger(__name__)

# A row whose index ends with this continues the row above it: the two are one
# coefficient's two solutions, and LaTeX spans the label over both with
# \multirow rather than repeating it. `smefit.tables` labels them.
CONTINUATION_SUFFIX = " (2)"

# Cells and headers are written for the HTML report, which pandoc renders
# without MathJax and so without \sqrt (see `smefit.tables._REACH_COLUMN`);
# in a real LaTeX document those compromises are unnecessary, so they are
# undone on the way out.
_LATEX_SUBSTITUTIONS = {
    r"$\Lambda/c_i^{1/2}$": r"$\Lambda/\sqrt{c_i}$",
    "—": r"\textemdash",
}

# What a document has to load for the emitted table to compile. amssymb is not
# about the table at all: some operator names carry \Box (``$c_{\varphi\Box}$``),
# which base LaTeX2e does not provide — it fails with "Command \Box not provided
# in base LaTeX2e". `tests/test_latex_table.py` compiles a table of every
# operator label there is against exactly this list.
_PREAMBLE = r"""% \usepackage{tabularx}
% \usepackage{multirow}
% \usepackage{amssymb}   % \Box, in operator names such as $c_{\varphi\Box}$"""

# Copies the LaTeX of one table. `navigator.clipboard` needs a secure context,
# which a report opened from disk over file:// is not — so the deprecated
# textarea trick is kept as the fallback that still works there.
_COPY_SCRIPT = """<script>
window.smefitCopyLatex = window.smefitCopyLatex || function (button, id) {
  var text = document.getElementById(id).textContent;
  var done = function () {
    var label = button.textContent;
    button.textContent = "Copied";
    setTimeout(function () { button.textContent = label; }, 1500);
  };
  var fallback = function () {
    var area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    try { document.execCommand("copy"); } catch (e) { /* nothing else to try */ }
    document.body.removeChild(area);
    done();
  };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(done, fallback);
  } else {
    fallback();
  }
};
</script>"""


def _cell(value: object) -> str:
    """One cell as LaTeX.

    Cells are written as LaTeX already — ``$c_{\\varphi D}$``, ``68\\% CL`` —
    so nothing is escaped here; only what the HTML rendering had to compromise
    on is put back.
    """
    text = str(value)
    for source, replacement in _LATEX_SUBSTITUTIONS.items():
        text = text.replace(source, replacement)
    return text


def _row_spans(index: list[str]) -> list[int]:
    """How many rows each row's first cell spans, 0 for a continuation row.

    A coefficient with two solutions occupies two rows and its name is written
    once, over both.
    """
    spans = []
    for position, label in enumerate(index):
        if label.endswith(CONTINUATION_SUFFIX):
            spans.append(0)
            continue
        span = 1
        while position + span < len(index) and index[position + span].endswith(
            CONTINUATION_SUFFIX
        ):
            span += 1
        spans.append(span)
    return spans


def to_latex(
    df: pd.DataFrame,
    caption: str | None = None,
    label: str | None = None,
    index_header: str = "Coefficient",
) -> str:
    """Render *df* as a LaTeX table in the house style.

    Parameters
    ----------
    df : pd.DataFrame
        The table. A two-level column index is written as a header of two
        rows, the upper one spanning each group with ``\\multicolumn``; a flat
        one is written as a single header row.
    caption : str, optional
        Caption, under the table. Omitted when not given.
    label : str, optional
        ``\\label`` to cross-reference the table by. Omitted when not given.
    index_header : str, optional
        Header of the first column, which holds the frame's index.

    Returns
    -------
    str
        The ``table`` float, ready to paste into a document.
    """
    index = [str(value) for value in df.index]
    grouped = df.columns.nlevels == 2

    lines = [
        _PREAMBLE,
        r"\begin{table}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{10pt}",
        r"\renewcommand{\arraystretch}{1.5}",
        # the first column stretches to fill the line, the numbers stay tight
        r"\begin{tabularx}{\linewidth}{|X|" + "c|" * df.shape[1] + "}",
        r"\hline",
    ]

    if grouped:
        groups: list[tuple[str, int]] = []
        for group in df.columns.get_level_values(0):
            if groups and groups[-1][0] == group:
                groups[-1] = (group, groups[-1][1] + 1)
            else:
                groups.append((group, 1))
        lines.append(
            " & ".join(
                [
                    rf"\multirow{{2}}{{*}}{{{index_header}}}",
                    *(
                        rf"\multicolumn{{{width}}}{{c|}}{{{_cell(group)}}}"
                        for group, width in groups
                    ),
                ]
            )
            + r" \\"
        )
        # a partial rule under the fit names only: it separates them from the
        # column names below without cutting through the \multirow'd corner
        # cell, which a full \hline would
        lines.append(rf"\cline{{2-{df.shape[1] + 1}}}")
        lines.append(
            " & ".join(["", *(_cell(name) for name in df.columns.get_level_values(1))])
            + r" \\"
        )
    else:
        lines.append(
            " & ".join([index_header, *(_cell(name) for name in df.columns)]) + r" \\"
        )

    # a double rule under the header, as the house style has it
    lines.extend([r"\hline", r"\hline"])

    spans = _row_spans(index)
    for position, (label_text, span) in enumerate(zip(index, spans)):
        if span == 0:  # a continuation row leaves the spanned cell empty
            first = ""
        elif span == 1:
            first = _cell(label_text)
        else:
            first = rf"\multirow{{{span}}}{{*}}{{{_cell(label_text)}}}"
        cells = [_cell(value) for value in df.iloc[position]]
        lines.append(" & ".join([first, *cells]) + r" \\")
        # rules separate coefficients, not the solutions of one
        if position + 1 == len(index) or spans[position + 1] != 0:
            lines.append(r"\hline")

    lines.append(r"\end{tabularx}")
    if caption is not None:
        lines.extend([r"\vspace{0.2cm}", rf"\caption{{{caption}}}"])
    if label is not None:
        lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


class LatexTable(Table):
    """A :class:`reportengine.table.Table` that also carries its LaTeX.

    Only :attr:`as_markdown` differs: the HTML table as before, preceded by a
    button and by the LaTeX it copies. The LaTeX rides in a
    ``<script type="application/x-latex">``, which a browser does not render
    and pandoc passes through untouched.
    """

    _metadata = [*Table._metadata, "latex", "latex_id"]

    @classmethod
    def fromdf(cls, df, *, description=None, path=None, latex="", latex_id="table"):
        res = super().fromdf(df, description=description, path=path)
        res.latex = latex
        res.latex_id = latex_id
        return res

    @property
    def as_markdown(self) -> str:
        # A <script> is a raw-text element: what it holds is not HTML, so
        # escaping it would put the escapes themselves on the clipboard. Only
        # the one sequence that would close the element early is broken up,
        # and `\/` is how JavaScript-adjacent text has always done that.
        payload = self.latex.replace("</script", r"<\/script")
        button = (
            f'<button type="button" class="copy-latex" '
            # single quotes inside: the attribute is delimited by double ones,
            # and a nested double quote ends it early — pandoc then fails to
            # read the tag at all and escapes the lot into the page as text
            f"onclick=\"smefitCopyLatex(this, '{self.latex_id}')\">"
            "Copy LaTeX</button>"
        )
        script = (
            f'<script type="application/x-latex" id="{self.latex_id}">'
            f"{payload}</script>"
        )
        # blank lines between them: each is a raw HTML block of its own, and
        # pandoc only passes a block through when it starts one
        return "\n\n".join([_COPY_SCRIPT, button, script, super().as_markdown])


def savelatextable(df, path):
    """Write the table as csv and as LaTeX, and keep the LaTeX for the report.

    The caption and label are read from ``df.attrs``, which is how a provider
    passes them: reportengine hands the object the provider returned straight
    to this, so nothing copies them away in between.
    """
    log.debug("Writing table %s", path)
    df.to_csv(str(path), sep="\t", na_rep="nan")

    latex = to_latex(
        df,
        caption=df.attrs.get("latex_caption"),
        label=df.attrs.get("latex_label"),
        index_header=df.attrs.get("latex_index_header", "Coefficient"),
    )
    tex_path = path.with_suffix(".tex")
    tex_path.write_text(latex + "\n")

    return LatexTable.fromdf(df, path=path, latex=latex, latex_id=f"latex-{path.stem}")


@add_highlight
def latex_table(f):
    """Save a table as csv and LaTeX, and offer the LaTeX in the report.

    ``reportengine.table.table`` with the LaTeX added; a provider swaps one
    decorator for the other and changes nothing else.
    """
    f.prepare = prepare_path
    f.final_action = savelatextable
    return f
