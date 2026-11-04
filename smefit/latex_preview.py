"""
smefit.latex_preview.py

Typesets a LaTeX file written by smefit as a document of its own, and compiles
it into a PDF: a preview of how it will look in a paper.

A helper module, not a reportengine provider: actions import its functions.
The file names are its arguments.
"""

import logging
import pathlib
import re
import shutil
import subprocess
import tempfile
from typing import Optional

log = logging.getLogger(__name__)

# The text width of a preview: a single-column paper with 2 cm margins.
_PREVIEW_TEXTWIDTH = "483pt"
# What LaTeX writes to its log when another run would change the output
_LATEX_RERUN = re.compile(r"Rerun (?:to get|LaTeX)")
# pdflatex runs past the first; longtable settles within two or three
_MAX_LATEX_RERUNS = 4
# Seconds a single pdflatex or bibtex run may take
_LATEX_TIMEOUT = 120


def preview_tex(source: str, bibliography: Optional[str] = None) -> str:
    """A document that typesets ``<source>.tex`` on its own, at the text width
    of a paper, with the bibliography of ``<bibliography>.bib`` when given.

    Both are file names without their extension, as ``\\input`` and
    ``\\bibliography`` take them.
    """
    lines = [
        f"% Written by smefit: a summary table of the datasets in TeX format.",
        r"\documentclass{article}",
        rf"\usepackage[textwidth={_PREVIEW_TEXTWIDTH}]{{geometry}}",
        r"\usepackage{amsmath,array,booktabs,xltabular}",
    ]
    if bibliography is not None:
        lines.append(r"\usepackage[numbers]{natbib}")
    lines.append(r"\usepackage{hyperref}")
    if bibliography is not None:
        # unsrtnat prints a DOI as plain text unless \doi is already defined
        lines.append(r"\newcommand{\doi}[1]{\href{https://doi.org/#1}{doi:#1}}")
    lines += [r"\begin{document}", rf"\input{{{source}}}"]
    if bibliography is not None:
        lines += [r"\bibliographystyle{unsrtnat}", rf"\bibliography{{{bibliography}}}"]
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


class _CompileError(Exception):
    """A LaTeX tool that failed, with what it said about it."""


def _run_tool(command, cwd, ok_codes=(0,)):
    """Run one LaTeX tool in *cwd*, raising :class:`_CompileError` on failure.

    bibtex exits with 1 on a mere warning (an INSPIRE entry with no author,
    say), so it passes ``ok_codes=(0, 1)``.
    """
    try:
        result = subprocess.run(
            command, cwd=cwd, capture_output=True, timeout=_LATEX_TIMEOUT, check=False
        )
    except subprocess.TimeoutExpired as e:
        raise _CompileError(f"{command[0]} timed out") from e
    if result.returncode in ok_codes:
        return
    output = result.stdout.decode("utf-8", errors="replace").splitlines()
    # pdflatex marks its errors with "!"; bibtex's last line counts them
    errors = [line for line in output if line.startswith("!")] or output[-1:]
    raise _CompileError(f"{command[0]} failed: {' '.join(errors[:2])}")


def compile_preview(
    folder, preview: str, source: str, bibliography: Optional[str] = None
):
    """Compile ``<preview>.tex`` of *folder* into ``<preview>.pdf`` next to it.

    ``<preview>.tex`` is a document from :func:`preview_tex`, reading
    ``<source>.tex`` and, when given, ``<bibliography>.bib`` of the same
    folder. The tools run in a scratch folder, so only the PDF comes back:
    pdflatex, bibtex when there is a bibliography, then pdflatex again until
    LaTeX stops asking for a rerun. Stopping earlier is not enough for a table:
    longtable settles its column widths over runs, and a table one run short
    overfills its line.

    A missing tool or a LaTeX error is warned about since the file being
    previewed is already written; on an error ``<preview>.log`` is copied
    next to it.
    """
    folder = pathlib.Path(folder)
    tools = ["pdflatex"] + (["bibtex"] if bibliography is not None else [])
    missing = [tool for tool in tools if shutil.which(tool) is None]
    if missing:
        log.warning(
            "%s not found: %s.pdf is not compiled.", " and ".join(missing), preview
        )
        return
    sources = [f"{source}.tex", f"{preview}.tex"]
    if bibliography is not None:
        sources.append(f"{bibliography}.bib")
    pdflatex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error"]
    pdflatex.append(f"{preview}.tex")

    with tempfile.TemporaryDirectory() as scratch:
        scratch = pathlib.Path(scratch)
        for name in sources:
            shutil.copy(folder / name, scratch)
        latex_log = scratch / f"{preview}.log"
        try:
            _run_tool(pdflatex, scratch)
            if bibliography is not None:
                _run_tool(["bibtex", preview], scratch, ok_codes=(0, 1))
            # after bibtex a rerun is due whatever the log says
            due = bibliography is not None
            reruns = 0
            while due or _LATEX_RERUN.search(
                latex_log.read_text(encoding="utf-8", errors="replace")
            ):
                if reruns == _MAX_LATEX_RERUNS:
                    log.warning(
                        "LaTeX still asks for a rerun after %d runs: "
                        "%s.pdf may not be final.",
                        reruns + 1,
                        preview,
                    )
                    break
                _run_tool(pdflatex, scratch)
                reruns += 1
                due = False
        except _CompileError as e:
            if latex_log.exists():
                shutil.copy(latex_log, folder)
            log.warning("%s: %s.pdf is not compiled (see %s.log).", e, preview, preview)
            return
        shutil.copy(scratch / f"{preview}.pdf", folder)
