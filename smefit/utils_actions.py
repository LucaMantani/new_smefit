"""
smefit.utils_actions.py

Reportengine utility actions for smefit.
"""

import logging
import pathlib
import re
import shutil
import subprocess
import tempfile

import numpy as np
import yaml

log = logging.getLogger(__name__)

# The text width of the dataset table preview: a single-column paper with 2 cm
# margins.
_PREVIEW_TEXTWIDTH = "483pt"
# What LaTeX writes to its log when another run would change the output
_LATEX_RERUN = re.compile(r"Rerun (?:to get|LaTeX)")
# pdflatex runs past the first; longtable settles within two or three
_MAX_LATEX_RERUNS = 4
# Seconds a single pdflatex or bibtex run may take
_LATEX_TIMEOUT = 120


def write_pseudodata(pseudodata, theory_path, output_path):
    """Write pseudodata DataGroup to YAML files under output_path/pseudodata/.

    For datasets with a '_proj' suffix, the corresponding theory JSON is also
    copied into output_path/pseudodata_theory/ so the projected datasets can be
    used directly in a subsequent fit runcard.
    """
    out_dir = pathlib.Path(output_path) / "pseudodata"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in pseudodata.datasets:
        syst = np.asarray(ds.syst_err)  # (n_sys, n_data)
        lumi = np.asarray(ds.luminosity)

        data_dict = {
            "dataset_name": ds.name,
            "num_data": ds.num_data,
            "data_central": np.asarray(ds.central_values).tolist(),
            "statistical_error": np.asarray(ds.stat_err).tolist(),
            "systematics": (
                syst.tolist() if ds.num_data > 1 else syst.flatten().tolist()
            ),
            "sys_names": list(ds.sys_names),
            "sys_type": list(ds.sys_types),
        }

        if not np.isnan(lumi).all():
            data_dict["luminosity"] = lumi.tolist() if len(lumi) > 1 else float(lumi[0])

        with open(out_dir / f"{ds.name}.yaml", "w") as f:
            yaml.dump(data_dict, f, sort_keys=False)

        if ds.name.endswith("_proj"):
            original_name = ds.name.removesuffix("_proj")
            src = pathlib.Path(theory_path) / f"{original_name}.json"
            if src.exists():
                th_dir = pathlib.Path(output_path) / "pseudodata_theory"
                th_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, th_dir / f"{ds.name}.json")


def run_pca(pca, output_path):
    """Print the principal-component analysis and save it."""
    pca.print_summary()
    pca.write(output_path)


def write_dataset_table(
    dataset_table_tex, dataset_bibliography, output_path, compile_pdf=False
):
    """Write the LaTeX table of the runcard's datasets, for a paper's appendix.

    ``dataset_table.tex`` holds the table. With ``fetch_bibtex: true`` the
    table cites its references with INSPIRE keys, and ``dataset_table.bib``
    holds their BibTeX. Otherwise — the default, or INSPIRE unreachable — no
    ``.bib`` is written, and the table links to its references and compiles on
    its own.

    With ``compile_pdf: true``, ``dataset_table_preview.tex`` typesets the table
    and its bibliography as a document of their own, compiled into
    ``dataset_table_preview.pdf`` when pdflatex is installed. No auxiliary file
    is left behind, and a failed compilation only warns.
    """
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "dataset_table.tex").write_text(dataset_table_tex, encoding="utf-8")
    with_bibliography = bool(dataset_bibliography.bibtex)
    if with_bibliography:
        (output_path / "dataset_table.bib").write_text(
            dataset_bibliography.bibtex, encoding="utf-8"
        )
    if compile_pdf:
        (output_path / "dataset_table_preview.tex").write_text(
            _preview_tex(with_bibliography), encoding="utf-8"
        )
        _compile_preview(output_path, with_bibliography)


def _preview_tex(with_bibliography: bool) -> str:
    """A document that typesets ``dataset_table.tex`` on its own, at the text
    width of a paper, with the bibliography of ``dataset_table.bib`` when
    there is one."""
    lines = [
        "% Written by smefit (write_dataset_table, compile_pdf: true).",
        r"\documentclass{article}",
        rf"\usepackage[textwidth={_PREVIEW_TEXTWIDTH}]{{geometry}}",
        r"\usepackage{amsmath,array,booktabs,xltabular}",
    ]
    if with_bibliography:
        lines.append(r"\usepackage[numbers]{natbib}")
    lines.append(r"\usepackage{hyperref}")
    if with_bibliography:
        # unsrtnat prints a DOI as plain text unless \doi is already defined
        lines.append(r"\newcommand{\doi}[1]{\href{https://doi.org/#1}{doi:#1}}")
    lines += [r"\begin{document}", r"\input{dataset_table}"]
    if with_bibliography:
        lines += [r"\bibliographystyle{unsrtnat}", r"\bibliography{dataset_table}"]
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


def _compile_preview(output_path, with_bibliography: bool):
    """Compile ``dataset_table_preview.tex`` of *output_path* into its PDF.

    The tools run in a scratch folder, so only the PDF comes back: pdflatex,
    bibtex when there is a bibliography, then pdflatex again until LaTeX stops
    asking for a rerun. Stopping earlier is not enough for the table: longtable
    settles its column widths over runs, and a table one run short overfills
    its line.

    A missing tool or a LaTeX error is warned about since
    the table itself is already written; on an error the ``.log`` is copied
    next to it.
    """
    output_path = pathlib.Path(output_path)
    tools = ["pdflatex"] + (["bibtex"] if with_bibliography else [])
    missing = [tool for tool in tools if shutil.which(tool) is None]
    if missing:
        log.warning(
            "%s not found: dataset_table_preview.pdf is not compiled.",
            " and ".join(missing),
        )
        return
    sources = ["dataset_table.tex", "dataset_table_preview.tex"]
    if with_bibliography:
        sources.append("dataset_table.bib")
    pdflatex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error"]
    pdflatex.append("dataset_table_preview.tex")

    with tempfile.TemporaryDirectory() as scratch:
        scratch = pathlib.Path(scratch)
        for name in sources:
            shutil.copy(output_path / name, scratch)
        latex_log = scratch / "dataset_table_preview.log"
        try:
            _run_tool(pdflatex, scratch)
            if with_bibliography:
                _run_tool(["bibtex", "dataset_table_preview"], scratch, ok_codes=(0, 1))
            # after bibtex a rerun is due whatever the log says
            due = with_bibliography
            reruns = 0
            while due or _LATEX_RERUN.search(
                latex_log.read_text(encoding="utf-8", errors="replace")
            ):
                if reruns == _MAX_LATEX_RERUNS:
                    log.warning(
                        "LaTeX still asks for a rerun after %d runs: "
                        "dataset_table_preview.pdf may not be final.",
                        reruns + 1,
                    )
                    break
                _run_tool(pdflatex, scratch)
                reruns += 1
                due = False
        except _CompileError as e:
            if latex_log.exists():
                shutil.copy(latex_log, output_path)
            log.warning(
                "%s: dataset_table_preview.pdf is not compiled (see "
                "dataset_table_preview.log).",
                e,
            )
            return
        shutil.copy(scratch / "dataset_table_preview.pdf", output_path)
