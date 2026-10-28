"""
smefit.dataset_info.py

Descriptive table about the datasets of a fit:
one row per dataset with its size, luminosity and
the publication it comes from, rendered as the LaTeX table
of an appendix.

The commondata YAML of a dataset carries its provenance (``description``,
``arxiv``, ``luminosity``) next to the numbers. This module reads the
provenance, and resolves the references to the citation keys and BibTeX entries
of INSPIRE, so that the table cites the papers in publication style.
"""

import json
import logging
import pathlib
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import yaml

log = logging.getLogger(__name__)

INSPIRE_API = "https://inspirehep.net/api/literature"
# References per INSPIRE query: keeps every request URL short.
_INSPIRE_BATCH = 50

# YYMM.NNNNN (YYMM.NNNN before 2015) and the pre-2007 archive/YYMMNNN form
_ARXIV_ID = re.compile(
    r"(?<![\d.])(\d{4}\.\d{4,5}|(?:hep-(?:ex|ph|th|lat)|nucl-(?:ex|th)|astro-ph"
    r"|gr-qc|physics)/\d{7})(?:v\d+)?(?![\d.])"
)
_INSPIRE_RECORD = re.compile(r"inspirehep\.net/(?:api/)?(?:literature|record)/(\d+)")
# A URL in a description: up to a space, or a comma or semicolon gluing it to
# the next words ("...record/ins1803608,Table 04")
_URL = r"https?://[^\s,;]+"
# What a link in a description reads as, by site; any other reads as its host.
_LINK_LABELS = {
    "hepdata.net": "HEPData",
    "arxiv.org": "arXiv",
    "inspirehep.net": "INSPIRE",
    "doi.org": "DOI",
    "indico.cern.ch": "Indico",
    "cds.cern.ch": "CDS",
}
# The characters LaTeX reserves that a plain-text description may contain. A
# backslash is not among them: in a description it starts a command.
_LATEX_SPECIAL = {
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "_": r"\_",
    "$": r"\$",
    "^": r"\^{}",
}
# The text width of the preview: a single-column paper with 2 cm margins.
_PREVIEW_TEXTWIDTH = "483pt"
# What LaTeX writes to its log when another run would change the output
_LATEX_RERUN = re.compile(r"Rerun (?:to get|LaTeX)")
# pdflatex runs past the first; longtable settles within two or three
_MAX_LATEX_RERUNS = 4
# Seconds a single pdflatex or bibtex run may take
_LATEX_TIMEOUT = 120


@dataclass(frozen=True)
class DatasetInfo:
    """Provenance of one dataset, read from its commondata file.

    Attributes
    ----------
    name : str
    num_data : int
    description : str
        Free text, LaTeX math included.
    luminosity : tuple of float
        The distinct integrated luminosities of the data points, in fb^-1;
        empty when the file gives none.
    references : tuple of str
        The publications the data come from, each as the INSPIRE search term
        that finds it: ``arxiv:<number>`` or ``recid:<record>``; or, for a
        source INSPIRE does not hold (a CERN repository record, a talk),
        ``url:<address>``, which is only ever linked.
    group : str or None
        The ``group:`` of the runcard's dataset entry.
    """

    name: str
    num_data: int
    description: str = ""
    luminosity: Tuple[float, ...] = ()
    references: Tuple[str, ...] = ()
    group: Optional[str] = None


@dataclass(frozen=True)
class Bibliography:
    """Citation keys and BibTeX for the references of a set of datasets.

    Attributes
    ----------
    texkeys : dict
        Reference (as in :attr:`DatasetInfo.references`) to its INSPIRE
        citation key.
    bibtex : str
        The BibTeX entries of those keys.

    A reference without a key is
    linked rather than cited, so the table compiles without the ``.bib``.
    """

    texkeys: dict = field(default_factory=dict)
    bibtex: str = ""

    def cite(self, references) -> str:
        """The LaTeX that cites *references*: one ``\\cite`` for those with a
        key, then a link to each of the others.

        Several pieces are stacked one per line, since a link cannot break and
        two side by side run out of their column.
        """
        keys = list(
            dict.fromkeys(self.texkeys[r] for r in references if r in self.texkeys)
        )
        pieces = [rf"\cite{{{','.join(keys)}}}"] if keys else []
        pieces += [_link(r) for r in references if r not in self.texkeys]
        if len(pieces) < 2:
            return "".join(pieces)
        stacked = r" \\ ".join(pieces)
        return rf"\begin{{tabular}}[t]{{@{{}}l@{{}}}}{stacked}\end{{tabular}}"


def _link(reference: str) -> str:
    """A ``\\href`` to *reference*, labelled with its identifier."""
    kind, _, identifier = reference.partition(":")
    if kind == "url":
        return _latex_link(identifier)
    if kind == "arxiv":
        return rf"\href{{https://arxiv.org/abs/{identifier}}}{{arXiv:{identifier}}}"
    return rf"\href{{https://inspirehep.net/literature/{identifier}}}{{INSPIRE:{identifier}}}"


def _is_blank(raw: Optional[str]) -> bool:
    """Whether a commondata field is missing, or written but left empty."""
    return raw is None or raw.strip().lower() in ("", "none", "null")


def _parse_references(raw: Optional[str], dataset: str) -> Tuple[str, ...]:
    """The references of the ``arxiv`` field of a commondata file.

    The field is free text: an arXiv number, several separated by commas, or
    a URL in their place. An INSPIRE record URL is looked up like an arXiv
    number; any other URL (a CERN repository record, a talk) is kept as a
    link. Anything else is left out with a warning, and so is an arXiv
    number that has lost its last digit.
    """
    if _is_blank(raw):
        return ()
    references = []
    for token in re.split(r"[,;\s]+", raw.strip()):
        if not token:
            continue
        record = _INSPIRE_RECORD.search(token)
        # in any other URL, an arXiv-like number is part of the path
        arxiv = (
            _ARXIV_ID.search(token)
            if "://" not in token or "arxiv.org/" in token
            else None
        )
        if record:
            references.append(f"recid:{record.group(1)}")
        elif re.fullmatch(_URL, token) and not arxiv:
            references.append(f"url:{token}")
        elif arxiv and re.fullmatch(r"(1[5-9]|[2-9]\d)\d{2}\.\d{4}", arxiv.group(1)):
            # numbers have five digits after the dot since 2015: this one was
            # read as a float and its trailing zero dropped
            log.warning(
                "%s: arXiv number %r is truncated; it is left out of the "
                "dataset table.",
                dataset,
                token,
            )
        elif arxiv:
            references.append(f"arxiv:{arxiv.group(1)}")
        else:
            log.warning(
                "%s: no arXiv number or INSPIRE record in its reference %r; "
                "it is left out of the dataset table.",
                dataset,
                token,
            )
    return tuple(dict.fromkeys(references))


def _parse_luminosity(raw, dataset: str) -> Tuple[float, ...]:
    """The distinct luminosities of a commondata ``luminosity`` field."""
    if not isinstance(raw, list) and _is_blank(raw):
        return ()
    values = raw if isinstance(raw, list) else [raw]
    try:
        return tuple(dict.fromkeys(float(value) for value in values))
    except ValueError:
        log.warning("%s: unreadable luminosity %r.", dataset, raw)
        return ()


def _load_dataset_info(data_path, dataset_entry) -> DatasetInfo:
    """Read the provenance of one runcard ``datasets`` entry."""
    name = dataset_entry["name"]
    path = pathlib.Path(data_path) / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Dataset {name} not found in {data_path}")
    # BaseLoader keeps every scalar as written: safe_load reads the arXiv
    # number 1609.03920 as the float 1609.0392.
    with open(path, encoding="utf-8") as file:
        commondata = yaml.load(file, Loader=yaml.BaseLoader)
    description = " ".join((commondata.get("description") or "").split())
    if _is_blank(commondata.get("arxiv")):
        references = _references_in_description(description, name)
    else:
        references = _parse_references(commondata["arxiv"], name)
    return DatasetInfo(
        name=name,
        num_data=int(commondata["num_data"]),
        description=description,
        luminosity=_parse_luminosity(commondata.get("luminosity"), name),
        references=references,
        group=dataset_entry.get("group"),
    )


def _references_in_description(description: str, dataset: str) -> Tuple[str, ...]:
    """The arXiv numbers a description mentions, for a file with no ``arxiv``.

    A guess, so each use is warned about: "projection from 1907.04311,
    rescaled by luminosity from 2503.19983" cites both papers.
    """
    numbers = [match.group(1) for match in _ARXIV_ID.finditer(description)]
    references = _parse_references(" ".join(numbers), dataset) if numbers else ()
    if references:
        log.warning(
            "%s: no arxiv field; the dataset table cites the arXiv numbers its "
            "description mentions: %s.",
            dataset,
            ", ".join(r.removeprefix("arxiv:") for r in references),
        )
    return references


def _inspire_query(terms, **params) -> bytes:
    """One INSPIRE literature search for any of *terms*."""
    query = urllib.parse.urlencode(
        {"q": " or ".join(terms), "size": len(terms), **params}
    )
    with urllib.request.urlopen(f"{INSPIRE_API}?{query}", timeout=30) as response:
        return response.read()


def _with_urls(bibtex: str, urls) -> str:
    """*bibtex* with a ``url`` field added to the entries keyed in *urls*."""

    def add(match):
        url = urls.get(match.group(2))
        return f'{match.group(0)}\n    url = "{url}",' if url else match.group(0)

    return re.sub(r"(?m)^(@\w+\{)([^,\s]+),", add, bibtex)


def _fetch_inspire_bibliography(references) -> Bibliography:
    """Look *references* up on INSPIRE, in batches of ``_INSPIRE_BATCH``.

    Every entry links to the paper: through its ``doi`` to the journal, or,
    for a paper with none, through a ``url`` added to its INSPIRE record. A
    ``url`` on the others too would be printed next to the DOI by the styles
    that show both.
    """
    references = list(dict.fromkeys(references))
    found = {}
    urls = {}
    entries = []
    for start in range(0, len(references), _INSPIRE_BATCH):
        batch = references[start : start + _INSPIRE_BATCH]
        hits = json.loads(
            _inspire_query(batch, fields="texkeys,arxiv_eprints,control_number,dois")
        )["hits"]["hits"]
        for hit in hits:
            record = hit["metadata"]
            if not record.get("texkeys"):
                continue
            # the first key is the current one; the others are kept by INSPIRE
            # only so that old citations still resolve
            texkey = record["texkeys"][0]
            found[f"recid:{record['control_number']}"] = texkey
            for eprint in record.get("arxiv_eprints", []):
                found[f"arxiv:{eprint['value']}"] = texkey
            if not record.get("dois"):
                urls[texkey] = (
                    f"https://inspirehep.net/literature/{record['control_number']}"
                )
        bibtex = _inspire_query(batch, format="bibtex").decode("utf-8").strip()
        entries.append(_with_urls(bibtex, urls))

    missing = [r for r in references if r not in found]
    if missing:
        log.warning(
            "INSPIRE has no record for %s: the dataset table links to them "
            "instead of citing them.",
            ", ".join(missing),
        )
    return Bibliography(
        texkeys={r: found[r] for r in references if r in found},
        bibtex="\n\n".join(entries) + "\n",
    )


def _latex_plain(text: str) -> str:
    """Plain *text* as LaTeX, its reserved characters escaped unless already
    escaped. The Unicode the database writes that pdflatex has no glyph for is
    spelled out: the minus sign, and the combining macron of ``tt̄``."""
    text = re.sub(r"(?<!\\)([&%#_$^])", lambda m: _LATEX_SPECIAL[m.group(1)], text)
    text = re.sub("(.)\u0304", r"\\={\1}", text)
    return text.replace("\u2212", "$-$")


def _latex_link(url: str) -> str:
    """A URL of a description as a ``\\href`` labelled with its site.

    Written out, a URL has no point a table column can break it at, and runs
    out of the column. Sentence punctuation the URL swallowed is put back
    after the link.
    """
    stripped = url.rstrip(".)")
    trailing = url[len(stripped) :]
    host = (urllib.parse.urlparse(stripped).hostname or "").removeprefix("www.")
    label = next(
        (
            label
            for site, label in _LINK_LABELS.items()
            if host == site or host.endswith(f".{site}")
        ),
        host,
    )
    # \href takes the URL verbatim but for these, which break a table cell
    target = re.sub(r"(?<!\\)([&%#])", r"\\\1", stripped)
    return rf"\href{{{target}}}{{{label}}}{trailing}"


def _latex_text(text: str) -> str:
    """*text* as LaTeX: inline math is kept as written, a URL becomes a short
    link (:func:`_latex_link`), and the rest goes through
    :func:`_latex_plain`."""
    pieces = re.split(rf"(\$[^$]+\$|{_URL})", text)
    return "".join(
        (
            piece
            if piece.startswith("$") and piece.endswith("$") and len(piece) > 1
            else (
                _latex_link(piece) if re.fullmatch(_URL, piece) else _latex_plain(piece)
            )
        )
        for piece in pieces
    )


def _by_group(infos) -> List[Tuple[Optional[str], List[DatasetInfo]]]:
    """*infos* split by group, groups in order of first appearance."""
    groups: dict = {}
    for info in infos:
        groups.setdefault(info.group, []).append(info)
    return list(groups.items())


def dataset_info(datasets, data_path) -> List[DatasetInfo]:
    """The provenance of each dataset of the runcard, in runcard order.
    Reads only the commondata files, a runcard with ``datasets`` and
    ``data_path`` is enough.
    """
    return [_load_dataset_info(data_path, entry) for entry in datasets]


def dataset_bibliography(dataset_info, fetch_bibtex=False) -> Bibliography:
    """Resolve the references of the datasets on INSPIRE, when asked to.

    Parameters
    ----------
    dataset_info : list of DatasetInfo
    fetch_bibtex : bool, optional
        Look the references up on INSPIRE, so the table cites them with
        INSPIRE keys and ships their BibTeX. Off by default: a run needs no
        network, and the table links to each reference instead.

    Returns
    -------
    Bibliography
        Empty when the lookup is off or INSPIRE cannot be reached.
    """
    references = [
        r for info in dataset_info for r in info.references if not r.startswith("url:")
    ]
    if not fetch_bibtex or not references:
        return Bibliography()
    try:
        return _fetch_inspire_bibliography(references)
    except (OSError, ValueError, KeyError) as e:
        log.warning(
            "Could not fetch the bibliography from INSPIRE (%s): the dataset "
            "table links to its references instead of citing them.",
            e,
        )
        return Bibliography()


def dataset_table_tex(dataset_info, dataset_bibliography) -> str:
    """The datasets as a LaTeX ``xltabular``, one section per ``group``.

    An ``xltabular`` is a ``longtable`` as wide as the line: the name and
    description columns share what the numbers and references leave, so the
    table fits the text width of whatever paper it is put in. A name breaks
    after its underscores.

    The table needs the ``array``, ``booktabs``, ``xltabular`` and
    ``hyperref`` packages, and its citations the BibTeX of
    ``dataset_bibliography``. Datasets without a group come in their own
    section with no heading.
    """
    lines = [
        "% Written by smefit (write_dataset_table).",
        r"% Needs \usepackage{array,booktabs,xltabular,hyperref}.",
        r"\begin{xltabular}{\linewidth}{"
        r">{\raggedright\arraybackslash\hsize=0.8\hsize}X "
        r">{\raggedright\arraybackslash\hsize=1.2\hsize}X r r l}",
        r"\caption{Datasets included in the fit.}",
        r"\label{tab:datasets} \\",
        r"\toprule",
        r"Dataset & Description & $n_{\rm dat}$ & $\mathcal{L}$ [fb$^{-1}$] "
        r"& Ref. \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Dataset & Description & $n_{\rm dat}$ & $\mathcal{L}$ [fb$^{-1}$] "
        r"& Ref. \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for i, (group, infos) in enumerate(_by_group(dataset_info)):
        if i > 0:
            lines.append(r"\midrule")
        if group is not None:
            lines.append(
                rf"\multicolumn{{5}}{{l}}{{\textbf{{{_latex_text(group)}}}}} \\"
            )
        for info in infos:
            luminosity = ", ".join(f"{value:g}" for value in info.luminosity) or "--"
            name = _latex_text(info.name).replace(r"\_", r"\_\allowbreak{}")
            lines.append(
                rf"\texttt{{{name}}} & "
                rf"{_latex_text(info.description)} & {info.num_data} & "
                rf"{luminosity} & {dataset_bibliography.cite(info.references)} \\"
            )
    total = sum(info.num_data for info in dataset_info)
    lines += [
        r"\midrule",
        rf"\multicolumn{{2}}{{l}}{{Total}} & {total} & & \\",
        r"\end{xltabular}",
    ]
    return "\n".join(lines) + "\n"


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
