"""
smefit.dataset_info.py

The datasets of a fit as a paper describes them: one row per dataset with its
size, luminosity and the publication it comes from, rendered as the LaTeX table
of an appendix.

The commondata YAML of a dataset carries its provenance (``description``,
``arxiv``, ``luminosity``) next to the numbers, but
:func:`smefit.loader.load_dataset` reads only the numbers. This module reads the
provenance, and resolves the references to the citation keys and BibTeX entries
of INSPIRE, so that the table cites the papers the way the rest of a paper does.
"""

import json
import logging
import pathlib
import re
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


@dataclass(frozen=True)
class DatasetInfo:
    """Provenance of one dataset, read from its commondata file.

    Attributes
    ----------
    name : str
    num_data : int
    description : str
        Free text as the commondata file has it, LaTeX math included.
    luminosity : tuple of float
        The distinct integrated luminosities of the data points, in fb^-1;
        empty when the file gives none.
    references : tuple of str
        The publications the data come from, each as the INSPIRE search term
        that finds it: ``arxiv:<number>`` or ``recid:<record>``.
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

    A reference without a key — every one, when the lookup was skipped — is
    linked rather than cited, so the table compiles without the ``.bib``.
    """

    texkeys: dict = field(default_factory=dict)
    bibtex: str = ""

    def cite(self, references) -> str:
        """The LaTeX that cites *references*: one ``\\cite`` for those with a
        key, then a link to each of the others."""
        keys = list(
            dict.fromkeys(self.texkeys[r] for r in references if r in self.texkeys)
        )
        pieces = [rf"\cite{{{','.join(keys)}}}"] if keys else []
        pieces += [_link(r) for r in references if r not in self.texkeys]
        return ", ".join(pieces)


def _link(reference: str) -> str:
    """A ``\\href`` to *reference*, labelled with its identifier."""
    kind, _, identifier = reference.partition(":")
    if kind == "arxiv":
        return rf"\href{{https://arxiv.org/abs/{identifier}}}{{arXiv:{identifier}}}"
    return rf"\href{{https://inspirehep.net/literature/{identifier}}}{{INSPIRE:{identifier}}}"


def _parse_references(raw: Optional[str], dataset: str) -> Tuple[str, ...]:
    """The INSPIRE search terms for the ``arxiv`` field of a commondata file.

    The field is free text: an arXiv number, several separated by commas, or
    an INSPIRE record URL in their place. Anything else — a link to a file on
    INSPIRE names no record, nor does a link to another site — is left out
    with a warning, and so is an arXiv number that has lost its last digit.
    """
    if raw is None or raw.strip().lower() in ("", "none", "null"):
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
    if raw is None:
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
    return DatasetInfo(
        name=name,
        num_data=int(commondata["num_data"]),
        description=" ".join((commondata.get("description") or "").split()),
        luminosity=_parse_luminosity(commondata.get("luminosity"), name),
        references=_parse_references(commondata.get("arxiv"), name),
        group=dataset_entry.get("group"),
    )


def _inspire_query(terms, **params) -> bytes:
    """One INSPIRE literature search for any of *terms*."""
    query = urllib.parse.urlencode(
        {"q": " or ".join(terms), "size": len(terms), **params}
    )
    with urllib.request.urlopen(f"{INSPIRE_API}?{query}", timeout=30) as response:
        return response.read()


def _fetch_inspire_bibliography(references) -> Bibliography:
    """Look *references* up on INSPIRE, in batches of ``_INSPIRE_BATCH``."""
    references = list(dict.fromkeys(references))
    found = {}
    entries = []
    for start in range(0, len(references), _INSPIRE_BATCH):
        batch = references[start : start + _INSPIRE_BATCH]
        hits = json.loads(
            _inspire_query(batch, fields="texkeys,arxiv_eprints,control_number")
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
        entries.append(_inspire_query(batch, format="bibtex").decode("utf-8").strip())

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

    Reads only the commondata files, so it needs neither theory predictions
    nor coefficients: a runcard with ``datasets`` and ``data_path`` is enough.
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
        Empty when the lookup is off or INSPIRE cannot be reached: a failed
        lookup degrades the table to links, with a warning, rather than
        failing the run.
    """
    references = [r for info in dataset_info for r in info.references]
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
    """The datasets as a LaTeX ``longtable``, one section per ``group``.

    The table needs the ``array``, ``booktabs``, ``longtable`` and
    ``hyperref`` packages, and its citations the BibTeX of
    ``dataset_bibliography``. Datasets without a group come in their own
    section with no heading.
    """
    lines = [
        "% Written by smefit (write_dataset_table).",
        r"% Needs \usepackage{array,booktabs,longtable,hyperref}.",
        r"\begin{longtable}{l >{\raggedright\arraybackslash}p{0.33\textwidth} r r l}",
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
            lines.append(
                rf"\texttt{{{_latex_text(info.name)}}} & "
                rf"{_latex_text(info.description)} & {info.num_data} & "
                rf"{luminosity} & {dataset_bibliography.cite(info.references)} \\"
            )
    total = sum(info.num_data for info in dataset_info)
    lines += [
        r"\midrule",
        rf"\multicolumn{{2}}{{l}}{{Total}} & {total} & & \\",
        r"\end{longtable}",
    ]
    return "\n".join(lines) + "\n"
