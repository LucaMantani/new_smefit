"""Unit tests for smefit.dataset_info — the provenance of the datasets and
their LaTeX table."""

import inspect
import json
import logging
import urllib.error

import pytest

from smefit import dataset_info as di
from smefit.dataset_info import (
    Bibliography,
    DatasetInfo,
    dataset_bibliography,
    dataset_info,
    dataset_table_tex,
)


def _write_commondata(directory, name, metadata):
    """A commondata file whose metadata lines are written verbatim, unquoted —
    the way the database writes them, which is what the loader must survive."""
    lines = [
        f"dataset_name: {name}",
        "num_data: 2",
        "data_central: [1.0, 2.0]",
        "statistical_error: [0.1, 0.2]",
        "systematics: [[0.05, 0.10]]",
        "sys_names: [UNCORR]",
        "sys_type: [ADD]",
        *metadata,
    ]
    (directory / f"{name}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Reading the provenance
# ---------------------------------------------------------------------------


def test_arxiv_number_keeps_its_trailing_zero(tmp_path):
    """safe_load would read 1609.03920 as the float 1609.0392 — another paper."""
    _write_commondata(tmp_path, "DS", ["arxiv: 1609.03920"])

    (info,) = dataset_info([{"name": "DS"}], tmp_path)

    assert info.references == ("arxiv:1609.03920",)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1606.02266, 1507.04548", ("arxiv:1606.02266", "arxiv:1507.04548")),
        ("arXiv:1511.04716", ("arxiv:1511.04716",)),
        ("https://arxiv.org/abs/1511.04716v2", ("arxiv:1511.04716",)),
        ("hep-ex/0509008", ("arxiv:hep-ex/0509008",)),
        ("https://inspirehep.net/literature/1323200", ("recid:1323200",)),
        ("1511.04716, 1511.04716", ("arxiv:1511.04716",)),
        ("1403.7366", ("arxiv:1403.7366",)),
        # the number in a link to another site is not an arXiv number: the
        # link is kept as it is
        (
            "2205.08553v2, https://indico.cern.ch/event/1439855/contributions/6461561",
            (
                "arxiv:2205.08553",
                "url:https://indico.cern.ch/event/1439855/contributions/6461561",
            ),
        ),
        (
            "https://repository.cern/records/3rj9z-94994",
            ("url:https://repository.cern/records/3rj9z-94994",),
        ),
        # a link to a file on INSPIRE names no record to look up
        (
            "https://inspirehep.net/files/82aa29e742f4324970d810e88817b0ac",
            ("url:https://inspirehep.net/files/82aa29e742f4324970d810e88817b0ac",),
        ),
        ("1812.059", ()),
        ("0509008", ()),
        ("none", ()),
        ("", ()),
        (None, ()),
    ],
)
def test_parse_references(raw, expected):
    assert di._parse_references(raw, "DS") == expected


def test_unreadable_reference_is_left_out_with_a_warning(caplog):
    """Neither an arXiv number nor a link: 1812.059 has lost two digits."""
    with caplog.at_level(logging.WARNING, logger="smefit.dataset_info"):
        assert di._parse_references("1812.059", "DS") == ()

    assert "DS" in caplog.text


def test_truncated_arxiv_number_is_left_out_with_a_warning(caplog):
    """Since 2015 numbers have five digits after the dot: 1609.0392 is
    1609.03920 written through a float, and links to no paper."""
    with caplog.at_level(logging.WARNING, logger="smefit.dataset_info"):
        assert di._parse_references("1609.0392", "DS") == ()

    assert "truncated" in caplog.text


def test_dataset_info_reads_metadata_and_runcard_group(tmp_path):
    _write_commondata(
        tmp_path,
        "DS",
        [
            "description: Charge asymmetry\n  differential in $m_{t\\bar t}$",
            "luminosity: 139 # units: fb^-1",
            "arxiv: 2208.12095",
        ],
    )

    (info,) = dataset_info(
        [{"name": "DS", "order": "NLO_QCD", "group": "Top"}], tmp_path
    )

    assert info == DatasetInfo(
        name="DS",
        num_data=2,
        description=r"Charge asymmetry differential in $m_{t\bar t}$",
        luminosity=(139.0,),
        references=("arxiv:2208.12095",),
        group="Top",
    )


def test_per_point_luminosity_keeps_its_distinct_values(tmp_path):
    _write_commondata(tmp_path, "DS", ["luminosity: [36.1, 36.1, 139]"])

    (info,) = dataset_info([{"name": "DS"}], tmp_path)

    assert info.luminosity == (36.1, 139.0)


def test_empty_luminosity_is_missing_without_a_warning(tmp_path, caplog):
    """LEP1_EWPOs_2006 writes `luminosity:` with no value."""
    _write_commondata(tmp_path, "DS", ["luminosity:"])

    with caplog.at_level(logging.WARNING, logger="smefit.dataset_info"):
        (info,) = dataset_info([{"name": "DS"}], tmp_path)

    assert info.luminosity == ()
    assert caplog.text == ""


def test_description_arxiv_numbers_are_cited_when_the_file_has_none(tmp_path, caplog):
    """The LCF projections name their sources only in the description. A
    guess, so it is warned about."""
    _write_commondata(
        tmp_path,
        "DS",
        [
            "description: Projection from 1907.04311 Table 6, rescaled by "
            "luminosity from 2503.19983 Table 3.",
        ],
    )

    with caplog.at_level(logging.WARNING, logger="smefit.dataset_info"):
        (info,) = dataset_info([{"name": "DS"}], tmp_path)

    assert info.references == ("arxiv:1907.04311", "arxiv:2503.19983")
    assert "DS: no arxiv field" in caplog.text


def test_description_is_not_read_for_references_when_the_file_has_some(tmp_path):
    _write_commondata(
        tmp_path,
        "DS",
        ["arxiv: 1511.04716", "description: Compared with 1907.04311."],
    )

    (info,) = dataset_info([{"name": "DS"}], tmp_path)

    assert info.references == ("arxiv:1511.04716",)


def test_missing_metadata_falls_back_to_empty(tmp_path):
    """The LEP datasets have no arxiv, luminosity or description at all."""
    _write_commondata(tmp_path, "DS", [])

    (info,) = dataset_info([{"name": "DS"}], tmp_path)

    assert info == DatasetInfo(name="DS", num_data=2)


def test_runcard_order_is_kept(tmp_path):
    for name in ("B", "A"):
        _write_commondata(tmp_path, name, [])

    infos = dataset_info([{"name": "B"}, {"name": "A"}], tmp_path)

    assert [info.name for info in infos] == ["B", "A"]


def test_missing_dataset_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="NOPE"):
        dataset_info([{"name": "NOPE"}], tmp_path)


def test_fixture_dataset_has_no_provenance(fixtures_data_path):
    """The fit fixture carries no provenance, and still loads."""
    (info,) = dataset_info([{"name": "TESTDATA"}], fixtures_data_path)

    assert info.num_data == 3
    assert info.luminosity == (1.5,)
    assert info.references == ()


# ---------------------------------------------------------------------------
# Citing
# ---------------------------------------------------------------------------


def test_cite_uses_keys_and_links_the_rest():
    bibliography = Bibliography(texkeys={"arxiv:1": "A:2020a", "recid:2": "B:2021b"})

    cited = bibliography.cite(("arxiv:1", "recid:2", "arxiv:3"))

    assert cited == (
        r"\begin{tabular}[t]{@{}l@{}}\cite{A:2020a,B:2021b} \\ "
        r"\href{https://arxiv.org/abs/3}{arXiv:3}\end{tabular}"
    )


def test_cite_stacks_links_one_per_line():
    """Two links side by side cannot break, and run out of the column
    (ATLAS_CMS_SSinc_RunI)."""
    assert Bibliography().cite(("arxiv:1", "arxiv:2")) == (
        r"\begin{tabular}[t]{@{}l@{}}\href{https://arxiv.org/abs/1}{arXiv:1} \\ "
        r"\href{https://arxiv.org/abs/2}{arXiv:2}\end{tabular}"
    )


def test_cite_keys_alone_are_one_cite():
    bibliography = Bibliography(texkeys={"arxiv:1": "A:2020a", "arxiv:2": "B:2021b"})

    assert bibliography.cite(("arxiv:1", "arxiv:2")) == r"\cite{A:2020a,B:2021b}"


def test_cite_links_inspire_records():
    assert Bibliography().cite(("recid:1323200",)) == (
        r"\href{https://inspirehep.net/literature/1323200}{INSPIRE:1323200}"
    )


def test_cite_links_another_site_by_its_name():
    reference = "url:https://repository.cern/records/3rj9z-94994"

    assert Bibliography().cite((reference,)) == (
        r"\href{https://repository.cern/records/3rj9z-94994}{repository.cern}"
    )


def test_cite_nothing_is_empty():
    assert Bibliography().cite(()) == ""


def _fake_inspire(monkeypatch, records, bibtex="@article{A:2020a,}"):
    """Answer INSPIRE queries from *records*, recording each query made."""
    queries = []

    def query(terms, **params):
        queries.append((list(terms), params))
        if params.get("format") == "bibtex":
            return bibtex.encode()
        return json.dumps(
            {"hits": {"hits": [{"metadata": r} for r in records]}}
        ).encode()

    monkeypatch.setattr(di, "_inspire_query", query)
    return queries


def test_bibliography_maps_references_to_their_current_texkey(monkeypatch):
    _fake_inspire(
        monkeypatch,
        [
            {
                "control_number": 1404878,
                "texkeys": ["ATLAS:2015lsn", "Aad:2015mbv"],
                "arxiv_eprints": [{"value": "1511.04716"}],
            },
            {"control_number": 1323200, "texkeys": ["CMS:2014ika"]},
        ],
    )
    infos = [
        DatasetInfo("DS1", 1, references=("arxiv:1511.04716",)),
        DatasetInfo("DS2", 1, references=("recid:1323200",)),
    ]

    bibliography = dataset_bibliography(infos, fetch_bibtex=True)

    assert bibliography.texkeys == {
        "arxiv:1511.04716": "ATLAS:2015lsn",
        "recid:1323200": "CMS:2014ika",
    }
    assert bibliography.bibtex == "@article{A:2020a,}\n"


def test_paper_without_doi_links_to_its_inspire_record(monkeypatch):
    """A paper with a DOI is linked to its journal through the ``doi`` field
    INSPIRE writes; one without gets a ``url`` to INSPIRE instead."""
    _fake_inspire(
        monkeypatch,
        [
            {
                "control_number": 1404878,
                "texkeys": ["ATLAS:2015lsn"],
                "dois": [{"value": "10.1140/epjc/s10052-016-3910-7"}],
            },
            {"control_number": 1323200, "texkeys": ["CMS:2014ika"]},
        ],
        bibtex=(
            '@article{ATLAS:2015lsn,\n    doi = "10.1140/epjc/s10052-016-3910-7"\n}'
            '\n\n@techreport{CMS:2014ika,\n    title = "x"\n}'
        ),
    )
    infos = [
        DatasetInfo("DS1", 1, references=("recid:1404878",)),
        DatasetInfo("DS2", 1, references=("recid:1323200",)),
    ]

    bibtex = dataset_bibliography(infos, fetch_bibtex=True).bibtex

    assert bibtex == (
        '@article{ATLAS:2015lsn,\n    doi = "10.1140/epjc/s10052-016-3910-7"\n}'
        "\n\n@techreport{CMS:2014ika,\n"
        '    url = "https://inspirehep.net/literature/1323200",\n'
        '    title = "x"\n}\n'
    )


def test_bibliography_queries_each_reference_once_in_batches(monkeypatch):
    queries = _fake_inspire(monkeypatch, [])
    monkeypatch.setattr(di, "_INSPIRE_BATCH", 2)
    infos = [
        DatasetInfo("DS1", 1, references=("arxiv:1", "arxiv:2")),
        DatasetInfo("DS2", 1, references=("arxiv:2", "arxiv:3")),
    ]

    dataset_bibliography(infos, fetch_bibtex=True)

    searched = [terms for terms, params in queries if "fields" in params]
    assert searched == [["arxiv:1", "arxiv:2"], ["arxiv:3"]]


def test_reference_inspire_does_not_know_is_warned_about(monkeypatch, caplog):
    _fake_inspire(monkeypatch, [])
    infos = [DatasetInfo("DS", 1, references=("arxiv:9999.99999",))]

    with caplog.at_level(logging.WARNING, logger="smefit.dataset_info"):
        bibliography = dataset_bibliography(infos, fetch_bibtex=True)

    assert bibliography.texkeys == {}
    assert "arxiv:9999.99999" in caplog.text


def test_inspire_lookup_is_opt_in(monkeypatch):
    """By default a run makes no request: the lookup needs fetch_bibtex."""

    def no_network(*args, **kwargs):
        raise AssertionError("INSPIRE was queried")

    monkeypatch.setattr(di, "_inspire_query", no_network)
    infos = [DatasetInfo("DS", 1, references=("arxiv:1511.04716",))]

    assert dataset_bibliography(infos) == Bibliography()
    assert dataset_bibliography(infos, fetch_bibtex=False) == Bibliography()


def test_links_to_other_sites_are_not_looked_up(monkeypatch):
    """INSPIRE holds no record for them: they are only ever linked."""

    def no_network(*args, **kwargs):
        raise AssertionError("INSPIRE was queried")

    monkeypatch.setattr(di, "_inspire_query", no_network)
    infos = [
        DatasetInfo("DS", 1, references=("url:https://repository.cern/records/x",))
    ]

    assert dataset_bibliography(infos, fetch_bibtex=True) == Bibliography()


def test_unreachable_inspire_degrades_to_links(monkeypatch, caplog):
    """A report should not fail because INSPIRE is down."""

    def offline(*args, **kwargs):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(di, "_inspire_query", offline)
    infos = [DatasetInfo("DS", 1, references=("arxiv:1511.04716",))]

    with caplog.at_level(logging.WARNING, logger="smefit.dataset_info"):
        assert dataset_bibliography(infos, fetch_bibtex=True) == Bibliography()

    assert "INSPIRE" in caplog.text


# ---------------------------------------------------------------------------
# The LaTeX table
# ---------------------------------------------------------------------------


def _rows(tex):
    """The body rows of a rendered table: the lines between its heads and total."""
    body = tex.split("\\endfoot\n", 1)[1].rsplit("\\midrule\n\\multicolumn{2}", 1)[0]
    return body.splitlines()


def test_table_row_escapes_text_and_keeps_math():
    info = DatasetInfo(
        "ATLAS_tt_8TeV",
        7,
        description=r"50% of $p_{\rm T}$ in t_t & more",
        luminosity=(20.3,),
        references=("arxiv:1511.04716",),
    )
    bibliography = Bibliography(texkeys={"arxiv:1511.04716": "ATLAS:2015lsn"})

    (row,) = _rows(dataset_table_tex([info], bibliography))

    assert row == (
        r"\texttt{ATLAS\_\allowbreak{}tt\_\allowbreak{}8TeV} & "
        r"50\% of $p_{\rm T}$ in t\_t \& more & 7 & "
        r"20.3 & \cite{ATLAS:2015lsn} \\"
    )


def test_already_escaped_characters_are_left_alone():
    assert di._latex_text(r"10\% and \_x") == r"10\% and \_x"


def test_unpaired_dollar_is_escaped():
    assert di._latex_text("costs $5") == r"costs \$5"


def test_url_becomes_a_link_labelled_with_its_site():
    """A written-out URL cannot break, so it would run out of its column. The
    comma glued to it is not part of it (ATLAS_Zjj_13TeV_dphijj)."""
    text = "from HEPData record:https://www.hepdata.net/record/ins1803608,Table 04."

    assert di._latex_text(text) == (
        r"from HEPData record:\href{https://www.hepdata.net/record/ins1803608}"
        r"{HEPData},Table 04."
    )


def test_unknown_site_reads_as_its_host_and_punctuation_stays_outside():
    text = "See https://example.org/a_b/c.pdf."

    assert di._latex_text(text) == (
        r"See \href{https://example.org/a_b/c.pdf}{example.org}."
    )


def test_link_target_escapes_what_breaks_a_table_cell():
    """\\& \\% \\# give back &, %, # in the PDF link; _ and ~ need nothing."""
    url = "https://www.hepdata.net/record/ins1?version=1&table=Table%202#t_1~x"

    assert di._latex_text(url) == (
        r"\href{https://www.hepdata.net/record/ins1?version=1\&table=Table\%202"
        r"\#t_1~x}{HEPData}"
    )


def test_unicode_pdflatex_cannot_typeset_is_spelled_out():
    """The two the database writes: a minus sign and the macron of tt̄."""
    assert di._latex_text("W− and tt̄Z") == r"W$-$ and t\={t}Z"


def test_table_sections_follow_groups_in_order_of_first_appearance():
    infos = [
        DatasetInfo("A", 1, group="Top"),
        DatasetInfo("B", 2, group="Higgs"),
        DatasetInfo("C", 3, group="Top"),
    ]

    rows = _rows(dataset_table_tex(infos, Bibliography()))

    assert rows[0] == r"\multicolumn{5}{l}{\textbf{Top}} \\"
    assert rows[1].startswith(r"\texttt{A}")
    assert rows[2].startswith(r"\texttt{C}")
    assert rows[3] == r"\midrule"
    assert rows[4] == r"\multicolumn{5}{l}{\textbf{Higgs}} \\"
    assert rows[5].startswith(r"\texttt{B}")


def test_ungrouped_table_has_no_section_headings():
    infos = [DatasetInfo("A", 1), DatasetInfo("B", 2)]

    rows = _rows(dataset_table_tex(infos, Bibliography()))

    assert [row.split(" & ")[0] for row in rows] == [r"\texttt{A}", r"\texttt{B}"]


def test_missing_luminosity_is_a_dash():
    (row,) = _rows(dataset_table_tex([DatasetInfo("A", 1)], Bibliography()))

    assert row.split(" & ")[3] == "--"


def test_table_ends_with_the_total_number_of_points():
    infos = [DatasetInfo("A", 7), DatasetInfo("B", 15)]

    tex = dataset_table_tex(infos, Bibliography())

    assert tex.endswith(
        "\\midrule\n\\multicolumn{2}{l}{Total} & 22 & & \\\\\n\\end{xltabular}\n"
    )


def test_table_spans_the_line_width():
    """The name and description columns take what the others leave, so the
    table fits any text width."""
    tex = dataset_table_tex([DatasetInfo("A", 1)], Bibliography())

    assert r"\begin{xltabular}{\linewidth}" in tex
    assert r"\usepackage{array,booktabs,xltabular,hyperref}" in tex


def test_preview_typesets_the_table_at_paper_width():
    tex = di._preview_tex(with_bibliography=False)

    assert r"\usepackage[textwidth=483pt]{geometry}" in tex
    assert r"\input{dataset_table}" in tex
    assert "natbib" not in tex
    assert r"\bibliography" not in tex


def test_preview_with_bibliography_links_its_dois():
    tex = di._preview_tex(with_bibliography=True)

    assert r"\usepackage[numbers]{natbib}" in tex
    assert r"\newcommand{\doi}[1]{\href{https://doi.org/#1}{doi:#1}}" in tex
    assert tex.endswith(
        "\\bibliographystyle{unsrtnat}\n\\bibliography{dataset_table}\n"
        "\\end{document}\n"
    )


def test_nodes_read_only_the_commondata():
    """The contract the skill references rely on: every node of this module
    needs only `datasets`, `data_path` and the other nodes here, so its
    actions run on a runcard with no theory or coefficients
    (`collect_commondata_actions` in scripts/generate_skill_reference.py)."""
    nodes = {
        name
        for name, obj in vars(di).items()
        if inspect.isfunction(obj)
        and obj.__module__ == di.__name__
        and not name.startswith("_")
    }
    for name in nodes:
        required = {
            p.name
            for p in inspect.signature(getattr(di, name)).parameters.values()
            if p.default is inspect.Parameter.empty
        }
        assert required <= nodes | {"datasets", "data_path"}, name
