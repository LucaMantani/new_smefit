"""Freshness tests for the handwritten reference docs bundled with the skills.

Most files under `.agents/skills/*/references/` are AUTO-GENERATED, and
`scripts/generate_skill_reference.py --check` keeps those honest in CI. The
schema tables in `smefit-analysis/references/output-layout.md` are not: they
are prose an agent is told to trust, describing JSON that only the code knows
the shape of. Nothing regenerates them, so a key added to or dropped from a
payload leaves the doc quietly wrong.

These tests close that gap by writing each payload and comparing the keys it
actually contains against the keys the doc tabulates.
"""

import json
import re
from pathlib import Path

import jax.numpy as jnp

import smefit.fit_result
from smefit.fit_result import FitResult, FitResultGroup

DOC = (
    Path(__file__).resolve().parents[1]
    / ".agents"
    / "skills"
    / "smefit-analysis"
    / "references"
    / "output-layout.md"
)

JOINT_HEADING = "## fit_results.json schema (joint fit)"
INDIVIDUAL_HEADING = "## fit_results.json schema (individual fits)"


def _documented_fields(heading):
    """The field names tabulated under *heading*, from the table's first column.

    A row names one or more fields in backticks (`num_data`, `n_free`, `ndof`),
    so every backticked token in that column is a documented field. The header
    and separator rows carry none and drop out on their own.
    """
    lines = DOC.read_text().splitlines()
    assert heading in lines, f"'{heading}' section is missing from {DOC.name}"

    fields = set()
    for line in lines[lines.index(heading) + 1 :]:
        if line.startswith("## "):
            break
        if line.startswith("|"):
            fields.update(re.findall(r"`([^`]+)`", line.split("|")[1]))
    return fields


def _written_payload(result_or_group, tmp_path, write):
    """The decoded fit_results.json that *write* leaves behind."""
    getattr(result_or_group, write)(tmp_path)
    with (tmp_path / "fit_results.json").open() as f:
        return json.load(f)


def _assert_documents(documented, written, wrote):
    undocumented = written - documented
    stale = documented - written
    assert (
        not undocumented
    ), f"{DOC.name} does not document {sorted(undocumented)}, which {wrote} writes."
    assert (
        not stale
    ), f"{DOC.name} documents {sorted(stale)}, which {wrote} no longer writes."


def test_joint_schema_table_matches_what_is_written(tmp_path):
    result = FitResult(
        free_parameters=["OpA"],
        best_fit_point={"OpA": 1.0},
        max_loglikelihood=-1.0,
        num_data=5,
    )

    _assert_documents(
        _documented_fields(JOINT_HEADING),
        set(_written_payload(result, tmp_path, "write")),
        "FitResult.write",
    )


def test_individual_schema_table_matches_what_is_written(tmp_path):
    result = FitResult(
        free_parameters=["OpA"],
        best_fit_point={"OpA": 1.0},
        max_loglikelihood=-1.0,
        num_data=5,
        samples={"OpA": jnp.array([0.9, 1.0, 1.1])},
    )

    _assert_documents(
        _documented_fields(INDIVIDUAL_HEADING),
        set(_written_payload(FitResultGroup([result]), tmp_path, "write_summary")),
        "FitResultGroup.write_summary",
    )


def test_the_doc_names_only_methods_that_exist():
    """The doc tells an agent which calls to make; a rename must not outlive it."""
    for cls_name, attr in re.findall(
        r"`(FitResultGroup|FitResult|Fit)\.(\w+)", DOC.read_text()
    ):
        cls = getattr(smefit.fit_result, cls_name)
        assert hasattr(
            cls, attr
        ), f"{DOC.name} refers to {cls_name}.{attr}, which no longer exists."
