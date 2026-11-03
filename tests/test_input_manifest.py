"""Unit tests for smefit.input_manifest."""

import hashlib
import shutil

import pytest

from smefit.input_manifest import InputManifest, _git_commit


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def runcard(tmp_path, fixtures_data_path, fixtures_theory_path):
    """A runcard on a writable copy of the fixture database."""
    shutil.copytree(fixtures_data_path, tmp_path / "commondata")
    shutil.copytree(fixtures_theory_path, tmp_path / "theory")
    return {
        "data_path": str(tmp_path / "commondata"),
        "theory_path": str(tmp_path / "theory"),
        "datasets": [{"name": "TESTDATA", "order": "NLO"}],
    }


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def test_from_runcard_hashes_every_input_file(runcard, tmp_path):
    module = tmp_path / "MyChi2.py"
    module.write_text("class MyChi2: ...\n")
    runcard["external_chi2"] = {"MyChi2": {"path": str(module), "extra": 1}}

    manifest = InputManifest.from_runcard(runcard)

    # keys are relative to the database, so manifests compare across machines
    assert manifest.files == {
        "commondata/TESTDATA.yaml": _sha(tmp_path / "commondata" / "TESTDATA.yaml"),
        "theory/TESTDATA.json": _sha(tmp_path / "theory" / "TESTDATA.json"),
        "external_chi2/MyChi2": _sha(module),
    }
    assert set(manifest.database_commit) == {"commondata", "theory"}


def test_from_runcard_resolves_prefix_paths(runcard, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "smefit.paths.load_user_paths", lambda: {"my_db": str(tmp_path)}
    )
    prefixed = {**runcard, "theory_path": "my_db/theory"}

    assert InputManifest.from_runcard(prefixed) == InputManifest.from_runcard(runcard)


def test_from_runcard_records_a_missing_file_as_none(runcard, tmp_path):
    (tmp_path / "theory" / "TESTDATA.json").unlink()

    manifest = InputManifest.from_runcard(runcard)

    assert manifest.files["theory/TESTDATA.json"] is None


def test_an_external_only_runcard_records_no_database():
    manifest = InputManifest.from_runcard({"external_chi2": {}})

    assert manifest.files == {}
    assert manifest.database_commit == {}


def test_git_commit_outside_a_checkout_is_none(tmp_path):
    assert _git_commit(tmp_path) is None


def test_git_commit_without_git_installed_is_none(tmp_path, monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr("smefit.input_manifest.subprocess.run", no_git)

    assert _git_commit(tmp_path) is None


def test_a_manifest_is_recorded_without_git(runcard, monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr("smefit.input_manifest.subprocess.run", no_git)

    manifest = InputManifest.from_runcard(runcard)

    assert all(manifest.files.values())  # the hashes need no git
    assert manifest.smefit_commit is None
    assert set(manifest.database_commit.values()) == {None}


# ---------------------------------------------------------------------------
# Comparing
# ---------------------------------------------------------------------------


def _manifest(files, db_commit="aaa", smefit_commit="s1"):
    return InputManifest(dict(files), {"theory": db_commit}, smefit_commit)


def test_identical_manifests_have_no_differences():
    manifest = _manifest({"theory/A.json": "h1"})

    assert manifest.differences(_manifest({"theory/A.json": "h1"})) == ([], [])


def test_changed_files_are_kept_apart_from_missing_ones():
    was = _manifest({"theory/A.json": "h1", "commondata/A.yaml": "h2"})
    now = _manifest({"theory/A.json": "XX", "commondata/A.yaml": None})

    assert was.differences(now) == (
        ["theory/A.json: changed"],
        ["commondata/A.yaml: missing"],
    )


def test_a_database_commit_alone_is_no_difference():
    """Unchanged input files reproduce the fit, whatever else the database holds."""
    was = _manifest({"theory/A.json": "h1"}, db_commit="aaa")
    now = _manifest({"theory/A.json": "h1"}, db_commit="bbb")

    assert was.differences(now) == ([], [])


def test_a_database_commit_is_noted_with_changed_files():
    was = _manifest({"theory/A.json": "h1"}, db_commit="aaa")
    now = _manifest({"theory/A.json": "h2"}, db_commit="bbb")

    assert "theory database commit: aaa -> bbb" in was.differences(now)[1]


def test_a_different_smefit_is_only_a_note():
    was = _manifest({}, smefit_commit="s1")
    now = _manifest({}, smefit_commit="s2")

    assert was.differences(now) == ([], ["smefit commit: s1 -> s2"])


@pytest.mark.parametrize("was, now", [("s1", None), (None, "s1")])
def test_an_unknown_commit_is_no_difference(was, now):
    """No git on one of the two machines says nothing about the code."""
    changed = ({"theory/A.json": "h1"}, {"theory/A.json": "h2"})

    diff = _manifest(changed[0], db_commit=was, smefit_commit=was).differences(
        _manifest(changed[1], db_commit=now, smefit_commit=now)
    )

    assert diff == (["theory/A.json: changed"], [])


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def test_write_load_roundtrip(runcard, tmp_path):
    manifest = InputManifest.from_runcard(runcard)
    out = tmp_path / "fit"

    manifest.write(out)

    assert (out / "input" / "manifest.json").is_file()
    assert InputManifest.load(out) == manifest


def test_load_of_a_fit_without_a_manifest_is_none(tmp_path):
    assert InputManifest.load(tmp_path) is None
