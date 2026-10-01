"""Tests for smefit.server_utils.

The WebDAV server is replaced by ``FakeWebDAV``, a stand-in for
``webdav3.client.Client`` backed by a local directory. It reproduces the
SURFdrive restrictions the module is written around — a PUT onto an existing
file, a MOVE onto an existing path, and a MKCOL/PUT/MOVE under a missing parent
all fail — so these tests also check that the module never relies on an
operation the real server rejects.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import shutil
import tarfile
import types
from collections.abc import Callable
from typing import Any

import pytest
import webdav3.client
import yaml
from webdav3.exceptions import (
    RemoteParentNotFound,
    RemoteResourceNotFound,
    ResponseErrorCode,
)

from smefit import server_utils
from smefit.server_utils import Downloader, ServerError, Uploader

PRIVATE_PROFILE = {
    "webdav_hostname": "https://example.org/webdav/",
    "webdav_login": "private-login",
    "webdav_password": "private-password",
    "name": "Private Tester",
}
PUBLIC_PROFILE = {
    "webdav_hostname": "https://example.org/public/",
    "webdav_login": "public-login",
    "webdav_password": "public-password",
    "name": "Public Tester",
}


# ---------------------------------------------------------------------------
# Fake WebDAV server
# ---------------------------------------------------------------------------


class FakeWebDAV:
    """Directory-backed stand-in for ``webdav3.client.Client``.

    Installed in place of the ``Client`` class: calling the instance records the
    options a client was built with and returns the instance itself, so every
    client the module creates talks to the same fake server.
    """

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.root.mkdir(parents=True)
        self.options: list[dict[str, str]] = []

    def __call__(self, options: dict[str, str]) -> FakeWebDAV:
        self.options.append(dict(options))
        return self

    def path(self, remote_path: str) -> pathlib.Path:
        return self.root / remote_path.strip("/")

    def check(self, remote_path: str = "/") -> bool:
        return self.path(remote_path).exists()

    def mkdir(self, remote_path: str) -> None:
        target = self.path(remote_path)
        if not target.parent.exists():
            raise RemoteParentNotFound(remote_path)
        target.mkdir(exist_ok=True)

    def list(self, remote_path: str = "/") -> list[str]:
        target = self.path(remote_path)
        if not target.is_dir():
            raise RemoteResourceNotFound(remote_path)
        return sorted(c.name + ("/" if c.is_dir() else "") for c in target.iterdir())

    def upload_sync(self, remote_path: str, local_path: str) -> None:
        target = self.path(remote_path)
        if not target.parent.exists():
            raise RemoteParentNotFound(remote_path)
        if target.exists():
            raise ResponseErrorCode(remote_path, 403, "overwrite rejected")
        shutil.copyfile(local_path, target)

    def download_sync(self, remote_path: str, local_path: str) -> None:
        source = self.path(remote_path)
        if not source.is_file():
            raise RemoteResourceNotFound(remote_path)
        shutil.copyfile(source, local_path)

    def move(
        self, remote_path_from: str, remote_path_to: str, overwrite: bool = False
    ) -> None:
        source = self.path(remote_path_from)
        target = self.path(remote_path_to)
        if not source.exists():
            raise RemoteResourceNotFound(remote_path_from)
        if not target.parent.exists():
            raise RemoteParentNotFound(remote_path_to)
        if target.exists() and not overwrite:
            raise ResponseErrorCode(remote_path_to, 412, "destination exists")
        source.rename(target)

    def clean(self, remote_path: str) -> None:
        target = self.path(remote_path)
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

    def free(self) -> int:
        return 123_456

    # -- test helpers, not part of the webdav3 API --------------------------

    def read_json(self, remote_path: str) -> Any:
        return json.loads(self.path(remote_path).read_text())

    def put_json(self, remote_path: str, payload: Any) -> None:
        target = self.path(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload))

    def put_archive(self, remote_path: str, source: pathlib.Path) -> None:
        """Place a tarball of *source* on the server, bypassing the module."""
        target = self.path(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(target, "w:gz") as tar:
            tar.add(source, arcname=source.name)


def _ticking_datetime() -> types.SimpleNamespace:
    """A ``datetime`` module whose clock advances one second per ``now()`` call.

    The module names registry backups by a second-resolution timestamp, so two
    registry writes in the same second would collide on the backup path.
    """
    current = [datetime.datetime(2026, 1, 1, 12, 0, 0)]

    class TickingDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz: datetime.tzinfo | None = None) -> datetime.datetime:
            current[0] += datetime.timedelta(seconds=1)
            return current[0]

    return types.SimpleNamespace(datetime=TickingDatetime)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def write_config(path: pathlib.Path, **profiles: dict[str, str]) -> None:
    path.write_text(yaml.safe_dump(profiles))


@pytest.fixture
def config_path(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> pathlib.Path:
    path = tmp_path / "server.yaml"
    monkeypatch.setattr(server_utils, "CONFIG_PATH", path)
    return path


@pytest.fixture
def fake_server(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    config_path: pathlib.Path,
) -> FakeWebDAV:
    """The fake server with no credentials configured."""
    fake = FakeWebDAV(tmp_path / "remote")
    monkeypatch.setattr(webdav3.client, "Client", fake)
    monkeypatch.setattr(server_utils, "datetime", _ticking_datetime())
    return fake


@pytest.fixture
def remote(fake_server: FakeWebDAV, config_path: pathlib.Path) -> FakeWebDAV:
    """The fake server with private write credentials configured."""
    write_config(config_path, private=PRIVATE_PROFILE)
    return fake_server


@pytest.fixture
def work(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "local"
    path.mkdir()
    return path


def make_fit(
    base: pathlib.Path,
    name: str = "fit_a",
    rge_rel: str | None = "rge_matrix.pkl",
    runcard: bool = True,
) -> pathlib.Path:
    """Create a fit directory with results, and optionally a runcard and RGE matrix."""
    fit = base / name
    fit.mkdir(parents=True)
    (fit / "fit_results.json").write_text(json.dumps({"name": name}))
    if runcard:
        (fit / "input").mkdir()
        (fit / "input" / "runcard.yaml").write_text("result_ID: " + name + "\n")
    if rge_rel is not None:
        rge = fit / rge_rel
        rge.parent.mkdir(parents=True, exist_ok=True)
        rge.write_bytes(b"rge-" + name.encode())
    return fit


def make_report(base: pathlib.Path, name: str = "report_a") -> pathlib.Path:
    report = base / name
    report.mkdir(parents=True)
    (report / "index.html").write_text("<html></html>")
    return report


def archive_members(path: pathlib.Path) -> set[str]:
    with tarfile.open(path, "r:gz") as tar:
        return {m.name for m in tar.getmembers()}


# ---------------------------------------------------------------------------
# Configuration and client selection
# ---------------------------------------------------------------------------


class TestConfig:
    def test_missing_config_is_empty(self, config_path: pathlib.Path) -> None:
        assert server_utils._load_config() == {}

    def test_empty_config_is_empty(self, config_path: pathlib.Path) -> None:
        config_path.write_text("")
        assert server_utils._load_config() == {}

    def test_config_is_parsed(self, config_path: pathlib.Path) -> None:
        write_config(config_path, private=PRIVATE_PROFILE)
        assert server_utils._load_config() == {"private": PRIVATE_PROFILE}

    @pytest.mark.parametrize(
        ("config", "need_write", "expected"),
        [
            ({"private": {}, "public": {}}, True, "private"),
            ({"private": {}}, False, "private"),
            ({"public": {}}, True, "public"),
            ({"public": {}}, False, "public"),
            ({}, False, "public"),
        ],
    )
    def test_auto_server(
        self, config: dict[str, Any], need_write: bool, expected: str
    ) -> None:
        assert server_utils._auto_server(config, need_write) == expected

    def test_auto_server_write_without_credentials(self) -> None:
        with pytest.raises(ServerError, match="No server credentials"):
            server_utils._auto_server({}, need_write=True)


class TestGetClient:
    def test_public_read_uses_bundled_credentials(
        self, fake_server: FakeWebDAV
    ) -> None:
        assert server_utils._get_client(None) is fake_server
        assert fake_server.options == [server_utils._BUNDLED_PUBLIC_SERVER]

    def test_public_read_ignores_configured_public_profile(
        self, fake_server: FakeWebDAV, config_path: pathlib.Path
    ) -> None:
        write_config(config_path, public=PUBLIC_PROFILE)
        server_utils._get_client("public")
        assert fake_server.options == [server_utils._BUNDLED_PUBLIC_SERVER]

    def test_public_write_uses_configured_profile(
        self, fake_server: FakeWebDAV, config_path: pathlib.Path
    ) -> None:
        write_config(config_path, public=PUBLIC_PROFILE)
        server_utils._get_client("public", need_write=True)
        assert fake_server.options == [PUBLIC_PROFILE]

    def test_auto_prefers_private(self, remote: FakeWebDAV) -> None:
        server_utils._get_client(None)
        assert remote.options == [PRIVATE_PROFILE]

    def test_public_write_without_profile(self, fake_server: FakeWebDAV) -> None:
        with pytest.raises(ServerError, match="Write access to the public server"):
            server_utils._get_client("public", need_write=True)

    def test_private_without_profile(self, fake_server: FakeWebDAV) -> None:
        with pytest.raises(ServerError, match="Private server credentials not found"):
            server_utils._get_client("private")

    def test_unknown_server(self, fake_server: FakeWebDAV) -> None:
        with pytest.raises(ServerError, match="Unknown server 'nope'"):
            server_utils._get_client("nope", need_write=True)

    def test_incomplete_profile(
        self, fake_server: FakeWebDAV, config_path: pathlib.Path
    ) -> None:
        profile = {k: v for k, v in PRIVATE_PROFILE.items() if k != "webdav_password"}
        write_config(config_path, private=profile)
        with pytest.raises(ServerError, match="Missing key 'webdav_password'"):
            server_utils._get_client("private")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("resource_type", "name", "expected"),
    [
        ("fit", "fit_a", "fits/fit_a.tar.gz"),
        ("report", "rep", "reports/rep.tar.gz"),
        ("misc", "dir/file.pkl", "misc/dir/file.pkl"),
    ],
)
def test_remote_path(resource_type: str, name: str, expected: str) -> None:
    assert server_utils._remote_path(resource_type, name) == expected


def test_compress_and_extract_round_trip(work: pathlib.Path) -> None:
    fit = make_fit(work / "src", rge_rel="sub/rge_matrix.pkl")
    archive = work / "fit.tar.gz"
    server_utils._compress(
        fit,
        archive,
        arcname="renamed",
        strip_paths={"sub/rge_matrix.pkl", "input/runcard.yaml"},
    )
    members = archive_members(archive)
    assert "renamed/fit_results.json" in members
    assert "renamed/sub/rge_matrix.pkl" not in members
    assert "renamed/input/runcard.yaml" not in members

    server_utils._extract(archive, work / "dest")
    assert (work / "dest" / "renamed" / "fit_results.json").read_text() == (
        fit / "fit_results.json"
    ).read_text()


def test_compress_defaults_to_source_name(work: pathlib.Path) -> None:
    fit = make_fit(work)
    archive = work / "fit.tar.gz"
    server_utils._compress(fit, archive)
    assert {
        "fit_a",
        "fit_a/fit_results.json",
        "fit_a/rge_matrix.pkl",
        "fit_a/input/runcard.yaml",
    } <= archive_members(archive)


def test_ensure_remote_path(remote: FakeWebDAV) -> None:
    server_utils._ensure_remote_path(remote, "a/b/c")
    assert remote.path("a/b/c").is_dir()
    # Idempotent
    server_utils._ensure_remote_path(remote, "a/b/c")


def test_detect_has_rge(work: pathlib.Path) -> None:
    assert server_utils._detect_has_rge(
        make_fit(work, "with", rge_rel="x/rge_matrix.pkl")
    )
    assert not server_utils._detect_has_rge(make_fit(work, "without", rge_rel=None))


def test_registry_backup_path(remote: FakeWebDAV) -> None:
    path = server_utils._registry_backup_path("registry.json")
    assert path == "bin/registry_backups/registry.json.20260101T120001.bak"


def test_list_resource_names(remote: FakeWebDAV) -> None:
    assert server_utils._list_resource_names(remote, "fit") == []
    for name in ("fits/a.tar.gz", "fits/b.tar.gz", "fits/notes.txt"):
        remote.put_json(name, {})
    remote.put_json("fits/rge_matrices/a.pkl", {})
    remote.put_json("misc/registry_misc.json", {})
    remote.put_json("misc/data.pkl", {})
    remote.put_json("misc/sub/x.pkl", {})
    assert server_utils._list_fit_names(remote) == ["a", "b"]
    assert server_utils._list_resource_names(remote, "misc") == ["data.pkl", "sub"]


def test_list_resource_names_skips_self_entry() -> None:
    """Some WebDAV servers list the directory itself as its first entry."""

    class SelfListingClient:
        def check(self, path: str) -> bool:
            return True

        def list(self, path: str) -> list[str]:
            return ["reports/", "", "r.tar.gz"]

    assert server_utils._list_resource_names(SelfListingClient(), "report") == ["r"]


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------


class TestRemoteRegistry:
    def test_absent_registry_is_empty(self, remote: FakeWebDAV) -> None:
        assert server_utils._read_registry(remote) == server_utils._empty_registry()

    def test_legacy_flat_registry_is_migrated(self, remote: FakeWebDAV) -> None:
        remote.put_json("registry.json", {"fit_a": {"has_rge": True}})
        assert server_utils._read_registry(remote) == {
            "fits": {"fit_a": {"has_rge": True}},
            "reports": {},
            "projects": [],
        }

    def test_missing_sections_are_filled(self, remote: FakeWebDAV) -> None:
        remote.put_json("registry.json", {"fits": {"fit_a": {}}})
        assert server_utils._read_registry(remote) == {
            "fits": {"fit_a": {}},
            "reports": {},
            "projects": [],
        }

    def test_write_backs_up_previous_registry(self, remote: FakeWebDAV) -> None:
        first = {"fits": {"a": {}}, "reports": {}, "projects": []}
        second = {"fits": {"b": {}}, "reports": {}, "projects": []}
        server_utils._write_registry(remote, first)
        assert not remote.check("bin/registry_backups")
        server_utils._write_registry(remote, second)
        assert server_utils._read_registry(remote) == second
        (backup,) = remote.list("bin/registry_backups")
        assert backup.startswith("registry.json.")
        assert remote.read_json(f"bin/registry_backups/{backup}") == first

    def test_misc_registry_round_trip(self, remote: FakeWebDAV) -> None:
        assert server_utils._read_misc_registry(remote) == {}
        remote.mkdir("misc")
        server_utils._write_misc_registry(remote, {"x": {"comment": "1"}})
        server_utils._write_misc_registry(remote, {"y": {}})
        assert server_utils._read_misc_registry(remote) == {"y": {}}
        (backup,) = remote.list("bin/registry_backups")
        assert backup.startswith("registry_misc.json.")

    def test_bin_registry_round_trip(self, remote: FakeWebDAV) -> None:
        assert server_utils._read_bin_registry(remote) == {}
        server_utils._write_bin_registry(remote, {"fit/a": {}})
        server_utils._write_bin_registry(remote, {"fit/b": {}})
        assert server_utils._read_bin_registry(remote) == {"fit/b": {}}
        (backup,) = remote.list("bin/registry_backups")
        assert backup.startswith("registry_bin.json.")


class TestLocalRegistry:
    def test_absent_registry_is_empty(self, work: pathlib.Path) -> None:
        assert server_utils._read_local_registry(work) == server_utils._empty_registry()

    def test_round_trip_creates_directory(self, work: pathlib.Path) -> None:
        results = work / "results"
        server_utils._write_local_registry(results, {"fits": {"a": {}}})
        assert server_utils._read_local_registry(results) == {
            "fits": {"a": {}},
            "reports": {},
            "projects": [],
        }


# ---------------------------------------------------------------------------
# Uploader
# ---------------------------------------------------------------------------


class TestUploader:
    def test_unknown_server(self, remote: FakeWebDAV) -> None:
        with pytest.raises(ServerError, match="Unknown server"):
            Uploader(server="nope")

    def test_requires_write_credentials(self, fake_server: FakeWebDAV) -> None:
        with pytest.raises(ServerError, match="No server credentials"):
            Uploader()

    def test_upload_fit_stores_rge_and_runcard_separately(
        self, remote: FakeWebDAV, work: pathlib.Path
    ) -> None:
        fit = make_fit(work, rge_rel="rge/rge_matrix.pkl")
        Uploader().upload("fit", "fit_a", fit, message="hello", project="proj")

        members = archive_members(remote.path("fits/fit_a.tar.gz"))
        assert "fit_a/fit_results.json" in members
        assert "fit_a/rge/rge_matrix.pkl" not in members
        assert "fit_a/input/runcard.yaml" not in members
        assert remote.path("fits/rge_matrices/fit_a.pkl").read_bytes() == b"rge-fit_a"
        assert remote.path("fits/runcards/fit_a.yaml").read_text() == (
            "result_ID: fit_a\n"
        )

        entry = server_utils._read_registry(remote)["fits"]["fit_a"]
        assert entry == {
            "created_at": entry["created_at"],
            "uploaded_by": "Private Tester",
            "has_rge": True,
            "rge_path": "rge/rge_matrix.pkl",
            "runcard_path": "input/runcard.yaml",
            "comment": "hello",
            "project": "proj",
        }

    def test_upload_fit_without_rge_or_runcard(
        self, remote: FakeWebDAV, work: pathlib.Path
    ) -> None:
        fit = make_fit(work, rge_rel=None, runcard=False)
        Uploader().upload("fit", "renamed", fit)
        assert "renamed/fit_results.json" in archive_members(
            remote.path("fits/renamed.tar.gz")
        )
        assert not remote.check("fits/rge_matrices")
        assert not remote.check("fits/runcards")
        entry = server_utils._read_registry(remote)["fits"]["renamed"]
        assert entry["has_rge"] is False
        assert entry["rge_path"] is None
        assert entry["runcard_path"] is None
        assert "comment" not in entry
        assert "project" not in entry

    def test_upload_defaults_to_cwd(
        self,
        remote: FakeWebDAV,
        work: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        make_fit(work / "nested")
        monkeypatch.chdir(work)
        Uploader().upload("fit", "nested/fit_a/")
        assert remote.check("fits/fit_a.tar.gz")
        assert "fit_a" in server_utils._read_registry(remote)["fits"]

    def test_upload_existing_requires_force(
        self, remote: FakeWebDAV, work: pathlib.Path
    ) -> None:
        fit = make_fit(work)
        Uploader().upload("fit", "fit_a", fit)
        with pytest.raises(ServerError, match="already exists"):
            Uploader().upload("fit", "fit_a", fit)

    def test_force_moves_previous_upload_aside(
        self, remote: FakeWebDAV, work: pathlib.Path
    ) -> None:
        fit = make_fit(work)
        Uploader().upload("fit", "fit_a", fit)
        (fit / "rge_matrix.pkl").write_bytes(b"new-rge")
        Uploader().upload("fit", "fit_a", fit, force=True)

        assert remote.path("fits/rge_matrices/fit_a.pkl").read_bytes() == b"new-rge"
        backups = remote.list("bin/fits")
        assert any(b.startswith("fit_a.tar.gz.") for b in backups)
        assert any(
            b.startswith("fit_a.pkl.") for b in remote.list("bin/fits/rge_matrices")
        )
        assert any(
            b.startswith("fit_a.yaml.") for b in remote.list("bin/fits/runcards")
        )

    def test_upload_report(self, remote: FakeWebDAV, work: pathlib.Path) -> None:
        report = make_report(work)
        Uploader().upload("report", "report_a", report, project="proj")
        assert "report_a/index.html" in archive_members(
            remote.path("reports/report_a.tar.gz")
        )
        entry = server_utils._read_registry(remote)["reports"]["report_a"]
        assert entry["project"] == "proj"
        assert "has_rge" not in entry

    def test_rejects_directory_without_marker(
        self, remote: FakeWebDAV, work: pathlib.Path
    ) -> None:
        fit = make_fit(work)
        with pytest.raises(ServerError, match="no index.html found"):
            Uploader().upload("report", "fit_a", fit)

    def test_rejects_missing_local_path(
        self, remote: FakeWebDAV, work: pathlib.Path
    ) -> None:
        with pytest.raises(ServerError, match="Local path does not exist"):
            Uploader().upload("fit", "fit_a", work / "missing")

    def test_upload_misc(self, remote: FakeWebDAV, work: pathlib.Path) -> None:
        local = work / "data.pkl"
        local.write_bytes(b"payload")
        Uploader().upload(
            "misc", "runs/run1/data.pkl", local, message="m", project="proj"
        )
        assert remote.path("misc/runs/run1/data.pkl").read_bytes() == b"payload"
        entry = server_utils._read_misc_registry(remote)["runs/run1/data.pkl"]
        assert entry == {
            "uploaded_at": entry["uploaded_at"],
            "uploaded_by": "Private Tester",
            "comment": "m",
            "project": "proj",
        }

    def test_upload_misc_defaults_to_basename_in_cwd(
        self,
        remote: FakeWebDAV,
        work: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (work / "data.pkl").write_bytes(b"payload")
        monkeypatch.chdir(work)
        Uploader().upload("misc", "runs/data.pkl")
        assert remote.path("misc/runs/data.pkl").read_bytes() == b"payload"

    def test_upload_misc_existing_requires_force(
        self, remote: FakeWebDAV, work: pathlib.Path
    ) -> None:
        local = work / "data.pkl"
        local.write_bytes(b"v1")
        Uploader().upload("misc", "data.pkl", local)
        local.write_bytes(b"v2")
        with pytest.raises(ServerError, match="already exists"):
            Uploader().upload("misc", "data.pkl", local)
        Uploader().upload("misc", "data.pkl", local, force=True)
        assert remote.path("misc/data.pkl").read_bytes() == b"v2"
        assert any(b.startswith("data.pkl.") for b in remote.list("bin/misc"))

    def test_upload_misc_missing_local_file(
        self, remote: FakeWebDAV, work: pathlib.Path
    ) -> None:
        with pytest.raises(ServerError, match="Local path does not exist"):
            Uploader().upload("misc", "data.pkl", work / "missing.pkl")


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------


class TestDownloader:
    def test_unknown_server(self, remote: FakeWebDAV) -> None:
        with pytest.raises(ServerError, match="Unknown server"):
            Downloader(server="nope")

    def test_registries(self, remote: FakeWebDAV) -> None:
        remote.put_json("registry.json", {"fits": {"a": {}}})
        remote.put_json("misc/registry_misc.json", {"m": {}})
        remote.put_json("bin/registry_bin.json", {"fit/b": {}})
        downloader = Downloader()
        assert downloader.get_registry()["fits"] == {"a": {}}
        assert downloader.get_misc_registry() == {"m": {}}
        assert downloader.get_bin_registry() == {"fit/b": {}}

    def test_list_resources(self, remote: FakeWebDAV, work: pathlib.Path) -> None:
        downloader = Downloader()
        assert downloader.list_resources("fit") == []
        Uploader().upload("fit", "fit_a", make_fit(work))
        Uploader().upload("report", "report_a", make_report(work))
        local = work / "data.pkl"
        local.write_bytes(b"x")
        Uploader().upload("misc", "data.pkl", local)
        assert downloader.list_resources("fit") == ["fit_a"]
        assert downloader.list_resources("report") == ["report_a"]
        assert downloader.list_resources("misc") == ["data.pkl"]

    def test_list_resources_skips_self_entry(self, remote: FakeWebDAV) -> None:
        remote.put_json("reports/r.tar.gz", {})
        downloader = Downloader()
        downloader._client = types.SimpleNamespace(
            check=lambda path: True,
            list=lambda path: ["reports/", "r.tar.gz", "notes.txt"],
        )
        assert downloader.list_resources("report") == ["r"]

    def test_fit_round_trip_reinjects_rge_and_runcard(
        self, remote: FakeWebDAV, work: pathlib.Path
    ) -> None:
        fit = make_fit(work / "src", rge_rel="rge/rge_matrix.pkl")
        Uploader().upload("fit", "fit_a", fit)

        dest = Downloader().download("fit", "fit_a", work / "dest")
        assert dest == work / "dest" / "fit_a"
        for rel in ("fit_results.json", "rge/rge_matrix.pkl", "input/runcard.yaml"):
            assert (dest / rel).read_bytes() == (fit / rel).read_bytes()

    def test_fit_without_registry_entry_uses_default_paths(
        self, remote: FakeWebDAV, work: pathlib.Path
    ) -> None:
        fit = make_fit(work / "src", rge_rel="rge/rge_matrix.pkl")
        Uploader().upload("fit", "fit_a", fit)
        remote.clean("registry.json")

        dest = Downloader().download("fit", "fit_a", work / "dest")
        assert (dest / "rge_matrix.pkl").read_bytes() == b"rge-fit_a"
        assert (dest / "input" / "runcard.yaml").exists()

    def test_download_defaults_to_cwd(
        self,
        remote: FakeWebDAV,
        work: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        Uploader().upload("report", "report_a", make_report(work / "src"))
        monkeypatch.chdir(work)
        assert Downloader().download("report", "report_a") == work / "report_a"
        assert (work / "report_a" / "index.html").exists()

    def test_download_missing(self, remote: FakeWebDAV, work: pathlib.Path) -> None:
        with pytest.raises(ServerError, match="not found"):
            Downloader().download("fit", "missing", work)

    def test_download_misc(
        self,
        remote: FakeWebDAV,
        work: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        remote.put_json("misc/runs/data.json", {"a": 1})
        dest = Downloader().download("misc", "runs/data.json", work / "out")
        assert dest == work / "out" / "data.json"
        assert json.loads(dest.read_text()) == {"a": 1}

        monkeypatch.chdir(work)
        assert Downloader().download("misc", "runs/data.json") == work / "data.json"

    def test_download_misc_missing(
        self, remote: FakeWebDAV, work: pathlib.Path
    ) -> None:
        with pytest.raises(ServerError, match="misc/missing not found"):
            Downloader().download("misc", "missing", work)

    def test_update_local_registry(
        self, remote: FakeWebDAV, work: pathlib.Path
    ) -> None:
        remote.put_json("registry.json", {"fits": {"fit_a": {"comment": "c"}}})
        results = work / "results"
        server_utils._write_local_registry(results, {"fits": {"old": {}}})
        downloader = Downloader()
        downloader.update_local_registry("fit", "fit_a", results)
        downloader.update_local_registry("misc", "data.pkl", results)
        assert server_utils._read_local_registry(results)["fits"] == {
            "old": {},
            "fit_a": {"comment": "c"},
        }


# ---------------------------------------------------------------------------
# Resource type validation shared by every entry point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda: server_utils.rename("bogus", "a", "b"),
        lambda: server_utils.delete("bogus", "a"),
        lambda: server_utils.trash("bogus", "a"),
        lambda: server_utils.restore("bogus", "a"),
        lambda: Uploader().upload("bogus", "a"),
        lambda: Downloader().download("bogus", "a"),
        lambda: Downloader().list_resources("bogus"),
    ],
    ids=["rename", "delete", "trash", "restore", "upload", "download", "list"],
)
def test_unknown_resource_type(remote: FakeWebDAV, call: Callable[[], Any]) -> None:
    with pytest.raises(ServerError, match="Unknown resource type 'bogus'"):
        call()


# ---------------------------------------------------------------------------
# RGE helpers
# ---------------------------------------------------------------------------


class TestRge:
    def test_download_rge_from_dedicated_dir(
        self,
        remote: FakeWebDAV,
        work: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        Uploader().upload("fit", "fit_a", make_fit(work / "src"))
        dest = server_utils.download_rge("fit_a", work / "out")
        assert dest == work / "out" / "rge_matrix.pkl"
        assert dest.read_bytes() == b"rge-fit_a"

        monkeypatch.chdir(work)
        assert server_utils.download_rge("fit_a") == work / "rge_matrix.pkl"

    def test_download_rge_from_legacy_archive(
        self, remote: FakeWebDAV, work: pathlib.Path
    ) -> None:
        remote.put_archive(
            "fits/legacy.tar.gz",
            make_fit(work / "src", "legacy", rge_rel="x/rge_matrix.pkl"),
        )
        dest = server_utils.download_rge("legacy", work / "out")
        assert dest.read_bytes() == b"rge-legacy"

    def test_download_rge_legacy_archive_without_rge(
        self, remote: FakeWebDAV, work: pathlib.Path
    ) -> None:
        remote.put_archive("fits/legacy.tar.gz", make_fit(work / "src", "legacy", None))
        with pytest.raises(ServerError, match="does not contain"):
            server_utils.download_rge("legacy", work / "out")

    def test_download_rge_missing_fit(
        self, remote: FakeWebDAV, work: pathlib.Path
    ) -> None:
        with pytest.raises(ServerError, match="not found"):
            server_utils.download_rge("missing", work)

    def test_list_fits_with_rge_legacy_archives(
        self, remote: FakeWebDAV, work: pathlib.Path
    ) -> None:
        remote.put_archive("fits/with.tar.gz", make_fit(work, "with"))
        remote.put_archive("fits/without.tar.gz", make_fit(work, "without", None))
        assert server_utils.list_fits_with_rge() == ["with"]

    @pytest.mark.xfail(
        strict=True,
        reason="Uploader strips rge_matrix.pkl from the archive and stores it under "
        "fits/rge_matrices/, which list_fits_with_rge does not look at",
    )
    def test_list_fits_with_rge_current_upload_layout(
        self, remote: FakeWebDAV, work: pathlib.Path
    ) -> None:
        Uploader().upload("fit", "fit_a", make_fit(work))
        assert server_utils.list_fits_with_rge() == ["fit_a"]


# ---------------------------------------------------------------------------
# rename / delete / trash / restore / mkdir
# ---------------------------------------------------------------------------


class TestRename:
    def test_rename_fit_moves_all_parts(
        self, remote: FakeWebDAV, work: pathlib.Path
    ) -> None:
        Uploader().upload("fit", "old", make_fit(work), message="c")
        server_utils.rename("fit", "old", "new")
        assert not remote.check("fits/old.tar.gz")
        assert remote.check("fits/new.tar.gz")
        assert remote.check("fits/rge_matrices/new.pkl")
        assert remote.check("fits/runcards/new.yaml")
        fits = server_utils._read_registry(remote)["fits"]
        assert set(fits) == {"new"}
        assert fits["new"]["comment"] == "c"

    def test_rename_with_comment(self, remote: FakeWebDAV, work: pathlib.Path) -> None:
        Uploader().upload("report", "old", make_report(work))
        server_utils.rename("report", "old", "new", comment="updated")
        assert server_utils._read_registry(remote)["reports"]["new"]["comment"] == (
            "updated"
        )

    def test_comment_only(self, remote: FakeWebDAV, work: pathlib.Path) -> None:
        Uploader().upload("fit", "fit_a", make_fit(work))
        server_utils.rename("fit", "fit_a", comment="note")
        server_utils.rename("fit", "fit_a", "fit_a", comment="note2")
        assert remote.check("fits/fit_a.tar.gz")
        assert server_utils._read_registry(remote)["fits"]["fit_a"]["comment"] == (
            "note2"
        )

    def test_comment_only_missing_resource(self, remote: FakeWebDAV) -> None:
        with pytest.raises(ServerError, match="not found"):
            server_utils.rename("fit", "missing", comment="note")

    def test_needs_name_or_comment(self, remote: FakeWebDAV) -> None:
        with pytest.raises(ServerError, match="Provide a new name"):
            server_utils.rename("fit", "fit_a")

    def test_missing_source(self, remote: FakeWebDAV) -> None:
        with pytest.raises(ServerError, match="not found"):
            server_utils.rename("fit", "missing", "new")

    def test_existing_target(self, remote: FakeWebDAV, work: pathlib.Path) -> None:
        Uploader().upload("fit", "a", make_fit(work, "a"))
        Uploader().upload("fit", "b", make_fit(work, "b"))
        with pytest.raises(ServerError, match="already exists"):
            server_utils.rename("fit", "a", "b")

    def test_rename_unregistered_leaves_registry_alone(
        self, remote: FakeWebDAV, work: pathlib.Path
    ) -> None:
        remote.put_archive("fits/old.tar.gz", make_fit(work, "old"))
        server_utils.rename("fit", "old", "new")
        assert remote.check("fits/new.tar.gz")
        assert not remote.check("registry.json")

    def test_rename_misc_into_new_directory(
        self, remote: FakeWebDAV, work: pathlib.Path
    ) -> None:
        local = work / "data.pkl"
        local.write_bytes(b"x")
        Uploader().upload("misc", "data.pkl", local, message="m")
        server_utils.rename("misc", "data.pkl", "deep/dir/data.pkl", comment="moved")
        assert remote.path("misc/deep/dir/data.pkl").read_bytes() == b"x"
        misc_reg = server_utils._read_misc_registry(remote)
        assert set(misc_reg) == {"deep/dir/data.pkl"}
        assert misc_reg["deep/dir/data.pkl"]["comment"] == "moved"


class TestDelete:
    def test_delete_fit(self, remote: FakeWebDAV, work: pathlib.Path) -> None:
        Uploader().upload("fit", "fit_a", make_fit(work))
        server_utils.delete("fit", "fit_a")
        assert not remote.check("fits/fit_a.tar.gz")
        assert not remote.check("fits/rge_matrices/fit_a.pkl")
        assert not remote.check("fits/runcards/fit_a.yaml")
        assert server_utils._read_registry(remote)["fits"] == {}

    def test_delete_unregistered(self, remote: FakeWebDAV, work: pathlib.Path) -> None:
        remote.put_archive("reports/r.tar.gz", make_report(work, "r"))
        server_utils.delete("report", "r")
        assert not remote.check("reports/r.tar.gz")

    def test_delete_misc(self, remote: FakeWebDAV, work: pathlib.Path) -> None:
        local = work / "data.pkl"
        local.write_bytes(b"x")
        Uploader().upload("misc", "data.pkl", local)
        server_utils.delete("misc", "data.pkl")
        assert not remote.check("misc/data.pkl")
        assert server_utils._read_misc_registry(remote) == {}

    def test_delete_missing(self, remote: FakeWebDAV) -> None:
        with pytest.raises(ServerError, match="not found"):
            server_utils.delete("fit", "missing")


class TestTrashAndRestore:
    def test_fit_round_trip(self, remote: FakeWebDAV, work: pathlib.Path) -> None:
        Uploader().upload("fit", "fit_a", make_fit(work), message="orig")
        original = server_utils._read_registry(remote)["fits"]["fit_a"]

        server_utils.trash("fit", "fit_a", message="bye")
        for path in (
            "fits/fit_a.tar.gz",
            "fits/rge_matrices/fit_a.pkl",
            "fits/runcards/fit_a.yaml",
        ):
            assert not remote.check(path)
            assert remote.check(f"bin/{path}")
        assert server_utils._read_registry(remote)["fits"] == {}
        bin_entry = server_utils._read_bin_registry(remote)["fit/fit_a"]
        assert bin_entry == {
            "deleted_at": bin_entry["deleted_at"],
            "deleted_by": "Private Tester",
            "resource_type": "fit",
            "resource_name": "fit_a",
            "original_meta": original,
            "comment": "bye",
        }

        server_utils.restore("fit", "fit_a")
        for path in (
            "fits/fit_a.tar.gz",
            "fits/rge_matrices/fit_a.pkl",
            "fits/runcards/fit_a.yaml",
        ):
            assert remote.check(path)
            assert not remote.check(f"bin/{path}")
        assert server_utils._read_registry(remote)["fits"]["fit_a"] == original
        assert server_utils._read_bin_registry(remote) == {}

    def test_misc_round_trip(self, remote: FakeWebDAV, work: pathlib.Path) -> None:
        local = work / "data.pkl"
        local.write_bytes(b"x")
        Uploader().upload("misc", "sub/data.pkl", local)
        original = server_utils._read_misc_registry(remote)["sub/data.pkl"]

        server_utils.trash("misc", "sub/data.pkl")
        assert remote.check("bin/misc/sub/data.pkl")
        assert server_utils._read_misc_registry(remote) == {}
        assert (
            "comment"
            not in server_utils._read_bin_registry(remote)["misc/sub/data.pkl"]
        )

        server_utils.restore("misc", "sub/data.pkl")
        assert remote.path("misc/sub/data.pkl").read_bytes() == b"x"
        assert server_utils._read_misc_registry(remote) == {"sub/data.pkl": original}

    def test_trash_unregistered(self, remote: FakeWebDAV, work: pathlib.Path) -> None:
        remote.put_archive("reports/r.tar.gz", make_report(work, "r"))
        server_utils.trash("report", "r")
        assert (
            server_utils._read_bin_registry(remote)["report/r"]["original_meta"] == {}
        )

    def test_trash_records_public_uploader(
        self,
        fake_server: FakeWebDAV,
        config_path: pathlib.Path,
        work: pathlib.Path,
    ) -> None:
        write_config(config_path, public=PUBLIC_PROFILE)
        Uploader().upload("report", "r", make_report(work, "r"))
        server_utils.trash("report", "r", server="public")
        assert server_utils._read_bin_registry(fake_server)["report/r"][
            "deleted_by"
        ] == ("Public Tester")

    def test_trash_missing(self, remote: FakeWebDAV) -> None:
        with pytest.raises(ServerError, match="not found"):
            server_utils.trash("fit", "missing")

    def test_restore_not_in_bin(self, remote: FakeWebDAV) -> None:
        with pytest.raises(ServerError, match="is not in the bin"):
            server_utils.restore("fit", "missing")

    def test_restore_out_of_sync_bin(
        self, remote: FakeWebDAV, work: pathlib.Path
    ) -> None:
        Uploader().upload("report", "r", make_report(work, "r"))
        server_utils.trash("report", "r")
        remote.clean("bin/reports/r.tar.gz")
        with pytest.raises(ServerError, match="Binned file not found"):
            server_utils.restore("report", "r")

    def test_restore_onto_occupied_location(
        self, remote: FakeWebDAV, work: pathlib.Path
    ) -> None:
        report = make_report(work, "r")
        Uploader().upload("report", "r", report)
        server_utils.trash("report", "r")
        Uploader().upload("report", "r", report)
        with pytest.raises(ServerError, match="already exists"):
            server_utils.restore("report", "r")


def test_mkdir_misc(remote: FakeWebDAV) -> None:
    server_utils.mkdir_misc("/a/b/")
    assert remote.path("misc/a/b").is_dir()


# ---------------------------------------------------------------------------
# Report viewing
# ---------------------------------------------------------------------------


class TestViewReport:
    @pytest.fixture
    def opened(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        urls: list[str] = []
        monkeypatch.setattr("webbrowser.open", urls.append)
        return urls

    def test_downloads_then_opens(
        self, remote: FakeWebDAV, work: pathlib.Path, opened: list[str]
    ) -> None:
        Uploader().upload("report", "r", make_report(work / "src", "r"))
        server_utils.download_and_view_report("r", work / "out")
        index = work / "out" / "r" / "index.html"
        assert index.exists()
        assert opened == [index.resolve().as_uri()]

    def test_skips_download_when_present(
        self,
        fake_server: FakeWebDAV,
        work: pathlib.Path,
        opened: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        make_report(work, "r")
        monkeypatch.chdir(work)
        server_utils.download_and_view_report("r")
        assert fake_server.options == []
        assert opened == [(work / "r" / "index.html").resolve().as_uri()]

    def test_missing_index(
        self, fake_server: FakeWebDAV, work: pathlib.Path, opened: list[str]
    ) -> None:
        (work / "r").mkdir()
        with pytest.raises(ServerError, match="No index.html"):
            server_utils.download_and_view_report("r", work)
        assert opened == []


# ---------------------------------------------------------------------------
# Registry sync
# ---------------------------------------------------------------------------


def test_sync_registry(remote: FakeWebDAV, work: pathlib.Path) -> None:
    Uploader().upload("fit", "new", make_fit(work, "new"), message="dropped")
    Uploader().upload("report", "rep", make_report(work, "rep"))
    remote.put_archive("fits/legacy_rge.tar.gz", make_fit(work, "legacy_rge"))
    remote.put_archive("fits/legacy.tar.gz", make_fit(work, "legacy", None))
    remote.path("fits/broken.tar.gz").write_bytes(b"not a tarball")
    before = server_utils._read_registry(remote)

    registry = server_utils.sync_registry()

    assert registry == server_utils._read_registry(remote)
    assert set(registry["fits"]) == {"new", "legacy_rge", "legacy"}
    assert registry["fits"]["new"] == {
        "created_at": before["fits"]["new"]["created_at"],
        "has_rge": True,
        "rge_path": "rge_matrix.pkl",
        "runcard_path": "input/runcard.yaml",
        "uploaded_by": "Private Tester",
    }
    assert registry["fits"]["legacy_rge"]["has_rge"] is True
    assert registry["fits"]["legacy"]["has_rge"] is False
    assert registry["fits"]["legacy"]["uploaded_by"] is None
    assert registry["reports"] == {
        "rep": {
            "created_at": before["reports"]["rep"]["created_at"],
            "uploaded_by": "Private Tester",
        }
    }
    assert registry["projects"] == []


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


class TestRemoteProjects:
    def test_lifecycle(self, remote: FakeWebDAV, work: pathlib.Path) -> None:
        assert server_utils.list_projects() == []
        server_utils.add_project("zeta")
        server_utils.add_project("alpha")
        assert server_utils.list_projects() == ["alpha", "zeta"]

        Uploader().upload("fit", "f", make_fit(work, "f"), project="alpha")
        Uploader().upload("report", "r", make_report(work, "r"), project="alpha")
        Uploader().upload("report", "s", make_report(work, "s"), project="zeta")
        server_utils.rename_project("alpha", "beta")
        registry = server_utils._read_registry(remote)
        assert registry["projects"] == ["beta", "zeta"]
        assert registry["fits"]["f"]["project"] == "beta"
        assert registry["reports"]["r"]["project"] == "beta"
        assert registry["reports"]["s"]["project"] == "zeta"

        server_utils.remove_project("beta")
        assert server_utils.list_projects() == ["zeta"]
        assert server_utils._read_registry(remote)["fits"]["f"]["project"] == "beta"

    def test_errors(self, remote: FakeWebDAV) -> None:
        server_utils.add_project("a")
        server_utils.add_project("b")
        with pytest.raises(ServerError, match="already exists"):
            server_utils.add_project("a")
        with pytest.raises(ServerError, match="not found"):
            server_utils.rename_project("missing", "c")
        with pytest.raises(ServerError, match="already exists"):
            server_utils.rename_project("a", "b")
        with pytest.raises(ServerError, match="not found"):
            server_utils.remove_project("missing")


class TestLocalProjects:
    def test_lifecycle(self, work: pathlib.Path) -> None:
        results = work / "results"
        assert server_utils.list_local_projects(results) == []
        server_utils.add_local_project("zeta", results)
        server_utils.add_local_project("alpha", results)
        assert server_utils.list_local_projects(results) == ["alpha", "zeta"]

        registry = server_utils._read_local_registry(results)
        registry["fits"]["f"] = {"project": "alpha"}
        registry["reports"]["r"] = {"project": "zeta"}
        server_utils._write_local_registry(results, registry)

        server_utils.rename_local_project("alpha", "beta", results)
        registry = server_utils._read_local_registry(results)
        assert registry["projects"] == ["beta", "zeta"]
        assert registry["fits"]["f"]["project"] == "beta"
        assert registry["reports"]["r"]["project"] == "zeta"

        server_utils.remove_local_project("beta", results)
        assert server_utils.list_local_projects(results) == ["zeta"]

    def test_errors(self, work: pathlib.Path) -> None:
        server_utils.add_local_project("a", work)
        server_utils.add_local_project("b", work)
        with pytest.raises(ServerError, match="already exists"):
            server_utils.add_local_project("a", work)
        with pytest.raises(ServerError, match="not found"):
            server_utils.rename_local_project("missing", "c", work)
        with pytest.raises(ServerError, match="already exists"):
            server_utils.rename_local_project("a", "b", work)
        with pytest.raises(ServerError, match="not found"):
            server_utils.remove_local_project("missing", work)


def test_get_free_space(fake_server: FakeWebDAV) -> None:
    assert server_utils.get_free_space() == 123_456
