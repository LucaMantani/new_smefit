"""
Tools to upload/download resources to/from the Nikhef SURFdrive server via WebDAV.

Two server profiles are supported:
  public  – read-only credentials are bundled with the package (no setup needed).
             Team members with write access store their credentials under the
             'public' key in ~/.config/smefit/server.yaml.
  private – requires a 'private' key in ~/.config/smefit/server.yaml.

Config file format (~/.config/smefit/server.yaml):
    public:
      webdav_hostname: https://surfdrive.surf.nl/public.php/webdav/
      webdav_login:    <write-token>
      webdav_password: <password>
    private:
      webdav_hostname: https://surfdrive.surf.nl/public.php/webdav/
      webdav_login:    <private-token>
      webdav_password: <password>

Run 'smefit_setup_server' to create this file interactively or from a YAML template.

Resource types and their remote directories:
    fit     -> fits/
    report  -> reports/

RGE matrices (rge_matrix.pkl) are stored inside fit directories, not as
standalone resources. Use list_fits_with_rge() and download_rge() to work
with them.
"""

import datetime
import json
import logging
import pathlib
import tarfile
import tempfile

import yaml

log = logging.getLogger(__name__)

RESOURCE_TYPES = ["fit", "report"]
REGISTRY_PATH = "registry.json"
SERVERS = ["public", "private"]
RGE_FILENAME = "rge_matrix.pkl"

_REMOTE_DIRS = {
    "fit": "fits",
    "report": "reports",
}

CONFIG_PATH = pathlib.Path.home() / ".config" / "smefit" / "server.yaml"

# Read-only credentials for the public server — safe to bundle since they grant read access only.
_BUNDLED_PUBLIC_SERVER = {
    "webdav_hostname": "https://surfdrive.surf.nl/public.php/webdav/",
    "webdav_login": "widktYypyLy9b8K",
    "webdav_password": "Cgt6fePMgK",
}


class ServerError(Exception):
    pass


def _load_config() -> dict:
    """Load ~/.config/smefit/server.yaml, returning an empty dict if absent."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def _auto_server(config: dict, need_write: bool) -> str:
    """Pick the best available server automatically.

    Private is preferred when credentials are configured. For read operations,
    falls back to the bundled public credentials so external users need no setup.
    """
    if "private" in config:
        return "private"
    if need_write:
        if "public" in config:
            return "public"
        raise ServerError(
            f"No server credentials found in {CONFIG_PATH}.\n"
            "Run 'smefit_setup_server' to configure your credentials."
        )
    return "public"


def _get_client(server: str | None, need_write: bool = False):
    """Return an authenticated WebDAV client for *server*.

    When *server* is None the best available server is chosen automatically:
    private if credentials are configured, otherwise the bundled public
    read-only credentials (no config file required).
    """
    from webdav3.client import Client

    config = _load_config()

    if server is None:
        server = _auto_server(config, need_write)

    if server == "public" and not need_write:
        return Client(_BUNDLED_PUBLIC_SERVER)

    hint = "Run 'smefit_setup_server' to configure your credentials."

    if server == "public":
        if "public" not in config:
            raise ServerError(
                f"Write access to the public server requires a 'public:' section in {CONFIG_PATH}.\n{hint}"
            )
    elif server == "private":
        if "private" not in config:
            raise ServerError(
                f"Private server credentials not found in {CONFIG_PATH}.\n{hint}"
            )
    else:
        raise ServerError(
            f"Unknown server '{server}'. Choose from: {', '.join(SERVERS)}"
        )

    profile = config[server]
    for key in ("webdav_hostname", "webdav_login", "webdav_password"):
        if key not in profile:
            raise ServerError(f"Missing key '{key}' under '{server}:' in {CONFIG_PATH}")
    return Client(profile)


def _remote_path(resource_type: str, resource_name: str) -> str:
    """Return the remote WebDAV path for a given resource."""
    return f"{_REMOTE_DIRS[resource_type]}/{resource_name}.tar.gz"


def _compress(
    source: pathlib.Path, archive_path: pathlib.Path, arcname: str | None = None
) -> None:
    log.info("Compressing %s ...", source)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source, arcname=arcname or source.name)


def _extract(archive_path: pathlib.Path, dest: pathlib.Path) -> None:
    log.info("Extracting to %s ...", dest)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dest)


def _detect_has_rge(local_path: pathlib.Path) -> bool:
    """Return True if rge_matrix.pkl exists anywhere under local_path."""
    return any(local_path.rglob(RGE_FILENAME))


def _empty_registry() -> dict:
    return {"fits": {}, "reports": {}}


def _read_registry(client) -> dict:
    """Download and parse the registry; return an empty registry if absent.

    Migrates the old flat format {name: meta} to {"fits": {name: meta}, "reports": {}}.
    """
    if not client.check(REGISTRY_PATH):
        return _empty_registry()
    with tempfile.TemporaryDirectory(prefix="smefit_registry_") as tmpdir:
        tmp = pathlib.Path(tmpdir) / "registry.json"
        client.download_sync(remote_path=REGISTRY_PATH, local_path=str(tmp))
        data = json.loads(tmp.read_text())
    if "fits" not in data and "reports" not in data:
        return {"fits": data, "reports": {}}
    return {**_empty_registry(), **data}


def _write_registry(client, registry: dict) -> None:
    """Serialize and upload the fit registry."""
    with tempfile.TemporaryDirectory(prefix="smefit_registry_") as tmpdir:
        tmp = pathlib.Path(tmpdir) / "registry.json"
        tmp.write_text(json.dumps(registry, indent=2, sort_keys=True))
        client.upload_sync(remote_path=REGISTRY_PATH, local_path=str(tmp))


def _list_fit_names(client) -> list[str]:
    """Return all fit names from the remote fits/ directory."""
    remote_dir = _REMOTE_DIRS["fit"]
    if not client.check(remote_dir):
        return []
    names = []
    for e in client.list(remote_dir):
        e = e.rstrip("/")
        if e in (remote_dir, ""):
            continue
        if e.endswith(".tar.gz"):
            e = e[: -len(".tar.gz")]
        names.append(e)
    return names


def list_fits_with_rge(server: str | None = None) -> list[str]:
    """Return names of fits on the server that contain an rge_matrix.pkl file.

    Each fit tarball is downloaded and inspected; this may be slow for large
    repositories.
    """
    client = _get_client(server, need_write=False)
    fit_names = _list_fit_names(client)
    results = []
    for fit_name in fit_names:
        remote = _remote_path("fit", fit_name)
        log.info("Checking %s ...", fit_name)
        with tempfile.TemporaryDirectory(prefix="smefit_rge_check_") as tmpdir:
            archive = pathlib.Path(tmpdir) / f"{fit_name}.tar.gz"
            client.download_sync(remote_path=remote, local_path=str(archive))
            with tarfile.open(archive, "r:gz") as tar:
                if any(
                    pathlib.Path(m.name).name == RGE_FILENAME for m in tar.getmembers()
                ):
                    results.append(fit_name)
    return results


def download_rge(
    fit_name: str,
    local_path: pathlib.Path | None = None,
    server: str | None = None,
) -> pathlib.Path:
    """Download rge_matrix.pkl from *fit_name* on the server.

    Only the RGE file is extracted from the fit tarball; the rest is discarded.
    Returns the path to the saved file.
    """
    client = _get_client(server, need_write=False)
    remote = _remote_path("fit", fit_name)
    if not client.check(remote):
        raise ServerError(f"Fit '{fit_name}' not found on server.")

    if local_path is None:
        local_path = pathlib.Path.cwd()
    local_path = pathlib.Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)

    log.info("Downloading fit archive for '%s' ...", fit_name)
    with tempfile.TemporaryDirectory(prefix="smefit_rge_dl_") as tmpdir:
        archive = pathlib.Path(tmpdir) / f"{fit_name}.tar.gz"
        client.download_sync(remote_path=remote, local_path=str(archive))
        with tarfile.open(archive, "r:gz") as tar:
            rge_member = next(
                (
                    m
                    for m in tar.getmembers()
                    if pathlib.Path(m.name).name == RGE_FILENAME
                ),
                None,
            )
            if rge_member is None:
                raise ServerError(
                    f"Fit '{fit_name}' does not contain a {RGE_FILENAME} file."
                )
            f = tar.extractfile(rge_member)
            dest = local_path / RGE_FILENAME
            log.info("Extracting %s -> %s ...", RGE_FILENAME, dest)
            dest.write_bytes(f.read())

    log.info("Download complete: %s", dest)
    return dest


def rename(
    resource_type: str,
    old_name: str,
    new_name: str,
    server: str | None = None,
) -> None:
    """Rename a resource on the server."""
    if resource_type not in RESOURCE_TYPES:
        raise ServerError(
            f"Unknown resource type '{resource_type}'. "
            f"Choose from: {', '.join(RESOURCE_TYPES)}"
        )
    client = _get_client(server, need_write=True)
    old_remote = _remote_path(resource_type, old_name)
    new_remote = _remote_path(resource_type, new_name)
    if not client.check(old_remote):
        raise ServerError(f"Resource '{old_name}' not found on server.")
    if client.check(new_remote):
        raise ServerError(
            f"'{new_name}' already exists on server. Choose a different name."
        )
    client.move(remote_path_from=old_remote, remote_path_to=new_remote)
    log.info("Renamed '%s' -> '%s'.", old_name, new_name)
    if resource_type == "fit":
        registry = _read_registry(client)
        if old_name in registry["fits"]:
            registry["fits"][new_name] = registry["fits"].pop(old_name)
            _write_registry(client, registry)


def delete(
    resource_type: str,
    resource_name: str,
    server: str | None = None,
) -> None:
    """Delete a resource from the server."""
    if resource_type not in RESOURCE_TYPES:
        raise ServerError(
            f"Unknown resource type '{resource_type}'. "
            f"Choose from: {', '.join(RESOURCE_TYPES)}"
        )
    client = _get_client(server, need_write=True)
    remote = _remote_path(resource_type, resource_name)
    if not client.check(remote):
        raise ServerError(f"Resource '{resource_name}' not found on server.")
    client.clean(remote)
    log.info("Deleted '%s'.", resource_name)
    if resource_type == "fit":
        registry = _read_registry(client)
        if resource_name in registry["fits"]:
            del registry["fits"][resource_name]
            _write_registry(client, registry)


class Uploader:
    """Upload resources to a server. Requires write credentials in the config file."""

    def __init__(self, server: str | None = None):
        if server is not None and server not in SERVERS:
            raise ServerError(
                f"Unknown server '{server}'. Choose from: {', '.join(SERVERS)}"
            )
        self._client = _get_client(server, need_write=True)
        self._server = server or "auto"

    def _ensure_remote_dir(self, resource_type: str) -> None:
        remote_dir = _REMOTE_DIRS[resource_type]
        if not self._client.check(remote_dir):
            self._client.mkdir(remote_dir)

    def _check_remote_exists(self, resource_type: str, resource_name: str) -> bool:
        return self._client.check(_remote_path(resource_type, resource_name))

    def upload(
        self,
        resource_type: str,
        resource_name: str,
        local_path: pathlib.Path | None = None,
        force: bool = False,
    ) -> None:
        """Upload *resource_name* of *resource_type* from *local_path*."""
        if resource_type not in RESOURCE_TYPES:
            raise ServerError(
                f"Unknown resource type '{resource_type}'. "
                f"Choose from: {', '.join(RESOURCE_TYPES)}"
            )

        if local_path is None:
            local_path = pathlib.Path.cwd() / resource_name
        local_path = pathlib.Path(local_path)

        if not local_path.exists():
            raise ServerError(f"Local path does not exist: {local_path}")

        self._ensure_remote_dir(resource_type)

        remote = _remote_path(resource_type, resource_name)
        if not force and self._check_remote_exists(resource_type, resource_name):
            raise ServerError(
                f"'{resource_name}' already exists on the {self._server} server. "
                "Use --force to overwrite."
            )

        with tempfile.TemporaryDirectory(prefix="smefit_upload_") as tmpdir:
            archive = pathlib.Path(tmpdir) / f"{resource_name}.tar.gz"
            _compress(local_path, archive, arcname=resource_name)
            log.info("Uploading %s -> %s ...", archive.name, remote)
            self._client.upload_sync(remote_path=remote, local_path=str(archive))
        log.info("Upload complete.")
        if resource_type == "fit":
            registry = _read_registry(self._client)
            registry["fits"][resource_name] = {
                "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "has_rge": _detect_has_rge(local_path),
            }
            _write_registry(self._client, registry)
            log.info("Registry updated.")


class Downloader:
    """Download resources from a server. Uses bundled credentials for the public server."""

    def __init__(self, server: str | None = None):
        if server is not None and server not in SERVERS:
            raise ServerError(
                f"Unknown server '{server}'. Choose from: {', '.join(SERVERS)}"
            )
        self._client = _get_client(server, need_write=False)
        self._server = server or "auto"

    def get_registry(self) -> dict:
        """Return the fit registry from the server, or {} if absent."""
        return _read_registry(self._client)

    def list_resources(self, resource_type: str) -> list[str]:
        """Return names of available resources of *resource_type* on the server."""
        if resource_type not in RESOURCE_TYPES:
            raise ServerError(
                f"Unknown resource type '{resource_type}'. "
                f"Choose from: {', '.join(RESOURCE_TYPES)}"
            )
        remote_dir = _REMOTE_DIRS[resource_type]
        if not self._client.check(remote_dir):
            return []
        names = []
        for e in self._client.list(remote_dir):
            e = e.rstrip("/")
            if e in (remote_dir, ""):
                continue
            if e.endswith(".tar.gz"):
                e = e[: -len(".tar.gz")]
            names.append(e)
        return names

    def download(
        self,
        resource_type: str,
        resource_name: str,
        local_path: pathlib.Path | None = None,
    ) -> pathlib.Path:
        """Download *resource_name* of *resource_type* to *local_path*.

        Returns the path to the downloaded resource.
        """
        if resource_type not in RESOURCE_TYPES:
            raise ServerError(
                f"Unknown resource type '{resource_type}'. "
                f"Choose from: {', '.join(RESOURCE_TYPES)}"
            )

        if local_path is None:
            local_path = pathlib.Path.cwd()
        local_path = pathlib.Path(local_path)
        local_path.mkdir(parents=True, exist_ok=True)

        remote = _remote_path(resource_type, resource_name)
        if not self._client.check(remote):
            raise ServerError(
                f"Resource '{resource_name}' (type: {resource_type}) "
                f"not found on the {self._server} server."
            )

        with tempfile.TemporaryDirectory(prefix="smefit_download_") as tmpdir:
            archive = pathlib.Path(tmpdir) / f"{resource_name}.tar.gz"
            log.info("Downloading %s ...", remote)
            self._client.download_sync(remote_path=remote, local_path=str(archive))
            _extract(archive, local_path)
        dest = local_path / resource_name
        log.info("Download complete: %s", dest)
        return dest


def download_and_view_report(
    report_name: str,
    local_path: pathlib.Path | None = None,
    server: str | None = None,
) -> None:
    """Download *report_name* if not already present, then open index.html in the browser."""
    import webbrowser

    if local_path is None:
        local_path = pathlib.Path.cwd()
    local_path = pathlib.Path(local_path)
    report_dir = local_path / report_name
    index = report_dir / "index.html"

    if not report_dir.exists():
        downloader = Downloader(server=server)
        downloader.download("report", report_name, local_path)
    else:
        log.info("Report already present at %s, skipping download.", report_dir)

    if not index.exists():
        raise ServerError(f"No index.html found in {report_dir}.")

    url = index.resolve().as_uri()
    log.info("Opening %s in browser.", url)
    webbrowser.open(url)


def sync_registry(server: str | None = None) -> dict:
    """Rebuild the fit registry by inspecting all fit archives on the server.

    Existing 'created_at' values are preserved where known; entries absent from
    the current registry receive the current timestamp as a fallback.
    Returns the new registry dict.
    """
    client = _get_client(server, need_write=True)
    fit_names = _list_fit_names(client)
    old_registry = _read_registry(client)
    now = datetime.datetime.now().isoformat(timespec="seconds")

    registry = _empty_registry()
    for fit_name in fit_names:
        remote = _remote_path("fit", fit_name)
        log.info("Inspecting %s ...", fit_name)
        with tempfile.TemporaryDirectory(prefix="smefit_sync_") as tmpdir:
            archive = pathlib.Path(tmpdir) / f"{fit_name}.tar.gz"
            client.download_sync(remote_path=remote, local_path=str(archive))
            with tarfile.open(archive, "r:gz") as tar:
                has_rge = any(
                    pathlib.Path(m.name).name == RGE_FILENAME for m in tar.getmembers()
                )
        created_at = old_registry["fits"].get(fit_name, {}).get("created_at", now)
        registry["fits"][fit_name] = {"created_at": created_at, "has_rge": has_rge}

    _write_registry(client, registry)
    log.info("Registry synced: %d fit(s).", len(registry["fits"]))
    return registry
