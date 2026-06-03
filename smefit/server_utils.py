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
standalone resources. Use list_fits_with_rge() and download_rge() to work with them.
"""

import datetime
import json
import logging
import pathlib
import tarfile
import tempfile

import yaml

log = logging.getLogger(__name__)

RESOURCE_TYPES = ["fit", "report", "misc"]
_ARCHIVABLE_TYPES = ["fit", "report"]  # resource types stored as tarballs
# Required marker files — upload is rejected if none is found anywhere in the directory
_RESOURCE_MARKERS = {
    "fit": "fit_results.json",
    "report": "index.html",
}
REGISTRY_PATH = "registry.json"
MISC_REGISTRY_PATH = "misc/registry_misc.json"
SERVERS = ["public", "private"]
RGE_FILENAME = "rge_matrix.pkl"
RUNCARD_FILENAME = "runcard.yaml"
# Standard relative path of the runcard inside a fit directory
RUNCARD_RELATIVE_PATH = pathlib.Path("input") / "runcard.yaml"
# Remote subdirectories for separately-stored fit components
RGE_MATRICES_REMOTE_DIR = "fits/rge_matrices"
RUNCARDS_REMOTE_DIR = "fits/runcards"

_REMOTE_DIRS = {
    "fit": "fits",
    "report": "reports",
    "misc": "misc",
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
    if resource_type == "misc":
        return f"misc/{resource_name}"
    return f"{_REMOTE_DIRS[resource_type]}/{resource_name}.tar.gz"


def _compress(
    source: pathlib.Path,
    archive_path: pathlib.Path,
    arcname: str | None = None,
    strip_paths: set | None = None,
) -> None:
    """Compress *source* into *archive_path*, optionally stripping files by relative path."""
    log.info("Compressing %s ...", source)
    _strip = {pathlib.PurePosixPath(p) for p in (strip_paths or [])}

    def _filter(tarinfo):
        p = pathlib.PurePosixPath(tarinfo.name)
        if _strip and len(p.parts) > 1:
            rel = pathlib.PurePosixPath(*p.parts[1:])
            if rel in _strip:
                return None
        return tarinfo

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source, arcname=arcname or source.name, filter=_filter)


def _extract(archive_path: pathlib.Path, dest: pathlib.Path) -> None:
    log.info("Extracting to %s ...", dest)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dest)


def _ensure_remote_path(client, path: str) -> None:
    """Create every directory component of *path* on the server if missing."""
    parts = pathlib.PurePosixPath(path).parts
    for i in range(1, len(parts) + 1):
        segment = str(pathlib.PurePosixPath(*parts[:i]))
        if not client.check(segment):
            client.mkdir(segment)


def _detect_has_rge(local_path: pathlib.Path) -> bool:
    """Return True if rge_matrix.pkl exists anywhere under local_path."""
    return any(local_path.rglob(RGE_FILENAME))


def _empty_registry() -> dict:
    return {"fits": {}, "reports": {}, "projects": []}


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
        return {"fits": data, "reports": {}, "projects": []}
    return {**_empty_registry(), **data}


def _write_registry(client, registry: dict) -> None:
    """Serialize and upload the fit registry."""
    with tempfile.TemporaryDirectory(prefix="smefit_registry_") as tmpdir:
        tmp = pathlib.Path(tmpdir) / "registry.json"
        tmp.write_text(json.dumps(registry, indent=2, sort_keys=True))
        client.upload_sync(remote_path=REGISTRY_PATH, local_path=str(tmp))


def _read_misc_registry(client) -> dict:
    """Download and parse misc/registry_misc.json; return {} if absent."""
    if not client.check(MISC_REGISTRY_PATH):
        return {}
    with tempfile.TemporaryDirectory(prefix="smefit_misc_reg_") as tmpdir:
        tmp = pathlib.Path(tmpdir) / "registry_misc.json"
        client.download_sync(remote_path=MISC_REGISTRY_PATH, local_path=str(tmp))
        return json.loads(tmp.read_text())


def _write_misc_registry(client, registry: dict) -> None:
    """Serialize and upload misc/registry_misc.json."""
    with tempfile.TemporaryDirectory(prefix="smefit_misc_reg_") as tmpdir:
        tmp = pathlib.Path(tmpdir) / "registry_misc.json"
        tmp.write_text(json.dumps(registry, indent=2, sort_keys=True))
        client.upload_sync(remote_path=MISC_REGISTRY_PATH, local_path=str(tmp))


def _list_resource_names(client, resource_type: str) -> list[str]:
    """Return all resource names of *resource_type* from the remote directory."""
    remote_dir = _REMOTE_DIRS[resource_type]
    if not client.check(remote_dir):
        return []
    names = []
    for e in client.list(remote_dir):
        e = e.rstrip("/")
        if e in (remote_dir, ""):
            continue
        if resource_type in _ARCHIVABLE_TYPES:
            # Only include actual archives; subdirectories (e.g. rge_matrices/) are skipped
            if not e.endswith(".tar.gz"):
                continue
            e = e[: -len(".tar.gz")]
        elif resource_type == "misc" and e == "registry_misc.json":
            continue
        names.append(e)
    return names


def _list_fit_names(client) -> list[str]:
    return _list_resource_names(client, "fit")


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

    Tries the dedicated rge_matrices/ directory first (fast). Falls back to
    extracting from the fit archive for fits uploaded before this structure existed.
    Returns the path to the saved file.
    """
    client = _get_client(server, need_write=False)

    if local_path is None:
        local_path = pathlib.Path.cwd()
    local_path = pathlib.Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)
    dest = local_path / RGE_FILENAME

    # Fast path: dedicated subdir
    rge_remote = f"{RGE_MATRICES_REMOTE_DIR}/{fit_name}.pkl"
    if client.check(rge_remote):
        log.info("Downloading rge_matrix for '%s' ...", fit_name)
        client.download_sync(remote_path=rge_remote, local_path=str(dest))
        log.info("Download complete: %s", dest)
        return dest

    # Fallback: extract from fit archive (legacy fits)
    archive_remote = _remote_path("fit", fit_name)
    if not client.check(archive_remote):
        raise ServerError(f"Fit '{fit_name}' not found on server.")
    log.info("Downloading fit archive for '%s' (legacy) ...", fit_name)
    with tempfile.TemporaryDirectory(prefix="smefit_rge_dl_") as tmpdir:
        archive = pathlib.Path(tmpdir) / f"{fit_name}.tar.gz"
        client.download_sync(remote_path=archive_remote, local_path=str(archive))
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
            dest.write_bytes(tar.extractfile(rge_member).read())
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
    if resource_type == "misc":
        _ensure_remote_path(client, str(pathlib.PurePosixPath(new_remote).parent))
    client.move(remote_path_from=old_remote, remote_path_to=new_remote)
    log.info("Renamed '%s' -> '%s'.", old_name, new_name)
    if resource_type == "fit":
        for old_sub, new_sub in [
            (
                f"{RGE_MATRICES_REMOTE_DIR}/{old_name}.pkl",
                f"{RGE_MATRICES_REMOTE_DIR}/{new_name}.pkl",
            ),
            (
                f"{RUNCARDS_REMOTE_DIR}/{old_name}.yaml",
                f"{RUNCARDS_REMOTE_DIR}/{new_name}.yaml",
            ),
        ]:
            if client.check(old_sub):
                client.move(remote_path_from=old_sub, remote_path_to=new_sub)
    if resource_type in _ARCHIVABLE_TYPES:
        registry = _read_registry(client)
        section = registry[f"{resource_type}s"]
        if old_name in section:
            section[new_name] = section.pop(old_name)
            _write_registry(client, registry)
    elif resource_type == "misc":
        misc_reg = _read_misc_registry(client)
        if old_name in misc_reg:
            misc_reg[new_name] = misc_reg.pop(old_name)
            _write_misc_registry(client, misc_reg)


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
        for sub in [
            f"{RGE_MATRICES_REMOTE_DIR}/{resource_name}.pkl",
            f"{RUNCARDS_REMOTE_DIR}/{resource_name}.yaml",
        ]:
            if client.check(sub):
                client.clean(sub)
    if resource_type in _ARCHIVABLE_TYPES:
        registry = _read_registry(client)
        section = registry[f"{resource_type}s"]
        if resource_name in section:
            del section[resource_name]
            _write_registry(client, registry)
    elif resource_type == "misc":
        misc_reg = _read_misc_registry(client)
        if resource_name in misc_reg:
            del misc_reg[resource_name]
            _write_misc_registry(client, misc_reg)


class Uploader:
    """Upload resources to a server. Requires write credentials in the config file."""

    def __init__(self, server: str | None = None):
        if server is not None and server not in SERVERS:
            raise ServerError(
                f"Unknown server '{server}'. Choose from: {', '.join(SERVERS)}"
            )
        config = _load_config()
        resolved_server = (
            server if server is not None else _auto_server(config, need_write=True)
        )
        self._client = _get_client(server, need_write=True)
        self._server = resolved_server
        self._uploader_name = config.get(resolved_server, {}).get("name")

    def _ensure_remote_dir(self, resource_type: str) -> None:
        remote_dir = _REMOTE_DIRS[resource_type]
        if not self._client.check(remote_dir):
            self._client.mkdir(remote_dir)

    def _ensure_dir(self, path: str) -> None:
        if not self._client.check(path):
            self._client.mkdir(path)

    def _check_remote_exists(self, resource_type: str, resource_name: str) -> bool:
        return self._client.check(_remote_path(resource_type, resource_name))

    def upload(
        self,
        resource_type: str,
        resource_name: str,
        local_path: pathlib.Path | None = None,
        force: bool = False,
        message: str | None = None,
        project: str | None = None,
    ) -> None:
        """Upload *resource_name* of *resource_type* from *local_path*.

        For fits, rge_matrix.pkl and input/runcard.yaml are extracted from the
        archive and stored separately under fits/rge_matrices/ and fits/runcards/.
        They are transparently re-injected on download.

        For misc, *resource_name* is the full path inside misc/ on the server
        (e.g. 'results/run1/output.pkl'). Intermediate directories are created
        automatically. *local_path* defaults to the basename of *resource_name*.
        """
        resource_name = resource_name.rstrip("/\\")

        if resource_type not in RESOURCE_TYPES:
            raise ServerError(
                f"Unknown resource type '{resource_type}'. "
                f"Choose from: {', '.join(RESOURCE_TYPES)}"
            )

        if resource_type == "misc":
            if local_path is None:
                local_path = pathlib.Path(resource_name).name
            local_path = pathlib.Path(local_path)
            if not local_path.exists():
                raise ServerError(f"Local path does not exist: {local_path}")
            remote = _remote_path("misc", resource_name)
            if not force and self._client.check(remote):
                raise ServerError(
                    f"'misc/{resource_name}' already exists on the {self._server} server. "
                    "Use --force to overwrite."
                )
            _ensure_remote_path(self._client, str(pathlib.PurePosixPath(remote).parent))
            log.info("Uploading %s -> %s ...", local_path, remote)
            self._client.upload_sync(remote_path=remote, local_path=str(local_path))
            log.info("Upload complete.")
            misc_reg = _read_misc_registry(self._client)
            entry = {
                "uploaded_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "uploaded_by": self._uploader_name,
            }
            if message:
                entry["comment"] = message
            if project:
                entry["project"] = project
            misc_reg[resource_name] = entry
            _write_misc_registry(self._client, misc_reg)
            log.info("To download: smefit_get misc %s", resource_name)
            return

        if local_path is None:
            local_path = pathlib.Path.cwd() / resource_name
        local_path = pathlib.Path(local_path)

        if not local_path.exists():
            raise ServerError(f"Local path does not exist: {local_path}")

        if resource_type in _RESOURCE_MARKERS:
            marker = _RESOURCE_MARKERS[resource_type]
            if not any(local_path.rglob(marker)):
                raise ServerError(
                    f"'{local_path.name}' does not look like a valid {resource_type}: "
                    f"no {marker} found."
                )

        self._ensure_remote_dir(resource_type)

        remote = _remote_path(resource_type, resource_name)
        if not force and self._check_remote_exists(resource_type, resource_name):
            raise ServerError(
                f"'{resource_name}' already exists on the {self._server} server. "
                "Use --force to overwrite."
            )

        # For fits: locate files to store separately and strip from archive
        strip_paths = set()
        rge_local = None
        rge_rel = None
        runcard_local = None
        if resource_type == "fit":
            rge_found = next(local_path.rglob(RGE_FILENAME), None)
            if rge_found:
                rge_local = rge_found
                rge_rel = rge_found.relative_to(local_path)
                strip_paths.add(pathlib.PurePosixPath(rge_rel))
            runcard_candidate = local_path / RUNCARD_RELATIVE_PATH
            if runcard_candidate.exists():
                runcard_local = runcard_candidate
                strip_paths.add(pathlib.PurePosixPath(RUNCARD_RELATIVE_PATH))

        with tempfile.TemporaryDirectory(prefix="smefit_upload_") as tmpdir:
            archive = pathlib.Path(tmpdir) / f"{resource_name}.tar.gz"
            _compress(
                local_path, archive, arcname=resource_name, strip_paths=strip_paths
            )
            log.info("Uploading %s -> %s ...", archive.name, remote)
            self._client.upload_sync(remote_path=remote, local_path=str(archive))
        log.info("Upload complete.")

        if rge_local:
            self._ensure_dir(RGE_MATRICES_REMOTE_DIR)
            rge_remote = f"{RGE_MATRICES_REMOTE_DIR}/{resource_name}.pkl"
            log.info("Uploading rge_matrix -> %s ...", rge_remote)
            self._client.upload_sync(remote_path=rge_remote, local_path=str(rge_local))

        if runcard_local:
            self._ensure_dir(RUNCARDS_REMOTE_DIR)
            runcard_remote = f"{RUNCARDS_REMOTE_DIR}/{resource_name}.yaml"
            log.info("Uploading runcard -> %s ...", runcard_remote)
            self._client.upload_sync(
                remote_path=runcard_remote, local_path=str(runcard_local)
            )

        registry = _read_registry(self._client)
        entry = {
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "uploaded_by": self._uploader_name,
        }
        if resource_type == "fit":
            entry["has_rge"] = rge_local is not None
            entry["rge_path"] = str(rge_rel) if rge_rel else None
            entry["runcard_path"] = (
                str(RUNCARD_RELATIVE_PATH) if runcard_local else None
            )
        if message:
            entry["comment"] = message
        if project:
            entry["project"] = project
        registry[f"{resource_type}s"][resource_name] = entry
        _write_registry(self._client, registry)
        log.info("Registry updated.")
        log.info("To download: smefit_get %s %s", resource_type, resource_name)
        if resource_type == "report":
            log.info("To view locally: view_report %s", resource_name)


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
        """Return the fit/report registry from the server, or {} if absent."""
        return _read_registry(self._client)

    def get_misc_registry(self) -> dict:
        """Return the misc registry from the server, or {} if absent."""
        return _read_misc_registry(self._client)

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
            if resource_type in _ARCHIVABLE_TYPES:
                if not e.endswith(".tar.gz"):
                    continue
                e = e[: -len(".tar.gz")]
            elif resource_type == "misc" and e == "registry_misc.json":
                continue
            names.append(e)
        return names

    def download(
        self,
        resource_type: str,
        resource_name: str,
        local_path: pathlib.Path | None = None,
    ) -> pathlib.Path:
        """Download *resource_name* of *resource_type* to *local_path*.

        For misc, *resource_name* is the full path inside misc/ on the server.
        The file is saved under its basename in *local_path*.
        Returns the path to the downloaded resource.
        """
        if resource_type not in RESOURCE_TYPES:
            raise ServerError(
                f"Unknown resource type '{resource_type}'. "
                f"Choose from: {', '.join(RESOURCE_TYPES)}"
            )

        if resource_type == "misc":
            remote = _remote_path("misc", resource_name)
            if not self._client.check(remote):
                raise ServerError(
                    f"misc/{resource_name} not found on the {self._server} server."
                )
            if local_path is None:
                local_path = pathlib.Path.cwd()
            local_path = pathlib.Path(local_path)
            local_path.mkdir(parents=True, exist_ok=True)
            dest = local_path / pathlib.Path(resource_name).name
            log.info("Downloading %s ...", remote)
            self._client.download_sync(remote_path=remote, local_path=str(dest))
            log.info("Download complete: %s", dest)
            return dest

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

        if resource_type == "fit":
            self._inject_fit_files(resource_name, local_path)

        dest = local_path / resource_name
        log.info("Download complete: %s", dest)
        return dest

    def _inject_fit_files(self, fit_name: str, local_path: pathlib.Path) -> None:
        """Re-inject rge_matrix and runcard into the extracted fit directory."""
        registry = _read_registry(self._client)
        fit_meta = registry.get("fits", {}).get(fit_name, {})
        fit_dir = local_path / fit_name

        rge_remote = f"{RGE_MATRICES_REMOTE_DIR}/{fit_name}.pkl"
        if self._client.check(rge_remote):
            rge_rel = fit_meta.get("rge_path") or RGE_FILENAME
            dest = fit_dir / rge_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            log.info("Downloading rge_matrix ...")
            self._client.download_sync(remote_path=rge_remote, local_path=str(dest))

        runcard_remote = f"{RUNCARDS_REMOTE_DIR}/{fit_name}.yaml"
        if self._client.check(runcard_remote):
            runcard_rel = fit_meta.get("runcard_path") or str(RUNCARD_RELATIVE_PATH)
            dest = fit_dir / runcard_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            log.info("Downloading runcard ...")
            self._client.download_sync(remote_path=runcard_remote, local_path=str(dest))


def mkdir_misc(path: str, server: str | None = None) -> None:
    """Create a directory (and all parents) under misc/ on the server."""
    client = _get_client(server, need_write=True)
    full_path = f"misc/{path.strip('/')}"
    _ensure_remote_path(client, full_path)
    log.info("Created misc/%s.", path.strip("/"))


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
    """Rebuild the registry by inspecting all resources on the server.

    For fits, archives are downloaded to re-detect has_rge.
    For reports, only names are listed (no archive inspection needed).
    Existing 'created_at' and 'uploaded_by' values are preserved where known.
    Returns the new registry dict.
    """
    client = _get_client(server, need_write=True)
    old_registry = _read_registry(client)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    registry = _empty_registry()

    for fit_name in _list_resource_names(client, "fit"):
        try:
            rge_remote = f"{RGE_MATRICES_REMOTE_DIR}/{fit_name}.pkl"
            if client.check(rge_remote):
                log.info("Checking %s (new structure) ...", fit_name)
                has_rge = True
            else:
                log.info("Inspecting archive for %s (legacy) ...", fit_name)
                with tempfile.TemporaryDirectory(prefix="smefit_sync_") as tmpdir:
                    archive = pathlib.Path(tmpdir) / f"{fit_name}.tar.gz"
                    client.download_sync(
                        remote_path=_remote_path("fit", fit_name),
                        local_path=str(archive),
                    )
                    with tarfile.open(archive, "r:gz") as tar:
                        has_rge = any(
                            pathlib.Path(m.name).name == RGE_FILENAME
                            for m in tar.getmembers()
                        )
            old = old_registry["fits"].get(fit_name, {})
            registry["fits"][fit_name] = {
                "created_at": old.get("created_at", now),
                "has_rge": has_rge,
                "rge_path": old.get("rge_path"),
                "runcard_path": old.get("runcard_path"),
                "uploaded_by": old.get("uploaded_by"),
            }
        except Exception as exc:
            log.warning("Skipping fit '%s': %s", fit_name, exc)

    for report_name in _list_resource_names(client, "report"):
        try:
            log.info("Registering report %s ...", report_name)
            old = old_registry["reports"].get(report_name, {})
            registry["reports"][report_name] = {
                "created_at": old.get("created_at", now),
                "uploaded_by": old.get("uploaded_by"),
            }
        except Exception as exc:
            log.warning("Skipping report '%s': %s", report_name, exc)

    _write_registry(client, registry)
    log.info(
        "Registry synced: %d fit(s), %d report(s).",
        len(registry["fits"]),
        len(registry["reports"]),
    )
    return registry


# ---------------------------------------------------------------------------
# Project management
# ---------------------------------------------------------------------------


def list_projects(server: str | None = None) -> list:
    """Return the list of project names from the registry."""
    client = _get_client(server, need_write=False)
    registry = _read_registry(client)
    return sorted(registry.get("projects", []))


def add_project(project_name: str, server: str | None = None) -> None:
    """Add *project_name* to the project list in the registry."""
    client = _get_client(server, need_write=True)
    registry = _read_registry(client)
    projects = registry.setdefault("projects", [])
    if project_name in projects:
        raise ServerError(f"Project '{project_name}' already exists.")
    projects.append(project_name)
    registry["projects"] = sorted(projects)
    _write_registry(client, registry)
    log.info("Added project '%s'.", project_name)


def rename_project(old_name: str, new_name: str, server: str | None = None) -> None:
    """Rename a project in the registry and update all resources that reference it."""
    client = _get_client(server, need_write=True)
    registry = _read_registry(client)
    projects = registry.setdefault("projects", [])
    if old_name not in projects:
        raise ServerError(f"Project '{old_name}' not found.")
    if new_name in projects:
        raise ServerError(f"Project '{new_name}' already exists.")
    projects[projects.index(old_name)] = new_name
    registry["projects"] = sorted(projects)
    for section in ("fits", "reports"):
        for meta in registry.get(section, {}).values():
            if meta.get("project") == old_name:
                meta["project"] = new_name
    _write_registry(client, registry)
    log.info("Renamed project '%s' -> '%s'.", old_name, new_name)


def remove_project(project_name: str, server: str | None = None) -> None:
    """Remove *project_name* from the project list.

    Resources that referenced the removed project retain their metadata but the
    project label will no longer appear in the valid project list.
    """
    client = _get_client(server, need_write=True)
    registry = _read_registry(client)
    projects = registry.setdefault("projects", [])
    if project_name not in projects:
        raise ServerError(f"Project '{project_name}' not found.")
    projects.remove(project_name)
    registry["projects"] = sorted(projects)
    _write_registry(client, registry)
    log.info("Removed project '%s'.", project_name)
