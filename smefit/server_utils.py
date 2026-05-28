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
    rge     -> rge/
"""

import logging
import pathlib
import tarfile
import tempfile

import yaml

log = logging.getLogger(__name__)

RESOURCE_TYPES = ["fit", "report", "rge"]
SERVERS = ["public", "private"]

_REMOTE_DIRS = {
    "fit": "fits",
    "report": "reports",
    "rge": "rge",
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
        raise ServerError(f"Unknown server '{server}'. Choose from: {', '.join(SERVERS)}")

    profile = config[server]
    for key in ("webdav_hostname", "webdav_login", "webdav_password"):
        if key not in profile:
            raise ServerError(f"Missing key '{key}' under '{server}:' in {CONFIG_PATH}")
    return Client(profile)


def _remote_path(resource_type: str, resource_name: str) -> str:
    """Return the remote WebDAV path for a given resource."""
    remote_dir = _REMOTE_DIRS[resource_type]
    if resource_type == "rge":
        return f"{remote_dir}/{resource_name}"
    return f"{remote_dir}/{resource_name}.tar.gz"


def _compress(source: pathlib.Path, archive_path: pathlib.Path) -> None:
    log.info("Compressing %s ...", source)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source, arcname=source.name)


def _extract(archive_path: pathlib.Path, dest: pathlib.Path) -> None:
    log.info("Extracting to %s ...", dest)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dest)


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

        if resource_type == "rge":
            self._upload_file(local_path, remote)
        else:
            self._upload_archive(local_path, resource_name, remote)

    def _upload_file(self, local_path: pathlib.Path, remote: str) -> None:
        log.info("Uploading %s -> %s ...", local_path, remote)
        self._client.upload_sync(remote_path=remote, local_path=str(local_path))
        log.info("Upload complete.")

    def _upload_archive(
        self, local_path: pathlib.Path, resource_name: str, remote: str
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="smefit_upload_") as tmpdir:
            archive = pathlib.Path(tmpdir) / f"{resource_name}.tar.gz"
            _compress(local_path, archive)
            log.info("Uploading %s -> %s ...", archive.name, remote)
            self._client.upload_sync(remote_path=remote, local_path=str(archive))
        log.info("Upload complete.")


class Downloader:
    """Download resources from a server. Uses bundled credentials for the public server."""

    def __init__(self, server: str | None = None):
        if server is not None and server not in SERVERS:
            raise ServerError(
                f"Unknown server '{server}'. Choose from: {', '.join(SERVERS)}"
            )
        self._client = _get_client(server, need_write=False)
        self._server = server or "auto"

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
        entries = self._client.list(remote_dir)
        names = []
        for e in entries:
            e = e.rstrip("/")
            if e == remote_dir or e == "":
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

        if resource_type == "rge":
            return self._download_file(remote, resource_name, local_path)
        else:
            return self._download_archive(remote, resource_name, local_path)

    def _download_file(
        self, remote: str, resource_name: str, local_path: pathlib.Path
    ) -> pathlib.Path:
        dest = local_path / resource_name
        log.info("Downloading %s -> %s ...", remote, dest)
        self._client.download_sync(remote_path=remote, local_path=str(dest))
        log.info("Download complete: %s", dest)
        return dest

    def _download_archive(
        self, remote: str, resource_name: str, local_path: pathlib.Path
    ) -> pathlib.Path:
        with tempfile.TemporaryDirectory(prefix="smefit_download_") as tmpdir:
            archive = pathlib.Path(tmpdir) / f"{resource_name}.tar.gz"
            log.info("Downloading %s ...", remote)
            self._client.download_sync(remote_path=remote, local_path=str(archive))
            _extract(archive, local_path)
        dest = local_path / resource_name
        log.info("Download complete: %s", dest)
        return dest
