"""
SMEFiT server management.

    smefit_server {setup,storage,sync,tutorial} [options]

  setup    – configure server credentials
  storage  – show free/used space on the server
  sync     – rebuild the registry from scratch
  tutorial – list all available smefit commands with a short description
"""

import argparse
import getpass
import logging
import pathlib
import sys

import yaml
from reportengine import colors

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(colors.ColorHandler())

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
_USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(text, *codes):
    return ("".join(codes) + str(text) + "\033[0m") if _USE_COLOR else str(text)


_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"


def _header(s):
    return _c(s, _BOLD, _CYAN)


def _bold(s):
    return _c(s, _BOLD)


def _dim(s):
    return _c(s, _DIM)


def _green(s):
    return _c(s, _GREEN)


# ---------------------------------------------------------------------------
# setup helpers (from smefit_setup_server)
# ---------------------------------------------------------------------------
_CONFIG_PATH = pathlib.Path.home() / ".config" / "smefit" / "server.yaml"
_REQUIRED_KEYS = ("webdav_hostname", "webdav_login", "webdav_password")
_PROFILES = ("public", "private")


def _validate(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("Credentials file must be a YAML mapping.")
    unknown = set(config) - set(_PROFILES)
    if unknown:
        raise ValueError(
            f"Unknown profile(s): {', '.join(sorted(unknown))}. "
            f"Expected: {', '.join(_PROFILES)}"
        )
    if not config:
        raise ValueError("Credentials file contains no profiles.")
    for profile, values in config.items():
        if not isinstance(values, dict):
            raise ValueError(f"Profile '{profile}' must be a YAML mapping.")
        for key in _REQUIRED_KEYS:
            if key not in values:
                raise ValueError(f"Missing key '{key}' under '{profile}:'.")


def _prompt_profile(profile: str) -> dict | None:
    print(f"\n--- {profile.upper()} server ---")
    if input(f"Configure '{profile}' profile? [y/N] ").strip().lower() != "y":
        return None
    hostname = input("  webdav_hostname: ").strip()
    login = input("  webdav_login:    ").strip()
    password = getpass.getpass("  webdav_password: ")
    name = input("  name (shown in registry, optional): ").strip()
    result = {
        "webdav_hostname": hostname,
        "webdav_login": login,
        "webdav_password": password,
    }
    if name:
        result["name"] = name
    return result


def _write_config(config: dict) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    _CONFIG_PATH.chmod(0o600)
    log.info("Credentials written to %s", _CONFIG_PATH)


# ---------------------------------------------------------------------------
# tutorial content
# ---------------------------------------------------------------------------
_COMMANDS = [
    ("smefit", "Run a fit from a runcard YAML file."),
    (
        "smefit_ls",
        "List server resources — fits, reports, registry, rge, misc.\n"
        "                        Use --project NAME to filter by project.",
    ),
    ("smefit_upload", "Upload a fit, report, or misc file to the server."),
    ("smefit_get", "Download a fit, report, rge matrix, or misc file."),
    ("smefit_mv", "Rename a resource on the server."),
    ("smefit_rm", "Delete one or more resources from the server."),
    ("smefit_mkdir", "Create a directory under misc/ on the server."),
    ("smefit_manage_project", "Manage the project list: add / rename / remove / list."),
    ("view_report", "Download (if needed) and open a report in the browser."),
    (
        "smefit_server",
        "Server management: setup credentials, check storage, sync registry.",
    ),
]


def _print_tutorial() -> None:
    cmd_w = max(len(cmd) for cmd, _ in _COMMANDS)
    print(f"\n  {_header('SMEFiT command reference')}\n")
    for cmd, desc in _COMMANDS:
        lines = desc.split("\n")
        print(f"  {_bold(cmd):{cmd_w + 10}}{lines[0]}")
        for extra in lines[1:]:
            print(f"  {' ' * (cmd_w + 10)}{_dim(extra)}")
    print(f"\n  {_dim('Run any command with --help for full usage.')}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # setup
    p_setup = sub.add_parser("setup", help="Configure server credentials.")
    p_setup.add_argument(
        "credentials_yaml",
        nargs="?",
        default=None,
        metavar="CREDENTIALS_YAML",
        help="Path to a YAML credentials file. Prompts interactively if omitted.",
    )
    p_setup.add_argument(
        "--force",
        action="store_true",
        help=f"Overwrite {_CONFIG_PATH} if it already exists.",
    )

    # storage
    p_storage = sub.add_parser("storage", help="Show free/used space on the server.")
    p_storage.add_argument(
        "--server",
        choices=["public", "private"],
        default=None,
        help="Server to query (auto-detected by default).",
    )

    # sync
    p_sync = sub.add_parser("sync", help="Rebuild the registry from scratch.")
    p_sync.add_argument(
        "--server",
        choices=["public", "private"],
        default=None,
        help="Server to sync (auto-detected by default).",
    )

    # tutorial
    sub.add_parser("tutorial", help="List all available smefit commands.")

    args = parser.parse_args()

    try:
        if args.command == "setup":
            if _CONFIG_PATH.exists() and not args.force:
                log.error("%s already exists. Use --force to overwrite.", _CONFIG_PATH)
                sys.exit(1)
            if args.credentials_yaml is not None:
                src = pathlib.Path(args.credentials_yaml)
                if not src.exists():
                    log.error("File not found: %s", src)
                    sys.exit(1)
                with open(src) as f:
                    config = yaml.safe_load(f)
                try:
                    _validate(config)
                except ValueError as e:
                    log.error("%s", e)
                    sys.exit(1)
            else:
                config = {}
                for profile in _PROFILES:
                    result = _prompt_profile(profile)
                    if result is not None:
                        config[profile] = result
                if not config:
                    log.error("No profiles configured. Exiting without writing.")
                    sys.exit(1)
            _write_config(config)

        elif args.command == "storage":
            from smefit.server_utils import ServerError, get_free_space

            try:
                _TOTAL_GB = 1000.0
                free_bytes = get_free_space(server=args.server)
                free_gb = free_bytes / (1024**3)
                used_gb = _TOTAL_GB - free_gb
                pct = used_gb / _TOTAL_GB
                bar_width = 30
                filled = round(pct * bar_width)
                bar = "█" * filled + "░" * (bar_width - filled)
                print(f"\n  {_header('Storage')}")
                print(
                    f"  {bar}  {_bold(f'{used_gb:.1f}')} / {_TOTAL_GB:.0f} GB used"
                    f"  ({_green(f'{free_gb:.1f} GB free')})\n"
                )
            except ServerError as e:
                log.error("%s", e)
                sys.exit(1)

        elif args.command == "sync":
            from smefit.server_utils import ServerError, sync_registry

            try:
                sync_registry(server=args.server)
            except ServerError as e:
                log.error("%s", e)
                sys.exit(1)

        elif args.command == "tutorial":
            _print_tutorial()

    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
