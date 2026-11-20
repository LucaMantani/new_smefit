"""
SMEFiT server management.

    smefit_server {setup,storage,sync,tutorial} [options]

  setup    – configure server credentials
  storage  – show free/used space on the server
  sync     – rebuild the registry from scratch
  tutorial – list all available smefit commands with a short description
"""

import argparse
import logging
import sys

from reportengine import colors

from smefit.scripts import smefit_setup_server

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
# tutorial content
# ---------------------------------------------------------------------------
_COMMANDS = [
    ("smefit", "Run a fit from a runcard YAML file."),
    (
        "smefit_ls",
        "List server resources — fit, report, rge, registry, misc, bin.\n"
        "Use --project NAME to filter by project.",
    ),
    ("smefit_upload", "Upload a fit, report, or misc file to the server."),
    ("smefit_get", "Download a fit, report, rge matrix, or misc file."),
    (
        "smefit_mv",
        "Rename a resource and/or update its comment on the server.\n"
        "NEW_NAME is optional when --comment / -c is given.",
    ),
    (
        "smefit_rm",
        "Move one or more resources to bin/ on the server.\n"
        "Prompts for a deletion comment. Use -f to skip confirmation.",
    ),
    ("smefit_restore", "Restore a resource from bin/ back to its original location."),
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
    smefit_setup_server.add_arguments(p_setup)

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
            smefit_setup_server.run(args)

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
