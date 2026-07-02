"""
Upload a resource to the SMEFiT server.

    smefit_upload RESOURCE_TYPE RESOURCE_NAME [LOCAL_PATH]

RESOURCE_TYPE must be one of: fit, report, misc.
RESOURCE_NAME is the name of the resource on the server.
LOCAL_PATH is the local path to the resource (defaults to ./<RESOURCE_NAME>).

Use --local to move the resource into smefit_results/ instead of uploading to
the server. The local registry.json is updated with the same metadata fields.
"""

import argparse
import logging
import sys

from reportengine import colors

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(colors.ColorHandler())


def _prompt_project(projects: list) -> str | None:
    """Interactively ask the user to pick a project from *projects*.

    Returns the chosen project name, or None if the user skips.
    """
    if not projects:
        return None
    print("Available projects:")
    for i, p in enumerate(projects, 1):
        print(f"  {i}. {p}")
    raw = input("Project number (press Enter to skip): ").strip()
    if not raw:
        return None
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(projects):
            return projects[idx]
        print("Invalid number — skipping project.", file=sys.stderr)
    except ValueError:
        print("Invalid input — skipping project.", file=sys.stderr)
    return None


def _local_upload(args, message: str | None) -> None:
    """Move a resource folder into smefit_results/ and update the local registry."""
    import datetime
    import getpass
    import pathlib
    import shutil

    from smefit.paths import get_local_results_dir
    from smefit.server_utils import (
        _RESOURCE_MARKERS,
        RGE_FILENAME,
        RUNCARD_RELATIVE_PATH,
        _detect_has_rge,
        _read_local_registry,
        _write_local_registry,
        list_local_projects,
    )

    if args.resource_type not in ("fit", "report"):
        log.error("--local only supports resource types 'fit' and 'report'.")
        sys.exit(1)

    src = pathlib.Path(args.local_path if args.local_path else args.resource_name)
    if not src.exists():
        log.error("Source path does not exist: %s", src)
        sys.exit(1)

    if args.resource_type in _RESOURCE_MARKERS:
        marker = _RESOURCE_MARKERS[args.resource_type]
        if not any(src.rglob(marker)):
            log.error(
                "'%s' does not look like a valid %s: no %s found.",
                src,
                args.resource_type,
                marker,
            )
            sys.exit(1)

    local_results_dir = get_local_results_dir()
    if local_results_dir is None:
        log.error(
            "smefit_results is not configured. Run 'smefit_setup_paths' to set it up."
        )
        sys.exit(1)

    project = args.project
    if project is None:
        try:
            projects = list_local_projects(local_results_dir)
            project = _prompt_project(projects)
        except EOFError:
            pass

    subdir = "fits" if args.resource_type == "fit" else "reports"
    dest = local_results_dir / subdir / args.resource_name

    if dest.exists():
        if not args.force:
            log.error(
                "'%s' already exists in %s. Use --force to overwrite.",
                args.resource_name,
                dest.parent,
            )
            sys.exit(1)
        shutil.rmtree(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    log.info("Moved '%s' -> %s", src, dest)

    entry = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "uploaded_by": getpass.getuser(),
    }
    if args.resource_type == "fit":
        has_rge = _detect_has_rge(dest)
        rge_file = next(dest.rglob(RGE_FILENAME), None)
        entry["has_rge"] = has_rge
        entry["rge_path"] = str(rge_file.relative_to(dest)) if rge_file else None
        runcard_candidate = dest / RUNCARD_RELATIVE_PATH
        entry["runcard_path"] = (
            str(RUNCARD_RELATIVE_PATH) if runcard_candidate.exists() else None
        )
    if message:
        entry["comment"] = message
    if project:
        entry["project"] = project

    local_reg = _read_local_registry(local_results_dir)
    local_reg[f"{args.resource_type}s"][args.resource_name] = entry
    _write_local_registry(local_results_dir, local_reg)
    log.info("Local registry updated: %s", local_results_dir / "registry.json")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "resource_type",
        choices=["fit", "report", "misc"],
        help="Type of resource to upload.",
    )
    parser.add_argument("resource_name", help="Name of the resource.")
    parser.add_argument(
        "local_path",
        nargs="?",
        default=None,
        help="Local path to the resource. Defaults to ./<resource_name>.",
    )
    parser.add_argument(
        "--server",
        choices=["public", "private"],
        default=None,
        help=(
            "Server to upload to. Defaults to private if credentials are configured, "
            "otherwise the public server (requires write credentials)."
        ),
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite if the resource already exists on the server.",
    )
    parser.add_argument(
        "-m",
        "--message",
        default=None,
        help="Comment stored in the registry (prompted interactively if omitted).",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Project name to tag this resource with (skips the interactive prompt).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "Move the resource folder into smefit_results/ instead of uploading to "
            "the server. Updates the local registry.json. Only valid for fit and report."
        ),
    )
    args = parser.parse_args()

    message = args.message
    if message is None:
        try:
            message = input("Comment (press Enter to skip): ").strip() or None
        except EOFError:
            message = None

    if args.local:
        _local_upload(args, message)
        return

    from smefit.server_utils import ServerError, Uploader, list_projects

    project = args.project
    if args.resource_type in ("fit", "report"):
        try:
            projects = list_projects(server=args.server)
            if project is not None:
                if project not in projects:
                    print(
                        f"Project '{project}' does not exist. "
                        "Use 'smefit_manage_project add <name>' to create a new project.",
                        file=sys.stderr,
                    )
                    project = _prompt_project(projects)
            else:
                project = _prompt_project(projects)
        except ServerError:
            pass
        except EOFError:
            pass

    try:
        uploader = Uploader(server=args.server)
        uploader.upload(
            args.resource_type,
            args.resource_name,
            args.local_path,
            args.force,
            message,
            project,
        )
    except ServerError as e:
        log.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
