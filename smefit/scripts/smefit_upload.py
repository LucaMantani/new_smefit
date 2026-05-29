"""
Upload a resource to the SMEFiT server.

    smefit_upload RESOURCE_TYPE RESOURCE_NAME [LOCAL_PATH]

RESOURCE_TYPE must be one of: fit, report, rge.
RESOURCE_NAME is the name of the resource on the server.
LOCAL_PATH is the local path to the resource (defaults to ./<RESOURCE_NAME>).
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
    args = parser.parse_args()

    message = args.message
    if message is None:
        try:
            message = input("Comment (press Enter to skip): ").strip() or None
        except EOFError:
            message = None

    from smefit.server_utils import ServerError, Uploader, list_projects

    project = args.project
    if project is None and args.resource_type in ("fit", "report"):
        try:
            projects = list_projects(server=args.server)
            project = _prompt_project(projects)
        except ServerError:
            # If we can't read the registry just skip the project prompt
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
