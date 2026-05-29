"""
Manage the project list on the SMEFiT server.

Projects are metadata labels that can be attached to fits and reports at upload
time. They are stored in the registry and never alter the resources themselves.

    smefit_project list
    smefit_project add    PROJECT_NAME
    smefit_project rename OLD_NAME NEW_NAME
    smefit_project remove PROJECT_NAME
"""

import argparse
import logging
import sys

from reportengine import colors

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(colors.ColorHandler())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--server",
        choices=["public", "private"],
        default=None,
        help="Server to operate on (auto-detected by default).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List available projects.")

    p_add = sub.add_parser("add", help="Add a new project.")
    p_add.add_argument("project_name", help="Name of the project to add.")

    p_rename = sub.add_parser("rename", help="Rename a project.")
    p_rename.add_argument("old_name", help="Current project name.")
    p_rename.add_argument("new_name", help="New project name.")

    p_remove = sub.add_parser("remove", help="Remove a project.")
    p_remove.add_argument("project_name", help="Name of the project to remove.")

    args = parser.parse_args()

    from smefit.server_utils import (
        ServerError,
        add_project,
        list_projects,
        remove_project,
        rename_project,
    )

    try:
        if args.command == "list":
            projects = list_projects(server=args.server)
            if projects:
                for p in projects:
                    print(p)
            else:
                print("(no projects defined)")

        elif args.command == "add":
            add_project(args.project_name, server=args.server)

        elif args.command == "rename":
            rename_project(args.old_name, args.new_name, server=args.server)

        elif args.command == "remove":
            remove_project(args.project_name, server=args.server)

    except ServerError as e:
        log.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
