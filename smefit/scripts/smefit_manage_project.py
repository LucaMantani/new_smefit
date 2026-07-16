"""
Manage the project list on the SMEFiT server or in the local registry.

Projects are metadata labels that can be attached to fits and reports at upload
time. They are stored in the registry and never alter the resources themselves.

    smefit_manage_project list
    smefit_manage_project add    PROJECT_NAME
    smefit_manage_project rename OLD_NAME NEW_NAME
    smefit_manage_project remove PROJECT_NAME

Add --local to operate on the local smefit_results/registry.json instead.
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
    parser.add_argument(
        "--local",
        action="store_true",
        help="Operate on the local smefit_results/registry.json instead of the server.",
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
        add_local_project,
        add_project,
        list_local_projects,
        list_projects,
        remove_local_project,
        remove_project,
        rename_local_project,
        rename_project,
    )

    try:
        if args.local:
            from smefit.paths import get_local_results_dir

            local_results_dir = get_local_results_dir()
            if local_results_dir is None:
                log.error(
                    "smefit_results is not configured. Run 'smefit_setup_local' to set it up."
                )
                sys.exit(1)

            if args.command == "list":
                projects = list_local_projects(local_results_dir)
                if projects:
                    for p in projects:
                        print(p)
                else:
                    print("(no local projects defined)")

            elif args.command == "add":
                add_local_project(args.project_name, local_results_dir)

            elif args.command == "rename":
                rename_local_project(args.old_name, args.new_name, local_results_dir)

            elif args.command == "remove":
                remove_local_project(args.project_name, local_results_dir)

        else:
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
