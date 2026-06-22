"""
Move one or more resources from the SMEFiT server into the remote bin/ folder.

    smefit_rm RESOURCE_TYPE NAME [NAME ...]

RESOURCE_TYPE must be one of: fit, report, misc.
Resources are moved to bin/<type>/<name> and removed from the registry.
The deletion is recorded in bin/registry_bin.json with the date, deleter, and comment.
Prompts for confirmation before moving. Use -f to skip.
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
        "resource_type",
        choices=["fit", "report", "misc"],
        help="Type of resource.",
    )
    parser.add_argument(
        "names",
        nargs="+",
        metavar="NAME",
        help="Name(s) of the resource(s) to move to bin.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    parser.add_argument(
        "-m",
        "--message",
        default=None,
        help="Deletion comment stored in the bin registry (prompted interactively if omitted).",
    )
    parser.add_argument(
        "--server",
        choices=["public", "private"],
        default=None,
        help=(
            "Server to operate on. Auto-detects by default: private if credentials "
            "are configured, otherwise public."
        ),
    )
    args = parser.parse_args()

    from smefit.server_utils import ServerError, trash

    message = args.message
    if message is None:
        try:
            message = input("Deletion comment (press Enter to skip): ").strip() or None
        except EOFError:
            message = None

    if not args.force:
        targets = ", ".join(f"'{n}'" for n in args.names)
        answer = input(f"Move {targets} to bin on server? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    try:
        for name in args.names:
            trash(args.resource_type, name, server=args.server, message=message)
    except ServerError as e:
        log.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
