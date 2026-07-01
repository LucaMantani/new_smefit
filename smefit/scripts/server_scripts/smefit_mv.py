"""
Rename a resource and/or update its comment on the SMEFiT server.

    smefit_mv RESOURCE_TYPE OLD_NAME [NEW_NAME] [--comment "new comment"]

RESOURCE_TYPE must be one of: fit, report, misc.

At least one of NEW_NAME or --comment must be supplied.
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
    parser.add_argument("old_name", help="Current name of the resource on the server.")
    parser.add_argument(
        "new_name",
        nargs="?",
        default=None,
        help="New name for the resource (optional if --comment is given).",
    )
    parser.add_argument(
        "--comment",
        "-c",
        default=None,
        help="New comment to associate with the resource.",
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

    if args.new_name is None and args.comment is None:
        parser.error("Provide a new name, --comment, or both.")

    from smefit.server_utils import ServerError, rename

    try:
        rename(
            args.resource_type,
            args.old_name,
            args.new_name,
            server=args.server,
            comment=args.comment,
        )
    except ServerError as e:
        log.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
