"""
Delete one or more resources from the SMEFiT server.

    smefit_rm RESOURCE_TYPE NAME [NAME ...]

RESOURCE_TYPE must be one of: fit, report, rge.
Prompts for confirmation before deleting. Use -f to skip.
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
        choices=["fit", "report"],
        help="Type of resource.",
    )
    parser.add_argument(
        "names",
        nargs="+",
        metavar="NAME",
        help="Name(s) of the resource(s) to delete.",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Skip confirmation prompt.",
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

    from smefit.server_utils import ServerError, delete

    if not args.force:
        targets = ", ".join(f"'{n}'" for n in args.names)
        answer = input(f"Delete {targets} from server? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    try:
        for name in args.names:
            delete(args.resource_type, name, server=args.server)
    except ServerError as e:
        log.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
