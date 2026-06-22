"""
Restore a resource from the bin back to its original location.

    smefit_restore RESOURCE_TYPE NAME

RESOURCE_TYPE must be one of: fit, report, misc.
The resource is moved from bin/ back to its original remote path and its
registry entry is reinstated. Prompts for confirmation. Use -f to skip.
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
        "name",
        metavar="NAME",
        help="Name of the resource to restore.",
    )
    parser.add_argument(
        "-f",
        "--force",
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

    from smefit.server_utils import ServerError, restore

    if not args.force:
        answer = (
            input(f"Restore '{args.name}' from bin to its original location? [y/N] ")
            .strip()
            .lower()
        )
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    try:
        restore(args.resource_type, args.name, server=args.server)
    except ServerError as e:
        log.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
