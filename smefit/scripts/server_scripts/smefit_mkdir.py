"""
Create a directory under misc/ on the SMEFiT server.

    smefit_mkdir PATH

PATH is relative to misc/ on the server.
Intermediate directories are created automatically.

Examples:
    smefit_mkdir results/2026
    smefit_mkdir matrices/run42/output
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
        "path",
        help="Directory path to create inside misc/ (e.g. 'results/2026').",
    )
    parser.add_argument(
        "--server",
        choices=["public", "private"],
        default=None,
        help="Server to operate on. Auto-detects by default.",
    )
    args = parser.parse_args()

    from smefit.server_utils import ServerError, mkdir_misc

    try:
        mkdir_misc(args.path, server=args.server)
    except ServerError as e:
        log.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
