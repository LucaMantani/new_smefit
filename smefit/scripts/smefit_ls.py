"""
List resources available on the SMEFiT server.

    smefit_ls RESOURCE_TYPE [--server public|private]

RESOURCE_TYPE must be one of: fit, report, rge.
The server is auto-detected: private if credentials are configured, else public.
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
        choices=["fit", "report", "rge"],
        help="Type of resource to list.",
    )
    parser.add_argument(
        "--server",
        choices=["public", "private"],
        default=None,
        help=(
            "Server to query. Auto-detects by default: private if credentials "
            "are configured, otherwise public."
        ),
    )
    args = parser.parse_args()

    from smefit.server_utils import Downloader, ServerError

    try:
        downloader = Downloader(server=args.server)
        resources = downloader.list_resources(args.resource_type)
        if resources:
            print(f"Available {args.resource_type}s on server:")
            for r in resources:
                print(f"  {r}")
        else:
            print(f"No {args.resource_type}s found on server.")
    except ServerError as e:
        log.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
