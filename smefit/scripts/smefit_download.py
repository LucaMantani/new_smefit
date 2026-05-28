"""
Download a resource from the SMEFiT server.

    smefit_download RESOURCE_TYPE RESOURCE_NAME [LOCAL_PATH]

RESOURCE_TYPE must be one of: fit, report, rge.
RESOURCE_NAME is the name of the resource on the server.
LOCAL_PATH is the directory where the resource will be saved (defaults to .).

Use --list to see available resources of a given type.
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
        help="Type of resource to download.",
    )
    parser.add_argument(
        "resource_name",
        nargs="?",
        default=None,
        help="Name of the resource to download. Omit when using --list.",
    )
    parser.add_argument(
        "local_path",
        nargs="?",
        default=None,
        help="Local directory where the resource will be saved. Defaults to ./.",
    )
    parser.add_argument(
        "--server",
        choices=["public", "private"],
        default="public",
        help="Server to download from (default: public).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available resources of RESOURCE_TYPE on the server and exit.",
    )
    args = parser.parse_args()

    from smefit.server_utils import Downloader, ServerError

    try:
        downloader = Downloader(server=args.server)

        if args.list:
            resources = downloader.list_resources(args.resource_type)
            if resources:
                print(f"Available {args.resource_type}s on server:")
                for r in resources:
                    print(f"  {r}")
            else:
                print(f"No {args.resource_type}s found on server.")
            return

        if args.resource_name is None:
            parser.error("RESOURCE_NAME is required unless --list is used.")

        downloader.download(args.resource_type, args.resource_name, args.local_path)

    except ServerError as e:
        log.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
