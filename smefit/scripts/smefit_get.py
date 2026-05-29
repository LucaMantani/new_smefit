"""
Download a resource from the SMEFiT server.

    smefit_get RESOURCE_TYPE RESOURCE_NAME [LOCAL_PATH]

RESOURCE_TYPE must be one of: fit, report, rge.
  fit/report  – downloads the full resource directory.
  rge         – downloads only the rge_matrix.pkl from the named fit.

LOCAL_PATH is the directory where the resource will be saved (defaults to .).

Use --view with 'report' to open the report in a browser after downloading.
Use 'smefit_ls RESOURCE_TYPE' to see what is available on the server.
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
        help="Name of the resource (or fit name when resource_type is rge).",
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
        default=None,
        help=(
            "Server to download from. Defaults to private if credentials are configured, "
            "otherwise the public server (no setup required)."
        ),
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open the report in a browser after downloading. Only valid for resource_type 'report'.",
    )
    args = parser.parse_args()

    if args.view and args.resource_type != "report":
        parser.error("--view is only valid for resource_type 'report'.")

    from smefit.server_utils import (
        Downloader,
        ServerError,
        download_and_view_report,
        download_rge,
    )

    try:
        if args.resource_type == "rge":
            download_rge(
                args.resource_name, local_path=args.local_path, server=args.server
            )
        elif args.view:
            download_and_view_report(
                args.resource_name, local_path=args.local_path, server=args.server
            )
        else:
            downloader = Downloader(server=args.server)
            downloader.download(args.resource_type, args.resource_name, args.local_path)
    except ServerError as e:
        log.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
