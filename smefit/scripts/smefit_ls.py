"""
List resources available on the SMEFiT server.

    smefit_ls RESOURCE_TYPE [--server public|private]

RESOURCE_TYPE must be one of: fit, report, rge.
  fit/report  – lists available resources of that type.
  rge         – lists fits that contain an rge_matrix.pkl file.
                Note: this downloads each fit archive to inspect it and may be slow.

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

    from smefit.server_utils import Downloader, ServerError, list_fits_with_rge

    try:
        if args.resource_type == "rge":
            resources = list_fits_with_rge(server=args.server)
            label = "fits with rge_matrix.pkl"
        else:
            downloader = Downloader(server=args.server)
            resources = downloader.list_resources(args.resource_type)
            label = f"{args.resource_type}s"

        if resources:
            print(f"Available {label} on server:")
            for r in resources:
                print(f"  {r}")
        else:
            print(f"No {label} found on server.")
    except ServerError as e:
        log.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
