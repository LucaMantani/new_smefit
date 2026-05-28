"""
Upload a resource to the SMEFiT server.

    smefit_upload RESOURCE_TYPE RESOURCE_NAME [LOCAL_PATH]

RESOURCE_TYPE must be one of: fit, report, rge.
RESOURCE_NAME is the name of the resource on the server.
LOCAL_PATH is the local path to the resource (defaults to ./<RESOURCE_NAME>).
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
        help="Type of resource to upload.",
    )
    parser.add_argument("resource_name", help="Name of the resource.")
    parser.add_argument(
        "local_path",
        nargs="?",
        default=None,
        help="Local path to the resource. Defaults to ./<resource_name>.",
    )
    parser.add_argument(
        "--server",
        choices=["public", "private"],
        default=None,
        help=(
            "Server to upload to. Defaults to private if credentials are configured, "
            "otherwise the public server (requires write credentials)."
        ),
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite if the resource already exists on the server.",
    )
    args = parser.parse_args()

    from smefit.server_utils import ServerError, Uploader

    try:
        uploader = Uploader(server=args.server)
        uploader.upload(args.resource_type, args.resource_name, args.local_path, args.force)
    except ServerError as e:
        log.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
