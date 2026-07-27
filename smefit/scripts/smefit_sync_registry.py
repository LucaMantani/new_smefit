"""
Rebuild the SMEFiT fit registry on the server.

    smefit_sync_registry

Downloads and inspects all fit archives to regenerate fits/registry.json.
Existing 'created_at' values are preserved where known.
Requires write credentials.
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
        "--server",
        choices=["public", "private"],
        default=None,
        help="Server to sync. Auto-detects by default.",
    )
    args = parser.parse_args()

    from smefit.server_utils import ServerError, sync_registry

    try:
        sync_registry(server=args.server)
    except ServerError as e:
        log.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
