"""
Download and open an SMEFiT report in a browser.

    view_report REPORT_NAME [LOCAL_PATH]

Shortcut for: smefit_download report REPORT_NAME [LOCAL_PATH] --view [--server ...]

If the report is already present locally it is opened directly without re-downloading.
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
    parser.add_argument("report_name", help="Name of the report.")
    parser.add_argument(
        "local_path",
        nargs="?",
        default=None,
        help="Local directory for the report. Defaults to ./.",
    )
    parser.add_argument(
        "--server",
        choices=["public", "private"],
        default=None,
        help=(
            "Server to download the report from. Defaults to private if credentials "
            "are configured, otherwise the public server (no setup required)."
        ),
    )
    args = parser.parse_args()

    from smefit.server_utils import ServerError, download_and_view_report

    try:
        download_and_view_report(
            args.report_name, local_path=args.local_path, server=args.server
        )
    except ServerError as e:
        log.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
