"""
Download and open an SMEFiT report in a browser.

    view_report REPORT_NAME [LOCAL_PATH]

REPORT_NAME is the name of the report on the server.
LOCAL_PATH is the directory where the report will be saved (defaults to .).

If the report is already present at LOCAL_PATH/REPORT_NAME, it is opened
directly without re-downloading.
"""

import argparse
import logging
import pathlib
import sys
import webbrowser

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
        default="public",
        help="Server to download the report from (default: public).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Download the report without opening a browser.",
    )
    args = parser.parse_args()

    if args.local_path is None:
        local_path = pathlib.Path.cwd()
    else:
        local_path = pathlib.Path(args.local_path)

    report_dir = local_path / args.report_name
    index = report_dir / "index.html"

    if not report_dir.exists():
        from smefit.server_utils import Downloader, ServerError

        try:
            downloader = Downloader(server=args.server)
            downloader.download("report", args.report_name, local_path)
        except ServerError as e:
            log.error("%s", e)
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nInterrupted by user.", file=sys.stderr)
            sys.exit(1)
    else:
        log.info("Report already present at %s, skipping download.", report_dir)

    if not index.exists():
        log.error("No index.html found in %s.", report_dir)
        sys.exit(1)

    if not args.no_browser:
        url = index.resolve().as_uri()
        log.info("Opening %s in browser.", url)
        webbrowser.open(url)


if __name__ == "__main__":
    main()
