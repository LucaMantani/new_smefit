"""
List resources available on the SMEFiT server.

    smefit_ls [RESOURCE_TYPE] [--server public|private]

RESOURCE_TYPE must be one of: fit, report, rge, registry (default: registry).
  fit/report  – lists available resources of that type.
  rge         – lists fits that have an rge_matrix.pkl, read from the registry.
  registry    – displays the full registry with all tracked metadata.

The server is auto-detected: private if credentials are configured, else public.
"""

import argparse
import logging
import sys

from reportengine import colors

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(colors.ColorHandler())


def _comment_str(meta: dict) -> str:
    comment = meta.get("comment")
    if not comment:
        return ""
    return f'  "{comment[:50]}{"..." if len(comment) > 50 else ""}"'


def _print_fits(fits: dict) -> None:
    if not fits:
        print("  (none)")
        return
    width = max(len(name) for name in fits)
    for name, meta in sorted(fits.items()):
        date = meta.get("created_at", "?")[:10]
        rge = "yes" if meta.get("has_rge") else "no"
        uploader = meta.get("uploaded_by")
        suffix = f"  by={uploader}" if uploader else ""
        print(f"  {name:<{width}}  {date}  rge={rge}{suffix}{_comment_str(meta)}")


def _print_reports(reports: dict) -> None:
    if not reports:
        print("  (none)")
        return
    width = max(len(name) for name in reports)
    for name, meta in sorted(reports.items()):
        date = meta.get("created_at", "?")[:10]
        uploader = meta.get("uploaded_by")
        suffix = f"  by={uploader}" if uploader else ""
        print(f"  {name:<{width}}  {date}{suffix}{_comment_str(meta)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "resource_type",
        choices=["fit", "report", "rge", "registry", "misc"],
        nargs="?",
        default="registry",
        help="Type of resource to list (default: registry).",
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
        if args.resource_type == "registry":
            downloader = Downloader(server=args.server)
            registry = downloader.get_registry()
            fits = registry.get("fits", {})
            reports = registry.get("reports", {})
            print(f"Fits ({len(fits)}):")
            _print_fits(fits)
            print(f"\nReports ({len(reports)}):")
            _print_reports(reports)

        elif args.resource_type == "rge":
            downloader = Downloader(server=args.server)
            registry = downloader.get_registry()
            rge_fits = {
                name: meta
                for name, meta in registry.get("fits", {}).items()
                if meta.get("has_rge")
            }
            print(f"Fits with rge_matrix.pkl ({len(rge_fits)}):")
            _print_fits(rge_fits) if rge_fits else print("  (none)")

        elif args.resource_type == "fit":
            downloader = Downloader(server=args.server)
            resources = downloader.list_resources("fit")
            registry = downloader.get_registry()
            if resources:
                width = max(len(r) for r in resources)
                print("Available fits on server:")
                for r in resources:
                    meta = registry["fits"].get(r)
                    if meta:
                        date = meta.get("created_at", "?")[:10]
                        rge = "yes" if meta.get("has_rge") else "no"
                        print(f"  {r:<{width}}  {date}  rge={rge}")
                    else:
                        print(f"  {r:<{width}}  (not in registry)")
            else:
                print("No fits found on server.")

        elif args.resource_type == "misc":
            downloader = Downloader(server=args.server)
            resources = downloader.list_resources("misc")
            if resources:
                print("Contents of misc/ on server:")
                for r in resources:
                    print(f"  {r}")
            else:
                print("misc/ is empty or does not exist on server.")

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
