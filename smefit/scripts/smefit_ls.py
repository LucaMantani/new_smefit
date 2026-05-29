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

# ---------------------------------------------------------------------------
# Terminal color helpers (disabled when stdout is not a TTY)
# ---------------------------------------------------------------------------
_USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(text, *codes):
    return ("".join(codes) + str(text) + "\033[0m") if _USE_COLOR else str(text)


_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_CYAN  = "\033[36m"
_GREEN = "\033[32m"


def _header(title: str) -> str:
    return _c(title, _BOLD, _CYAN)


def _name(s: str) -> str:
    return _c(s, _BOLD)


def _date(s: str) -> str:
    return _c(s, _DIM)


def _rge(val: bool) -> str:
    return _c("rge=yes", _GREEN) if val else _c("rge=no", _DIM)


def _by(uploader: str) -> str:
    return _c("by=", _DIM) + uploader


def _comment(text: str) -> str:
    truncated = text[:50] + ("..." if len(text) > 50 else "")
    return _c(f'"{truncated}"', _DIM)


# ---------------------------------------------------------------------------
# Row formatters
# ---------------------------------------------------------------------------

def _print_fits(fits: dict) -> None:
    if not fits:
        print(_c("  (none)", _DIM))
        return
    width = max(len(n) for n in fits)
    for name, meta in sorted(fits.items()):
        date = _date(meta.get("created_at", "?")[:10])
        rge  = _rge(meta.get("has_rge", False))
        parts = [f"  {_name(name):<{width + 8}}  {date}  {rge}"]
        if meta.get("uploaded_by"):
            parts.append(_by(meta["uploaded_by"]))
        c = meta.get("comment")
        if c:
            parts.append(_comment(c))
        print("  ".join(parts))


def _print_reports(reports: dict) -> None:
    if not reports:
        print(_c("  (none)", _DIM))
        return
    width = max(len(n) for n in reports)
    for name, meta in sorted(reports.items()):
        date = _date(meta.get("created_at", "?")[:10])
        parts = [f"  {_name(name):<{width + 8}}  {date}"]
        if meta.get("uploaded_by"):
            parts.append(_by(meta["uploaded_by"]))
        c = meta.get("comment")
        if c:
            parts.append(_comment(c))
        print("  ".join(parts))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
            fits    = registry.get("fits", {})
            reports = registry.get("reports", {})
            print(_header(f"Fits ({len(fits)})"))
            _print_fits(fits)
            print()
            print(_header(f"Reports ({len(reports)})"))
            _print_reports(reports)

        elif args.resource_type == "rge":
            downloader = Downloader(server=args.server)
            rge_fits = {
                name: meta
                for name, meta in downloader.get_registry().get("fits", {}).items()
                if meta.get("has_rge")
            }
            print(_header(f"Fits with rge_matrix.pkl ({len(rge_fits)})"))
            _print_fits(rge_fits)

        elif args.resource_type == "fit":
            downloader = Downloader(server=args.server)
            resources = downloader.list_resources("fit")
            registry  = downloader.get_registry()
            print(_header(f"Fits ({len(resources)})"))
            if resources:
                width = max(len(r) for r in resources)
                for r in resources:
                    meta = registry["fits"].get(r)
                    if meta:
                        date = _date(meta.get("created_at", "?")[:10])
                        rge  = _rge(meta.get("has_rge", False))
                        print(f"  {_name(r):<{width + 8}}  {date}  {rge}")
                    else:
                        print(f"  {_name(r):<{width + 8}}  {_c('(not in registry)', _DIM)}")
            else:
                print(_c("  (none)", _DIM))

        elif args.resource_type == "misc":
            downloader = Downloader(server=args.server)
            entries  = downloader.list_resources("misc")
            misc_reg = downloader.get_misc_registry()
            print(_header(f"misc/ ({len(entries)} entries)"))
            if entries:
                width = max(len(e) for e in entries)
                for name in sorted(entries):
                    meta = misc_reg.get(name)
                    if meta:
                        date = _date(meta.get("uploaded_at", "?")[:10])
                        parts = [f"  {_name(name):<{width + 8}}  {date}"]
                        if meta.get("uploaded_by"):
                            parts.append(_by(meta["uploaded_by"]))
                        c = meta.get("comment")
                        if c:
                            parts.append(_comment(c))
                        print("  ".join(parts))
                    else:
                        print(f"  {_name(name):<{width + 8}}  {_c('(no metadata)', _DIM)}")
            else:
                print(_c("  (empty)", _DIM))

        else:
            downloader = Downloader(server=args.server)
            resources = downloader.list_resources(args.resource_type)
            label = f"{args.resource_type}s"
            print(_header(f"{label.capitalize()} ({len(resources)})"))
            if resources:
                for r in resources:
                    print(f"  {_name(r)}")
            else:
                print(_c("  (none)", _DIM))

    except ServerError as e:
        log.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
