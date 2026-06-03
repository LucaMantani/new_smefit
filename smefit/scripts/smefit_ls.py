"""
List resources available on the SMEFiT server.

    smefit_ls [RESOURCE_TYPE] [--server public|private] [--project NAME]

RESOURCE_TYPE must be one of: fit, report, rge, registry, misc (default: registry).
  fit/report  – lists available resources of that type.
  rge         – lists fits that have an rge_matrix.pkl, read from the registry.
  registry    – displays the full registry with all tracked metadata.
  misc        – lists contents of misc/ with metadata from registry_misc.json.

Use --project NAME to restrict the output to resources belonging to that project.
If the name does not match any known project, an interactive list is shown instead.

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
# Color helpers (disabled when stdout is not a TTY)
# ---------------------------------------------------------------------------
_USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(text, *codes):
    return ("".join(codes) + str(text) + "\033[0m") if _USE_COLOR else str(text)


_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"


def _header(s):
    return _c(s, _BOLD, _CYAN)


def _bold(s):
    return _c(s, _BOLD)


def _dim(s):
    return _c(s, _DIM)


def _green(s):
    return _c(s, _GREEN)


def _rge(val):
    return _green("yes") if val else _dim("no")


def _cmt(s):
    return _dim(f'"{s[:50]}{"..." if len(s) > 50 else ""}"')


# ---------------------------------------------------------------------------
# Table renderer
# ---------------------------------------------------------------------------


def _table(title: str, headers: list, plain_rows: list, colored_rows: list) -> None:
    """Print a box-drawing table.

    *plain_rows* and *colored_rows* must be parallel lists of equal-length rows.
    Column widths are computed from *plain_rows* so ANSI codes don't affect layout.
    """
    print(f"\n  {_header(title)}")
    if not plain_rows:
        print(_dim("  (none)"))
        return

    widths = [len(h) for h in headers]
    for row in plain_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _hline(l, m, r):
        return "  " + l + m.join("─" * (w + 2) for w in widths) + r

    def _row_line(plain_cells, colored_cells):
        parts = []
        for plain, colored, w in zip(plain_cells, colored_cells, widths):
            parts.append(f" {colored}{' ' * (w - len(plain))} ")
        return "  │" + "│".join(parts) + "│"

    print(_hline("┌", "┬", "┐"))
    print(_row_line(headers, [_bold(h) for h in headers]))
    print(_hline("├", "┼", "┤"))
    for p_row, c_row in zip(plain_rows, colored_rows):
        print(_row_line(p_row, c_row))
    print(_hline("└", "┴", "┘"))


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def _fit_rows(fits: dict):
    has_comment = any(m.get("comment") for m in fits.values())
    has_project = any(m.get("project") for m in fits.values())
    headers = ["Name", "Date", "RGE", "Uploaded by"]
    if has_project:
        headers.append("Project")
    if has_comment:
        headers.append("Comment")

    plain, colored = [], []
    for name, meta in sorted(fits.items()):
        date = meta.get("created_at", "")[:10]
        rge_val = meta.get("has_rge", False)
        rge_str = "yes" if rge_val else "no"
        uploader = meta.get("uploaded_by") or ""
        p = [name, date, rge_str, uploader]
        c = [_bold(name), _dim(date), _rge(rge_val), uploader]
        if has_project:
            proj = meta.get("project") or ""
            p.append(proj)
            c.append(_green(proj) if proj else "")
        if has_comment:
            cmt = meta.get("comment") or ""
            cmt_plain = f'"{cmt[:50]}{"..." if len(cmt) > 50 else ""}"' if cmt else ""
            p.append(cmt_plain)
            c.append(_cmt(cmt) if cmt else "")
        plain.append(p)
        colored.append(c)
    return headers, plain, colored


def _report_rows(reports: dict):
    has_comment = any(m.get("comment") for m in reports.values())
    has_project = any(m.get("project") for m in reports.values())
    headers = ["Name", "Date", "Uploaded by"]
    if has_project:
        headers.append("Project")
    if has_comment:
        headers.append("Comment")

    plain, colored = [], []
    for name, meta in sorted(reports.items()):
        date = meta.get("created_at", "")[:10]
        uploader = meta.get("uploaded_by") or ""
        p = [name, date, uploader]
        c = [_bold(name), _dim(date), uploader]
        if has_project:
            proj = meta.get("project") or ""
            p.append(proj)
            c.append(_green(proj) if proj else "")
        if has_comment:
            cmt = meta.get("comment") or ""
            cmt_plain = f'"{cmt[:50]}{"..." if len(cmt) > 50 else ""}"' if cmt else ""
            p.append(cmt_plain)
            c.append(_cmt(cmt) if cmt else "")
        plain.append(p)
        colored.append(c)
    return headers, plain, colored


def _misc_rows(entries: list, misc_reg: dict):
    has_comment = any(misc_reg.get(e, {}).get("comment") for e in entries)
    headers = ["Path", "Date", "Uploaded by"]
    if has_comment:
        headers.append("Comment")

    plain, colored = [], []
    for name in sorted(entries):
        meta = misc_reg.get(name, {})
        date = meta.get("uploaded_at", "")[:10]
        uploader = meta.get("uploaded_by") or ""
        p = [name, date if date else "—", uploader if uploader else "—"]
        c = [
            _bold(name),
            _dim(date) if date else _dim("—"),
            uploader if uploader else _dim("—"),
        ]
        if has_comment:
            cmt = meta.get("comment") or ""
            cmt_plain = f'"{cmt[:50]}{"..." if len(cmt) > 50 else ""}"' if cmt else ""
            p.append(cmt_plain)
            c.append(_cmt(cmt) if cmt else "")
        plain.append(p)
        colored.append(c)
    return headers, plain, colored


# ---------------------------------------------------------------------------
# Interactive project picker
# ---------------------------------------------------------------------------


def _pick_project(projects: list) -> str | None:
    """Print a numbered project list and return the user's choice, or None."""
    if not projects:
        print(
            _dim(
                "  No projects defined. Use 'smefit_manage_project add <name>' to create one."
            )
        )
        return None
    print(f"\n  {_header(f'Projects ({len(projects)})')}")
    for i, p in enumerate(projects, 1):
        print(f"    {i}. {_green(p)}")
    try:
        raw = input("\nProject number (press Enter to cancel): ").strip()
    except EOFError:
        return None
    if not raw:
        return None
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(projects):
            return projects[idx]
        print("Invalid number.", file=sys.stderr)
    except ValueError:
        print("Invalid input.", file=sys.stderr)
    return None


def _resolve_project(downloader, requested: str) -> str | None:
    """Validate *requested* against the registry's project list.

    Returns the validated project name, or a user-chosen one if *requested* was
    not found. Returns None if the user cancels the interactive picker.
    """
    registry = downloader.get_registry()
    projects = sorted(registry.get("projects", []))
    if requested in projects:
        return requested
    print(
        f"  Project '{requested}' does not exist. "
        "Use 'smefit_manage_project add <name>' to create a new project.",
        file=sys.stderr,
    )
    return _pick_project(projects)


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
    parser.add_argument(
        "--project",
        default=None,
        metavar="NAME",
        help="Restrict output to resources belonging to this project.",
    )
    args = parser.parse_args()

    from smefit.server_utils import Downloader, ServerError

    try:
        downloader = Downloader(server=args.server)

        # Resolve project filter once — validates the name or offers interactive pick.
        project_filter = None
        if args.project is not None:
            project_filter = _resolve_project(downloader, args.project)
            if project_filter is None:
                sys.exit(0)

        def _filter_fits(fits):
            if project_filter is None:
                return fits
            return {k: v for k, v in fits.items() if v.get("project") == project_filter}

        def _filter_reports(reports):
            if project_filter is None:
                return reports
            return {
                k: v for k, v in reports.items() if v.get("project") == project_filter
            }

        title_suffix = f" [{project_filter}]" if project_filter else ""

        if args.resource_type == "registry":
            registry = downloader.get_registry()
            fits = _filter_fits(registry.get("fits", {}))
            reports = _filter_reports(registry.get("reports", {}))
            _table(f"Fits ({len(fits)}){title_suffix}", *_fit_rows(fits))
            _table(f"Reports ({len(reports)}){title_suffix}", *_report_rows(reports))
            if not project_filter:
                projects = sorted(registry.get("projects", []))
                print(f"\n  {_header(f'Projects ({len(projects)})')}")
                if projects:
                    for p in projects:
                        print(f"    {_green(p)}")
                else:
                    print(_dim("  (none)"))
            print()

        elif args.resource_type == "rge":
            registry = downloader.get_registry()
            rge_fits = _filter_fits(
                {k: v for k, v in registry.get("fits", {}).items() if v.get("has_rge")}
            )
            _table(
                f"Fits with rge_matrix.pkl ({len(rge_fits)}){title_suffix}",
                *_fit_rows(rge_fits),
            )
            print()

        elif args.resource_type == "fit":
            resources = downloader.list_resources("fit")
            registry = downloader.get_registry()
            fits = _filter_fits(
                {r: registry["fits"][r] for r in resources if r in registry["fits"]}
            )
            _table(f"Fits ({len(fits)}){title_suffix}", *_fit_rows(fits))
            if not project_filter:
                for r in resources:
                    if r not in registry["fits"]:
                        print(f"  {_bold(r)}  {_dim('(not in registry)')}")
            print()

        elif args.resource_type == "report":
            registry = downloader.get_registry()
            reports = _filter_reports(registry.get("reports", {}))
            _table(f"Reports ({len(reports)}){title_suffix}", *_report_rows(reports))
            print()

        elif args.resource_type == "misc":
            entries = downloader.list_resources("misc")
            misc_reg = downloader.get_misc_registry()
            _table(f"misc/ ({len(entries)} entries)", *_misc_rows(entries, misc_reg))
            print()

    except ServerError as e:
        log.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
