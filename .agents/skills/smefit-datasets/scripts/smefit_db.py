#!/usr/bin/env python
"""Locate and query the smefit_database (datasets, operators, external likelihoods).

Prefers a local clone of https://github.com/LHCfitNikhef/smefit_database; falls
back to fetching the small catalog files from GitHub when no clone is found.

The clone location is read from smefit's machine-specific `.config/paths.yaml`
(key `smefit_database`, created by the interactive `smefit_setup_local` command),
with fallbacks for unconfigured setups.

Subcommands:
    locate                 find the local smefit_database clone
    search KEYWORD         find datasets matching a keyword (name or experiment group)
    info DATASET           show metadata + operators entering a dataset
    operators [PATTERN]    list implemented Wilson coefficients (optionally filtered)
    ext [PATTERN]          list external likelihoods with ready-to-paste runcard blocks
    clone [DEST]           print the setup/clone commands (never executes them)

Common flags: --json (machine-readable output), --offline (never touch the network).

Exit codes: 0 = ok, 1 = query found nothing, 2 = bad usage/input, 3 = no database
available (neither local clone nor network).
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml is required (pip install pyyaml).")
    sys.exit(2)

REPO_URL = "https://github.com/LHCfitNikhef/smefit_database"
RAW_BASE = "https://raw.githubusercontent.com/LHCfitNikhef/smefit_database/main"
CACHE_DIR = Path(tempfile.gettempdir()) / "smefit_db_cache"
CACHE_MAX_AGE = 24 * 3600  # seconds

CATALOGS = {
    "data": "data_summary.yaml",
    "operators": "operators_implemented.yaml",
    "ext": "ext_likelihood_summary.yaml",
}


def is_database(path):
    return (
        path.is_dir()
        and (path / "commondata").is_dir()
        and (path / "theory").is_dir()
        and (path / "data_summary.yaml").is_file()
    )


def find_paths_config():
    """Locate smefit's machine-specific .config/paths.yaml (see smefit_setup_local).

    Mirrored verbatim in ../../smefit-runcard/scripts/validate_runcard.py — each
    skill directory has to stand alone (see the skills README.md), so keep
    the two copies in sync rather than factoring one out.
    """
    # 1. Walk up from cwd (covers running inside the repo or a work dir below it).
    for parent in [Path.cwd()] + list(Path.cwd().parents):
        candidate = parent / ".config" / "paths.yaml"
        if candidate.is_file():
            return candidate
    # 2. Script-relative repo root (.agents/skills/<skill>/scripts/ -> repo).
    candidate = Path(__file__).resolve().parents[4] / ".config" / "paths.yaml"
    if candidate.is_file():
        return candidate
    # 3. Ask an installed smefit where its config lives (smefit.paths only
    #    imports pathlib+yaml, so this subprocess is cheap).
    try:
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                "from smefit.paths import USER_PATHS_CONFIG; print(USER_PATHS_CONFIG)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.returncode == 0:
            candidate = Path(out.stdout.strip())
            if candidate.is_file():
                return candidate
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def find_database():
    """Return (path, how) for the first valid smefit_database clone found, else (None, None)."""
    # Official mechanism: smefit_database key in .config/paths.yaml.
    config = find_paths_config()
    if config is not None:
        try:
            user_paths = yaml.safe_load(config.read_text()) or {}
        except (yaml.YAMLError, OSError):
            user_paths = {}
        db = user_paths.get("smefit_database")
        if db:
            p = Path(str(db)).expanduser()
            if is_database(p):
                return p, f"smefit_database in {config}"

    # Fallbacks for unconfigured setups.
    # Scan runcards in cwd (and ./runcards) for an absolute data_path into a clone.
    candidates = list(Path.cwd().glob("*.yaml"))
    runcard_dir = Path.cwd() / "runcards"
    if runcard_dir.is_dir():
        candidates += list(runcard_dir.glob("*.yaml"))
    for runcard in sorted(candidates):
        try:
            content = yaml.safe_load(runcard.read_text())
        except (yaml.YAMLError, OSError, UnicodeDecodeError):
            continue
        if not isinstance(content, dict):
            continue
        data_path = content.get("data_path")
        if not data_path:
            continue
        parent = Path(str(data_path)).expanduser().parent
        if is_database(parent):
            return parent, f"data_path in {runcard.name}"

    for probe in (
        Path.cwd().parent / "smefit_database",
        Path.cwd() / "smefit_database",
        Path.home() / "smefit_database",
    ):
        if is_database(probe):
            return probe, f"probed {probe}"

    return None, None


def fetch_remote(filename, offline):
    """Fetch a catalog file from GitHub raw, with a small on-disk cache."""
    if offline:
        return None
    cached = CACHE_DIR / filename
    if cached.is_file() and time.time() - cached.stat().st_mtime < CACHE_MAX_AGE:
        return cached.read_text()
    url = f"{RAW_BASE}/{filename}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            text = resp.read().decode()
    except (urllib.error.URLError, OSError):
        return None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_text(text)
    return text


def load_catalog(kind, db_path, offline):
    """Load a catalog YAML from the local clone or GitHub. Returns (data, source)."""
    filename = CATALOGS[kind]
    if db_path is not None:
        local = db_path / filename
        if local.is_file():
            return yaml.safe_load(local.read_text()), str(local)
    text = fetch_remote(filename, offline)
    if text is not None:
        return yaml.safe_load(text), f"{RAW_BASE}/{filename} (GitHub fallback)"
    return None, None


SETUP_HINT = (
    "Run smefit's interactive local setup (after confirming with the user) —\n"
    "it records the path in .config/paths.yaml and offers to clone the database:\n"
    "    smefit_setup_local\n"
    f"Alternatively clone manually and re-run the setup:\n    git clone {REPO_URL}"
)


def no_database_exit(offline):
    print(
        "No local smefit_database clone found"
        + (" and --offline set." if offline else " and GitHub is unreachable.")
    )
    print(SETUP_HINT)
    return 3


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------


def cmd_locate(args):
    db, how = find_database()
    if db is None:
        print("No local smefit_database clone found.")
        print(SETUP_HINT)
        return 3
    # Prefix form only works once paths.yaml is configured; otherwise recommend
    # absolute paths (and suggest running the setup).
    configured = "paths.yaml" in how
    if args.json:
        print(
            json.dumps(
                {
                    "database": str(db),
                    "data_path": (
                        "smefit_database/commondata"
                        if configured
                        else str(db / "commondata")
                    ),
                    "theory_path": (
                        "smefit_database/theory" if configured else str(db / "theory")
                    ),
                    "resolved_data_path": str(db / "commondata"),
                    "resolved_theory_path": str(db / "theory"),
                    "projections_path": str(db / "commondata_projections_L0"),
                    "prefix_form_available": configured,
                    "found_via": how,
                }
            )
        )
        return 0
    print(f"database: {db}   (via {how})")
    if configured:
        print("Use the shareable prefix form in runcards:")
        print("  data_path: smefit_database/commondata")
        print("  theory_path: smefit_database/theory")
        print(f"(resolved via .config/paths.yaml to {db})")
    else:
        print("Runcard values (absolute — .config/paths.yaml is not set up):")
        print(f"  data_path: {db / 'commondata'}")
        print(f"  theory_path: {db / 'theory'}")
        print(
            "Tip: run 'smefit_setup_local' once (after confirming with the user) "
            "to enable shareable prefix paths like smefit_database/commondata."
        )
    return 0


def iter_datasets(summary):
    """Yield (group, entry) for each dataset entry in data_summary.yaml."""
    for group, entries in summary.items():
        if not isinstance(entries, list):
            continue  # skip the Info: string
        for entry in entries:
            if isinstance(entry, dict) and "name" in entry:
                yield group, entry


def cmd_search(args):
    db, _ = find_database()
    summary, source = load_catalog("data", db, args.offline)
    if summary is None:
        return no_database_exit(args.offline)
    keyword = args.keyword.lower()
    matches = [
        {"group": group, **entry}
        for group, entry in iter_datasets(summary)
        if keyword in entry["name"].lower() or keyword in group.lower()
    ]
    if args.json:
        print(json.dumps({"source": source, "matches": matches}, indent=2))
        return 0 if matches else 1
    if not matches:
        print(f"No dataset matching '{args.keyword}' (catalog: {source})")
        return 1
    width = max(len(m["name"]) for m in matches)
    print(f"{len(matches)} dataset(s) matching '{args.keyword}' (catalog: {source}):\n")
    for m in matches:
        allowed = ", ".join(m.get("allowed_orders", []))
        print(
            f"  {m['name']:<{width}}  group={m['group']}  default_order={m.get('order', '?')}  allowed=[{allowed}]"
        )
    return 0


def cmd_info(args):
    db, _ = find_database()
    name = args.dataset

    # Catalog entry (allowed orders)
    summary, _ = load_catalog("data", db, args.offline)
    entry = None
    if summary is not None:
        for group, e in iter_datasets(summary):
            if e["name"] == name:
                entry = {"group": group, **e}
                break

    # Commondata metadata
    commondata = None
    if db is not None and (db / "commondata" / f"{name}.yaml").is_file():
        commondata = yaml.safe_load((db / "commondata" / f"{name}.yaml").read_text())
    elif (
        db is not None and (db / "commondata_projections_L0" / f"{name}.yaml").is_file()
    ):
        commondata = yaml.safe_load(
            (db / "commondata_projections_L0" / f"{name}.yaml").read_text()
        )
    else:
        text = fetch_remote(f"commondata/{name}.yaml", args.offline)
        if text is not None:
            commondata = yaml.safe_load(text)

    if entry is None and commondata is None:
        print(f"Dataset '{name}' not found in the catalog or commondata.")
        return 1

    info = {"name": name}
    if entry:
        info["group"] = entry["group"]
        info["default_order"] = entry.get("order")
        info["allowed_orders"] = entry.get("allowed_orders")
    if commondata:
        for key in (
            "description",
            "arxiv",
            "doi",
            "hepdata",
            "units",
            "luminosity",
            "num_data",
            "num_sys",
        ):
            if key in commondata:
                info[key] = commondata[key]

    # Operators entering the predictions (local theory JSON only — files are large)
    if db is not None and (db / "theory" / f"{name}.json").is_file():
        theory = json.loads((db / "theory" / f"{name}.json").read_text())
        orders = [
            k
            for k in theory
            if k not in {"best_sm", "scales"} and not k.startswith("theory_cov")
        ]
        info["orders_in_theory_file"] = sorted(orders)
        info["theory_cov_types"] = sorted(
            k[len("theory_cov_") :] for k in theory if k.startswith("theory_cov_")
        )
        if orders:
            pred = theory[orders[0]]
            linear = sorted(k for k in pred if k != "SM" and "*" not in k)
            info["operators"] = linear
            info["has_quadratic_terms"] = any("*" in k for k in pred)
    elif db is None:
        info["note"] = (
            "theory-file details skipped (no local clone; theory JSONs are large)"
        )

    if args.json:
        print(json.dumps(info, indent=2))
        return 0
    for key, value in info.items():
        if isinstance(value, list) and len(value) > 8:
            print(f"{key}:")
            for v in value:
                print(f"  - {v}")
        else:
            print(f"{key}: {value}")
    return 0


def cmd_operators(args):
    db, _ = find_database()
    catalog, source = load_catalog("operators", db, args.offline)
    if catalog is None:
        return no_database_exit(args.offline)
    operators = catalog.get("operators", {})
    pattern = (args.pattern or "").lower()
    matches = {
        name: definition
        for name, definition in operators.items()
        if pattern in name.lower()
    }
    if args.json:
        print(json.dumps({"source": source, "operators": matches}, indent=2))
        return 0 if matches else 1
    if not matches:
        print(f"No operator matching '{args.pattern}' (catalog: {source})")
        return 1
    print(
        f"{len(matches)} operator(s) (catalog: {source}; definitions in WCxf Warsaw basis):\n"
    )
    width = max(len(n) for n in matches)
    for name, definition in matches.items():
        print(f"  {name:<{width}}  =  {definition}")
    return 0


def cmd_ext(args):
    db, _ = find_database()
    catalog, source = load_catalog("ext", db, args.offline)
    if catalog is None:
        return no_database_exit(args.offline)
    likelihoods = catalog.get("external_chi2", {})
    pattern = (args.pattern or "").lower()
    matches = {
        name: config for name, config in likelihoods.items() if pattern in name.lower()
    }
    # Rewrite the placeholder paths to the located clone, when available.
    if db is not None:
        for config in matches.values():
            if isinstance(config, dict) and isinstance(config.get("path"), str):
                config["path"] = config["path"].replace(
                    "/path/to/smefit_database", str(db)
                )
    if args.json:
        print(json.dumps({"source": source, "external_chi2": matches}, indent=2))
        return 0 if matches else 1
    if not matches:
        print(f"No external likelihood matching '{args.pattern}' (catalog: {source})")
        return 1
    print(f"{len(matches)} external likelihood(s) (catalog: {source}).")
    print("Paste the ones you need under 'external_chi2:' in the runcard:\n")
    print(
        yaml.safe_dump(
            {"external_chi2": matches}, sort_keys=False, default_flow_style=False
        )
    )
    return 0


def cmd_clone(args):
    print("Preferred (after confirming with the user): run smefit's interactive")
    print("setup — it configures .config/paths.yaml and offers the clone itself:")
    print("    smefit_setup_local")
    dest = args.dest or "smefit_database"
    print("Manual alternative (then re-run smefit_setup_local to record the path):")
    print(f"    git clone {REPO_URL} {dest}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--offline", action="store_true", help="never touch the network"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("locate", help="find the local smefit_database clone")
    p.set_defaults(func=cmd_locate)

    p = sub.add_parser("search", help="find datasets by keyword")
    p.add_argument("keyword")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("info", help="show dataset metadata and operators")
    p.add_argument("dataset")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("operators", help="list implemented Wilson coefficients")
    p.add_argument("pattern", nargs="?", default="")
    p.set_defaults(func=cmd_operators)

    p = sub.add_parser("ext", help="list external likelihoods")
    p.add_argument("pattern", nargs="?", default="")
    p.set_defaults(func=cmd_ext)

    p = sub.add_parser("clone", help="print the git clone command")
    p.add_argument("dest", nargs="?", default=None)
    p.set_defaults(func=cmd_clone)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
