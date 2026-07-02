"""User-specific path configuration for smefit.

Paths are stored in <new_smefit>/.config/paths.yaml with one key per directory:

    new_smefit:      /path/to/new_smefit
    smefit_database: /path/to/smefit_database
    smefit_results:  /path/to/smefit_results

The file is auto-created on first use, assuming the three directories are
siblings of each other. Edit it directly or run 'smefit_setup_local' to update.

In runcards, use the key name as a prefix for relative paths:
    data_path:  smefit_database/commondata
    theory_path: smefit_database/theory
    path:        new_smefit/external_chi2/drell_yan/MyModule.py
    rg_matrix:   smefit_results/fits/my_fit/rge_matrix.pkl

Users can add extra aliases for any directory (e.g. an alternative database):
    lhc_database: /data/shared/lhc_database_v2

and then use them in runcards the same way:
    data_path: lhc_database/commondata
"""

import pathlib

import yaml

# Repo-local config — machine-specific, listed in .gitignore
USER_PATHS_CONFIG = pathlib.Path(__file__).parents[1] / ".config" / "paths.yaml"

_STANDARD_PREFIXES = ("new_smefit", "smefit_database", "smefit_results")


def load_user_paths() -> dict:
    """Load .config/paths.yaml, returning an empty dict if absent."""
    if not USER_PATHS_CONFIG.exists():
        return {}
    with open(USER_PATHS_CONFIG) as f:
        return yaml.safe_load(f) or {}


def write_user_paths(paths: dict) -> None:
    USER_PATHS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with open(USER_PATHS_CONFIG, "w") as f:
        yaml.dump(paths, f, default_flow_style=False)


def _ensure_smefit_paths() -> None:
    """Seed the three standard path keys on first use, assuming sibling layout."""
    new_smefit_dir = pathlib.Path(__file__).parents[1]
    workspace = new_smefit_dir.parent
    defaults = {
        "new_smefit": str(new_smefit_dir),
        "smefit_database": str(workspace / "smefit_database"),
        "smefit_results": str(workspace / "smefit_results"),
    }
    existing = dict(load_user_paths())
    changed = False
    for key, val in defaults.items():
        if key not in existing:
            existing[key] = val
            changed = True
    if changed:
        write_user_paths(existing)


def get_local_results_dir() -> pathlib.Path | None:
    """Return the configured smefit_results directory, or None if not set."""
    user_paths = load_user_paths()
    return (
        pathlib.Path(user_paths["smefit_results"])
        if "smefit_results" in user_paths
        else None
    )


def fetch_fit_if_missing(resolved_path: pathlib.Path) -> None:
    """If *resolved_path* is missing and lives under smefit_results/fits/<name>/,
    attempt to download it from the server before raising.

    For rge_matrix.pkl only the RGE file is fetched (fast path). For any other
    file the full fit archive is downloaded.
    """
    if resolved_path.exists():
        return

    user_paths = load_user_paths()
    results_str = user_paths.get("smefit_results")
    if not results_str:
        return

    fits_dir = pathlib.Path(results_str) / "fits"
    try:
        rel = resolved_path.relative_to(fits_dir)
    except ValueError:
        return  # not under smefit_results/fits/ — let the caller raise naturally

    fit_name = rel.parts[0]
    fit_dir = fits_dir / fit_name

    import logging

    log = logging.getLogger(__name__)
    log.info(
        "Fit '%s' not found locally — attempting to download from server ...",
        fit_name,
    )

    from smefit.server_utils import Downloader, ServerError

    try:
        downloader = Downloader()
        downloader.download("fit", fit_name, fits_dir)
        downloader.update_local_registry("fit", fit_name, pathlib.Path(results_str))
    except ServerError as e:
        raise FileNotFoundError(
            f"'{resolved_path}' does not exist locally and could not be "
            f"downloaded from the server: {e}"
        ) from e


def resolve_path(path_str: str) -> str:
    """Resolve a prefix-relative path using the user paths config.

    Any key in paths.yaml is a valid prefix. User-defined aliases
    (e.g. lhc_database) are resolved the same way as the standard prefixes.

    Paths that do not start with any known prefix are returned unchanged.
    Run 'smefit_setup_local' to create paths.yaml if it does not exist yet.
    """
    user_paths = load_user_paths()

    # Try all configured keys — longest first to avoid prefix shadowing
    for key in sorted(user_paths, key=len, reverse=True):
        if path_str == key or path_str.startswith(key + "/"):
            rest = path_str[len(key) :]
            return user_paths[key].rstrip("/") + rest

    # Standard prefix matched but not in config — paths.yaml is missing or incomplete
    for prefix in _STANDARD_PREFIXES:
        if path_str == prefix or path_str.startswith(prefix + "/"):
            raise ValueError(
                f"Path '{path_str}' starts with '{prefix}' but '{prefix}' "
                f"is not set in {USER_PATHS_CONFIG}. "
                "Run 'smefit_setup_local' to configure it."
            )

    return path_str
