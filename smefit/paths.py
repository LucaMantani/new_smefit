"""User-specific path configuration for smefit.

Paths are stored in ~/.config/smefit/paths.yaml:

    path_to_smefit: /path/to/smefit/  # parent directory of new_smefit, smefit_database, smefit_results

Run 'smefit_setup_paths' to create or update this file.

In runcards, use short prefix-relative paths such as:
    data_path: smefit_database/commondata
    theory_path: smefit_database/theory
    path: new_smefit/external_chi2/drell_yan/MyModule.py
"""

import pathlib

import yaml

USER_PATHS_CONFIG = pathlib.Path.home() / ".config" / "smefit" / "paths.yaml"

_SMEFIT_PATH_KEY = "path_to_smefit"
_KNOWN_PREFIXES = ("smefit_database", "new_smefit", "smefit_results")


def load_user_paths() -> dict:
    """Load ~/.config/smefit/paths.yaml, returning an empty dict if absent."""
    if not USER_PATHS_CONFIG.exists():
        return {}
    with open(USER_PATHS_CONFIG) as f:
        return yaml.safe_load(f) or {}


def write_user_paths(paths: dict) -> None:
    USER_PATHS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with open(USER_PATHS_CONFIG, "w") as f:
        yaml.dump(paths, f, default_flow_style=False)
    USER_PATHS_CONFIG.chmod(0o600)


def _ensure_smefit_paths() -> None:
    """Create smefit_results/ next to the repo and seed path_to_smefit on first use."""
    smefit_parent = pathlib.Path(__file__).parents[2]
    (smefit_parent / "smefit_results").mkdir(exist_ok=True)
    existing = dict(load_user_paths())
    if _SMEFIT_PATH_KEY not in existing:
        existing[_SMEFIT_PATH_KEY] = str(smefit_parent)
        write_user_paths(existing)


def resolve_path(path_str: str) -> str:
    """If path_str starts with a known prefix, prepend path_to_smefit from the cache."""
    for prefix in _KNOWN_PREFIXES:
        if path_str == prefix or path_str.startswith(prefix + "/"):
            _ensure_smefit_paths()
            user_paths = load_user_paths()
            if _SMEFIT_PATH_KEY not in user_paths:
                raise ValueError(
                    f"Path '{path_str}' starts with '{prefix}' but '{_SMEFIT_PATH_KEY}' "
                    f"is not set in {USER_PATHS_CONFIG}. Run 'smefit_setup_paths' to configure it."
                )
            base = user_paths[_SMEFIT_PATH_KEY].rstrip("/")
            return f"{base}/{path_str}"
    return path_str
