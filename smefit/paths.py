"""User-specific path configuration for smefit.

Paths are stored in ~/.config/smefit/paths.yaml:

    path_to_smefit_database: /path/to/smefit_database
    path_to_new_smefit: /path/to/new_smefit
    path_to_smefit_results: /path/to/smefit_results

Run 'smefit_setup_paths' to create or update this file.

In runcards, use short prefix-relative paths such as:
    data_path: smefit_database/commondata
    theory_path: smefit_database/theory
    path: new_smefit/external_chi2/drell_yan/MyModule.py
"""

import pathlib

import yaml

USER_PATHS_CONFIG = pathlib.Path.home() / ".config" / "smefit" / "paths.yaml"

_PREFIX_TO_KEY = {
    "smefit_database": "path_to_smefit_database",
    "new_smefit": "path_to_new_smefit",
    "smefit_results": "path_to_smefit_results",
}


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


def resolve_path(path_str: str) -> str:
    """If path_str starts with a known prefix, prepend the cached base dir."""
    for prefix, key in _PREFIX_TO_KEY.items():
        if path_str == prefix or path_str.startswith(prefix + "/"):
            user_paths = load_user_paths()
            if key not in user_paths:
                raise ValueError(
                    f"Path '{path_str}' starts with '{prefix}' but '{key}' is not set "
                    f"in {USER_PATHS_CONFIG}. Run 'smefit_setup_paths' to configure it."
                )
            base = user_paths[key].rstrip("/")
            suffix = path_str[len(prefix) :]
            return base + suffix
    return path_str
