"""User-specific path configuration for smefit.

Paths are stored in ~/.config/smefit/paths.yaml:

    data_path: /path/to/smefit_database/commondata
    theory_path: /path/to/smefit_database/theory

Run 'smefit_setup_paths' to create or update this file.
Use 'data_path: auto' and 'theory_path: auto' in runcards to read from this file.
"""

import pathlib

import yaml

USER_PATHS_CONFIG = pathlib.Path.home() / ".config" / "smefit" / "paths.yaml"


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
