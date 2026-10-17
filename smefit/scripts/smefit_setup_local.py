"""
Interactive setup for smefit path configuration.

    smefit_setup_local

Creates or updates <new_smefit>/.config/paths.yaml with the three standard
directory paths. For each key the inferred default is shown; press Enter to
accept it or type a custom path.

If smefit_database does not exist at the configured location, the script
offers to clone it from GitHub automatically.
"""

import logging
import pathlib
import subprocess

from reportengine import colors

from smefit.paths import (
    USER_PATHS_CONFIG,
    _ensure_smefit_paths,
    load_user_paths,
    write_user_paths,
)

_DATABASE_URL = "https://github.com/LHCfitNikhef/smefit_database.git"

_KEYS = ("new_smefit", "smefit_database", "smefit_results")
_DESCRIPTIONS = {
    "new_smefit": "Path to the new_smefit repository",
    "smefit_database": "Path to the smefit_database repository",
    "smefit_results": "Path to the smefit_results directory",
}

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(colors.ColorHandler())


def _ask(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else default


def _clone_database(target: pathlib.Path) -> bool:
    """Clone smefit_database into *target*. Returns True on success."""
    log.info("Cloning %s into %s ...", _DATABASE_URL, target)
    result = subprocess.run(
        ["git", "clone", _DATABASE_URL, str(target)],
        check=False,
    )
    if result.returncode != 0:
        log.error("git clone failed (exit code %d).", result.returncode)
        return False
    log.info("smefit_database cloned successfully.")
    return True


def main():
    # Seed defaults from package location so we have something to show
    _ensure_smefit_paths()
    current = load_user_paths()

    print("\nSetting up smefit paths. Press Enter to accept the default.\n")

    new_paths = dict(current)
    for key in _KEYS:
        default = current.get(key, "")
        new_paths[key] = _ask(_DESCRIPTIONS[key], default)

    # Check smefit_database exists; offer to clone if not
    db_path = pathlib.Path(new_paths["smefit_database"])
    if not db_path.exists():
        answer = (
            input(
                f"\nsmefit_database not found at {db_path}.\n"
                "Clone it from GitHub now? [Y/n]: "
            )
            .strip()
            .lower()
        )
        if answer in ("", "y", "yes"):
            if not _clone_database(db_path):
                log.warning(
                    "Clone failed — you can clone manually and re-run this command."
                )
        else:
            log.warning(
                "smefit_database not cloned. Clone it manually:\n" "  git clone %s %s",
                _DATABASE_URL,
                db_path,
            )

    # Create smefit_results directory if it doesn't exist
    results_path = pathlib.Path(new_paths["smefit_results"])
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
        log.info("Created smefit_results directory at %s", results_path)

    write_user_paths(new_paths)

    # Append a commented-out example for custom aliases so the user sees the syntax
    with open(USER_PATHS_CONFIG, "a") as f:
        f.write(
            "\n"
            "# You can add custom path aliases below, but prefer the three standard\n"
            "# keys above whenever possible — extra keys make runcards harder to share,\n"
            "# since collaborators need the same key defined in their own paths.yaml.\n"
            "# Example:\n"
            "#   my_extra_database: /path/to/my_extra_database\n"
            "# Then in a runcard:\n"
            "#   data_path: my_extra_database/commondata\n"
        )

    log.info("Paths saved to %s", USER_PATHS_CONFIG)
    print(
        f"\nSetup complete. You can add custom path aliases by editing:\n"
        f"  {USER_PATHS_CONFIG}\n"
        f"\nNote: prefer the three standard keys whenever possible — extra aliases\n"
        f"make runcards harder to share, as collaborators need the same key\n"
        f"defined in their own paths.yaml."
    )
