"""
Configure user-specific data and theory paths for SMEFiT.

    smefit_setup_paths [PATHS_YAML]

If PATHS_YAML is provided, its contents are validated and written to
~/.config/smefit/paths.yaml. Otherwise the script prompts interactively.

Expected YAML format:

    data_path: /path/to/smefit_database/commondata
    theory_path: /path/to/smefit_database/theory
    server_download_path: /path/to/download/directory

Once configured:
  - Set 'data_path: auto' and 'theory_path: auto' in any runcard to use
    those paths automatically.
  - 'smefit_get' will download resources to server_download_path by default
    when no local path is specified on the command line.
"""

import argparse
import logging
import pathlib
import sys

import yaml
from reportengine import colors

from smefit.paths import USER_PATHS_CONFIG, write_user_paths

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(colors.ColorHandler())

_KEYS = ("data_path", "theory_path", "server_download_path")


def _validate(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("Paths file must be a YAML mapping.")
    unknown = set(config) - set(_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown key(s): {', '.join(sorted(unknown))}. Expected: {', '.join(_KEYS)}"
        )
    if not config:
        raise ValueError("Paths file contains no entries.")
    for key, value in config.items():
        p = pathlib.Path(value)
        if not p.exists():
            raise ValueError(f"Path for '{key}' does not exist: {value}")


def _prompt() -> dict:
    config = {}
    for key in _KEYS:
        value = input(f"  {key}: ").strip()
        if value:
            config[key] = value
    return config


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths_yaml",
        nargs="?",
        default=None,
        metavar="PATHS_YAML",
        help="Path to a YAML file with data/theory paths. Prompts interactively if omitted.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"Overwrite {USER_PATHS_CONFIG} if it already exists.",
    )
    args = parser.parse_args()

    if USER_PATHS_CONFIG.exists() and not args.force:
        log.error("%s already exists. Use --force to overwrite.", USER_PATHS_CONFIG)
        sys.exit(1)

    if args.paths_yaml is not None:
        src = pathlib.Path(args.paths_yaml)
        if not src.exists():
            log.error("File not found: %s", src)
            sys.exit(1)
        with open(src) as f:
            config = yaml.safe_load(f)
    else:
        print("Enter absolute paths to your smefit database directories.")
        config = _prompt()
        if not config:
            log.error("No paths provided. Exiting without writing.")
            sys.exit(1)

    try:
        _validate(config)
    except ValueError as e:
        log.error("%s", e)
        sys.exit(1)

    write_user_paths(config)
    log.info("Paths written to %s", USER_PATHS_CONFIG)


if __name__ == "__main__":
    main()
