"""
Configure SMEFiT server credentials.

    smefit_setup_server [CREDENTIALS_YAML]

If CREDENTIALS_YAML is provided, its contents are validated and written to
<new_smefit>/.config/server.yaml. This is the recommended workflow for team members:
share a credentials template privately, then each member runs:

    smefit_setup_server my_credentials.yaml

Without an argument the script prompts interactively for each profile.

Expected YAML format (include only the profiles you have access to):

    public:
      webdav_hostname: https://surfdrive.surf.nl/public.php/webdav/
      webdav_login:    <write-token>
      webdav_password: <password>
    private:
      webdav_hostname: https://surfdrive.surf.nl/public.php/webdav/
      webdav_login:    <private-token>
      webdav_password: <password>
"""

import argparse
import getpass
import logging
import pathlib
import sys

import yaml
from reportengine import colors

from smefit import server_utils

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(colors.ColorHandler())

_REQUIRED_KEYS = ("webdav_hostname", "webdav_login", "webdav_password")
_PROFILES = ("public", "private")


def _validate(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("Credentials file must be a YAML mapping.")
    unknown = set(config) - set(_PROFILES)
    if unknown:
        raise ValueError(
            f"Unknown profile(s): {', '.join(sorted(unknown))}. Expected: {', '.join(_PROFILES)}"
        )
    if not config:
        raise ValueError("Credentials file contains no profiles.")
    for profile, values in config.items():
        if not isinstance(values, dict):
            raise ValueError(f"Profile '{profile}' must be a YAML mapping.")
        for key in _REQUIRED_KEYS:
            if key not in values:
                raise ValueError(f"Missing key '{key}' under '{profile}:'.")


def _prompt_profile(profile: str) -> dict | None:
    """Interactively collect credentials for one profile. Returns None if skipped."""
    print(f"\n--- {profile.upper()} server ---")
    answer = input(f"Configure '{profile}' profile? [y/N] ").strip().lower()
    if answer != "y":
        return None
    hostname = input("  webdav_hostname: ").strip()
    login = input("  webdav_login:    ").strip()
    password = getpass.getpass("  webdav_password: ")
    name = input("  name (shown in registry, optional): ").strip()
    result = {
        "webdav_hostname": hostname,
        "webdav_login": login,
        "webdav_password": password,
    }
    if name:
        result["name"] = name
    return result


def _write_config(config: dict) -> None:
    path = server_utils.CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    path.chmod(0o600)
    log.info("Credentials written to %s", path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "credentials_yaml",
        nargs="?",
        default=None,
        metavar="CREDENTIALS_YAML",
        help="Path to a YAML file with server credentials. Prompts interactively if omitted.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"Overwrite {server_utils.CONFIG_PATH} if it already exists.",
    )
    args = parser.parse_args()

    if server_utils.CONFIG_PATH.exists() and not args.force:
        log.error(
            "%s already exists. Use --force to overwrite.", server_utils.CONFIG_PATH
        )
        sys.exit(1)

    if args.credentials_yaml is not None:
        src = pathlib.Path(args.credentials_yaml)
        if not src.exists():
            log.error("File not found: %s", src)
            sys.exit(1)
        with open(src) as f:
            config = yaml.safe_load(f)
        try:
            _validate(config)
        except ValueError as e:
            log.error("%s", e)
            sys.exit(1)
    else:
        config = {}
        for profile in _PROFILES:
            result = _prompt_profile(profile)
            if result is not None:
                config[profile] = result
        if not config:
            log.error("No profiles configured. Exiting without writing.")
            sys.exit(1)

    _write_config(config)


if __name__ == "__main__":
    main()
