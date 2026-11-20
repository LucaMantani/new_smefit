"""
Configure SMEFiT server credentials.

    smefit_setup_server [CREDENTIALS_YAML] [--name NAME] [--force]

If CREDENTIALS_YAML is provided, its contents are validated and written to
<new_smefit>/.config/server.yaml. This is the recommended workflow for team members:
share a credentials template privately, then each member runs:

    smefit_setup_server my_credentials.yaml --name Alice

Without CREDENTIALS_YAML the script prompts interactively for each profile.

Expected YAML format (include only the profiles you have access to):

    name: Alice
    public:
      webdav_hostname: https://surfdrive.surf.nl/public.php/webdav/
      webdav_login:    <write-token>
      webdav_password: <password>
    private:
      webdav_hostname: https://surfdrive.surf.nl/public.php/webdav/
      webdav_login:    <private-token>
      webdav_password: <password>

'name' is what the registry shows next to everything you upload or delete: the
same on either server, and for local uploads. It is personal, so a shared
template normally leaves it out: pass it with --name, or answer the prompt.
Without one your OS login is used.

'smefit_server setup' runs the same code.
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

_REQUIRED_KEYS = ("webdav_hostname", "webdav_login", "webdav_password")
_PROFILES = ("public", "private")


def _validate(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("Credentials file must be a YAML mapping.")
    unknown = set(config) - set(_PROFILES) - {"name"}
    if unknown:
        raise ValueError(
            f"Unknown key(s): {', '.join(sorted(map(str, unknown)))}. "
            f"Expected: {', '.join(_PROFILES)}, name"
        )
    if not set(config) & set(_PROFILES):
        raise ValueError("Credentials file contains no profiles.")
    if "name" in config:
        name = config["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("'name' must be a non-empty string.")
    for profile in _PROFILES:
        if profile not in config:
            continue
        values = config[profile]
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
    return {
        "webdav_hostname": hostname,
        "webdav_login": login,
        "webdav_password": password,
    }


def _prompt_name() -> str | None:
    """Interactively ask for the name shown in the registry. Returns None if skipped."""
    answer = input(
        f"\nName shown in the registry (press Enter to use '{getpass.getuser()}'): "
    )
    return answer.strip() or None


def _write_config(config: dict) -> None:
    path = server_utils.CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    path.chmod(0o600)
    log.info("Credentials written to %s", path)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the setup arguments to *parser*, for this script or a subcommand."""
    parser.add_argument(
        "credentials_yaml",
        nargs="?",
        default=None,
        metavar="CREDENTIALS_YAML",
        help="Path to a YAML file with server credentials. Prompts interactively if omitted.",
    )
    parser.add_argument(
        "--name",
        default=None,
        metavar="NAME",
        help=(
            "Name shown in the registry next to what you upload or delete. "
            "Prompted for when neither this nor CREDENTIALS_YAML gives one."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"Overwrite {server_utils.CONFIG_PATH} if it already exists.",
    )


def run(args: argparse.Namespace) -> None:
    """Collect the credentials *args* describe and write them; exit on error."""
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

    # One name per user. --name wins; a file that already carries one is left
    # alone; otherwise ask, and an empty answer leaves the OS login in charge.
    name = args.name
    if not name and server_utils._configured_name(config) is None:
        name = _prompt_name()
    if name:
        config["name"] = name

    _write_config(config)


def main():
    log.setLevel(logging.INFO)
    log.addHandler(colors.ColorHandler())
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
