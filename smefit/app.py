"""
smefit.app.py

Module contains the main class for the smefit app.
"""

from reportengine.app import App
from smefit.config import smefitConfig
from smefit.environment import smefitEnvironment
import pathlib


smefit_providers = ["smefit.utils"]


class smefitApp(App):
    config_class = smefitConfig
    environment_class = smefitEnvironment

    def __init__(self, name="smefit", providers=[]):
        super().__init__(name, smefit_providers + providers)

    @property
    def argparser(self):
        """Parser arguments for smefit."""
        parser = super().argparser

        parser.add_argument(
            "-o",
            "--output",
            nargs="?",
            default=None,
            help="Name of the output directory.",
        )

        return parser

    def get_commandline_arguments(self, cmdline=None):
        """Get commandline arguments"""
        args = super().get_commandline_arguments(cmdline)
        if args["output"] is None:
            args["output"] = pathlib.Path(args["config_yml"]).stem
        return args


def main():
    a = smefitApp(name="smefit")
    a.main()


if __name__ == "__main__":
    main()
