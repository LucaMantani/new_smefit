"""
smefit.app.py

Module contains the main class for the smefit app.
"""

import argparse
import pathlib
from collections.abc import Sequence
from typing import Any

from reportengine.app import App

from smefit.config import smefitConfig
from smefit.environment import smefitEnvironment

smefit_providers = [
    "smefit.utils",
    "smefit.analytic_fit",
    "smefit.gradient_descent",
    "smefit.hessian_fit",
    "smefit.ultranest_fit",
    "smefit.blackjax_fit",
    "smefit.individual_fit",
    "smefit.chi2_scan",
    "smefit.fit_actions",
    "smefit.fisher",
    "smefit.dataset_chi2",
    "smefit.pca",
    "smefit.dataset_info",
    "smefit.tables",
    "smefit.figures",
    "smefit.utils_actions",
    "reportengine.report",
]


class smefitApp(App):
    config_class = smefitConfig
    environment_class = smefitEnvironment

    def __init__(self, name: str = "smefit", providers: Sequence[str] = ()) -> None:
        super().__init__(name, smefit_providers + list(providers))

    @property
    def argparser(self) -> argparse.ArgumentParser:
        """Parser arguments for smefit."""
        parser = super().argparser

        parser.add_argument(
            "-o",
            "--output",
            nargs="?",
            default=None,
            help="Name of the output directory.",
        )

        parser.add_argument(
            "-f32",
            "--float32",
            action="store_true",
            help="Use float32 precision for the computation",
        )

        return parser

    def get_commandline_arguments(
        self, cmdline: Sequence[str] | None = None
    ) -> dict[str, Any]:
        """Get commandline arguments"""
        args = super().get_commandline_arguments(cmdline)
        if args["output"] is None:
            args["output"] = pathlib.Path(args["config_yml"]).stem
        return args


def main() -> None:
    a = smefitApp(name="smefit")
    a.main()


if __name__ == "__main__":
    main()
