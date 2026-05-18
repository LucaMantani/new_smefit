"""
smefit.environment.py

Environment module of smefit
"""

import logging
import os

import jax
from jax.extend import backend as jbackend
from reportengine.environment import Environment

log = logging.getLogger(__name__)


def _configure_pandoc_path():
    """Ensure the pandoc executable directory is available on PATH."""
    try:
        import pypandoc

        pandoc_path = pypandoc.get_pandoc_path()
        pandoc_dir = os.path.dirname(pandoc_path)
        current_path = os.environ.get("PATH", "")
        path_entries = current_path.split(os.pathsep) if current_path else []

        if pandoc_dir and pandoc_dir not in path_entries:
            os.environ["PATH"] = (
                pandoc_dir
                if not current_path
                else pandoc_dir + os.pathsep + current_path
            )
    except Exception as exc:
        log.debug("Unable to configure pandoc PATH: %s", exc)


_configure_pandoc_path()


class smefitEnvironment(Environment):
    """smefit Environment class."""

    def __init__(self, float32=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.float32 = float32

        if self.float32:
            log.info("Using float32 precision")
            jax.config.update("jax_enable_x64", False)
        else:
            log.info("Using float64 precision")
            jax.config.update("jax_enable_x64", True)

        log.info(f"Running with backend: {jbackend.get_backend().platform}")
