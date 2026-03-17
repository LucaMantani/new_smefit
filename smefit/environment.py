"""
smefit.environment.py

Environment module of smefit
"""

import logging

import jax
from jax.extend import backend as jbackend
from reportengine.environment import Environment

log = logging.getLogger(__name__)


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
