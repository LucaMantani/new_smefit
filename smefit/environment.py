"""
smefit.environment.py

Environment module of smefit
"""

import logging

import jax
from jax.extend import backend as jbackend
from reportengine.environment import Environment

log = logging.getLogger(__name__)

# The precision chosen explicitly by an environment in this process, if any.
_chosen_float32 = None


class smefitEnvironment(Environment):
    """smefit Environment class.

    ``float32`` sets the JAX precision of the whole process. ``None`` keeps the
    one an earlier environment chose explicitly, float64 if none did: that is
    how ``smefitAPI`` builds its environment, at import, and importing it from
    within a ``smefit -f32`` run (an external chi2 module, a Bayesian update)
    must not switch the run to float64.
    """

    def __init__(self, float32=None, *args, **kwargs):
        global _chosen_float32
        super().__init__(*args, **kwargs)

        if float32 is None:
            float32 = bool(_chosen_float32)
        else:
            _chosen_float32 = float32
        self.float32 = float32

        if self.float32:
            log.info("Using float32 precision")
            jax.config.update("jax_enable_x64", False)
        else:
            log.info("Using float64 precision")
            jax.config.update("jax_enable_x64", True)

        log.info(f"Running with backend: {jbackend.get_backend().platform}")
