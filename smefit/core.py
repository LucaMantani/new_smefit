"""
smefit.core.py

Core module of smefit, containing the main data classes for the framework.
"""

from dataclasses import dataclass
from typing import List, Optional

import jax.numpy as jnp


@dataclass
class Dataset:
    """Class representing a dataset in smefit."""

    name: str
    num_data: int
    central_values: jnp.ndarray
    stat_err: jnp.ndarray
    syst_err: jnp.ndarray
    sys_names: List[str]
    sys_types: List[str]
    luminosity: Optional[jnp.ndarray] = None
