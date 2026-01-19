"""
smefit.loader.py

Loader module of smefit
"""

import logging

import jax.numpy as jnp
import yaml

from smefit.core import Dataset
from smefit.utils import ensure_list

log = logging.getLogger(__name__)


def load_dataset(data_path, dataset_name):
    """Load dataset from given path."""
    dataset_path = data_path / f"{dataset_name}.yaml"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset {dataset_name} not found in {data_path}")

    log.info("Loading dataset %s", dataset_name)

    with open(dataset_path) as file:
        dataset = yaml.safe_load(file)

    name = dataset["dataset_name"]
    num_data = int(dataset["num_data"])

    central_values = jnp.atleast_1d(jnp.asarray(dataset["data_central"], dtype=float))
    stat_err = jnp.atleast_1d(jnp.asarray(dataset["statistical_error"], dtype=float))
    syst_err = jnp.atleast_1d(jnp.asarray(dataset["systematics"], dtype=float))

    for arr, label in [
        (central_values, "data_central"),
        (stat_err, "statistical_error"),
        (syst_err, "systematics"),
    ]:
        if len(arr) != num_data:
            raise ValueError(
                f"{name}: {label} length {len(arr)} does not match num_data {num_data}"
            )

    sys_names = ensure_list(dataset["sys_names"])
    sys_types = ensure_list(dataset["sys_type"])

    if len(sys_names) != len(sys_types):
        raise ValueError(f"{name}: sys_names and sys_type length mismatch")

    luminosity = dataset.get("luminosity", None)
    if luminosity is not None:
        if isinstance(luminosity, list):
            if len(luminosity) != num_data:
                raise ValueError(
                    f"{name}: Luminosity length {len(luminosity)} does not match num_data {num_data}"
                )
            luminosity = jnp.asarray(luminosity, dtype=float)
        elif isinstance(luminosity, (int, float)):
            luminosity = jnp.full(num_data, float(luminosity))
        else:
            raise ValueError(f"{name}: Invalid luminosity format")
    else:
        luminosity = jnp.full(num_data, jnp.nan)

    return Dataset(
        name=name,
        num_data=num_data,
        central_values=central_values,
        stat_err=stat_err,
        syst_err=syst_err,
        sys_names=sys_names,
        sys_types=sys_types,
        luminosity=luminosity,
    )
