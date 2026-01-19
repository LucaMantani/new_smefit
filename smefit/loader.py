"""
smefit.loader.py

Loader module of smefit
"""

import logging

import jax.numpy as jnp
import yaml

from smefit.core import Dataset

log = logging.getLogger(__name__)


def load_dataset(data_path, dataset_name):
    """Load dataset from given path."""
    dataset_path = data_path / f"{dataset_name}.yaml"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset {dataset_name} not found in {data_path}")

    log.info(f"Loading dataset {dataset_name}")
    with open(dataset_path) as file:
        dataset = yaml.safe_load(file)

    name = dataset["dataset_name"]
    num_data = dataset["num_data"]
    central_values = jnp.atleast_1d(dataset["data_central"])
    stat_err = jnp.atleast_1d(dataset["statistical_error"])
    syst_err = jnp.atleast_1d(dataset["systematics"])
    sys_names = dataset["sys_names"]
    sys_types = dataset["sys_type"]

    # Load luminosity if present
    luminosity = dataset.get("luminosity", None)
    if luminosity is not None:
        if isinstance(luminosity, list):
            # check it is same length as num_data
            if len(luminosity) != num_data:
                raise ValueError(
                    f"{name}: Luminosity length {len(luminosity)} does not match num_data {num_data}"
                )
            luminosity = jnp.array(luminosity)
        elif isinstance(luminosity, (int, float)):
            luminosity = jnp.array([luminosity] * num_data)
        else:
            raise ValueError(f"{name}: Invalid luminosity format")

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
