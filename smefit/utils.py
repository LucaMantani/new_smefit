"""
smefit.utils.py

Utility functions for the smefit framework.
"""

import json
import pathlib

import jax
import jax.numpy as jnp


def write_fit_result(fit_result, output_path):
    """Serialise a FitResult to JSON and write it to the output folder.

    Parameters
    ----------
    fit_result : FitResult
    output_path : pathlib.Path
        Directory where ``fit_result.json`` will be written.
    """
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    unc = fit_result.std
    payload = {
        "free_parameters": fit_result.free_parameters,
        "num_data": fit_result.num_data,
        "n_free": fit_result.n_free,
        "ndof": fit_result.ndof,
        "max_loglikelihood": fit_result.max_loglikelihood,
        "chi2": fit_result.chi2_val,
        "chi2_ndof": fit_result.chi2_ndof,
        "logz": fit_result.logz,
        "best_fit_point": fit_result.best_fit_point,
        "std": unc,
        "samples": (
            {name: vals.tolist() for name, vals in fit_result.samples.items()}
            if fit_result.samples is not None
            else None
        ),
    }

    out_file = output_path / "fit_results.json"
    with out_file.open("w") as f:
        json.dump(payload, f, indent=2)


def ensure_list(x):
    """Ensure the input is a list.
    If the input is not a list, wrap it in a list.
    """
    if isinstance(x, list):
        return x
    return [x]


def run_test(eft_model, chi2):

    print(eft_model.coefficients.free_names)
    free = jnp.array([5.0, -3.0])

    print(eft_model.coefficients.resolve(free))

    print(eft_model.forward_map(free))

    print(chi2(free))
    # compute gradient
    grad_chi2 = jax.grad(chi2)(free)
    print(grad_chi2)
