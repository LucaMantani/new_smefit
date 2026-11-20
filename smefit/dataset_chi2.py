"""
smefit.dataset_chi2.py

The chi2 of each dataset of a fit at its best-fit point and at its baseline, and
the same for every fit of a runcard, for the tables comparing them.
"""

from collections.abc import Mapping, Sequence

import jax.numpy as jnp
import pandas as pd
from reportengine import collect

from smefit.core import CoefficientGroup

# The group of a dataset the runcard gives no ``group:`` to.
UNGROUPED = "Other"


def datasets_chi2_values(
    datasets_chi2: Sequence,
    best_fit_point: jnp.ndarray,
    coefficients: CoefficientGroup,
    data_groups: Mapping | None,
) -> pd.DataFrame:
    """The chi2 of each dataset at ``best_fit_point`` and at the baseline of
    ``coefficients``, with its group and size.

    Under ``fits::fitcontext`` the datasets, groups, coefficients and point are
    the fit's own, so this is each of its datasets as the fit was run: at its
    stored best fit, and at its own baseline (the ``baseline_value`` of each free
    coefficient, zero unless set), whatever another fit's is. The baseline is
    read from ``coefficients``, not from each chi2's ``baseline``: an external
    likelihood's is not set.

    Each dataset is evaluated on its diagonal block of ``fit_covmat``, as
    ``datasets_chi2`` builds it: correlations between datasets are left out.
    External likelihoods are rows of their own, as for the Fisher matrices.

    Parameters
    ----------
    datasets_chi2 : list of Chi2
        Per-dataset chi2 objects, external likelihoods included.
    best_fit_point : jnp.ndarray
        The best-fit values of the free coefficients.
    coefficients : smefit.core.CoefficientGroup
        Whose ``baseline_free`` is the baseline point.
    data_groups : dict[str, list[str]] or None
        Group name to its datasets. A dataset in no group is put in
        :data:`UNGROUPED`.

    Returns
    -------
    pd.DataFrame
        Index = dataset names, in the order of ``datasets_chi2``; columns
        ``group``, ``ndat``, ``chi2_best_fit`` and ``chi2_baseline`` (totals,
        not divided by ``ndat``).
    """
    group_of = {
        name: group
        for group, members in (data_groups or {}).items()
        for name in members
    }
    baseline = coefficients.baseline_free
    rows = {
        chi2.name: {
            "group": group_of.get(chi2.name, UNGROUPED),
            "ndat": int(chi2.num_data),
            "chi2_best_fit": float(chi2(best_fit_point)),
            "chi2_baseline": float(chi2(baseline)),
        }
        for chi2 in datasets_chi2
    }
    return pd.DataFrame.from_dict(
        rows,
        orient="index",
        columns=["group", "ndat", "chi2_best_fit", "chi2_baseline"],
    )


# one entry per fit, each evaluated in the fit's own setup
fits_datasets_chi2_values = collect("datasets_chi2_values", ("fits", "fitcontext"))
