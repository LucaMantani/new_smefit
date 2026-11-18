"""
smefit.dataset_chi2.py

The chi2 of each dataset of a fit at its best-fit point, and the same for every
fit of a runcard, for the tables comparing them.
"""

import pandas as pd
from reportengine import collect

# The group of a dataset the runcard gives no ``group:`` to.
UNGROUPED = "Other"


def best_fit_datasets_chi2(datasets_chi2, best_fit_point, data_groups):
    """The chi2 of each dataset at ``best_fit_point``, with its group and size.

    Under ``fits::fitcontext`` the datasets, groups and point are the fit's own
    (its stored best fit), so this is the chi2 of each of its datasets as the fit
    was run. Each dataset is evaluated on its diagonal block of ``fit_covmat``,
    as ``datasets_chi2`` builds it: correlations between datasets are left out.
    External likelihoods are rows of their own, as for the Fisher matrices.

    Parameters
    ----------
    datasets_chi2 : list of Chi2
        Per-dataset chi2 objects, external likelihoods included.
    best_fit_point : jnp.ndarray
        The free coefficients at which each chi2 is evaluated.
    data_groups : dict[str, list[str]] or None
        Group name to its datasets. A dataset in no group is put in
        :data:`UNGROUPED`.

    Returns
    -------
    pd.DataFrame
        Index = dataset names, in the order of ``datasets_chi2``; columns
        ``group``, ``ndat`` and ``chi2`` (the total, not divided by ``ndat``).
    """
    group_of = {
        name: group
        for group, members in (data_groups or {}).items()
        for name in members
    }
    rows = {
        chi2.name: {
            "group": group_of.get(chi2.name, UNGROUPED),
            "ndat": int(chi2.num_data),
            "chi2": float(chi2(best_fit_point)),
        }
        for chi2 in datasets_chi2
    }
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=["group", "ndat", "chi2"]
    )


# one entry per fit, each evaluated in the fit's own setup
fits_best_fit_datasets_chi2 = collect("best_fit_datasets_chi2", ("fits", "fitcontext"))
