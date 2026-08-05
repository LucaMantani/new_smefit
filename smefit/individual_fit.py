"""
smefit.individual_fit

One-at-a-time individual parameter fits using reportengine's collect + NSList.

Each free coefficient gets its own DAG namespace via ``individual_fit_coefficients``.
The ``produce_individual_*`` methods in ``config.py`` rebuild the coefficient group,
EFT model, chi2, and prior per namespace — the existing ``analytic_fit`` and
``ultranest_fit`` functions are called directly with zero reimplemented fit logic.
"""

import copy
import logging
import pathlib

from reportengine import collect

from smefit.analytic_fit import analytic_fit
from smefit.blackjax_fit import blackjax_fit
from smefit.gradient_descent import gd_best_fit
from smefit.hessian_fit import hessian_fit
from smefit.ultranest_fit import ultranest_fit

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider functions — one call per free coefficient via namespace expansion
# ---------------------------------------------------------------------------


def _name_after_coefficient(fit, coefficient):
    """Name a member of an individual fit after the coefficient it fitted.

    That is the subdirectory it is written to — ``individual_fits/<coefficient>``
    — so it is the name the member is loaded back under.
    """
    fit.fit_name = coefficient
    return fit


def individual_analytic_fit(
    individual_eft_model,
    data,
    fit_covmat,
    individual_chi2,
    individual_fit_coefficient,
    n_samples=10000,
    seed=42,
):
    """Analytic fit for a single free coefficient.

    Pure pass-through to ``analytic_fit`` — the DAG has already built an
    ``individual_eft_model`` and ``individual_chi2`` scoped to one free parameter.
    """
    return _name_after_coefficient(
        analytic_fit(
            individual_eft_model, data, fit_covmat, individual_chi2, n_samples, seed
        ),
        individual_fit_coefficient,
    )


def individual_ultranest_fit(
    individual_prior,
    individual_chi2,
    individual_coefficients,
    ultranest_settings,
    individual_fit_coefficient,
    use_quad=False,
):
    """UltraNest fit for a single free coefficient.

    Customises the log directory to a per-coefficient subdirectory, then
    delegates to ``ultranest_fit``.
    """
    settings = copy.deepcopy(ultranest_settings)
    settings["ReactiveNS_settings"]["log_dir"] = str(
        pathlib.Path(settings["ReactiveNS_settings"]["log_dir"])
        / individual_fit_coefficient
    )
    settings["ReactiveNS_settings"]["resume"] = "overwrite"

    return _name_after_coefficient(
        ultranest_fit(
            individual_prior,
            individual_chi2,
            individual_coefficients,
            settings,
            use_quad=use_quad,
        ),
        individual_fit_coefficient,
    )


def individual_blackjax_fit(
    individual_prior,
    individual_chi2,
    individual_coefficients,
    blackjax_settings,
    individual_fit_coefficient,
    use_quad=False,
):
    """BlackJAX fit for a single free coefficient.

    Redirects log_dir to a per-coefficient subdirectory, then delegates
    to ``blackjax_fit``.
    """
    settings = copy.deepcopy(blackjax_settings)
    settings["log_dir"] = str(
        pathlib.Path(settings["log_dir"]) / individual_fit_coefficient
    )
    return _name_after_coefficient(
        blackjax_fit(
            individual_prior,
            individual_chi2,
            individual_coefficients,
            settings,
            use_quad=use_quad,
        ),
        individual_fit_coefficient,
    )


def individual_gd_best_fit(individual_chi2, optimizer, gradient_descent_settings):
    """Best-fit point for a single free coefficient.

    Pure pass-through to ``gd_best_fit`` — the DAG has already built an
    ``individual_chi2`` scoped to one free parameter.
    """
    return gd_best_fit(individual_chi2, optimizer, gradient_descent_settings)


def individual_hessian_fit(
    individual_eft_model,
    individual_chi2,
    individual_gd_best_fit,
    hessian_settings,
    individual_fit_coefficient,
):
    """Hessian fit for a single free coefficient.

    Pure pass-through to ``hessian_fit`` — the DAG has already built an
    ``individual_eft_model``, ``individual_chi2``, and ``individual_gd_best_fit``
    scoped to one free parameter.
    """
    return _name_after_coefficient(
        hessian_fit(
            individual_eft_model,
            individual_chi2,
            individual_gd_best_fit,
            hessian_settings,
        ),
        individual_fit_coefficient,
    )


# ---------------------------------------------------------------------------
# collect instances — reportengine auto-discovers these at module level
# ---------------------------------------------------------------------------

individual_analytic_fits = collect(
    "individual_analytic_fit", ("individual_fit_coefficients",)
)
individual_ultranest_fits = collect(
    "individual_ultranest_fit", ("individual_fit_coefficients",)
)
individual_blackjax_fits = collect(
    "individual_blackjax_fit", ("individual_fit_coefficients",)
)
individual_hessian_fits = collect(
    "individual_hessian_fit", ("individual_fit_coefficients",)
)
