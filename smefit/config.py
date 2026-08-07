"""
smefit.config.py

Config module of smefit
"""

import logging
import os
import pathlib
from collections.abc import Mapping

import jax.numpy as jnp
import optax
from reportengine.configparser import ConfigError, element_of, explicit_node
from reportengine.namespaces import NSList
from reportengine.report import Config

from smefit.chi2 import Chi2, build_chi2, build_datasets_chi2
from smefit.core import Coefficient, CoefficientGroup, DataGroup, TheoryGroup
from smefit.external_chi2 import load_external_chi2
from smefit.fit_result import Fit
from smefit.loader import load_dataset, load_theory
from smefit.model import EFTModel
from smefit.paths import (
    fetch_fit_if_missing,
    resolve_fit_dir,
    resolve_path,
)
from smefit.priors import Prior, _build_dist, _UniformDist
from smefit.projections import Projection
from smefit.rge import load_rge_matrix
from smefit.utils import build_exact_posterior_prior
from smefit.whitening import (
    _whitening_baseline_shift,
    _whitening_gradient_descent_shift,
)

log = logging.getLogger(__name__)


class smefitConfig(Config):
    """smefit Config class."""

    def parse_data_path(self, data_path):
        """Parse `data_path`, the commondata directory holding `<dataset>.yaml`.

        Accepts an absolute path, or the shareable prefix form
        (`smefit_database/commondata`) resolved through the machine-specific
        `<new_smefit>/.config/paths.yaml` written by `smefit_setup_local`.
        Raises if the resolved directory does not exist.
        """
        data_path = resolve_path(data_path)
        data_path = pathlib.Path(data_path)
        if not data_path.exists():
            log.error(f"data_path {data_path} does not exist.")
            raise ValueError(f"data_path {data_path} does not exist.")
        log.info(f"Using data path: {data_path}")
        return data_path

    def parse_theory_path(self, theory_path):
        """Parse `theory_path`, the directory holding `<dataset>.json` predictions.

        Same resolution rules as `data_path`: absolute, or the shareable
        prefix form (`smefit_database/theory`) resolved through
        `<new_smefit>/.config/paths.yaml`. Every dataset listed in the runcard
        needs a matching JSON here, with the requested `order` as a key.
        """
        theory_path = resolve_path(theory_path)
        theory_path = pathlib.Path(theory_path)
        if not theory_path.exists():
            log.error(f"theory_path {theory_path} does not exist.")
            raise ValueError(f"theory_path {theory_path} does not exist.")
        log.info(f"Using theory path: {theory_path}")
        return theory_path

    def produce_data(self, datasets, data_path):
        """Produce data group object."""
        if hasattr(self, "_cached_data_group"):
            return self._cached_data_group

        parsed_datasets = []
        for ds in datasets:
            dataset = load_dataset(data_path, ds)
            parsed_datasets.append(dataset)

        self._cached_data_group = DataGroup(parsed_datasets)
        return self._cached_data_group

    def produce_data_groups(self, datasets):
        """Build data_groups from inline group: keys on each dataset entry."""
        groups: dict = {}
        for ds in datasets:
            group = ds.get("group")
            if group is not None:
                groups.setdefault(group, []).append(ds["name"])
        for name, group in getattr(self, "_ext_chi2_groups", {}).items():
            groups.setdefault(group, []).append(name)
        return groups if groups else None

    def parse_rge(self, rge):
        """Parse and validate RGE settings."""
        known_keys = {
            "init_scale",
            "obs_scale",
            "smeft_accuracy",
            "yukawa",
            "adm_QCD",
            "rg_matrix",
            "scale_variation",
        }
        for k in set(rge.keys()) - known_keys:
            log.warning("Unknown key '%s' in rge settings.", k)
        if "init_scale" not in rge:
            raise ConfigError("rge", rge, "rge block requires 'init_scale'")
        obs_scale = rge.get("obs_scale", "dynamic")
        if not isinstance(obs_scale, (int, float)) and obs_scale != "dynamic":
            raise ConfigError(
                "obs_scale", obs_scale, "obs_scale must be a float/int or 'dynamic'"
            )
        if "rg_matrix" in rge:
            rge["rg_matrix"] = resolve_path(rge["rg_matrix"])
            fetch_fit_if_missing(pathlib.Path(rge["rg_matrix"]))
        return rge

    def produce_init_scale(self, rge):
        """Produce the initial scale (in GeV) at which Wilson coefficients are defined."""
        return float(rge["init_scale"])

    def produce_rge_matrix(self, coefficients, theory, rge=None, output_path=None):
        """Produce the stacked RGE matrix for all data points.

        Returns ``None`` when the runcard has no ``rge:`` block.

        ``output_path`` is only used to cache the matrix to disk, so it is
        optional: under the `smefit` API there is no output folder (it is a
        reportengine environment attribute, supplied by the CLI's ``-o`` flag)
        and the matrix is simply not cached.
        """
        if rge is None:
            return None

        if hasattr(self, "_cached_rge_matrix"):
            return self._cached_rge_matrix

        coeff_list = sorted(coefficients.names)

        rge_matrix = load_rge_matrix(
            rge_dict=rge,
            coeff_list=coeff_list,
            theory_group=theory,
            save_path=output_path,
        )
        log.info(
            "RGE matrix computed: shape %s, obs operators: %s",
            rge_matrix.stacked_mats.shape,
            rge_matrix.obs_operators,
        )
        self._cached_rge_matrix = rge_matrix
        return rge_matrix

    def produce_theory(self, datasets, theory_path):
        """Produce theory group object."""
        if hasattr(self, "_cached_theory_group"):
            return self._cached_theory_group

        parsed_theories = [load_theory(theory_path, ds) for ds in datasets]

        self._cached_theory_group = TheoryGroup(parsed_theories)
        return self._cached_theory_group

    def produce_fit_covmat(self, data, theory, use_t0=False, use_theory_covmat=False):
        """Produce the covariance matrix to be used in the fit."""
        if hasattr(self, "_cached_fit_covmat"):
            return self._cached_fit_covmat

        if data.names != theory.names:
            raise ValueError(
                f"DataGroup and TheoryGroup contain different datasets.\n"
                f"  data:   {data.names}\n"
                f"  theory: {theory.names}"
            )
        if data.num_data != theory.n_data:
            raise ValueError(
                f"DataGroup and TheoryGroup have different total number of data points: "
                f"{data.num_data} vs {theory.n_data}"
            )
        if use_t0:
            log.info("Using t0 covariance matrix.")
            # build t0 covmat using theory predictions
            theory_predictions = theory.sm_pred
            covmat = data.t0_covmat(theory_predictions)
        else:
            log.info("Using experimental covariance matrix.")
            covmat = data.exp_covmat

        if use_theory_covmat:
            log.info("Adding theory covariance matrix to data covariance matrix.")
            covmat += theory.sm_covmat

        offset = 0
        singular_datasets = []
        for name, n in zip(data.names, data.ndata_list):
            block = covmat[offset : offset + n, offset : offset + n]
            sign, _ = jnp.linalg.slogdet(block)
            if sign == 0:
                singular_datasets.append(name)
            offset += n

        if singular_datasets:
            raise ValueError(
                "The covariance matrix block is singular for the following "
                f"dataset(s): {', '.join(singular_datasets)}."
            )

        self._cached_fit_covmat = covmat

        return covmat

    def parse_coefficients(self, coefficients):
        """Parse the `coefficients:` mapping into a CoefficientGroup.

        Each entry is `Name: {…}`, whose sub-keys are passed verbatim to
        `smefit.core.Coefficient` — so the dataclass fields are the allowed
        sub-keys, and `Coefficient.__post_init__` enforces which combination
        is legal for each kind:

        free (fitted)
            `free: True` (the default); `value`, `expr` and `vars` are
            forbidden. `prior` is optional here — `Coefficient` itself does
            not require it; it is only needed later by `produce_prior`/
            `produce_individual_prior` for sampler actions.
        fixed constant
            `free: False` plus `value`; `prior`, `expr`, `vars` forbidden.
        expression-constrained
            `free: False` plus `expr` and a non-empty `vars` naming other
            coefficients in this same mapping; `prior` and `value` forbidden.

        `baseline_value` (default 0.0) applies to free coefficients only: it
        is the gradient-descent starting point, and the vector returned
        directly under `gradient_descent_settings.sm_solution: True`.
        """
        coeffs = []
        for coeff_name, coeff_info in coefficients.items():
            coeffs.append(Coefficient(name=coeff_name, **coeff_info))
        group = CoefficientGroup(coeffs)
        return group

    def parse_whitening(self, whitening):
        """Parse and validate the optional whitening block."""
        known_keys = {"sigma_prior", "eps", "shift"}
        for k in set(whitening.keys()) - known_keys:
            log.warning("Unknown key '%s' in whitening settings.", k)
        shift = whitening.get("shift", "baseline")
        allowed_shifts = {"baseline", "gradient_descent"}
        if shift not in allowed_shifts:
            raise ConfigError(
                f"whitening.shift must be one of {sorted(allowed_shifts)}, "
                f"got '{shift}'"
            )
        return {
            "sigma_prior": float(whitening.get("sigma_prior", 5.0)),
            "eps": float(whitening.get("eps", 1e-8)),
            "shift": shift,
        }

    @explicit_node
    def produce_whitening_transformation(self, whitening=None):
        """Dispatch to the correct whitening-transform builder.

        This must stay an ExplicitNode-returning method on smefitConfig (not
        a plain provider function) so the decision of whether gd_best_fit is
        needed can be made dynamically: only ``whitening["shift"] ==
        "gradient_descent"`` triggers a dependency on ``gd_best_fit`` (and
        hence ``gradient_descent_settings``). When whitening is disabled, the
        worker must still be a zero-argument callable (reportengine's
        ExplicitNode dispatch calls inspect.signature on it), so a plain
        ``None`` cannot be returned directly.
        """
        if whitening is None:
            return lambda: None
        if whitening["shift"] == "gradient_descent":
            log.info("Whitening: centering on the gradient-descent best-fit point.")
            return _whitening_gradient_descent_shift
        log.info("Whitening: centering on the coefficients' baseline point.")
        return _whitening_baseline_shift

    def produce_eft_model(self, theory, coefficients, rge_matrix, use_quad=False):
        """Produce EFT model mapping coefficients to theory predictions.

        ``rge_matrix`` is deliberately required: giving it a ``None`` default
        would let reportengine swallow any failure to build it (it catches
        ``KeyError`` and substitutes the default) and hand back a model with no
        RGE running, silently.
        """
        return EFTModel(theory, coefficients, use_quad, rge_matrix)

    def parse_external_chi2(self, external_chi2):
        """Parse the `external_chi2:` mapping of custom likelihood modules.

        Each entry is `ClassName: {path: …, …}`, where `path` points at the
        Python module defining that class (prefix-relative paths are resolved
        here, as is `rg_matrix`). Every other key is forwarded verbatim to the
        class constructor, which must also accept `coefficients=` and
        `rge_dict=` and expose `compute_chi2`, `num_data` and `param_names`.

        `group` is the one exception: it is stripped here and kept aside for
        report/Fisher aggregation rather than forwarded.

        A runcard may define `external_chi2` with no `datasets:` at all, in
        which case the fit runs on the external likelihoods alone.
        """
        self._ext_chi2_groups = {}
        cleaned = {}
        for name, cfg in external_chi2.items():
            group = cfg.get("group")
            if group is not None:
                self._ext_chi2_groups[name] = group
            entry = {k: v for k, v in cfg.items() if k != "group"}
            if "path" in entry:
                entry["path"] = resolve_path(entry["path"])
            cleaned[name] = entry
        return cleaned

    def produce_ext_chi2_func(self, coefficients, external_chi2, rge=None):
        """Load and wrap external chi2 modules into Chi2 objects."""
        return load_external_chi2(external_chi2, coefficients, rge_dict=rge)

    def _build_chi2_impl(
        self,
        eft_model=None,
        data=None,
        fit_covmat=None,
        ext_chi2_func=None,
        coefficients=None,
    ):
        """Shared chi2 build logic used by both joint and individual producers."""

        if data is None and ext_chi2_func is None:
            raise ConfigError(
                "chi2",
                None,
                "No data provided and no external_chi2 configured. "
                "At least one source of chi2 is required.",
            )

        if data is not None:
            if eft_model is None or fit_covmat is None:
                raise ValueError(
                    "Data provided but eft_model or fit_covmat is None. "
                    "These are required to build the base chi2 from datasets. "
                    "It is possible that an error occurred in producing one of these nodes, "
                    "so check for errors in their production."
                )
            base_chi2 = build_chi2(eft_model, data, fit_covmat)
        else:
            base_chi2 = None

        free_names = (
            eft_model.coefficients.free_names
            if eft_model is not None
            else ext_chi2_func[0].param_names
        )

        # baseline ("default") values of the free coefficients; a property of the
        # coefficients dictionary, so it applies to external-chi2-only fits too.
        baseline = coefficients.baseline_free if coefficients is not None else None

        if ext_chi2_func is None:
            return Chi2(
                base_chi2,
                param_names=free_names,
                num_data=data.num_data,
                baseline=baseline,
            )

        if base_chi2 is None:

            def total_fn(coeffs):
                return sum(ext(coeffs) for ext in ext_chi2_func)

        else:

            def total_fn(coeffs):
                return base_chi2(coeffs) + sum(ext(coeffs) for ext in ext_chi2_func)

        tot_num_data = (data.num_data if data else 0) + sum(
            ext.num_data for ext in ext_chi2_func
        )

        return Chi2(
            total_fn,
            param_names=free_names,
            num_data=tot_num_data,
            has_external=True,
            baseline=baseline,
        )

    def produce_chi2(
        self,
        eft_model=None,
        data=None,
        fit_covmat=None,
        ext_chi2_func=None,
        coefficients=None,
    ):
        """Produce the chi2 function, optionally combining with external chi2s.

        When no datasets are provided, base chi2 is skipped and only external
        contributions are summed.
        """
        return self._build_chi2_impl(
            eft_model, data, fit_covmat, ext_chi2_func, coefficients
        )

    def produce_datasets_chi2(
        self,
        eft_model=None,
        data=None,
        fit_covmat=None,
        ext_chi2_func=None,
    ):
        """Produce a list of per-dataset chi2 objects.

        Regular datasets each get their own chi2 (diagonal block of fit_covmat).
        External chi2 contributions are appended as individual entries.
        """

        chi2_list = build_datasets_chi2(eft_model, data, fit_covmat)

        if ext_chi2_func is not None:
            chi2_list.extend(ext_chi2_func)

        return chi2_list

    def parse_ultranest_settings(
        self,
        settings,
        output_path=None,
    ):
        """For a Nested Sampling fit, parses the ultranest_settings namespace from the runcard,
        and ensures the choice of settings is valid.

        ``output_path`` is optional because it is a reportengine environment
        attribute that only the CLI supplies; without it there is no folder to
        derive ``log_dir`` from, so the user must set it explicitly.
        """

        # Warn about unknown keys
        known_keys = {
            "ReactiveNS_settings",
            "Run_settings",
            "SliceSampler_settings",
            "ultranest_seed",
            "sampler_plot",
        }
        for k in settings.keys() - known_keys:
            log.warning(
                ConfigError(
                    f"Key '{k}' in ultranest_settings not known.", k, known_keys
                )
            )

        # Defaults for ReactiveNS_settings (log_dir depends on output_path)
        reactive_defaults = {
            "log_dir": (
                str(output_path / "ultranest_logs") if output_path is not None else None
            ),
            "resume": False,
            "vectorized": False,
        }
        reactive_user = settings.get("ReactiveNS_settings", {})

        ultranest_settings = {
            "ultranest_seed": settings.get("ultranest_seed", 123456),
            "sampler_plot": settings.get("sampler_plot", False),
            "ReactiveNS_settings": {**reactive_defaults, **reactive_user},
            "Run_settings": settings.get("Run_settings", {}),
            "SliceSampler_settings": settings.get("SliceSampler_settings", {}),
        }

        # Validate resume: if resuming, the logs directory must exist
        reactive = ultranest_settings["ReactiveNS_settings"]
        if reactive["resume"]:
            if not os.path.exists(reactive["log_dir"]):
                raise FileNotFoundError(
                    f"Could not find previous ultranest fit at {reactive['log_dir']}."
                )
            log.info(f"Resuming ultranest fit from {reactive['log_dir']}.")
        else:
            # UltraNest expects "overwrite" instead of False
            reactive["resume"] = "overwrite"

        return ultranest_settings

    def parse_blackjax_settings(self, settings, output_path=None):
        """For a BlackJAX fit, parses the blackjax_settings namespace from the runcard,
        and ensures the choice of settings is valid.

        ``output_path`` is optional for the same reason as in
        ``parse_ultranest_settings``: without an output folder there is nothing
        to derive ``log_dir`` from, so the user must set it explicitly.
        """

        # Begin by checking that the user-supplied keys are known; warn the user otherwise.
        known_keys = {
            "n_posterior_samples",
            "n_live",
            "repeats",
            "delete_fraction",
            "log_precision",
            "posterior_resampling_seed",
            "seed",
            "log_dir",
        }

        kdiff = settings.keys() - known_keys
        for k in kdiff:
            log.warning(
                ConfigError(f"Key '{k}' in blackjax_settings not known.", k, known_keys)
            )

        # Now construct the blackjax_settings dictionary
        blackjax_settings = {}

        # Extract settings and set default values
        blackjax_settings["n_posterior_samples"] = settings.get(
            "n_posterior_samples", 1000
        )
        blackjax_settings["n_live"] = settings.get("n_live", 500)
        blackjax_settings["repeats"] = settings.get("repeats", 3)
        blackjax_settings["delete_fraction"] = settings.get("delete_fraction", 0.5)
        blackjax_settings["log_precision"] = settings.get("log_precision", -2)
        blackjax_settings["seed"] = settings.get("seed", 0)
        blackjax_settings["posterior_resampling_seed"] = settings.get(
            "posterior_resampling_seed", 123456
        )
        # Set directory where blackjax_logs will be saved
        blackjax_settings["log_dir"] = settings.get(
            "log_dir",
            str(output_path / "blackjax_logs") if output_path is not None else None,
        )

        return blackjax_settings

    def parse_optimizer_settings(self, settings):
        """Parse the optimizer_settings block.

        Keys
        ----
        optimizer : str, default "adam"
            Name of any ``optax`` optimizer factory (e.g. "adam", "sgd").
        optimizer_hyperparams : dict, default {"learning_rate": 1e-2}
            Keyword arguments forwarded verbatim to the optimizer constructor.
        clipnorm : float or None, default None
            If set, gradients are clipped to this global norm before the update.
        scheduler : dict or None, default None
            Learning-rate schedule.  Sub-keys:
              name   – name of any ``optax`` schedule factory
              params – kwargs forwarded to the schedule factory
        """
        known_keys = {"optimizer", "optimizer_hyperparams", "clipnorm", "scheduler"}
        for k in set(settings.keys()) - known_keys:
            log.warning("Unknown key '%s' in optimizer_settings.", k)
        return dict(settings)

    def produce_optimizer(self, optimizer_settings=None):
        """Build and return an optax optimizer from optimizer_settings.

        When ``optimizer_settings`` is absent from the runcard, falls back to
        Adam with learning_rate=1e-2.
        """
        settings = optimizer_settings or {}
        opt_name = settings.get("optimizer", "adam")
        hyperparams = dict(
            settings.get("optimizer_hyperparams", {"learning_rate": 1e-2})
        )

        # Inject learning-rate schedule if requested
        scheduler = settings.get("scheduler")
        if scheduler is not None:
            schedule_fn = getattr(optax, scheduler["name"])(**scheduler["params"])
            hyperparams["learning_rate"] = schedule_fn

        base_opt = getattr(optax, opt_name)(**hyperparams)

        clipnorm = settings.get("clipnorm")
        if clipnorm is not None:
            log.info(
                "Optimizer: %s(%s) with gradient clipping (clipnorm=%.3g)",
                opt_name,
                ", ".join(f"{k}={v}" for k, v in hyperparams.items()),
                clipnorm,
            )
            return optax.chain(optax.clip_by_global_norm(float(clipnorm)), base_opt)

        log.info(
            "Optimizer: %s(%s)%s",
            opt_name,
            ", ".join(f"{k}={v}" for k, v in hyperparams.items()),
            f" with {scheduler['name']} scheduler" if scheduler is not None else "",
        )
        return base_opt

    def parse_gradient_descent_settings(self, settings):
        """Parse optional settings for the gradient-descent best-fit node.

        Keys
        ----
        sm_solution : bool, default False
            If True, skip optimisation and use coefficients baseline values as the
            best-fit point.
        n_steps : int, default 2000
            Maximum number of gradient-descent steps.
        tol : float, default 1e-8
            Gradient-norm convergence threshold.
        """
        known_keys = {"sm_solution", "n_steps", "tol"}
        for k in set(settings.keys()) - known_keys:
            log.warning("Unknown key '%s' in gradient_descent_settings.", k)
        return {
            "sm_solution": bool(settings.get("sm_solution", False)),
            "n_steps": int(settings.get("n_steps", 2000)),
            "tol": float(settings.get("tol", 1e-8)),
        }

    def parse_hessian_settings(self, settings):
        """Parse optional settings for the Hessian fit.

        Keys
        ----
        n_samples : int, default 10000
            Number of Gaussian posterior samples to draw.
        seed : int, default 42
            Random seed for sample generation.
        """
        known_keys = {"n_samples", "seed"}
        for k in set(settings.keys()) - known_keys:
            log.warning("Unknown key '%s' in hessian_settings.", k)
        return {
            "n_samples": int(settings.get("n_samples", 10000)),
            "seed": int(settings.get("seed", 42)),
        }

    def parse_bayesian_update_path(self, bayesian_update_path):
        """Parse and validate the path to a previous fit for Bayesian updating."""
        p = pathlib.Path(bayesian_update_path)
        if not p.exists():
            raise ConfigError(f"Directory not found at {bayesian_update_path}")
        if not (p / "fit_results.json").exists():
            raise ConfigError(f"fit_results.json not found at {bayesian_update_path}")
        if not (p / "input" / "runcard.yaml").exists():
            raise ConfigError(f"input/runcard.yaml not found at {bayesian_update_path}")
        return p

    def _build_prior_impl(self, coefficients):
        """Shared prior build logic used by both joint and individual producers."""
        specs = coefficients.prior_specs()
        for name, spec in specs.items():
            if spec is None:
                raise ValueError(f"Free coefficient '{name}' has no prior defined.")
        dists = [_build_dist(spec) for spec in specs.values()]
        return Prior(dists, coefficients.free_names, specs=specs)

    def produce_prior(
        self,
        coefficients,
        datasets=None,
        external_chi2=None,
        whitening=None,
        bayesian_update_path=None,
    ):
        """Produce joint prior over all free coefficients.

        When ``bayesian_update_path`` is set, returns an ExactPosteriorPrior
        that encodes the exact posterior from the previous fit.
        """
        if bayesian_update_path is not None:
            log.info(
                f"Producing ExactPosteriorPrior from previous fit at {bayesian_update_path}"
            )
            if whitening is not None:
                raise ConfigError(
                    "whitening is not compatible with bayesian_update_path: "
                    "ExactPosteriorPrior is defined in physical space and cannot be whitened."
                )
            return build_exact_posterior_prior(
                bayesian_update_path, coefficients, datasets, external_chi2
            )

        if whitening is not None:
            sigma = whitening["sigma_prior"]
            spec = {"dist": "uniform", "low": -sigma, "high": sigma}
            specs = {name: spec for name in coefficients.free_names}
            dists = [_UniformDist(-sigma, sigma) for _ in coefficients.free_names]
            return Prior(dists, coefficients.free_names, specs=specs)

        return self._build_prior_impl(coefficients)

    def parse_pseudodata_settings(self, pseudodata_settings):
        """Parse pseudodata projection settings."""
        known_keys = {"lumi_new", "noise", "seed", "fred_tot", "fred_sys"}
        for k in set(pseudodata_settings.keys()) - known_keys:
            log.warning("Unknown key '%s' in pseudodata_settings.", k)
        noise = pseudodata_settings.get("noise", "L0")
        if noise not in {"L0", "L1"}:
            raise ConfigError("noise", noise, "noise must be 'L0' or 'L1'")
        return {
            "lumi_new": pseudodata_settings.get("lumi_new", None),
            "noise": noise,
            "seed": pseudodata_settings.get("seed", None),
            "fred_tot": float(pseudodata_settings.get("fred_tot", 1.0)),
            "fred_sys": float(pseudodata_settings.get("fred_sys", 1.0)),
        }

    def produce_pseudodata(
        self,
        data,
        theory,
        pseudodata_settings,
        use_theory_covmat=False,
        eft_model=None,
    ):
        """Produce a pseudodata DataGroup via projections."""
        return Projection(
            data=data,
            theory=theory,
            pseudodata_settings=pseudodata_settings,
            use_theory_covmat=use_theory_covmat,
            eft_model=eft_model,
        ).build_data_group()

    # ------------------------------------------------------------------
    # Previously run fits
    # ------------------------------------------------------------------

    @element_of("fits")
    def parse_fit(self, fit: str | Mapping) -> Fit:
        """Load one previously run fit from disk into a :class:`Fit`.

        ``fits`` is the list form of this key, generated by ``element_of``, and
        is what a runcard writes: a list of the entries below, each loaded into
        a :class:`Fit`. It is a namespace list, so a provider can take a single
        ``fit`` and be collected over ``("fits",)``.

        Each entry is the name of a fit, or a mapping

            - name: my_fit                     # mandatory, the fit directory name
              path: smefit_results/fits        # optional, where to look for it
              label: '$\\mathrm{My\\ fit}$'      # optional, the legend label

        Without ``path`` the fit is looked up in ``smefit_results/fits/`` and
        downloaded from the server if it is not there yet. ``path`` is resolved
        through ``.config/paths.yaml`` like any other path.

        The name is how the fit is referred to everywhere downstream: it is the
        key of the per-fit plot settings, and the legend label when no ``label``
        is given. A ``label`` is passed to matplotlib verbatim, so it can be raw
        LaTeX (quote it in YAML to keep the backslashes).
        """
        entry = {"name": fit} if isinstance(fit, str) else dict(fit)
        if "name" not in entry:
            raise ConfigError(f"Each fits entry requires a 'name': {entry}")

        known_keys = {"name", "path", "label"}
        for k in set(entry.keys()) - known_keys:
            log.warning("Unknown key '%s' in fits entry.", k)

        label = entry.get("label")
        if label is not None and not isinstance(label, str):
            raise ConfigError(
                f"The 'label' of fit '{entry['name']}' must be a string, "
                f"got {label!r}."
            )

        try:
            path = resolve_fit_dir(entry["name"], entry.get("path"))
        except (FileNotFoundError, ValueError) as e:
            raise ConfigError(str(e)) from e

        try:
            # A label is how the runcard chooses to present the fit, not
            # something the fit directory knows about.
            return Fit.from_folder(path, label=label)
        except (KeyError, OSError, ValueError) as e:
            raise ConfigError(
                f"Could not load fit '{entry['name']}' from {path}: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Individual-fit producers — one free coefficient at a time
    # ------------------------------------------------------------------

    def produce_individual_fit_coefficients(self, coefficients):
        """Produce an NSList of free coefficient names for individual fits."""
        return NSList(coefficients.free_names, nskey="individual_fit_coefficient")

    def produce_individual_coefficients(self, coefficients, individual_fit_coefficient):
        """Produce a single-free-parameter coefficient group for individual fits."""
        return coefficients.single_free(individual_fit_coefficient)

    def produce_individual_eft_model(
        self, theory, individual_coefficients, rge_matrix, use_quad=False
    ):
        """Produce EFT model for a single-free-parameter individual fit.

        ``rge_matrix`` is required for the same reason as in
        ``produce_eft_model``.
        """
        return EFTModel(theory, individual_coefficients, use_quad, rge_matrix)

    def produce_individual_ext_chi2_func(
        self, individual_coefficients, external_chi2, rge=None
    ):
        """Load and wrap external chi2 modules for a single-free-parameter individual fit."""
        return load_external_chi2(external_chi2, individual_coefficients, rge_dict=rge)

    def produce_individual_chi2(
        self,
        individual_eft_model=None,
        data=None,
        fit_covmat=None,
        individual_ext_chi2_func=None,
        individual_coefficients=None,
    ):
        """Produce chi2 for a single-free-parameter individual fit."""
        return self._build_chi2_impl(
            individual_eft_model,
            data,
            fit_covmat,
            individual_ext_chi2_func,
            individual_coefficients,
        )

    def produce_individual_prior(self, individual_coefficients):
        """Produce prior for a single-free-parameter individual fit."""
        return self._build_prior_impl(individual_coefficients)
