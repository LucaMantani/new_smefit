"""
smefit.config.py

Config module of smefit
"""

import logging
import os
import pathlib

import jax
import jax.numpy as jnp
import optax
from reportengine.configparser import ConfigError
from reportengine.namespaces import NSList
from reportengine.report import Config

from smefit.chi2 import Chi2, build_chi2, build_datasets_chi2
from smefit.core import Coefficient, CoefficientGroup, DataGroup, TheoryGroup
from smefit.external_chi2 import load_external_chi2
from smefit.loader import load_dataset, load_theory
from smefit.model import EFTModel
from smefit.paths import USER_PATHS_CONFIG, load_user_paths, resolve_path
from smefit.priors import Prior, _build_dist, _UniformDist
from smefit.projections import Projection
from smefit.rge import load_rge_matrix
from smefit.utils import build_exact_posterior_prior

log = logging.getLogger(__name__)


class smefitConfig(Config):
    """smefit Config class."""

    def parse_data_path(self, data_path):
        """Parse data path, resolving prefix-relative paths from ~/.config/smefit/paths.yaml."""
        data_path = resolve_path(data_path)
        data_path = pathlib.Path(data_path)
        if not data_path.exists():
            log.error(f"data_path {data_path} does not exist.")
            raise ValueError(f"data_path {data_path} does not exist.")
        log.info(f"Using data path: {data_path}")
        return data_path

    def parse_theory_path(self, theory_path):
        """Parse theory path, resolving prefix-relative paths from ~/.config/smefit/paths.yaml."""
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
            dataset = load_dataset(data_path, ds["name"])
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
        return rge

    def produce_init_scale(self, rge):
        """Produce the initial scale (in GeV) at which Wilson coefficients are defined."""
        return float(rge["init_scale"])

    def produce_rge_matrix(self, rge, coefficients, theory, output_path):
        """Produce the stacked RGE matrix for all data points."""
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

        parsed_theories = [
            load_theory(
                theory_path, ds["name"], ds["order"], ds.get("theory_cov", "current")
            )
            for ds in datasets
        ]

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

        self._cached_fit_covmat = covmat

        return covmat

    def parse_coefficients(self, coefficients):
        """Parse coefficients configuration."""
        coeffs = []
        for coeff_name, coeff_info in coefficients.items():
            coeffs.append(Coefficient(name=coeff_name, **coeff_info))
        group = CoefficientGroup(coeffs)
        return group

    def parse_whitening(self, whitening):
        """Parse and validate the optional whitening block."""
        known_keys = {"sigma_prior", "eps"}
        for k in set(whitening.keys()) - known_keys:
            log.warning("Unknown key '%s' in whitening settings.", k)
        return {
            "sigma_prior": float(whitening.get("sigma_prior", 5.0)),
            "eps": float(whitening.get("eps", 1e-8)),
        }

    def produce_whitening_matrix(self, chi2, whitening=None):
        """Produce the whitening matrix W from the chi2 Hessian at c=0.

        Uses the plain chi2 node (already built by produce_chi2) to compute
        H = d²chi2/dc² at c=0, then returns W = L^{-T} where H = L L^T
        (Cholesky). When whitening is disabled (no whitening block in the
        runcard), returns None.
        """
        if whitening is None:
            return None
        eps = whitening["eps"]
        zeros = jnp.zeros(chi2.nparam)
        H = jax.hessian(chi2)(zeros) + eps * jnp.eye(chi2.nparam)
        log.info("Hessian whitening: cond(H) = %.3e", float(jnp.linalg.cond(H)))
        L = jnp.linalg.cholesky(H)
        return jnp.linalg.solve(L.T, jnp.eye(chi2.nparam))  # W = L^{-T}

    def produce_eft_model(self, theory, coefficients, use_quad=False, rge_matrix=None):
        """Produce EFT model mapping coefficients to theory predictions."""
        return EFTModel(theory, coefficients, use_quad, rge_matrix)

    def parse_external_chi2(self, external_chi2):
        """Pass-through parser. Strips 'group' keys and resolves prefix-relative paths."""
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

        if ext_chi2_func is None:
            return Chi2(base_chi2, param_names=free_names, num_data=data.num_data)

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
        )

    def produce_chi2(
        self,
        eft_model=None,
        data=None,
        fit_covmat=None,
        ext_chi2_func=None,
    ):
        """Produce the chi2 function, optionally combining with external chi2s.

        When no datasets are provided, base chi2 is skipped and only external
        contributions are summed.
        """
        return self._build_chi2_impl(eft_model, data, fit_covmat, ext_chi2_func)

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
        output_path,
    ):
        """For a Nested Sampling fit, parses the ultranest_settings namespace from the runcard,
        and ensures the choice of settings is valid.
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
            "log_dir": str(output_path / "ultranest_logs"),
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

    def parse_blackjax_settings(self, settings, output_path):
        """For a BlackJAX fit, parses the blackjax_settings namespace from the runcard,
        and ensures the choice of settings is valid.
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
            "log_dir", str(output_path / "blackjax_logs")
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
            If True, skip optimisation and use c=0 (SM point) as the
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
    # Individual-fit producers — one free coefficient at a time
    # ------------------------------------------------------------------

    def produce_individual_fit_coefficients(self, coefficients):
        """Produce an NSList of free coefficient names for individual fits."""
        return NSList(coefficients.free_names, nskey="individual_fit_coefficient")

    def produce_individual_coefficients(self, coefficients, individual_fit_coefficient):
        """Produce a single-free-parameter coefficient group for individual fits."""
        return coefficients.single_free(individual_fit_coefficient)

    def produce_individual_eft_model(
        self, theory, individual_coefficients, use_quad=False, rge_matrix=None
    ):
        """Produce EFT model for a single-free-parameter individual fit."""
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
    ):
        """Produce chi2 for a single-free-parameter individual fit."""
        return self._build_chi2_impl(
            individual_eft_model,
            data,
            fit_covmat,
            individual_ext_chi2_func,
        )

    def produce_individual_prior(self, individual_coefficients):
        """Produce prior for a single-free-parameter individual fit."""
        return self._build_prior_impl(individual_coefficients)
