"""
smefit.config.py

Config module of smefit
"""

import logging
import os
import pathlib

from reportengine.configparser import Config, ConfigError
from reportengine.namespaces import NSList

from smefit.chi2 import Chi2, build_chi2
from smefit.core import Coefficient, CoefficientGroup, DataGroup, TheoryGroup
from smefit.external_chi2 import load_external_chi2
from smefit.loader import load_dataset, load_theory
from smefit.model import EFTModel
from smefit.priors import Prior, _build_dist
from smefit.rge import load_rge_matrix

log = logging.getLogger(__name__)


class smefitConfig(Config):
    """smefit Config class."""

    def parse_data_path(self, data_path):
        """Parse data path."""
        data_path = pathlib.Path(data_path)
        # Verify it exists
        if not data_path.exists():
            log.error(f"data_path {data_path} does not exist.")
            raise ValueError(f"data_path {data_path} does not exist.")
        log.info(f"Using data path: {data_path}")
        return data_path

    def parse_theory_path(self, theory_path):
        """Parse theory path."""
        theory_path = pathlib.Path(theory_path)
        # Verify it exists
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

        obs_scale = rge.get("obs_scale", "dynamic")
        if isinstance(obs_scale, (float, int)):
            scales = [float(obs_scale)]
        else:
            # dynamic: use per-data-point scales owned by TheoryGroup
            scale_variation = rge.get("scale_variation", 1.0)
            scales = theory.scales.tolist()

            if scale_variation != 1.0:
                log.info("Applying scale variation of %s.", scale_variation)
                scales = [s * scale_variation for s in scales]

        rge_matrix = load_rge_matrix(
            rge_dict=rge, coeff_list=coeff_list, scales=scales, save_path=output_path
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
            load_theory(theory_path, ds["name"], ds["order"]) for ds in datasets
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
        # Validate that all vars in constrained coefficients refer to free coefficients
        free_names = set(group.free_names)
        for coeff in group.fixed_coeffs:
            if coeff.vars:
                unknown = [v for v in coeff.vars if v not in free_names]
                if unknown:
                    raise ValueError(
                        f"Coefficient '{coeff.name}': vars {unknown} are not free coefficients."
                    )
        return group

    def produce_eft_model(self, theory, coefficients, use_quad=False, rge_matrix=None):
        """Produce EFT model mapping coefficients to theory predictions."""
        return EFTModel(theory, coefficients, use_quad, rge_matrix)

    def parse_external_chi2(self, external_chi2):
        """Pass-through parser so reportengine can resolve external_chi2 as a node."""
        return external_chi2

    def _build_chi2_impl(
        self,
        coefficients,
        datasets=None,
        eft_model=None,
        data=None,
        fit_covmat=None,
        external_chi2=None,
        rge=None,
    ):
        """Shared chi2 build logic used by both joint and individual producers."""
        if not datasets and not external_chi2:
            raise ConfigError(
                "chi2",
                None,
                "No datasets provided and no external_chi2 configured. "
                "At least one source of chi2 is required.",
            )

        if datasets:
            base_chi2 = build_chi2(eft_model, data, fit_covmat)
        else:
            base_chi2 = None

        if not external_chi2:
            return Chi2(base_chi2)

        ext_modules = load_external_chi2(external_chi2, coefficients, rge_dict=rge)

        if base_chi2 is None:

            def total_fn(coeffs):
                return sum(ext(coeffs) for ext in ext_modules)

        else:

            def total_fn(coeffs):
                return base_chi2(coeffs) + sum(ext(coeffs) for ext in ext_modules)

        return Chi2(total_fn, has_external=True)

    def produce_chi2(
        self,
        coefficients,
        datasets=None,
        eft_model=None,
        data=None,
        fit_covmat=None,
        external_chi2=None,
        rge=None,
    ):
        """Produce the chi2 function, optionally combining with external chi2s.

        When no datasets are provided, base chi2 is skipped and only external
        contributions are summed.
        """
        return self._build_chi2_impl(
            coefficients, datasets, eft_model, data, fit_covmat, external_chi2, rge
        )

    def parse_ultranest_settings(
        self,
        settings,
        output_path,
    ):
        """For a Nested Sampling fit, parses the ultranest_settings namespace from the runcard,
        and ensures the choice of settings is valid.
        """

        # Begin by checking that the user-supplied keys are known; warn the user otherwise.
        known_keys = {
            "ReactiveNS_settings",
            "Run_settings",
            "SliceSampler_settings",
            "ultranest_seed",
            "sampler_plot",
        }

        kdiff = settings.keys() - known_keys
        for k in kdiff:
            log.warning(
                ConfigError(
                    f"Key '{k}' in ultranest_settings not known.", k, known_keys
                )
            )

        # Now construct the ultranest_settings dictionary, checking the parameter combinations are
        # valid
        ultranest_settings = {}

        # Set the ultranest seed
        ultranest_settings["ultranest_seed"] = settings.get("ultranest_seed", 123456)

        # Parse internal settings, if they are not mentioned, set to empty dict
        ultranest_settings["ReactiveNS_settings"] = settings.get(
            "ReactiveNS_settings", {}
        )
        ultranest_settings["Run_settings"] = settings.get("Run_settings", {})
        ultranest_settings["SliceSampler_settings"] = settings.get(
            "SliceSampler_settings", {}
        )

        # set sampler plot to False by default
        ultranest_settings["sampler_plot"] = settings.get("sampler_plot", False)

        # Check that the ReactiveNS_settings key was provided, if not set to default
        if ultranest_settings["ReactiveNS_settings"]:
            # Set the directory where the ultranest logs will be stored; by default
            # they are stored in output_path/ultranest_logs
            ultranest_settings["ReactiveNS_settings"]["log_dir"] = settings[
                "ReactiveNS_settings"
            ].get("log_dir", str(output_path / "ultranest_logs"))

            ultranest_settings["ReactiveNS_settings"]["resume"] = settings[
                "ReactiveNS_settings"
            ].get("resume", False)

            ultranest_settings["ReactiveNS_settings"]["vectorized"] = settings[
                "ReactiveNS_settings"
            ].get("vectorized", False)
        else:
            ultranest_settings["ReactiveNS_settings"]["log_dir"] = str(
                output_path / "ultranest_logs"
            )
            ultranest_settings["ReactiveNS_settings"]["resume"] = False
            ultranest_settings["ReactiveNS_settings"]["vectorized"] = False

        # In the case that the fit is resuming from a previous ultranest fit, the logs
        # directory must exist
        if ultranest_settings["ReactiveNS_settings"]["resume"]:
            if not os.path.exists(ultranest_settings["ReactiveNS_settings"]["log_dir"]):
                raise FileNotFoundError(
                    "Could not find previous ultranest fit at "
                    + str(ultranest_settings["ReactiveNS_settings"]["log_dir"])
                    + "."
                )

            log.info(
                "Resuming ultranest fit from "
                + str(ultranest_settings["ReactiveNS_settings"]["log_dir"])
                + "."
            )

        # If the resume option is false, ultranest expects "overwrite" instead
        if not ultranest_settings["ReactiveNS_settings"]["resume"]:
            ultranest_settings["ReactiveNS_settings"]["resume"] = "overwrite"

        return ultranest_settings

    def produce_individual_fit_coefficients(self, coefficients):
        """Produce an NSList of free coefficient names for individual fits."""
        return NSList(coefficients.free_names, nskey="individual_fit_coefficient")

    def _build_prior_impl(self, coefficients):
        """Shared prior build logic used by both joint and individual producers."""
        prior_specs = coefficients.prior_specs()
        dists = []
        for name in coefficients.free_names:
            spec = prior_specs[name]
            if spec is None:
                raise ValueError(f"Free coefficient '{name}' has no prior defined.")
            dists.append(_build_dist(spec))
        return Prior(dists, coefficients.free_names)

    def produce_prior(self, coefficients):
        """Produce joint prior over all free coefficients."""
        return self._build_prior_impl(coefficients)

    # ------------------------------------------------------------------
    # Individual-fit producers — one free coefficient at a time
    # ------------------------------------------------------------------

    def produce_individual_coefficients(self, coefficients, individual_fit_coefficient):
        """Produce a single-free-parameter coefficient group for individual fits."""
        return coefficients.single_free(individual_fit_coefficient)

    def produce_individual_eft_model(
        self, theory, individual_coefficients, use_quad=False, rge_matrix=None
    ):
        """Produce EFT model for a single-free-parameter individual fit."""
        return EFTModel(theory, individual_coefficients, use_quad, rge_matrix)

    def produce_individual_chi2(
        self,
        individual_coefficients,
        datasets=None,
        individual_eft_model=None,
        data=None,
        fit_covmat=None,
        external_chi2=None,
        rge=None,
    ):
        """Produce chi2 for a single-free-parameter individual fit."""
        return self._build_chi2_impl(
            individual_coefficients,
            datasets,
            individual_eft_model,
            data,
            fit_covmat,
            external_chi2,
            rge,
        )

    def produce_individual_prior(self, individual_coefficients):
        """Produce prior for a single-free-parameter individual fit."""
        return self._build_prior_impl(individual_coefficients)
