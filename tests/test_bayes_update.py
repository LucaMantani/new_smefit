"""Tests for the Bayesian update feature.

Covers:
- _WhitenedToPhysicalPrior  (priors.py)
- ExactPosteriorPrior        (priors.py)
- FitResult.from_json        (fit_result.py)
- parse_bayesian_update_path (config.py)
- produce_prior with bayesian_update_path (config.py)
- build_exact_posterior_prior (utils.py)
"""

import json
import math
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest
import yaml
from reportengine.configparser import ConfigError

from smefit.config import smefitConfig
from smefit.fit_result import FitResult, _format_prior
from smefit.priors import (
    ExactPosteriorPrior,
    Prior,
    _UniformDist,
    _WhitenedToPhysicalPrior,
)
from smefit.utils import build_exact_posterior_prior
from smefit.whitening import WhitenTransform

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform_prior(names, low=-1.0, high=1.0):
    dists = [_UniformDist(low, high) for _ in names]
    specs = {n: {"dist": "uniform", "low": low, "high": high} for n in names}
    return Prior(dists, names, specs=specs)


def _make_fit_dir(
    tmp_path,
    free_params,
    samples,
    whitening_active=False,
    whitening_transformation=None,
    datasets=None,
    external_chi2=None,
):
    """Write a minimal previous-fit directory usable by build_exact_posterior_prior."""
    fit_dir = tmp_path / "prev_fit"
    fit_dir.mkdir()
    (fit_dir / "input").mkdir()

    fit_json = {
        "free_parameters": free_params,
        "best_fit_point": {p: 0.0 for p in free_params},
        "max_loglikelihood": -1.0,
        "num_data": 5,
        "logz": None,
        "n_free": len(free_params),
        "ndof": 5 - len(free_params),
        "chi2": 2.0,
        "chi2_ndof": None,
        "bic": None,
        "aic": None,
        "std": {p: 0.1 for p in free_params},
        "samples": {p: list(samples[p]) for p in free_params},
        "prior_specs": {
            p: {"dist": "uniform", "low": -1.0, "high": 1.0} for p in free_params
        },
        "whitening_transformation": (
            whitening_transformation.to_dict()
            if whitening_transformation is not None
            else None
        ),
        "whitening_active": whitening_active,
    }
    with (fit_dir / "fit_results.json").open("w") as f:
        json.dump(fit_json, f)

    rc = {
        "coefficients": {
            p: {"free": True, "prior": {"dist": "uniform", "low": -1.0, "high": 1.0}}
            for p in free_params
        },
    }
    if datasets is not None:
        rc["datasets"] = datasets
    if external_chi2 is not None:
        rc["external_chi2"] = external_chi2

    with (fit_dir / "input" / "runcard.yaml").open("w") as f:
        yaml.dump(rc, f)

    return fit_dir


# ---------------------------------------------------------------------------
# _WhitenedToPhysicalPrior
# ---------------------------------------------------------------------------


def test_whitened_to_physical_identity_W():
    """With W=I, log_prob in physical space equals the wrapped prior's log_prob."""
    base = _uniform_prior(["OpA", "OpB"])
    transform = WhitenTransform(matrix=jnp.eye(2), shift=jnp.zeros(2))
    wrapped = _WhitenedToPhysicalPrior(base, transform)
    x = jnp.array([0.0, 0.5])
    assert float(wrapped.log_prob(x)) == pytest.approx(
        float(base.log_prob(x)), rel=1e-5
    )


def test_whitened_to_physical_diagonal_W():
    """With diagonal W=diag(2,2), physical params are half the whitened ones.
    The Jacobian factor is log|det(W^{-1})| = log(1/4) = -log(4)."""
    base = _uniform_prior(["OpA", "OpB"], low=-2.0, high=2.0)
    transform = WhitenTransform(matrix=2.0 * jnp.eye(2), shift=jnp.zeros(2))
    wrapped = _WhitenedToPhysicalPrior(base, transform)
    # x_phys in support of base after W^{-1} mapping (W^{-1} x = 0.5*x)
    x_phys = jnp.array([0.5, 0.5])
    x_w = 0.5 * x_phys  # = [0.25, 0.25], inside [-2, 2]
    expected = float(base.log_prob(x_w)) + math.log(0.25)
    assert float(wrapped.log_prob(x_phys)) == pytest.approx(expected, rel=1e-5)


def test_whitened_to_physical_inherits_param_names():
    base = _uniform_prior(["OpA"])
    transform = WhitenTransform(matrix=jnp.eye(1), shift=jnp.zeros(1))
    wrapped = _WhitenedToPhysicalPrior(base, transform)
    assert wrapped.param_names == ["OpA"]


def test_whitened_to_physical_shift_does_not_affect_jacobian():
    """A nonzero shift only translates coordinates; the log-det term is unchanged."""
    base = _uniform_prior(["OpA", "OpB"], low=-2.0, high=2.0)
    matrix = 2.0 * jnp.eye(2)
    no_shift = WhitenTransform(matrix=matrix, shift=jnp.zeros(2))
    shifted = WhitenTransform(matrix=matrix, shift=jnp.array([0.2, -0.2]))
    wrapped_no_shift = _WhitenedToPhysicalPrior(base, no_shift)
    wrapped_shifted = _WhitenedToPhysicalPrior(base, shifted)
    assert float(wrapped_no_shift._log_abs_det_matrix_inv) == pytest.approx(
        float(wrapped_shifted._log_abs_det_matrix_inv)
    )
    # log_prob at the corresponding physical point (offset by the shift) should match.
    x_w = jnp.array([0.25, 0.25])
    x_phys_no_shift = no_shift.to_physical(x_w)
    x_phys_shifted = shifted.to_physical(x_w)
    assert float(wrapped_no_shift.log_prob(x_phys_no_shift)) == pytest.approx(
        float(wrapped_shifted.log_prob(x_phys_shifted)), rel=1e-5
    )


# ---------------------------------------------------------------------------
# ExactPosteriorPrior
# ---------------------------------------------------------------------------


@pytest.fixture
def exact_prior_2d():
    """ExactPosteriorPrior with 2 params, 10 samples, quadratic log-likelihood."""
    base = _uniform_prior(["OpA", "OpB"])
    log_lik = lambda theta: -0.5 * jnp.sum(theta**2)
    samples = {"OpA": jnp.linspace(-0.5, 0.5, 10), "OpB": jnp.linspace(-0.3, 0.3, 10)}
    return ExactPosteriorPrior(
        base, log_lik, samples, ["OpA", "OpB"], source_path="/prev"
    )


def test_exact_prior_log_prob(exact_prior_2d):
    """log_prob = base_prior.log_prob + log_likelihood."""
    base = _uniform_prior(["OpA", "OpB"])
    log_lik = lambda theta: -0.5 * jnp.sum(theta**2)
    x = jnp.array([0.1, 0.2])
    expected = float(base.log_prob(x)) + float(log_lik(x))
    assert float(exact_prior_2d.log_prob(x)) == pytest.approx(expected, rel=1e-5)


def test_exact_prior_log_prob_out_of_support(exact_prior_2d):
    """Point outside the base prior support returns -inf."""
    x = jnp.array([2.0, 0.0])  # outside [-1, 1]
    assert float(exact_prior_2d.log_prob(x)) == float(-jnp.inf)


def test_exact_prior_sample_shape(exact_prior_2d):
    key = jax.random.PRNGKey(0)
    samples = exact_prior_2d.sample(key, n_samples=5)
    assert samples.shape == (5, 2)


def test_exact_prior_sample_replacement_warning(exact_prior_2d, caplog):
    """Requesting more samples than available logs a warning and still returns the right shape."""
    import logging

    key = jax.random.PRNGKey(1)
    with caplog.at_level(logging.WARNING, logger="smefit.priors"):
        samples = exact_prior_2d.sample(key, n_samples=20)
    assert samples.shape == (20, 2)
    assert any("replacement" in msg.lower() for msg in caplog.messages)


def test_exact_prior_sample_no_replacement(exact_prior_2d):
    """Requesting fewer samples than available should not raise."""
    key = jax.random.PRNGKey(2)
    samples = exact_prior_2d.sample(key, n_samples=5)
    assert samples.shape == (5, 2)


def test_exact_prior_prior_specs(exact_prior_2d):
    specs = exact_prior_2d.prior_specs
    assert specs["OpA"]["dist"] == "exact_posterior"
    assert specs["OpB"]["dist"] == "exact_posterior"
    assert specs["OpA"]["source"] == "/prev"


def test_exact_prior_param_names(exact_prior_2d):
    assert exact_prior_2d.param_names == ["OpA", "OpB"]


# ---------------------------------------------------------------------------
# _format_prior for exact_posterior spec
# ---------------------------------------------------------------------------


def test_format_prior_exact_posterior():
    spec = {"dist": "exact_posterior", "source": "/some/path"}
    assert _format_prior(spec) == "ExactPosterior"


# ---------------------------------------------------------------------------
# FitResult.from_json
# ---------------------------------------------------------------------------


def test_fit_result_from_json_roundtrip(tmp_path):
    """write() followed by from_json() should recover the original data."""
    samples = {"OpA": jnp.array([0.1, 0.2, 0.3]), "OpB": jnp.array([-0.1, 0.0, 0.1])}
    fr = FitResult(
        free_parameters=["OpA", "OpB"],
        best_fit_point={"OpA": 0.15, "OpB": 0.0},
        max_loglikelihood=-2.0,
        num_data=8,
        samples=samples,
        prior_specs={"OpA": {"dist": "uniform", "low": -1.0, "high": 1.0}},
    )
    fr.write(tmp_path)
    recovered = FitResult.from_json(tmp_path)

    assert recovered.free_parameters == ["OpA", "OpB"]
    assert recovered.best_fit_point["OpA"] == pytest.approx(0.15)
    assert recovered.max_loglikelihood == pytest.approx(-2.0)
    assert recovered.num_data == 8
    assert list(recovered.samples["OpA"]) == pytest.approx([0.1, 0.2, 0.3])
    assert recovered.whitening_transformation is None
    assert recovered.whitening_active is False


def test_fit_result_from_json_with_whitening_transformation(tmp_path):
    """whitening_transformation is round-tripped correctly."""
    W = jnp.array([[2.0, 0.5], [0.0, 1.0]])
    transform = WhitenTransform(matrix=W, shift=jnp.zeros(2))
    samples = {"OpA": jnp.array([0.1]), "OpB": jnp.array([0.2])}
    fr = FitResult(
        free_parameters=["OpA", "OpB"],
        best_fit_point={"OpA": 0.0, "OpB": 0.0},
        max_loglikelihood=-1.0,
        num_data=5,
        samples=samples,
        whitening_transformation=transform,
        whitening_active=True,
    )
    fr.write(tmp_path)
    recovered = FitResult.from_json(tmp_path)

    assert recovered.whitening_active is True
    assert recovered.whitening_transformation is not None
    assert recovered.whitening_transformation.matrix.shape == (2, 2)
    assert float(recovered.whitening_transformation.matrix[0, 0]) == pytest.approx(2.0)
    assert float(recovered.whitening_transformation.matrix[0, 1]) == pytest.approx(0.5)
    assert jnp.allclose(recovered.whitening_transformation.shift, jnp.zeros(2))


def test_fit_result_from_json_with_whitening_transformation_and_shift(tmp_path):
    """Nonzero shift is round-tripped correctly."""
    W = jnp.array([[2.0, 0.5], [0.0, 1.0]])
    shift = jnp.array([0.3, -0.7])
    transform = WhitenTransform(matrix=W, shift=shift)
    samples = {"OpA": jnp.array([0.1]), "OpB": jnp.array([0.2])}
    fr = FitResult(
        free_parameters=["OpA", "OpB"],
        best_fit_point={"OpA": 0.0, "OpB": 0.0},
        max_loglikelihood=-1.0,
        num_data=5,
        samples=samples,
        whitening_transformation=transform,
        whitening_active=True,
    )
    fr.write(tmp_path)
    recovered = FitResult.from_json(tmp_path)

    assert jnp.allclose(recovered.whitening_transformation.shift, shift)


def test_fit_result_from_json_no_samples(tmp_path):
    """from_json handles the case where samples is null."""
    fr = FitResult(
        free_parameters=["OpA"],
        best_fit_point={"OpA": 0.0},
        max_loglikelihood=-1.0,
        num_data=5,
        samples=None,
    )
    fr.write(tmp_path)
    recovered = FitResult.from_json(tmp_path)
    assert recovered.samples is None


# ---------------------------------------------------------------------------
# parse_bayesian_update_path
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg(tmp_path):
    c = smefitConfig.__new__(smefitConfig)
    c.output_path = tmp_path
    return c


def test_parse_bayesian_update_path_valid(cfg, tmp_path):
    fit_dir = tmp_path / "fit1"
    fit_dir.mkdir()
    (fit_dir / "fit_results.json").write_text("{}")
    (fit_dir / "input").mkdir()
    (fit_dir / "input" / "runcard.yaml").write_text("")
    result = cfg.parse_bayesian_update_path(str(fit_dir))
    assert result == fit_dir


def test_parse_bayesian_update_path_missing_dir(cfg, tmp_path):
    with pytest.raises(ConfigError, match="Directory not found"):
        cfg.parse_bayesian_update_path(str(tmp_path / "no_such_dir"))


def test_parse_bayesian_update_path_missing_fit_results(cfg, tmp_path):
    fit_dir = tmp_path / "fit1"
    fit_dir.mkdir()
    (fit_dir / "input").mkdir()
    (fit_dir / "input" / "runcard.yaml").write_text("")
    with pytest.raises(ConfigError, match="fit_results.json not found"):
        cfg.parse_bayesian_update_path(str(fit_dir))


def test_parse_bayesian_update_path_missing_runcard(cfg, tmp_path):
    fit_dir = tmp_path / "fit1"
    fit_dir.mkdir()
    (fit_dir / "fit_results.json").write_text("{}")
    with pytest.raises(ConfigError, match="runcard.yaml not found"):
        cfg.parse_bayesian_update_path(str(fit_dir))


# ---------------------------------------------------------------------------
# produce_prior with bayesian_update_path
# ---------------------------------------------------------------------------


def test_produce_prior_bayesian_update_returns_exact_posterior(cfg, coeff_group):
    """produce_prior delegates to build_exact_posterior_prior when path is given."""
    mock_epp = MagicMock(spec=ExactPosteriorPrior)
    with patch("smefit.config.build_exact_posterior_prior", return_value=mock_epp):
        result = cfg.produce_prior(
            coeff_group,
            bayesian_update_path="/some/path",
        )
    assert result is mock_epp


def test_produce_prior_bayesian_update_with_whitening_raises(cfg, coeff_group):
    """Combining bayesian_update_path with whitening is not allowed."""
    with pytest.raises(ConfigError, match="whitening is not compatible"):
        cfg.produce_prior(
            coeff_group,
            whitening={"sigma_prior": 3.0, "eps": 1e-8},
            bayesian_update_path="/some/path",
        )


# ---------------------------------------------------------------------------
# build_exact_posterior_prior
# ---------------------------------------------------------------------------


def _mock_coeff_group(free_names):
    cg = MagicMock()
    cg.free_names = free_names
    return cg


def test_build_param_mismatch_raises(tmp_path):
    """Mismatched free parameters between fit1 and current runcard raise ConfigError."""
    fit_dir = _make_fit_dir(
        tmp_path,
        free_params=["OpA"],
        samples={"OpA": [0.1, 0.2]},
    )
    cg = _mock_coeff_group(["OpA", "OpB"])  # different
    with pytest.raises(ConfigError, match="Free parameters mismatch"):
        build_exact_posterior_prior(fit_dir, cg, datasets=None)


def test_build_no_samples_raises(tmp_path):
    """Previous fit without posterior samples raises ConfigError."""
    fit_dir = _make_fit_dir(
        tmp_path,
        free_params=["OpA"],
        samples={"OpA": [0.1]},
    )
    # Overwrite fit_results.json to null out samples
    path = fit_dir / "fit_results.json"
    d = json.loads(path.read_text())
    d["samples"] = None
    path.write_text(json.dumps(d))

    cg = _mock_coeff_group(["OpA"])
    with pytest.raises(ConfigError, match="no posterior samples"):
        build_exact_posterior_prior(fit_dir, cg, datasets=None)


def test_build_dataset_overlap_raises(tmp_path):
    """Datasets appearing in both fits raise a double-counting ConfigError."""
    fit_dir = _make_fit_dir(
        tmp_path,
        free_params=["OpA"],
        samples={"OpA": [0.1, 0.2]},
        datasets=[{"name": "DS_SHARED", "order": "LO"}],
    )
    cg = _mock_coeff_group(["OpA"])
    current_datasets = [{"name": "DS_SHARED", "order": "LO"}]
    with pytest.raises(ConfigError, match="double-count"):
        build_exact_posterior_prior(fit_dir, cg, datasets=current_datasets)


def test_build_external_chi2_overlap_raises(tmp_path):
    """Overlapping external_chi2 keys raise a double-counting ConfigError."""
    fit_dir = _make_fit_dir(
        tmp_path,
        free_params=["OpA"],
        samples={"OpA": [0.1, 0.2]},
        external_chi2={"EXT_A": {}},
    )
    cg = _mock_coeff_group(["OpA"])
    with pytest.raises(ConfigError, match="double-count"):
        build_exact_posterior_prior(
            fit_dir, cg, datasets=None, external_chi2={"EXT_A": {}}
        )


def test_build_returns_exact_posterior_prior(tmp_path):
    """Happy path: returns an ExactPosteriorPrior with the correct param names."""
    fit_dir = _make_fit_dir(
        tmp_path,
        free_params=["OpA", "OpB"],
        samples={
            "OpA": [0.1, 0.2, 0.3],
            "OpB": [-0.1, 0.0, 0.1],
        },
    )
    cg = _mock_coeff_group(["OpA", "OpB"])
    base_prior = _uniform_prior(["OpA", "OpB"])
    mock_chi2_fn = lambda theta: jnp.sum(theta**2)

    with patch("smefit.api.smefitAPI") as mock_api:
        mock_api.chi2.return_value = mock_chi2_fn
        mock_api.prior.return_value = base_prior
        result = build_exact_posterior_prior(fit_dir, cg, datasets=None)

    assert isinstance(result, ExactPosteriorPrior)
    assert result.param_names == ["OpA", "OpB"]


def _build_with_mocked_api(fit_dir, names):
    """Run build_exact_posterior_prior with the DAG rebuild stubbed out.

    Returns the runcard dict the API was handed, which is what the RGE-reuse
    logic mutates.
    """
    cg = _mock_coeff_group(names)
    with patch("smefit.api.smefitAPI") as mock_api:
        mock_api.chi2.return_value = lambda theta: jnp.sum(theta**2)
        mock_api.prior.return_value = _uniform_prior(names)
        build_exact_posterior_prior(fit_dir, cg, datasets=None)
    return mock_api.chi2.call_args.kwargs


def test_build_reuses_the_previous_fits_rge_matrix(tmp_path):
    """Rebuilding fit1's chi2 must not re-run the RGE evolution it already did."""
    fit_dir = _make_fit_dir(tmp_path, free_params=["OpA"], samples={"OpA": [0.1, 0.2]})
    rc_path = fit_dir / "input" / "runcard.yaml"
    rc = yaml.safe_load(rc_path.read_text())
    rc["rge"] = {"init_scale": 1000.0}
    rc_path.write_text(yaml.dump(rc))
    (fit_dir / "rge_matrix.pkl").write_bytes(b"")

    passed_rc = _build_with_mocked_api(fit_dir, ["OpA"])

    assert passed_rc["rge"]["rg_matrix"] == str(fit_dir / "rge_matrix.pkl")


def test_build_does_not_override_an_explicit_rg_matrix(tmp_path):
    fit_dir = _make_fit_dir(tmp_path, free_params=["OpA"], samples={"OpA": [0.1, 0.2]})
    rc_path = fit_dir / "input" / "runcard.yaml"
    rc = yaml.safe_load(rc_path.read_text())
    rc["rge"] = {"init_scale": 1000.0, "rg_matrix": "/somewhere/else.pkl"}
    rc_path.write_text(yaml.dump(rc))
    (fit_dir / "rge_matrix.pkl").write_bytes(b"")

    passed_rc = _build_with_mocked_api(fit_dir, ["OpA"])

    assert passed_rc["rge"]["rg_matrix"] == "/somewhere/else.pkl"


def test_build_without_rge_block_is_untouched(tmp_path):
    fit_dir = _make_fit_dir(tmp_path, free_params=["OpA"], samples={"OpA": [0.1, 0.2]})
    (fit_dir / "rge_matrix.pkl").write_bytes(b"")

    passed_rc = _build_with_mocked_api(fit_dir, ["OpA"])

    assert "rge" not in passed_rc


def test_build_whitening_wraps_prior(tmp_path):
    """When the previous fit used whitening, the prior is wrapped in _WhitenedToPhysicalPrior."""
    transform = WhitenTransform(matrix=jnp.eye(2), shift=jnp.zeros(2))
    fit_dir = _make_fit_dir(
        tmp_path,
        free_params=["OpA", "OpB"],
        samples={"OpA": [0.1, 0.2], "OpB": [-0.1, 0.0]},
        whitening_active=True,
        whitening_transformation=transform,
    )
    cg = _mock_coeff_group(["OpA", "OpB"])
    base_prior = _uniform_prior(["OpA", "OpB"])
    mock_chi2_fn = lambda theta: jnp.sum(theta**2)

    with patch("smefit.api.smefitAPI") as mock_api:
        mock_api.chi2.return_value = mock_chi2_fn
        mock_api.prior.return_value = base_prior
        result = build_exact_posterior_prior(fit_dir, cg, datasets=None)

    # The prior inside ExactPosteriorPrior should be a _WhitenedToPhysicalPrior
    from smefit.priors import _WhitenedToPhysicalPrior

    assert isinstance(result._base_prior, _WhitenedToPhysicalPrior)


def test_build_whitening_active_but_no_matrix_raises(tmp_path):
    """whitening_active=True without a saved transformation raises ConfigError."""
    fit_dir = _make_fit_dir(
        tmp_path,
        free_params=["OpA"],
        samples={"OpA": [0.1, 0.2]},
        whitening_active=True,
        whitening_transformation=None,
    )
    cg = _mock_coeff_group(["OpA"])
    base_prior = _uniform_prior(["OpA"])
    mock_chi2_fn = lambda theta: jnp.sum(theta**2)

    with patch("smefit.api.smefitAPI") as mock_api:
        mock_api.chi2.return_value = mock_chi2_fn
        mock_api.prior.return_value = base_prior
        with pytest.raises(ConfigError, match="whitening_transformation was saved"):
            build_exact_posterior_prior(fit_dir, cg, datasets=None)
