"""Unit tests for smefit/config.py."""

import inspect
import json
import logging
import os
import pathlib
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest
import yaml
from reportengine.configparser import ConfigError, ExplicitNode

from smefit.chi2 import Chi2
from smefit.config import smefitConfig
from smefit.core import (
    Coefficient,
    CoefficientGroup,
    DataGroup,
    Dataset,
    Theory,
    TheoryGroup,
)
from smefit.model import EFTModel
from smefit.priors import Prior
from smefit.whitening import (
    _whitening_baseline_shift,
    _whitening_gradient_descent_shift,
)

# ---------------------------------------------------------------------------
# Fixture: bare smefitConfig instance (no reportengine DAG)
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg(tmp_path):
    """smefitConfig instance created via __new__, bypassing reportengine's __init__."""
    c = smefitConfig.__new__(smefitConfig)
    c.output_path = tmp_path
    return c


# ---------------------------------------------------------------------------
# parse_data_path / parse_theory_path
# ---------------------------------------------------------------------------


def test_parse_data_path_valid(cfg, tmp_path):
    result = cfg.parse_data_path(str(tmp_path))
    assert result == tmp_path


def test_parse_data_path_missing(cfg, tmp_path):
    missing = str(tmp_path / "nonexistent")
    with pytest.raises(ValueError, match="does not exist"):
        cfg.parse_data_path(missing)


def test_parse_data_path_prefix_resolved(cfg, tmp_path):
    (tmp_path / "smefit_database" / "commondata").mkdir(parents=True)
    with patch(
        "smefit.paths.load_user_paths",
        return_value={"smefit_database": str(tmp_path / "smefit_database")},
    ):
        result = cfg.parse_data_path("smefit_database/commondata")
    assert result == tmp_path / "smefit_database" / "commondata"


def test_parse_data_path_prefix_missing_key(cfg):
    with patch("smefit.paths.load_user_paths", return_value={}):
        with pytest.raises(ValueError, match="smefit_setup_local"):
            cfg.parse_data_path("smefit_database/commondata")


def test_parse_theory_path_valid(cfg, tmp_path):
    result = cfg.parse_theory_path(str(tmp_path))
    assert result == tmp_path


def test_parse_theory_path_missing(cfg, tmp_path):
    missing = str(tmp_path / "nonexistent")
    with pytest.raises(ValueError, match="does not exist"):
        cfg.parse_theory_path(missing)


def test_parse_theory_path_prefix_resolved(cfg, tmp_path):
    (tmp_path / "smefit_database" / "theory").mkdir(parents=True)
    with patch(
        "smefit.paths.load_user_paths",
        return_value={"smefit_database": str(tmp_path / "smefit_database")},
    ):
        result = cfg.parse_theory_path("smefit_database/theory")
    assert result == tmp_path / "smefit_database" / "theory"


def test_parse_theory_path_prefix_missing_key(cfg):
    with patch("smefit.paths.load_user_paths", return_value={}):
        with pytest.raises(ValueError, match="smefit_setup_local"):
            cfg.parse_theory_path("smefit_database/theory")


# ---------------------------------------------------------------------------
# parse_coefficients
# ---------------------------------------------------------------------------


def test_parse_coefficients_builds_group(cfg):
    raw = {
        "OpA": {"free": True, "prior": {"dist": "uniform", "low": -1.0, "high": 1.0}},
        "OpB": {"free": False, "value": 2.0},
    }
    group = cfg.parse_coefficients(raw)
    assert isinstance(group, CoefficientGroup)
    assert "OpA" in group.free_names
    assert "OpB" not in group.free_names


def test_parse_coefficients_with_valid_expr(cfg):
    raw = {
        "OpA": {"free": True, "prior": {"dist": "uniform", "low": -1.0, "high": 1.0}},
        "OpC": {"free": False, "vars": ["OpA"], "expr": "OpA**2"},
    }
    group = cfg.parse_coefficients(raw)
    assert "OpC" in group.names
    assert "OpC" not in group.free_names


def test_parse_coefficients_invalid_vars_raises(cfg):
    """vars may not reference another expression coefficient."""
    raw = {
        "OpA": {"free": True, "prior": {"dist": "uniform", "low": -1.0, "high": 1.0}},
        "OpB": {"free": False, "vars": ["OpA"], "expr": "OpA**2"},
        "OpC": {"free": False, "vars": ["OpB"], "expr": "OpB**2"},
    }
    with pytest.raises(
        ValueError, match="must be a free coefficient or a fixed coefficient"
    ):
        cfg.parse_coefficients(raw)


# ---------------------------------------------------------------------------
# parse_rge
# ---------------------------------------------------------------------------


def test_parse_rge_valid(cfg):
    rge = {"init_scale": 1000.0, "obs_scale": "dynamic"}
    result = cfg.parse_rge(rge)
    assert result == rge


def test_parse_rge_missing_init_scale_raises(cfg):
    with pytest.raises(ConfigError):
        cfg.parse_rge({"obs_scale": "dynamic"})


def test_parse_rge_invalid_obs_scale_raises(cfg):
    with pytest.raises(ConfigError):
        cfg.parse_rge({"init_scale": 1000.0, "obs_scale": "bad_value"})


def test_parse_rge_fixed_obs_scale_ok(cfg):
    rge = {"init_scale": 1000.0, "obs_scale": 91.2}
    result = cfg.parse_rge(rge)
    assert result["obs_scale"] == 91.2


# ---------------------------------------------------------------------------
# parse_ultranest_settings
# ---------------------------------------------------------------------------


def test_parse_ultranest_defaults(cfg, tmp_path):
    result = cfg.parse_ultranest_settings({}, tmp_path)
    assert result["ultranest_seed"] == 123456
    assert result["sampler_plot"] is False
    assert result["ReactiveNS_settings"]["resume"] == "overwrite"
    assert result["ReactiveNS_settings"]["vectorized"] is False
    assert "ultranest_logs" in result["ReactiveNS_settings"]["log_dir"]


def test_parse_ultranest_resume_false_becomes_overwrite(cfg, tmp_path):
    settings = {"ReactiveNS_settings": {"resume": False}}
    result = cfg.parse_ultranest_settings(settings, tmp_path)
    assert result["ReactiveNS_settings"]["resume"] == "overwrite"


def test_parse_ultranest_resume_true_existing_dir(cfg, tmp_path):
    log_dir = str(tmp_path / "ultranest_logs")
    os.makedirs(log_dir)
    settings = {"ReactiveNS_settings": {"log_dir": log_dir, "resume": True}}
    result = cfg.parse_ultranest_settings(settings, tmp_path)
    assert result["ReactiveNS_settings"]["resume"] is True


def test_parse_ultranest_resume_missing_dir_raises(cfg, tmp_path):
    missing_dir = str(tmp_path / "no_such_dir")
    settings = {"ReactiveNS_settings": {"log_dir": missing_dir, "resume": True}}
    with pytest.raises(FileNotFoundError):
        cfg.parse_ultranest_settings(settings, tmp_path)


# ---------------------------------------------------------------------------
# parse_blackjax_settings
# ---------------------------------------------------------------------------


def test_parse_blackjax_defaults(cfg, tmp_path):
    result = cfg.parse_blackjax_settings({}, tmp_path)
    assert result["n_live"] == 500
    assert result["n_posterior_samples"] == 1000
    assert result["log_precision"] == -2
    assert result["seed"] == 0
    assert "blackjax_logs" in result["log_dir"]


def test_parse_blackjax_custom_values(cfg, tmp_path):
    settings = {"n_live": 200, "seed": 42}
    result = cfg.parse_blackjax_settings(settings, tmp_path)
    assert result["n_live"] == 200
    assert result["seed"] == 42
    assert result["n_posterior_samples"] == 1000  # still default


# ---------------------------------------------------------------------------
# produce_fit_covmat
# ---------------------------------------------------------------------------


def test_produce_fit_covmat_exp(cfg, dataset_a, theory_a):
    data = DataGroup([dataset_a])
    theory = TheoryGroup([theory_a])
    covmat = cfg.produce_fit_covmat(data, theory)
    assert covmat.shape == (3, 3)


def test_produce_fit_covmat_with_theory_covmat(cfg, dataset_a, theory_a):
    data = DataGroup([dataset_a])
    theory = TheoryGroup([theory_a])
    covmat_no_th = cfg.produce_fit_covmat(data, theory, use_theory_covmat=False)
    # use a fresh cfg to bypass the cache
    cfg2 = smefitConfig.__new__(smefitConfig)
    covmat_with_th = cfg2.produce_fit_covmat(data, theory, use_theory_covmat=True)
    # adding sm_covmat makes entries larger
    assert jnp.all(covmat_with_th >= covmat_no_th)


def test_produce_fit_covmat_caching(cfg, dataset_a, theory_a):
    data = DataGroup([dataset_a])
    theory = TheoryGroup([theory_a])
    result1 = cfg.produce_fit_covmat(data, theory)
    result2 = cfg.produce_fit_covmat(data, theory)
    assert result1 is result2


def test_produce_fit_covmat_name_mismatch_raises(cfg, dataset_a, theory_b):
    data = DataGroup([dataset_a])
    theory = TheoryGroup([theory_b])
    with pytest.raises(ValueError, match="different datasets"):
        cfg.produce_fit_covmat(data, theory)


def test_produce_fit_covmat_singular_dataset_raises(cfg, dataset_a, theory_a):
    # zero out DS_A's stat and syst uncertainties, and its theory covmat: its
    # diagonal block becomes all zeros, i.e. singular
    zero_dataset = Dataset(
        name=dataset_a.name,
        num_data=dataset_a.num_data,
        central_values=dataset_a.central_values,
        stat_err=jnp.zeros_like(dataset_a.stat_err),
        syst_err=jnp.zeros_like(dataset_a.syst_err),
        sys_names=dataset_a.sys_names,
        sys_types=dataset_a.sys_types,
        luminosity=dataset_a.luminosity,
    )
    zero_theory = Theory(
        name=theory_a.name,
        order=theory_a.order,
        sm_pred=theory_a.sm_pred,
        eft_pred=theory_a.eft_pred,
        sm_covmat=jnp.zeros_like(theory_a.sm_covmat),
        scales=theory_a.scales,
        operators=list(theory_a.operators),
    )
    data = DataGroup([zero_dataset])
    theory = TheoryGroup([zero_theory])
    with pytest.raises(ValueError, match="singular.*DS_A"):
        cfg.produce_fit_covmat(data, theory, use_theory_covmat=True)


def test_produce_fit_covmat_lists_all_singular_datasets(
    cfg, dataset_a, dataset_b, theory_a, theory_b
):
    # zero out both DS_A and DS_B (stat, syst, and theory covmat), leaving
    # nothing else in the fit: both blocks are singular and should be listed
    zero_dataset_a = Dataset(
        name=dataset_a.name,
        num_data=dataset_a.num_data,
        central_values=dataset_a.central_values,
        stat_err=jnp.zeros_like(dataset_a.stat_err),
        syst_err=jnp.zeros_like(dataset_a.syst_err),
        sys_names=dataset_a.sys_names,
        sys_types=dataset_a.sys_types,
        luminosity=dataset_a.luminosity,
    )
    zero_dataset_b = Dataset(
        name=dataset_b.name,
        num_data=dataset_b.num_data,
        central_values=dataset_b.central_values,
        stat_err=jnp.zeros_like(dataset_b.stat_err),
        syst_err=jnp.zeros_like(dataset_b.syst_err),
        sys_names=dataset_b.sys_names,
        sys_types=dataset_b.sys_types,
        luminosity=dataset_b.luminosity,
    )
    zero_theory_a = Theory(
        name=theory_a.name,
        order=theory_a.order,
        sm_pred=theory_a.sm_pred,
        eft_pred=theory_a.eft_pred,
        sm_covmat=jnp.zeros_like(theory_a.sm_covmat),
        scales=theory_a.scales,
        operators=list(theory_a.operators),
    )
    zero_theory_b = Theory(
        name=theory_b.name,
        order=theory_b.order,
        sm_pred=theory_b.sm_pred,
        eft_pred=theory_b.eft_pred,
        sm_covmat=jnp.zeros_like(theory_b.sm_covmat),
        scales=theory_b.scales,
        operators=list(theory_b.operators),
    )
    data = DataGroup([zero_dataset_a, zero_dataset_b])
    theory = TheoryGroup([zero_theory_a, zero_theory_b])
    with pytest.raises(ValueError, match="DS_A") as exc_info:
        cfg.produce_fit_covmat(data, theory, use_theory_covmat=True)
    assert "DS_B" in str(exc_info.value)


# ---------------------------------------------------------------------------
# _build_chi2_impl
# ---------------------------------------------------------------------------


def test_build_chi2_no_data_no_ext_raises(cfg):
    with pytest.raises(ConfigError):
        cfg._build_chi2_impl()


def test_build_chi2_base_only(cfg):
    mock_eft = MagicMock()
    mock_eft.coefficients.free_names = ["OpA"]
    mock_data = MagicMock()
    mock_data.num_data = 5

    with patch("smefit.config.build_chi2", return_value=lambda c: jnp.sum(c**2)):
        result = cfg._build_chi2_impl(
            eft_model=mock_eft, data=mock_data, fit_covmat=jnp.eye(3)
        )
    assert isinstance(result, Chi2)
    assert result.num_data == 5
    assert not result.has_external


def test_build_chi2_ext_only(cfg):
    ext = Chi2(lambda c: jnp.sum(c**2), param_names=["OpA"], num_data=7)
    result = cfg._build_chi2_impl(ext_chi2_func=[ext])
    assert isinstance(result, Chi2)
    assert result.num_data == 7
    assert result.has_external


def test_build_chi2_baseline_from_coefficients(cfg):
    """baseline flows from the coefficients node into the Chi2 (base-only path)."""
    prior = {"dist": "uniform", "low": -1.0, "high": 1.0}
    coefficients = CoefficientGroup(
        [
            Coefficient(name="OpA", free=True, prior=prior, baseline_value=0.7),
            Coefficient(name="OpB", free=True, prior=prior, baseline_value=-0.3),
        ]
    )
    mock_eft = MagicMock()
    mock_eft.coefficients.free_names = ["OpA", "OpB"]
    mock_data = MagicMock()
    mock_data.num_data = 5

    with patch("smefit.config.build_chi2", return_value=lambda c: jnp.sum(c**2)):
        result = cfg._build_chi2_impl(
            eft_model=mock_eft,
            data=mock_data,
            fit_covmat=jnp.eye(3),
            coefficients=coefficients,
        )
    assert jnp.allclose(result.baseline, jnp.array([0.7, -0.3]))


def test_build_chi2_ext_only_baseline_from_coefficients(cfg):
    """External-chi2-only fits still honor baseline_value (no eft_model)."""
    prior = {"dist": "uniform", "low": -1.0, "high": 1.0}
    coefficients = CoefficientGroup(
        [Coefficient(name="OpA", free=True, prior=prior, baseline_value=1.5)]
    )
    ext = Chi2(lambda c: jnp.sum(c**2), param_names=["OpA"], num_data=7)
    result = cfg._build_chi2_impl(ext_chi2_func=[ext], coefficients=coefficients)
    assert result.has_external
    assert jnp.allclose(result.baseline, jnp.array([1.5]))


def test_build_chi2_combined(cfg):
    mock_eft = MagicMock()
    mock_eft.coefficients.free_names = ["OpA"]
    mock_data = MagicMock()
    mock_data.num_data = 5
    ext = Chi2(lambda c: jnp.sum(c**2), param_names=["OpA"], num_data=7)

    with patch("smefit.config.build_chi2", return_value=lambda c: jnp.sum(c**2)):
        result = cfg._build_chi2_impl(
            eft_model=mock_eft,
            data=mock_data,
            fit_covmat=jnp.eye(3),
            ext_chi2_func=[ext],
        )
    assert result.num_data == 12  # 5 + 7
    assert result.has_external


# ---------------------------------------------------------------------------
# produce_datasets_chi2
# ---------------------------------------------------------------------------

_PRIOR = {"dist": "uniform", "low": -1.0, "high": 1.0}


def test_produce_datasets_chi2_returns_list(cfg, dataset_a, theory_a):
    data = DataGroup([dataset_a])
    theory = TheoryGroup([theory_a])
    model = EFTModel(
        theory,
        CoefficientGroup([Coefficient("OpA", free=True, prior=_PRIOR)]),
        use_quad=False,
    )

    result = cfg.produce_datasets_chi2(
        eft_model=model, data=data, fit_covmat=jnp.eye(3)
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], Chi2)
    assert result[0].name == "DS_A"
    assert result[0].num_data == 3


def test_produce_datasets_chi2_appends_external(cfg, dataset_a, theory_a):
    data = DataGroup([dataset_a])
    theory = TheoryGroup([theory_a])
    model = EFTModel(
        theory,
        CoefficientGroup([Coefficient("OpA", free=True, prior=_PRIOR)]),
        use_quad=False,
    )
    ext = Chi2(lambda c: jnp.sum(c**2), ["OpA"], num_data=5, name="EXT")

    result = cfg.produce_datasets_chi2(
        eft_model=model, data=data, fit_covmat=jnp.eye(3), ext_chi2_func=[ext]
    )

    assert len(result) == 2
    assert result[0].name == "DS_A"
    assert result[1] is ext


# ---------------------------------------------------------------------------
# produce_prior
# ---------------------------------------------------------------------------


def test_produce_prior_no_whitening(cfg, coeff_group):
    result = cfg.produce_prior(coeff_group)
    assert isinstance(result, Prior)
    assert result.param_names == ["OpA"]


def test_produce_prior_with_whitening(cfg, coeff_group):
    whitening = {"sigma_prior": 3.0, "eps": 1e-8}
    result = cfg.produce_prior(coeff_group, whitening=whitening)
    assert isinstance(result, Prior)
    # Each distribution should be uniform on [-3, 3]
    from smefit.priors import _UniformDist

    assert isinstance(result.dists[0], _UniformDist)
    assert result.dists[0].low == pytest.approx(-3.0)
    assert result.dists[0].high == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# parse_whitening / produce_whitening_transformation
# ---------------------------------------------------------------------------


def test_parse_whitening_defaults(cfg):
    result = cfg.parse_whitening({})
    assert result == {"sigma_prior": 5.0, "eps": 1e-8, "shift": "baseline"}


def test_parse_whitening_shift_gradient_descent(cfg):
    result = cfg.parse_whitening({"shift": "gradient_descent"})
    assert result["shift"] == "gradient_descent"


def test_parse_whitening_shift_invalid_raises(cfg):
    with pytest.raises(ConfigError, match="whitening.shift"):
        cfg.parse_whitening({"shift": "bogus"})


def test_parse_whitening_unknown_key_warns(cfg, caplog):
    with caplog.at_level("WARNING"):
        cfg.parse_whitening({"bogus": 1})
    assert "Unknown key 'bogus'" in caplog.text


def test_produce_whitening_transformation_disabled(cfg):
    node = cfg.produce_whitening_transformation(whitening=None)
    assert isinstance(node, ExplicitNode)
    # zero-argument worker: no dependency on gd_best_fit or anything else
    assert inspect.signature(node.value).parameters == {}
    assert node.value() is None


def test_produce_whitening_transformation_baseline_shift(cfg):
    whitening = {"sigma_prior": 5.0, "eps": 1e-8, "shift": "baseline"}
    node = cfg.produce_whitening_transformation(whitening=whitening)
    assert isinstance(node, ExplicitNode)
    assert node.value is _whitening_baseline_shift
    # must not depend on gd_best_fit, so gradient_descent_settings is never required
    assert "gd_best_fit" not in inspect.signature(node.value).parameters


def test_produce_whitening_transformation_gradient_descent_shift(cfg):
    whitening = {"sigma_prior": 5.0, "eps": 1e-8, "shift": "gradient_descent"}
    node = cfg.produce_whitening_transformation(whitening=whitening)
    assert isinstance(node, ExplicitNode)
    assert node.value is _whitening_gradient_descent_shift
    assert "gd_best_fit" in inspect.signature(node.value).parameters


def test_baseline_shift_worker_centers_at_baseline():
    """With no baseline set, chi2.baseline defaults to zeros."""
    chi2 = Chi2(lambda c: jnp.sum(c**2), param_names=["OpA", "OpB"], num_data=1)
    whitening = {"sigma_prior": 5.0, "eps": 1e-8, "shift": "baseline"}
    transform = _whitening_baseline_shift(chi2, whitening)
    assert jnp.allclose(transform.shift, jnp.zeros(2))
    # Hessian of sum(c**2) is 2*I -> H = L L^T with L = sqrt(2)*I -> W = L^-T = I/sqrt(2)
    expected = jnp.eye(2) / jnp.sqrt(2.0)
    assert jnp.allclose(transform.matrix, expected, atol=1e-5)


def test_gradient_descent_shift_worker_centers_at_gd_best_fit():
    chi2 = Chi2(lambda c: jnp.sum(c**2), param_names=["OpA", "OpB"], num_data=1)
    whitening = {"sigma_prior": 5.0, "eps": 1e-8, "shift": "gradient_descent"}
    gd_best_fit = jnp.array([1.0, 2.0])
    transform = _whitening_gradient_descent_shift(chi2, gd_best_fit, whitening)
    assert jnp.allclose(transform.shift, gd_best_fit)
    expected = jnp.eye(2) / jnp.sqrt(2.0)
    assert jnp.allclose(transform.matrix, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# parse_gradient_descent_settings
# ---------------------------------------------------------------------------


def test_parse_gradient_descent_settings_defaults(cfg):
    result = cfg.parse_gradient_descent_settings({})
    assert result["sm_solution"] is False
    assert result["n_steps"] == 2000
    assert result["tol"] == pytest.approx(1e-8)


def test_parse_gradient_descent_settings_custom(cfg):
    result = cfg.parse_gradient_descent_settings(
        {"sm_solution": True, "n_steps": 500, "tol": 1e-6}
    )
    assert result["sm_solution"] is True
    assert result["n_steps"] == 500
    assert result["tol"] == pytest.approx(1e-6)


def test_parse_gradient_descent_settings_unknown_key_warns(cfg, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="smefit.config"):
        cfg.parse_gradient_descent_settings({"unknown_key": 99})
    assert any("unknown_key" in r.message for r in caplog.records)


# parse_hessian_settings
# ---------------------------------------------------------------------------


def test_parse_hessian_settings_defaults(cfg):
    result = cfg.parse_hessian_settings({})
    assert result["n_samples"] == 10000
    assert result["seed"] == 42


def test_parse_hessian_settings_custom(cfg):
    result = cfg.parse_hessian_settings({"n_samples": 500, "seed": 7})
    assert result["n_samples"] == 500
    assert result["seed"] == 7


def test_parse_hessian_settings_unknown_key_warns(cfg, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="smefit.config"):
        cfg.parse_hessian_settings({"unknown_key": 99})
    assert any("unknown_key" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# parse_optimizer_settings / produce_optimizer
# ---------------------------------------------------------------------------


def test_parse_optimizer_settings_passthrough(cfg):
    settings = {"optimizer": "sgd", "optimizer_hyperparams": {"learning_rate": 0.1}}
    result = cfg.parse_optimizer_settings(settings)
    assert result["optimizer"] == "sgd"
    assert result["optimizer_hyperparams"]["learning_rate"] == pytest.approx(0.1)


def test_parse_optimizer_settings_unknown_key_warns(cfg, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="smefit.config"):
        cfg.parse_optimizer_settings({"bad_key": 1})
    assert any("bad_key" in r.message for r in caplog.records)


def test_produce_optimizer_default(cfg):
    """No optimizer_settings → default Adam is returned."""
    opt = cfg.produce_optimizer()
    assert hasattr(opt, "init") and hasattr(opt, "update")


def test_produce_optimizer_with_settings(cfg):
    """Custom hyperparams are accepted without error."""
    settings = {"optimizer": "adam", "optimizer_hyperparams": {"learning_rate": 5e-3}}
    opt = cfg.produce_optimizer(settings)
    assert hasattr(opt, "init") and hasattr(opt, "update")


def test_produce_optimizer_with_clipnorm(cfg):
    """Clipnorm wraps the optimizer in a chain."""
    import optax

    settings = {
        "optimizer": "adam",
        "optimizer_hyperparams": {"learning_rate": 1e-2},
        "clipnorm": 1.0,
    }
    opt = cfg.produce_optimizer(settings)
    # optax.chain returns a GradientTransformationExtraArgs / named tuple;
    # the key check is that init/update exist and the object is not plain adam
    assert hasattr(opt, "init") and hasattr(opt, "update")
    # A chain wraps multiple transforms; verify by initialising on a dummy param
    import jax.numpy as jnp

    state = opt.init(jnp.zeros(2))
    assert state is not None


def test_produce_optimizer_with_scheduler(cfg):
    """A scheduler is injected as learning_rate without error."""
    settings = {
        "optimizer": "adam",
        "optimizer_hyperparams": {},
        "scheduler": {
            "name": "linear_schedule",
            "params": {
                "init_value": 1e-2,
                "end_value": 1e-4,
                "transition_steps": 1000,
            },
        },
    }
    opt = cfg.produce_optimizer(settings)
    assert hasattr(opt, "init") and hasattr(opt, "update")


# ---------------------------------------------------------------------------
# produce_data_groups
# ---------------------------------------------------------------------------


def test_produce_data_groups_no_group_keys_returns_none(cfg):
    datasets = [{"name": "DS_A", "order": "LO"}, {"name": "DS_B", "order": "NLO_QCD"}]
    assert cfg.produce_data_groups(datasets) is None


def test_produce_data_groups_all_grouped(cfg):
    datasets = [
        {"name": "DS_A", "order": "LO", "group": "G1"},
        {"name": "DS_B", "order": "LO", "group": "G2"},
        {"name": "DS_C", "order": "LO", "group": "G1"},
    ]
    result = cfg.produce_data_groups(datasets)
    assert result == {"G1": ["DS_A", "DS_C"], "G2": ["DS_B"]}


def test_produce_data_groups_partial_grouping(cfg):
    """Datasets without a group key are simply omitted from the result."""
    datasets = [
        {"name": "DS_A", "order": "LO", "group": "G1"},
        {"name": "DS_B", "order": "LO"},
    ]
    result = cfg.produce_data_groups(datasets)
    assert result == {"G1": ["DS_A"]}


def test_produce_data_groups_preserves_insertion_order(cfg):
    datasets = [
        {"name": "DS_C", "order": "LO", "group": "G3"},
        {"name": "DS_A", "order": "LO", "group": "G1"},
        {"name": "DS_B", "order": "LO", "group": "G2"},
    ]
    result = cfg.produce_data_groups(datasets)
    assert list(result.keys()) == ["G3", "G1", "G2"]


def test_produce_data_groups_merges_ext_chi2_groups(cfg):
    cfg._ext_chi2_groups = {"EXT_DS": "G2"}
    datasets = [{"name": "DS_A", "order": "LO", "group": "G1"}]
    result = cfg.produce_data_groups(datasets)
    assert result == {"G1": ["DS_A"], "G2": ["EXT_DS"]}


def test_produce_data_groups_no_ext_chi2_attr(cfg):
    """Works correctly when _ext_chi2_groups was never set (no external_chi2 in runcard)."""
    datasets = [{"name": "DS_A", "order": "LO", "group": "G1"}]
    result = cfg.produce_data_groups(datasets)
    assert result == {"G1": ["DS_A"]}


def test_produce_data_groups_ext_chi2_appended_to_existing_group(cfg):
    cfg._ext_chi2_groups = {"EXT_DS": "G1"}
    datasets = [{"name": "DS_A", "order": "LO", "group": "G1"}]
    result = cfg.produce_data_groups(datasets)
    assert result == {"G1": ["DS_A", "EXT_DS"]}


# ---------------------------------------------------------------------------
# parse_external_chi2
# ---------------------------------------------------------------------------


def test_parse_external_chi2_strips_group(cfg):
    raw = {
        "MyExt": {"path": "/some/path.py", "group": "G1", "use_quad": True},
    }
    result = cfg.parse_external_chi2(raw)
    assert "group" not in result["MyExt"]
    assert result["MyExt"]["path"] == "/some/path.py"
    assert result["MyExt"]["use_quad"] is True


def test_parse_external_chi2_caches_groups(cfg):
    raw = {
        "ExtA": {"path": "/a.py", "group": "G1"},
        "ExtB": {"path": "/b.py", "group": "G2"},
    }
    cfg.parse_external_chi2(raw)
    assert cfg._ext_chi2_groups == {"ExtA": "G1", "ExtB": "G2"}


def test_parse_external_chi2_no_group_key(cfg):
    raw = {"MyExt": {"path": "/some/path.py", "use_quad": False}}
    result = cfg.parse_external_chi2(raw)
    assert result == {"MyExt": {"path": "/some/path.py", "use_quad": False}}
    assert cfg._ext_chi2_groups == {}


def test_parse_external_chi2_mixed_group_and_no_group(cfg):
    raw = {
        "ExtA": {"path": "/a.py", "group": "G1"},
        "ExtB": {"path": "/b.py"},
    }
    result = cfg.parse_external_chi2(raw)
    assert "group" not in result["ExtA"]
    assert result["ExtB"] == {"path": "/b.py"}
    assert cfg._ext_chi2_groups == {"ExtA": "G1"}


def test_parse_external_chi2_resolves_prefix_path(cfg):
    raw = {"MyExt": {"path": "new_smefit/external_chi2/foo.py"}}
    with patch(
        "smefit.paths.load_user_paths",
        return_value={"new_smefit": "/home/user/smefit/new_smefit"},
    ):
        result = cfg.parse_external_chi2(raw)
    assert (
        result["MyExt"]["path"] == "/home/user/smefit/new_smefit/external_chi2/foo.py"
    )


# ---------------------------------------------------------------------------
# parse_fits
# ---------------------------------------------------------------------------


def _write_fit_dir(path, use_quad=False, action="run_analytic_fit"):
    """Minimal fit directory, as written by a smefit run."""
    (path / "input").mkdir(parents=True)
    (path / "fit_results.json").write_text(
        json.dumps(
            {
                "free_parameters": ["OpA", "OpB"],
                "best_fit_point": {"OpA": 0.0, "OpB": 1.0},
                "max_loglikelihood": -1.0,
                "num_data": 5,
                "samples": {"OpA": [0.0], "OpB": [1.0]},
            }
        )
    )
    (path / "input" / "runcard.yaml").write_text(
        yaml.dump({"use_quad": use_quad, "actions_": [action]})
    )
    return path


def test_parse_fits_accepts_plain_names(cfg, tmp_path):
    fit = _write_fit_dir(tmp_path / "fits" / "fit_a")

    with patch(
        "smefit.paths.load_user_paths",
        return_value={"smefit_results": str(tmp_path)},
    ):
        result = cfg.parse_fits(["fit_a"])

    assert result == [{"name": "fit_a", "path": fit, "label": None}]


def test_parse_fits_accepts_mappings(cfg, tmp_path):
    fit = _write_fit_dir(tmp_path / "elsewhere" / "fit_a")

    result = cfg.parse_fits(
        [{"name": "fit_a", "path": str(tmp_path / "elsewhere"), "label": "$A$"}]
    )

    assert result[0]["path"] == fit
    assert result[0]["label"] == "$A$"


def test_parse_fits_keeps_a_latex_label_verbatim(cfg, tmp_path):
    """A label is passed to matplotlib as given, backslashes and all."""
    _write_fit_dir(tmp_path / "elsewhere" / "fit_a")
    label = r"$\mathrm{FCC}\textnormal{-}\mathrm{ee\ descoped}$"

    result = cfg.parse_fits(
        [{"name": "fit_a", "path": str(tmp_path / "elsewhere"), "label": label}]
    )

    assert result[0]["label"] == label


def test_parse_fits_rejects_a_non_string_label(cfg, tmp_path):
    _write_fit_dir(tmp_path / "elsewhere" / "fit_a")

    with pytest.raises(ConfigError, match="must be a string"):
        cfg.parse_fits(
            [{"name": "fit_a", "path": str(tmp_path / "elsewhere"), "label": ["$A$"]}]
        )


def test_parse_fits_requires_a_name(cfg):
    with pytest.raises(ConfigError, match="requires a .name."):
        cfg.parse_fits([{"label": "$A$"}])


def test_parse_fits_missing_fit_raises(cfg, tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        cfg.parse_fits([{"name": "does_not_exist", "path": str(tmp_path)}])


def test_parse_fits_unknown_key_warns(cfg, tmp_path, caplog):
    _write_fit_dir(tmp_path / "elsewhere" / "fit_a")

    with caplog.at_level(logging.WARNING, logger="smefit.config"):
        cfg.parse_fits(
            [{"name": "fit_a", "path": str(tmp_path / "elsewhere"), "unknown_key": 1}]
        )

    assert any("unknown_key" in r.message for r in caplog.records)


def test_parse_fits_resolves_prefix_path(cfg, tmp_path):
    fit = _write_fit_dir(tmp_path / "my_fits" / "my_fit")

    with patch(
        "smefit.paths.load_user_paths",
        return_value={"smefit_results": str(tmp_path)},
    ):
        result = cfg.parse_fits([{"name": "my_fit", "path": "smefit_results/my_fits"}])

    assert result[0]["path"] == fit
