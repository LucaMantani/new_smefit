"""Unit tests for external_chi2/low_energy/superallowed_beta_decay.py."""

from __future__ import annotations

import importlib.util
import pathlib
from types import ModuleType

import jax
import jax.numpy as jnp
import pytest

from smefit.core import Coefficient, CoefficientGroup
from smefit.external_chi2 import load_external_chi2

MODULE_PATH = (
    pathlib.Path(__file__).parents[1]
    / "external_chi2"
    / "low_energy"
    / "superallowed_beta_decay.py"
)
UNIFORM = {"dist": "uniform", "low": -1.0, "high": 1.0}

# Ft involves ~1e24 conversion factors that overflow float32; smefit fits
# run in float64 by default.
jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def bd() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "superallowed_beta_decay", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def eta_group() -> CoefficientGroup:
    # Nuisances only: no SMEFT operator, so no RGE matching is needed.
    return CoefficientGroup(
        [
            Coefficient(name="eta2", free=True, prior=UNIFORM),
            Coefficient(name="eta3", free=True, prior=UNIFORM),
        ]
    )


def _chi2(bd: ModuleType, eta2: float, eta3: float, pre2: float, pre3: float):
    return bd._chi2_smeft(0.02471, 0.0, eta2, eta3, 0.97367, 0.0, pre2, pre3)


def test_prefactors_scale_their_eta(bd: ModuleType) -> None:
    # Only the products eta * prefactor enter the chi2.
    assert _chi2(bd, 2.0, 0.0, 1e-4, 0.0) == pytest.approx(
        _chi2(bd, 1.0, 0.0, 2e-4, 0.0)
    )
    assert _chi2(bd, 0.0, 2.0, 0.0, 1e-5) == pytest.approx(
        _chi2(bd, 0.0, 1.0, 0.0, 2e-5)
    )


def test_runcard_prefactors_reach_the_chi2(
    bd: ModuleType, eta_group: CoefficientGroup
) -> None:
    config = {
        "SA_beta_decays": {
            "path": str(MODULE_PATH),
            "eta2_prefactor": 1e-4,
            "eta3_prefactor": 2e-5,
        }
    }
    (chi2,) = load_external_chi2(config, eta_group, {})

    expected = _chi2(bd, 0.5, 0.3, 1e-4, 2e-5)
    assert chi2(jnp.array([0.5, 0.3])) == pytest.approx(float(expected))


def test_prefactors_default_when_omitted(
    bd: ModuleType, eta_group: CoefficientGroup
) -> None:
    (chi2,) = load_external_chi2(
        {"SA_beta_decays": {"path": str(MODULE_PATH)}}, eta_group, {}
    )

    expected = _chi2(
        bd, 0.5, 0.3, bd._DEFAULT_ETA2_PREFACTOR, bd._DEFAULT_ETA3_PREFACTOR
    )
    assert chi2(jnp.array([0.5, 0.3])) == pytest.approx(float(expected))
