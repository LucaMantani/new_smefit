"""Unit tests for external_chi2/low_energy/gaussian_constraints.py."""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import pytest

from smefit.core import Coefficient, CoefficientGroup
from smefit.external_chi2 import load_external_chi2

MODULE_PATH = (
    pathlib.Path(__file__).parents[1]
    / "external_chi2"
    / "low_energy"
    / "gaussian_constraints.py"
)
UNIFORM = {"dist": "uniform", "low": -1.0, "high": 1.0}


@pytest.fixture
def nuisance_group() -> CoefficientGroup:
    # A fixed coefficient first, so free and full indices differ.
    return CoefficientGroup(
        [
            Coefficient(name="DRV", free=False, value=0.02),
            Coefficient(name="eta1", free=True, prior=UNIFORM),
            Coefficient(name="eta2", free=True, prior=UNIFORM),
        ]
    )


def _load(entries: dict, coefficients: CoefficientGroup) -> list:
    config = {name: {"path": str(MODULE_PATH), **kw} for name, kw in entries.items()}
    return load_external_chi2(config, coefficients, {})


def test_constraint_reads_its_own_free_parameter(
    nuisance_group: CoefficientGroup,
) -> None:
    (chi2,) = _load(
        {"GaussConstraintEta2": {"central": 0.1, "sigma": 0.5}}, nuisance_group
    )

    # free vector is (eta1, eta2); only eta2 enters
    assert chi2(jnp.array([0.7, 0.6])) == pytest.approx(1.0)
    assert chi2.num_data == 1


def test_constraint_on_fixed_coefficient_uses_its_value(
    nuisance_group: CoefficientGroup,
) -> None:
    (chi2,) = _load(
        {"GaussConstraintDRV": {"central": 0.01, "sigma": 0.005}}, nuisance_group
    )

    assert chi2(jnp.array([0.0, 0.0])) == pytest.approx(4.0)


def test_constraint_hessian_is_diagonal_inverse_variance(
    nuisance_group: CoefficientGroup,
) -> None:
    chi2s = _load(
        {
            "GaussConstraintEta1": {"central": 0.0, "sigma": 2.0},
            "GaussConstraintEta2": {"central": 0.0, "sigma": 0.5},
        },
        nuisance_group,
    )
    hess = jax.hessian(lambda c: sum(chi2(c) for chi2 in chi2s))(jnp.zeros(2))

    assert jnp.allclose(hess, jnp.diag(jnp.array([2.0 / 4.0, 2.0 / 0.25])))


def test_missing_coefficient_raises(nuisance_group: CoefficientGroup) -> None:
    with pytest.raises(ValueError, match="'Vud' is not declared"):
        _load({"GaussConstraintVud": {"central": 0.97, "sigma": 0.001}}, nuisance_group)


def test_non_positive_sigma_raises(nuisance_group: CoefficientGroup) -> None:
    with pytest.raises(ValueError, match="sigma must be positive"):
        _load({"GaussConstraintEta1": {"central": 0.0, "sigma": 0.0}}, nuisance_group)
