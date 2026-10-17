"""Unit tests for smefit.chi2 — Chi2 class and build_chi2 function."""

import jax.numpy as jnp
import pytest

from smefit.chi2 import Chi2, build_chi2, build_datasets_chi2
from smefit.core import Coefficient, CoefficientGroup, DataGroup, Dataset, TheoryGroup
from smefit.model import EFTModel
from smefit.whitening import WhitenTransform

_PRIOR = {"dist": "uniform", "low": -5.0, "high": 5.0}


def _free(name):
    return Coefficient(name=name, free=True, prior=_PRIOR)


def _make_dataset(cv, name="CHI2_DS"):
    """Minimal dataset with given central values."""
    n = len(cv)
    return Dataset(
        name=name,
        num_data=n,
        central_values=jnp.array(cv, dtype=float),
        stat_err=jnp.ones(n),
        syst_err=jnp.zeros((1, n)),
        sys_names=["UNCORR"],
        sys_types=["ADD"],
        luminosity=jnp.ones(n),
    )


# ---------------------------------------------------------------------------
# build_chi2 (raw function)
# ---------------------------------------------------------------------------


def test_chi2_zero_when_data_equals_pred(theory_a):
    # data = sm_pred → residuals are zero at coeffs=0
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory_a, cg, use_quad=False)
    data = DataGroup([_make_dataset([10.0, 20.0, 30.0])])
    chi2_fn = build_chi2(model, data, jnp.eye(3))
    result = chi2_fn(jnp.array([0.0]))
    assert jnp.allclose(result, 0.0)


def test_chi2_known_value(theory_a):
    # data = sm_pred + 1 → residuals = [1,1,1], identity covmat → chi2 = 3
    cg = CoefficientGroup([_free("OpA")])
    model = EFTModel(theory_a, cg, use_quad=False)
    data = DataGroup([_make_dataset([11.0, 21.0, 31.0])])
    chi2_fn = build_chi2(model, data, jnp.eye(3))
    result = chi2_fn(jnp.array([0.0]))
    assert jnp.allclose(result, 3.0)


# ---------------------------------------------------------------------------
# Chi2 class metadata and callable
# ---------------------------------------------------------------------------


def test_chi2_metadata():
    fn = lambda c: jnp.sum(c**2)
    chi2 = Chi2(fn, ["OpA", "OpB"], num_data=10, has_external=False)
    assert chi2.nparam == 2
    assert chi2.num_data == 10
    assert chi2.param_names == ["OpA", "OpB"]
    assert chi2.has_external is False


def test_chi2_callable():
    fn = lambda c: jnp.sum(c**2)
    chi2 = Chi2(fn, ["OpA"], num_data=5)
    result = chi2(jnp.array([3.0]))
    # Result must be a scalar with value 9.0
    assert result.shape == ()
    assert float(result) == pytest.approx(9.0)


def test_chi2_baseline_default_zeros():
    chi2 = Chi2(lambda c: jnp.sum(c**2), ["OpA", "OpB"], num_data=5)
    assert jnp.allclose(chi2.baseline, jnp.zeros(2))


def test_chi2_baseline_explicit():
    baseline = jnp.array([0.7, -0.3])
    chi2 = Chi2(lambda c: jnp.sum(c**2), ["OpA", "OpB"], num_data=5, baseline=baseline)
    assert jnp.allclose(chi2.baseline, baseline)


# ---------------------------------------------------------------------------
# Chi2.whitened
# ---------------------------------------------------------------------------


def test_chi2_whitened_evaluates_at_physical_point():
    """Whitened chi2 at c_w equals the original at matrix @ c_w + shift."""
    transform = WhitenTransform(
        matrix=jnp.array([[2.0, 0.0], [0.0, 3.0]]), shift=jnp.array([0.5, -1.0])
    )
    chi2 = Chi2(lambda c: jnp.sum(c**2), ["OpA", "OpB"], num_data=7)

    whitened = chi2.whitened(transform)

    c_w = jnp.array([1.0, 1.0])
    assert float(whitened(c_w)) == pytest.approx(
        float(chi2(transform.to_physical(c_w)))
    )


def test_chi2_whitened_baseline_in_whitened_coordinates():
    baseline = jnp.array([1.0, -2.0])
    transform = WhitenTransform(
        matrix=jnp.array([[2.0, 0.0], [0.0, 3.0]]), shift=jnp.array([0.5, -1.0])
    )
    chi2 = Chi2(lambda c: jnp.sum(c**2), ["OpA", "OpB"], num_data=7, baseline=baseline)

    whitened = chi2.whitened(transform)

    assert jnp.allclose(whitened.baseline, transform.to_whitened(baseline))
    assert jnp.allclose(transform.to_physical(whitened.baseline), baseline)


def test_chi2_whitened_baseline_is_origin_when_centred_on_it():
    """With shift = baseline (the default whitening centre), the whitened
    baseline is the origin — what the sampler expects to start from."""
    baseline = jnp.array([1.0, -2.0])
    transform = WhitenTransform(matrix=jnp.eye(2) * 2.0, shift=baseline)
    chi2 = Chi2(lambda c: jnp.sum(c**2), ["OpA", "OpB"], num_data=7, baseline=baseline)

    assert jnp.allclose(chi2.whitened(transform).baseline, jnp.zeros(2))


def test_chi2_whitened_preserves_metadata():
    transform = WhitenTransform(matrix=jnp.eye(2), shift=jnp.zeros(2))
    chi2 = Chi2(
        lambda c: jnp.sum(c**2),
        ["OpA", "OpB"],
        num_data=7,
        has_external=True,
        name="total",
    )

    whitened = chi2.whitened(transform)

    assert isinstance(whitened, Chi2)
    assert whitened.param_names == ["OpA", "OpB"]
    assert whitened.nparam == 2
    assert whitened.num_data == 7
    assert whitened.has_external is True
    assert whitened.name == "total"


# ---------------------------------------------------------------------------
# build_datasets_chi2
# ---------------------------------------------------------------------------


def test_build_datasets_chi2_structure(theory_a, theory_b):
    # Two 3-point datasets → 2 Chi2 objects with matching names and data counts
    data = DataGroup(
        [
            _make_dataset([10.0, 20.0, 30.0], name="DS_A"),
            _make_dataset([40.0, 50.0, 60.0], name="DS_B"),
        ]
    )
    theory = TheoryGroup([theory_a, theory_b])
    model = EFTModel(theory, CoefficientGroup([_free("OpA")]), use_quad=False)

    result = build_datasets_chi2(model, data, jnp.eye(6))

    assert len(result) == 2
    assert all(isinstance(c, Chi2) for c in result)
    assert [c.name for c in result] == ["DS_A", "DS_B"]
    assert [c.num_data for c in result] == [3, 3]


def test_build_datasets_chi2_zero_when_data_equals_pred(theory_a, theory_b):
    # data = sm_pred for each dataset → chi2 = 0 at zero coefficients
    data = DataGroup(
        [
            _make_dataset([10.0, 20.0, 30.0], name="DS_A"),
            _make_dataset([40.0, 50.0, 60.0], name="DS_B"),
        ]
    )
    theory = TheoryGroup([theory_a, theory_b])
    model = EFTModel(theory, CoefficientGroup([_free("OpA")]), use_quad=False)

    chi2s = build_datasets_chi2(model, data, jnp.eye(6))
    coeffs = jnp.array([0.0])

    for chi2 in chi2s:
        assert jnp.allclose(chi2(coeffs), 0.0)


def test_build_datasets_chi2_known_values(theory_a, theory_b):
    # DS_A: data = SM + 1 → residuals [1,1,1], identity covmat → chi2 = 3
    # DS_B: data = SM + 2 → residuals [2,2,2], identity covmat → chi2 = 12
    data = DataGroup(
        [
            _make_dataset([11.0, 21.0, 31.0], name="DS_A"),
            _make_dataset([42.0, 52.0, 62.0], name="DS_B"),
        ]
    )
    theory = TheoryGroup([theory_a, theory_b])
    model = EFTModel(theory, CoefficientGroup([_free("OpA")]), use_quad=False)

    chi2s = build_datasets_chi2(model, data, jnp.eye(6))
    coeffs = jnp.array([0.0])

    assert jnp.allclose(chi2s[0](coeffs), 3.0)
    assert jnp.allclose(chi2s[1](coeffs), 12.0)
