"""Unit tests for smefit.projections — Projection.build_data_group scenarios."""

import jax.numpy as jnp
import numpy as np
import pytest

from smefit.core import (
    Coefficient,
    CoefficientGroup,
    DataGroup,
    Dataset,
    Theory,
    TheoryGroup,
)
from smefit.model import EFTModel
from smefit.projections import Projection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(lumi_new=None, noise="L0", seed=None, fred_sys=1.0, fred_tot=1.0):
    return {
        "lumi_new": lumi_new,
        "noise": noise,
        "seed": seed,
        "fred_sys": fred_sys,
        "fred_tot": fred_tot,
    }


def _theory(name, sm_pred, eft_pred=None, operators=None):
    """Minimal Theory; uses a single placeholder operator if none provided."""
    if operators is None:
        operators = ["OpA"]
    if eft_pred is None:
        eft_pred = {op: [0.0] * len(sm_pred) for op in operators}
    n = len(sm_pred)
    return Theory(
        name=name,
        order="LO",
        sm_pred=jnp.array(sm_pred, dtype=float),
        eft_pred=eft_pred,
        sm_covmat=jnp.eye(n) * 0.01,
        scales=jnp.full(n, 100.0),
        operators=operators,
    )


# ---------------------------------------------------------------------------
# Fixtures: aligned SM (cv_exp == sm_pred, ratio = 1 everywhere)
# ---------------------------------------------------------------------------


@pytest.fixture
def ds_aligned():
    """3-point dataset whose CV exactly matches theory sm_pred → ratio = 1."""
    return Dataset(
        name="PROJ_DS",
        num_data=3,
        central_values=jnp.array([100.0, 200.0, 300.0]),
        stat_err=jnp.array([10.0, 20.0, 30.0]),
        syst_err=jnp.array([[5.0, 10.0, 15.0]]),  # (1, 3) ADD
        sys_names=["SYS1"],
        sys_types=["ADD"],
        luminosity=jnp.array([10.0, 10.0, 10.0]),
    )


@pytest.fixture
def th_aligned():
    """Theory for PROJ_DS: sm_pred == cv_exp → stat ratio = 1."""
    return _theory(
        "PROJ_DS",
        sm_pred=[100.0, 200.0, 300.0],
        eft_pred={"OpA": [1.0, 2.0, 3.0]},
        operators=["OpA"],
    )


@pytest.fixture
def data_aligned(ds_aligned):
    return DataGroup([ds_aligned])


@pytest.fixture
def theory_aligned(th_aligned):
    return TheoryGroup([th_aligned])


@pytest.fixture
def eft_model_aligned(theory_aligned):
    """EFTModel with OpA fixed at 0.5 (all-fixed → empty free-params array)."""
    cg = CoefficientGroup([Coefficient(name="OpA", free=False, value=0.5)])
    return EFTModel(theory_aligned, cg, use_quad=False)


@pytest.fixture
def theory_aligned_expr():
    """Theory for PROJ_DS with operators OpD and OpWB (for expr-coefficient tests)."""
    return TheoryGroup(
        [
            _theory(
                "PROJ_DS",
                sm_pred=[100.0, 200.0, 300.0],
                eft_pred={"OpWB": [1.0, 2.0, 3.0], "OpD": [10.0, 20.0, 30.0]},
                operators=["OpD", "OpWB"],
            )
        ]
    )


@pytest.fixture
def eft_model_expr(theory_aligned_expr):
    """EFTModel: OpWB fixed=0.1, OpD fixed via expr=0.5*OpWB**2 → 0.005."""
    cg = CoefficientGroup(
        [
            Coefficient(name="OpWB", free=False, value=0.1),
            Coefficient(name="OpD", free=False, vars=["OpWB"], expr="0.5*OpWB**2"),
        ]
    )
    return EFTModel(theory_aligned_expr, cg, use_quad=False)


@pytest.fixture
def theory_aligned_quad():
    """Theory for PROJ_DS with linear and quadratic OpA terms."""
    return TheoryGroup(
        [
            _theory(
                "PROJ_DS",
                sm_pred=[100.0, 200.0, 300.0],
                eft_pred={"OpA": [1.0, 2.0, 3.0], "OpA*OpA": [0.5, 1.0, 1.5]},
                operators=["OpA"],
            )
        ]
    )


@pytest.fixture
def eft_model_quad(theory_aligned_quad):
    """EFTModel with use_quad=True; OpA fixed at 0.5."""
    cg = CoefficientGroup([Coefficient(name="OpA", free=False, value=0.5)])
    return EFTModel(theory_aligned_quad, cg, use_quad=True)


# ---------------------------------------------------------------------------
# SM closure test (lumi_new=None, noise="L0")
# ---------------------------------------------------------------------------


class TestSmClosure:
    def test_cv_equals_sm_pred(self, data_aligned, theory_aligned):
        dg = Projection(data_aligned, theory_aligned, _settings()).build_data_group()
        assert jnp.allclose(dg.datasets[0].central_values, theory_aligned.sm_pred)

    def test_dataset_name_unchanged(self, data_aligned, theory_aligned):
        dg = Projection(data_aligned, theory_aligned, _settings()).build_data_group()
        assert dg.datasets[0].name == "PROJ_DS"

    def test_stat_unchanged_when_ratio_is_one(self, data_aligned, theory_aligned):
        """sm_pred == cv_exp → sqrt(ratio) = 1 → stat unchanged."""
        dg = Projection(data_aligned, theory_aligned, _settings()).build_data_group()
        assert jnp.allclose(dg.datasets[0].stat_err, jnp.array([10.0, 20.0, 30.0]))

    def test_add_sys_unchanged(self, data_aligned, theory_aligned):
        dg = Projection(data_aligned, theory_aligned, _settings()).build_data_group()
        assert jnp.allclose(dg.datasets[0].syst_err, jnp.array([[5.0, 10.0, 15.0]]))

    def test_returns_data_group_with_correct_num_data(
        self, data_aligned, theory_aligned
    ):
        dg = Projection(data_aligned, theory_aligned, _settings()).build_data_group()
        assert isinstance(dg, DataGroup)
        assert dg.num_data == 3

    def test_all_sys_types_are_add(self, data_aligned, theory_aligned):
        dg = Projection(data_aligned, theory_aligned, _settings()).build_data_group()
        assert all(t == "ADD" for t in dg.datasets[0].sys_types)


# ---------------------------------------------------------------------------
# Luminosity projection (lumi_new given, noise="L0")
# ---------------------------------------------------------------------------


class TestLuminosityProjection:
    def test_dataset_name_has_proj_suffix(self, data_aligned, theory_aligned):
        dg = Projection(
            data_aligned, theory_aligned, _settings(lumi_new=40.0)
        ).build_data_group()
        assert dg.datasets[0].name == "PROJ_DS_proj"

    def test_stat_scaled_by_lumi_ratio(self, data_aligned, theory_aligned):
        """stat_red = stat * sqrt(lumi_old / lumi_new) = stat * sqrt(10/40) = stat * 0.5."""
        dg = Projection(
            data_aligned, theory_aligned, _settings(lumi_new=40.0)
        ).build_data_group()
        assert jnp.allclose(dg.datasets[0].stat_err, jnp.array([5.0, 10.0, 15.0]))

    def test_sys_scaled_by_fred_sys(self, data_aligned, theory_aligned):
        dg = Projection(
            data_aligned, theory_aligned, _settings(lumi_new=40.0, fred_sys=0.5)
        ).build_data_group()
        assert jnp.allclose(dg.datasets[0].syst_err, jnp.array([[2.5, 5.0, 7.5]]))

    def test_luminosity_updated(self, data_aligned, theory_aligned):
        dg = Projection(
            data_aligned, theory_aligned, _settings(lumi_new=40.0)
        ).build_data_group()
        assert jnp.allclose(dg.datasets[0].luminosity, jnp.full(3, 40.0))

    def test_cv_equals_sm_pred_in_l0(self, data_aligned, theory_aligned):
        """L0 CV is always the theory prediction regardless of lumi scaling."""
        dg = Projection(
            data_aligned, theory_aligned, _settings(lumi_new=40.0)
        ).build_data_group()
        assert jnp.allclose(dg.datasets[0].central_values, theory_aligned.sm_pred)


# ---------------------------------------------------------------------------
# EFT hypothesis (all-fixed CoefficientGroup → empty free-params array)
# ---------------------------------------------------------------------------


class TestEftHypothesis:
    def test_cv_includes_eft_correction(
        self, data_aligned, theory_aligned, eft_model_aligned
    ):
        """CV = sm_pred + lin_corr @ [0.5] for OpA lin = [1, 2, 3]."""
        dg = Projection(
            data_aligned, theory_aligned, _settings(), eft_model=eft_model_aligned
        ).build_data_group()
        expected = jnp.array([100.5, 201.0, 301.5])
        assert jnp.allclose(dg.datasets[0].central_values, expected)

    def test_sm_cv_without_eft_model(self, data_aligned, theory_aligned):
        dg = Projection(
            data_aligned, theory_aligned, _settings(), eft_model=None
        ).build_data_group()
        assert jnp.allclose(dg.datasets[0].central_values, theory_aligned.sm_pred)

    def test_cv_with_expr_coefficient(
        self, data_aligned, theory_aligned_expr, eft_model_expr
    ):
        """CV = sm + OpWB_lin*0.1 + OpD_lin*0.005, where OpD=0.5*OpWB**2=0.005."""
        dg = Projection(
            data_aligned, theory_aligned_expr, _settings(), eft_model=eft_model_expr
        ).build_data_group()
        # sm=[100,200,300] + [1,2,3]*0.1 + [10,20,30]*0.005 = [100.15, 200.3, 300.45]
        expected = jnp.array([100.15, 200.3, 300.45])
        assert jnp.allclose(dg.datasets[0].central_values, expected, atol=1e-5)

    def test_cv_includes_quadratic_correction(
        self, data_aligned, theory_aligned_quad, eft_model_quad
    ):
        """CV = sm + lin*0.5 + quad*0.5**2, with OpA lin=[1,2,3], quad=[0.5,1,1.5]."""
        dg = Projection(
            data_aligned, theory_aligned_quad, _settings(), eft_model=eft_model_quad
        ).build_data_group()
        # sm=[100,200,300] + [1,2,3]*0.5 + [0.5,1,1.5]*0.25 = [100.625, 201.25, 301.875]
        expected = jnp.array([100.625, 201.25, 301.875])
        assert jnp.allclose(dg.datasets[0].central_values, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# MULT systematics: t0 conversion and ratio rescaling
# ---------------------------------------------------------------------------


class TestMultSystematics:
    """cv_exp = [100, 200, 300], sm_pred = [110, 220, 330] (10% above exp).

    syst_err_mult = syst_err / cv_exp = [0.05, 0.05, 0.05]
    After t0: syst_err_mult * sm_pred = [5.5, 11.0, 16.5]
    ratio = sm_pred / cv_exp = [1.1, 1.1, 1.1]
    stat after ratio rescaling = stat * sqrt(1.1)
    """

    @pytest.fixture
    def ds_mult(self):
        return Dataset(
            name="MULT_DS",
            num_data=3,
            central_values=jnp.array([100.0, 200.0, 300.0]),
            stat_err=jnp.array([10.0, 20.0, 30.0]),
            syst_err=jnp.array([[5.0, 10.0, 15.0]]),  # (1, 3) MULT
            sys_names=["SYS_MULT"],
            sys_types=["MULT"],
            luminosity=jnp.array([10.0, 10.0, 10.0]),
        )

    @pytest.fixture
    def th_mult(self):
        """sm_pred is 10% above cv_exp so the t0 conversion is non-trivial."""
        return _theory("MULT_DS", sm_pred=[110.0, 220.0, 330.0])

    @pytest.fixture
    def proj_mult(self, ds_mult, th_mult):
        return Projection(DataGroup([ds_mult]), TheoryGroup([th_mult]), _settings())

    def test_mult_sys_converted_using_theory_cv(self, proj_mult):
        dg = proj_mult.build_data_group()
        expected = jnp.array([[5.5, 11.0, 16.5]])
        assert jnp.allclose(dg.datasets[0].syst_err, expected)

    def test_output_sys_types_all_add(self, proj_mult):
        dg = proj_mult.build_data_group()
        assert all(t == "ADD" for t in dg.datasets[0].sys_types)

    def test_stat_rescaled_by_sm_exp_ratio(self, proj_mult):
        """stat *= sqrt(sm_pred / cv_exp) = sqrt(1.1)."""
        dg = proj_mult.build_data_group()
        scale = float(jnp.sqrt(jnp.array(1.1)))
        expected = jnp.array([10.0, 20.0, 30.0]) * scale
        assert jnp.allclose(dg.datasets[0].stat_err, expected)


# ---------------------------------------------------------------------------
# No systematics: dummy UNCORR column injected
# ---------------------------------------------------------------------------


class TestNoSystematics:
    @pytest.fixture
    def ds_nosys(self):
        return Dataset(
            name="NO_SYS",
            num_data=3,
            central_values=jnp.array([1.0, 2.0, 3.0]),
            stat_err=jnp.array([0.1, 0.2, 0.3]),
            syst_err=jnp.zeros((0, 3)),
            sys_names=[],
            sys_types=[],
            luminosity=jnp.full(3, 1.0),
        )

    @pytest.fixture
    def th_nosys(self):
        return _theory("NO_SYS", sm_pred=[1.0, 2.0, 3.0])

    def test_dummy_uncorr_column_added(self, ds_nosys, th_nosys):
        dg = Projection(
            DataGroup([ds_nosys]), TheoryGroup([th_nosys]), _settings()
        ).build_data_group()
        ds = dg.datasets[0]
        assert ds.sys_names == ["UNCORR"]
        assert ds.sys_types == ["ADD"]
        assert ds.syst_err.shape == (1, 3)
        assert jnp.all(ds.syst_err == 0.0)


# ---------------------------------------------------------------------------
# L1 noise
# ---------------------------------------------------------------------------


class TestL1Noise:
    def test_cv_deviates_from_theory_pred(self, data_aligned, theory_aligned):
        """L1 fluctuates CVs; they should not equal the theory prediction exactly."""
        dg = Projection(
            data_aligned, theory_aligned, _settings(lumi_new=40.0, noise="L1", seed=0)
        ).build_data_group()
        assert not jnp.allclose(dg.datasets[0].central_values, theory_aligned.sm_pred)

    def test_reproducible_with_same_seed(self, data_aligned, theory_aligned):
        settings = _settings(lumi_new=40.0, noise="L1", seed=99)
        dg1 = Projection(data_aligned, theory_aligned, settings).build_data_group()
        dg2 = Projection(data_aligned, theory_aligned, settings).build_data_group()
        assert jnp.allclose(
            dg1.datasets[0].central_values, dg2.datasets[0].central_values
        )

    def test_different_seeds_give_different_cv(self, data_aligned, theory_aligned):
        dg1 = Projection(
            data_aligned, theory_aligned, _settings(lumi_new=40.0, noise="L1", seed=1)
        ).build_data_group()
        dg2 = Projection(
            data_aligned, theory_aligned, _settings(lumi_new=40.0, noise="L1", seed=2)
        ).build_data_group()
        assert not jnp.allclose(
            dg1.datasets[0].central_values, dg2.datasets[0].central_values
        )

    def test_stat_and_sys_unchanged_by_noise(self, data_aligned, theory_aligned):
        """Noise affects only central values, not uncertainties."""
        dg_l0 = Projection(
            data_aligned, theory_aligned, _settings(lumi_new=40.0, noise="L0", seed=7)
        ).build_data_group()
        dg_l1 = Projection(
            data_aligned, theory_aligned, _settings(lumi_new=40.0, noise="L1", seed=7)
        ).build_data_group()
        assert jnp.allclose(dg_l0.datasets[0].stat_err, dg_l1.datasets[0].stat_err)
        assert jnp.allclose(dg_l0.datasets[0].syst_err, dg_l1.datasets[0].syst_err)


# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_nan_luminosity_with_lumi_new_raises(self, theory_aligned):
        ds_nan = Dataset(
            name="PROJ_DS",
            num_data=3,
            central_values=jnp.array([100.0, 200.0, 300.0]),
            stat_err=jnp.array([10.0, 20.0, 30.0]),
            syst_err=jnp.array([[5.0, 10.0, 15.0]]),
            sys_names=["SYS1"],
            sys_types=["ADD"],
            luminosity=jnp.full(3, jnp.nan),
        )
        proj = Projection(DataGroup([ds_nan]), theory_aligned, _settings(lumi_new=40.0))
        with pytest.raises(ValueError, match="NaN luminosity"):
            proj.build_data_group()

    def test_negative_ratio_clamped_to_one(self):
        """Negative CV in data → ratio clamped to 1, stat not NaN."""
        ds = Dataset(
            name="NEG_CV",
            num_data=2,
            central_values=jnp.array([-1.0, 2.0]),  # negative first point
            stat_err=jnp.array([0.5, 1.0]),
            syst_err=jnp.zeros((1, 2)),
            sys_names=["UNCORR"],
            sys_types=["ADD"],
            luminosity=jnp.full(2, 1.0),
        )
        th = _theory("NEG_CV", sm_pred=[1.0, 2.0])
        dg = Projection(
            DataGroup([ds]), TheoryGroup([th]), _settings()
        ).build_data_group()
        assert not jnp.any(jnp.isnan(dg.datasets[0].stat_err))
        assert not jnp.any(jnp.isinf(dg.datasets[0].stat_err))

    def test_zero_stat_uses_fred_tot(self):
        """When all stats are zero, fred_tot is applied to sys instead of fred_sys."""
        ds = Dataset(
            name="ZERO_STAT",
            num_data=2,
            central_values=jnp.array([1.0, 2.0]),
            stat_err=jnp.array([0.0, 0.0]),  # all zero
            syst_err=jnp.array([[10.0, 20.0]]),
            sys_names=["SYS1"],
            sys_types=["ADD"],
            luminosity=jnp.full(2, 1.0),
        )
        th = _theory("ZERO_STAT", sm_pred=[1.0, 2.0])
        dg = Projection(
            DataGroup([ds]),
            TheoryGroup([th]),
            _settings(lumi_new=4.0, fred_sys=0.5, fred_tot=0.25),
        ).build_data_group()
        # fred_tot=0.25 applied (not fred_sys=0.5) because stats are zero
        assert jnp.allclose(dg.datasets[0].syst_err, jnp.array([[2.5, 5.0]]))


# ---------------------------------------------------------------------------
# Multi-dataset: correct offset tracking
# ---------------------------------------------------------------------------


class TestMultiDataset:
    @pytest.fixture
    def ds_b(self):
        return Dataset(
            name="DS_B",
            num_data=2,
            central_values=jnp.array([50.0, 100.0]),
            stat_err=jnp.array([5.0, 10.0]),
            syst_err=jnp.array([[2.0, 4.0]]),
            sys_names=["SYS1"],
            sys_types=["ADD"],
            luminosity=jnp.array([5.0, 5.0]),
        )

    @pytest.fixture
    def th_b(self):
        return _theory("DS_B", sm_pred=[55.0, 110.0])  # 10% above cv_exp

    def test_both_datasets_present(self, ds_aligned, th_aligned, ds_b, th_b):
        dg = Projection(
            DataGroup([ds_aligned, ds_b]),
            TheoryGroup([th_aligned, th_b]),
            _settings(),
        ).build_data_group()
        assert set(dg.names) == {"PROJ_DS", "DS_B"}

    def test_total_num_data(self, ds_aligned, th_aligned, ds_b, th_b):
        dg = Projection(
            DataGroup([ds_aligned, ds_b]),
            TheoryGroup([th_aligned, th_b]),
            _settings(),
        ).build_data_group()
        assert dg.num_data == 5  # 3 + 2

    def test_each_dataset_cv_correct(self, ds_aligned, th_aligned, ds_b, th_b):
        """Each dataset's CV comes from the correct slice of theory.sm_pred."""
        dg = Projection(
            DataGroup([ds_aligned, ds_b]),
            TheoryGroup([th_aligned, th_b]),
            _settings(),
        ).build_data_group()
        by_name = {ds.name: ds for ds in dg.datasets}
        assert jnp.allclose(
            by_name["PROJ_DS"].central_values, jnp.array([100.0, 200.0, 300.0])
        )
        assert jnp.allclose(by_name["DS_B"].central_values, jnp.array([55.0, 110.0]))

    def test_lumi_projection_name_suffix(self, ds_aligned, th_aligned, ds_b, th_b):
        dg = Projection(
            DataGroup([ds_aligned, ds_b]),
            TheoryGroup([th_aligned, th_b]),
            _settings(lumi_new=20.0),
        ).build_data_group()
        assert set(dg.names) == {"PROJ_DS_proj", "DS_B_proj"}
