"""
Shared fixtures for the SMEFiT unit test suite.
All fixtures are fully in-memory; no real I/O or reportengine integration.
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib as mpl
import pytest

from smefit.core import (
    Coefficient,
    CoefficientGroup,
    DataGroup,
    Dataset,
    Theory,
    TheoryGroup,
)

# ---------------------------------------------------------------------------
# Matplotlib
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_rc_params():
    """Keep one test's rc settings out of the next test's figures.

    The figure routines set the report's font and ``text.usetex`` globally, as
    matplotlib's rc is global; without this a test that only ever draws a bare
    ``plt.subplots()`` still renders through whatever the last figure routine
    left behind.
    """
    with mpl.rc_context():
        yield


@pytest.fixture
def draw():
    """Render a figure, so that what the tick formatters produce can be read.

    Rendering goes through mathtext rather than LaTeX: the figure routines turn
    ``text.usetex`` on for the report's own labels, and CI has no LaTeX
    installed, while every label these tests read back is plain enough for
    matplotlib to typeset itself.

    The rc setting alone is not enough — a Text takes its ``usetex`` from the
    rc in force when it was *created*, which for a tick label is when the axes
    were built — so the labels already in the figure are turned over one by
    one, and the rc covers whatever the draw creates.
    """

    def _draw(fig):
        mpl.rcParams["text.usetex"] = False
        for text in fig.findobj(mpl.text.Text):
            text.set_usetex(False)
        fig.canvas.draw()

    return _draw


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_data_path():
    return FIXTURES_DIR / "commondata"


@pytest.fixture
def fixtures_theory_path():
    return FIXTURES_DIR / "theory"


# ---------------------------------------------------------------------------
# Dataset fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset_a():
    """3-point dataset with UNCORR and CORR (intra-dataset) systematics."""
    return Dataset(
        name="DS_A",
        num_data=3,
        central_values=jnp.array([1.0, 2.0, 3.0]),
        stat_err=jnp.array([0.1, 0.2, 0.3]),
        syst_err=jnp.array([[0.05, 0.10, 0.15], [0.02, 0.04, 0.06]]),  # (2, 3)
        sys_names=["UNCORR", "CORR"],
        sys_types=["ADD", "ADD"],
        luminosity=jnp.array([1.0, 1.0, 1.0]),
    )


@pytest.fixture
def dataset_b():
    """3-point dataset with a cross-dataset systematic (not in INTRA_DATASET_SYS_NAME)."""
    return Dataset(
        name="DS_B",
        num_data=3,
        central_values=jnp.array([4.0, 5.0, 6.0]),
        stat_err=jnp.array([0.4, 0.5, 0.6]),
        syst_err=jnp.array([[0.10, 0.20, 0.30]]),  # (1, 3)
        sys_names=["CROSS_SYS"],
        sys_types=["ADD"],
        luminosity=jnp.array([2.0, 2.0, 2.0]),
    )


# ---------------------------------------------------------------------------
# Theory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def theory_a():
    """Theory for DS_A with operators OpA, OpB and quadratic OpA*OpA, OpA*OpB."""
    return Theory(
        name="DS_A",
        order="NLO",
        sm_pred=jnp.array([10.0, 20.0, 30.0]),
        eft_pred={
            "OpA": [1.0, 2.0, 3.0],
            "OpB": [0.5, 1.0, 1.5],
            "OpA*OpA": [0.1, 0.2, 0.3],
            "OpA*OpB": [0.05, 0.10, 0.15],
        },
        sm_covmat=jnp.diag(jnp.array([0.01, 0.04, 0.09])),
        scales=jnp.array([100.0, 100.0, 100.0]),
        operators=["OpA", "OpB"],
    )


@pytest.fixture
def theory_b():
    """Theory for DS_B with operators OpA, OpC — OpC is absent from theory_a (union test)."""
    return Theory(
        name="DS_B",
        order="NLO",
        sm_pred=jnp.array([40.0, 50.0, 60.0]),
        eft_pred={
            "OpA": [2.0, 3.0, 4.0],
            "OpC": [0.3, 0.6, 0.9],
            "OpA*OpA": [0.01, 0.02, 0.03],
        },
        sm_covmat=jnp.diag(jnp.array([0.16, 0.25, 0.36])),
        scales=jnp.array([200.0, 200.0, 200.0]),
        operators=["OpA", "OpC"],
    )


# ---------------------------------------------------------------------------
# Coefficient fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def coeff_free():
    return Coefficient(
        name="OpA",
        free=True,
        prior={"dist": "uniform", "low": -1.0, "high": 1.0},
    )


@pytest.fixture
def coeff_fixed():
    return Coefficient(name="OpB", free=False, value=2.0)


@pytest.fixture
def coeff_expr():
    return Coefficient(name="OpC", free=False, vars=["OpA"], expr="OpA**2")


@pytest.fixture
def coeff_group(coeff_free, coeff_fixed, coeff_expr):
    return CoefficientGroup([coeff_free, coeff_fixed, coeff_expr])


# ---------------------------------------------------------------------------
# Sampler fixtures (minimal chi2 / prior for ultranest / blackjax tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_chi2():
    """Trivial Chi2 over a single free parameter for testing sampler wrappers."""
    from smefit.chi2 import Chi2

    fn = lambda c: jnp.sum(c**2)
    return Chi2(fn, param_names=["OpA"], num_data=10)


@pytest.fixture
def minimal_prior():
    """Uniform prior over a single free parameter for testing sampler wrappers."""
    from smefit.priors import Prior, _UniformDist

    dist = _UniformDist(-1.0, 1.0)
    return Prior(
        [dist],
        ["OpA"],
        specs={"OpA": {"dist": "uniform", "low": -1.0, "high": 1.0}},
    )
