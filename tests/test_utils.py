"""Unit tests for pure functions in smefit.utils."""

import ast
import inspect

import jax
import jax.numpy as jnp

from smefit import utils
from smefit.core import Coefficient, CoefficientGroup
from smefit.utils import ensure_list, resolve_posterior

_PRIOR = {"dist": "uniform", "low": -5.0, "high": 5.0}


def _free(name):
    return Coefficient(name=name, free=True, prior=_PRIOR)


# ---------------------------------------------------------------------------
# utils is a low-level module (issue #90)
# ---------------------------------------------------------------------------


def test_utils_imports_no_smefit_module():
    """smefit.utils must not import other smefit modules, at any level.

    It sits at the bottom of the import graph so every module can use it;
    importing e.g. fit_result or priors here brings back circular imports.
    """
    tree = ast.parse(inspect.getsource(utils))
    offending = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offending += [
                a.name for a in node.names if a.name.split(".")[0] == "smefit"
            ]
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] == "smefit":
                offending.append(node.module or ".")
    assert not offending, f"smefit.utils imports smefit modules: {offending}"


# ---------------------------------------------------------------------------
# ensure_list
# ---------------------------------------------------------------------------


def test_ensure_list_passthrough():
    lst = [1, 2, 3]
    assert ensure_list(lst) is lst


def test_ensure_list_wraps_scalar():
    assert ensure_list(42) == [42]


def test_ensure_list_wraps_string():
    assert ensure_list("abc") == ["abc"]


# ---------------------------------------------------------------------------
# resolve_posterior
# ---------------------------------------------------------------------------


def test_resolve_posterior_shapes():
    """samples dict has shape (n_samples,) per name; best_fit_point is a dict."""
    cg = CoefficientGroup([_free("OpA"), _free("OpB")])

    n_samples = 20
    posterior_free = jax.random.normal(jax.random.PRNGKey(0), (n_samples, 2))
    best_free = jnp.array([0.5, -0.3])

    samples, best_fit_point = resolve_posterior(cg, posterior_free, best_free)

    for name in cg.names:
        assert name in samples
        assert samples[name].shape == (n_samples,)

    assert isinstance(best_fit_point, dict)
    for name in cg.names:
        assert name in best_fit_point
        assert isinstance(best_fit_point[name], float)
