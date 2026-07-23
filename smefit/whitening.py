"""
smefit.whitening.py

Builds the affine whitening transform c = matrix @ c_w + shift, used to
reparametrise free coefficients into a better-conditioned space for sampling.
"""

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp

log = logging.getLogger(__name__)


@dataclass(eq=False)
class WhitenTransform:
    """Affine map between whitened (c_w) and physical (c) coefficient space.

    c = matrix @ c_w + shift            (to_physical)
    c_w = matrix^{-1} @ (c - shift)      (to_whitened)

    Attributes
    ----------
    matrix : jnp.ndarray, shape (n_free, n_free)
        Cholesky-derived unwhitening matrix, W = L^{-T} where H = L L^T.
    shift : jnp.ndarray, shape (n_free,)
        Translation applied after the linear map. Zero vector when no shift
        was requested.
    """

    matrix: jnp.ndarray
    shift: jnp.ndarray

    def to_physical(self, c_w: jnp.ndarray) -> jnp.ndarray:
        return self.matrix @ c_w + self.shift

    def to_whitened(self, c_phys: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.solve(self.matrix, c_phys - self.shift)

    def to_dict(self) -> dict:
        """JSON-serialisable representation, used by FitResult.write()."""
        return {"matrix": self.matrix.tolist(), "shift": self.shift.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "WhitenTransform":
        return cls(matrix=jnp.array(d["matrix"]), shift=jnp.array(d["shift"]))


def _build_matrix(chi2, whitening, center):
    """Hessian-of-chi2 whitening matrix, evaluated at *center*.

    H = d²chi2/dc² |_{center} + eps*I = L L^T (Cholesky); returns W = L^{-T}.
    """
    eps = whitening["eps"]
    H = jax.hessian(chi2)(center) + eps * jnp.eye(chi2.nparam)
    log.info("Hessian whitening: cond(H) = %.3e", float(jnp.linalg.cond(H)))
    L = jnp.linalg.cholesky(H)
    return jnp.linalg.solve(L.T, jnp.eye(chi2.nparam))


def _no_shift(chi2, whitening):
    """Hessian evaluated at c=0, no shift."""
    zeros = jnp.zeros(chi2.nparam)
    return WhitenTransform(matrix=_build_matrix(chi2, whitening, zeros), shift=zeros)


def _with_shift(chi2, gd_best_fit, whitening):
    """Hessian evaluated at the gradient-descent best-fit point.

    Only reached when whitening["shift"] is True; this is the sole path
    through which gd_best_fit (and therefore gradient_descent_settings)
    enters the DAG for a whitening-enabled runcard.
    """
    shift = gd_best_fit
    return WhitenTransform(matrix=_build_matrix(chi2, whitening, shift), shift=shift)


def _disabled():
    """Returned when whitening is disabled entirely; takes no dependencies."""
    return None


def apply_whitening(chi2, coefficients, whitening_transformation):
    """Transform chi2 and coefficients into whitened space.

    Returns the transformed chi2 callable and whitened CoefficientGroup.
    """
    _chi2 = lambda c_w: chi2(whitening_transformation.to_physical(c_w))
    resolve_coeffs = coefficients.whitened(whitening_transformation)
    return _chi2, resolve_coeffs
