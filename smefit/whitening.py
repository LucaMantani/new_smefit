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
        Translation applied after the linear map: the coefficients' baseline
        point by default, or the gradient-descent best-fit point when
        ``whitening.shift: gradient_descent`` is requested.
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

    H = d²chi2/dc² |_{center} + eps*I = L L^T (Cholesky); returns W = L^{-T},
    so that the curvature seen in whitened space, W^T H W, is the identity.
    """
    eps = whitening["eps"]
    H = jax.hessian(chi2)(center)

    eigvals = jnp.linalg.eigvalsh(H)
    lam_min, lam_max = float(eigvals[0]), float(eigvals[-1])

    if lam_min + eps <= 0:
        # An exactly flat direction only comes out as O(machine_eps * lam_max)
        # from the AD Hessian, so a slightly negative eigenvalue is noise; one
        # far below that floor is real negative curvature, i.e. a saddle.
        cause = (
            "the centre is a saddle point, not a minimum"
            if lam_min < -100 * float(jnp.finfo(H.dtype).eps) * abs(lam_max)
            else "a flat direction came out numerically negative"
        )
        raise ValueError(
            f"Whitening: H + eps*I is not positive definite (smallest eigenvalue "
            f"{lam_min:.3e}, whitening.eps = {eps:.1e}): {cause}. Raising eps above "
            f"{abs(lam_min):.1e} makes the factorisation succeed, but then eps "
            f"rather than the data sets that direction's scale; prefer centring on "
            f"the fitted minimum with 'whitening: {{shift: gradient_descent}}'."
        )

    # eps floors the curvature of any flat direction, so it, and not the data,
    # sets how far the whitened prior reaches there: sigma_prior/sqrt(eps).
    n_flat = int(jnp.sum(eigvals < eps))
    log.log(
        logging.WARNING if n_flat else logging.INFO,
        "Whitening: eigenvalues of H in [%.3e, %.3e], cond(H + eps*I) = %.3e%s",
        lam_min,
        lam_max,
        (lam_max + eps) / (lam_min + eps),
        (
            f", {n_flat}/{chi2.nparam} below eps={eps:.1e} — unconstrained, with "
            f"the prior reaching ~{whitening['sigma_prior'] / eps**0.5:.1e}"
            if n_flat
            else ""
        ),
    )

    L = jnp.linalg.cholesky(H + eps * jnp.eye(chi2.nparam))
    return jnp.linalg.solve(L.T, jnp.eye(chi2.nparam))


def _whitening_baseline_shift(chi2, whitening):
    """Hessian evaluated at the coefficients' baseline point.

    Reached when whitening["shift"] == "baseline" (the default).
    """
    baseline = chi2.baseline
    return WhitenTransform(
        matrix=_build_matrix(chi2, whitening, baseline), shift=baseline
    )


def _whitening_gradient_descent_shift(chi2, gd_best_fit, whitening):
    """Hessian evaluated at the gradient-descent best-fit point.

    Only reached when whitening["shift"] == "gradient_descent".
    """
    shift = gd_best_fit
    return WhitenTransform(matrix=_build_matrix(chi2, whitening, shift), shift=shift)
