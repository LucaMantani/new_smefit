"""
smefit.priors — Prior distributions for the sampler-based fits.
"""

import logging
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

log = logging.getLogger(__name__)


def _logit_eps(z):
    """Smallest offset from 0/1 that survives z's dtype.

    Must be dtype-aware: a hardcoded 1e-12 rounds away entirely in float32
    (``1 - 1e-12 == 1``), so the clip below would not actually guard log(0).
    """
    return jnp.finfo(jnp.asarray(z).dtype).eps


# --- Internal 1D distribution classes ---


class _Distribution(ABC):
    @abstractmethod
    def ppf(self, u): ...  # inverse CDF

    @abstractmethod
    def log_prob(self, x): ...  # log density

    @abstractmethod
    def sample(self, rng_key, shape): ...

    @abstractmethod
    def __str__(self) -> str: ...  # short human-readable label, e.g. "U[-1, 1]"

    # --- Bijector to an unconstrained space, used by gradient-based samplers ---

    @abstractmethod
    def to_unconstrained(self, x): ...  # constrained x -> u in R

    @abstractmethod
    def from_unconstrained(self, u): ...  # u in R -> constrained x

    @abstractmethod
    def log_det_jacobian(self, u): ...  # log |dx/du| at u


class _UniformDist(_Distribution):
    def __init__(self, low, high):
        self.low, self.high = float(low), float(high)

    def ppf(self, u):
        return self.low + (self.high - self.low) * u

    def log_prob(self, x):
        in_support = (x >= self.low) & (x <= self.high)
        return jnp.where(in_support, -jnp.log(self.high - self.low), -jnp.inf)

    def sample(self, rng_key, shape):
        return jax.random.uniform(rng_key, shape, minval=self.low, maxval=self.high)

    def __str__(self) -> str:
        return f"U[{self.low}, {self.high}]"

    def from_unconstrained(self, u):
        """Logit bijector: x = low + (high - low) * sigmoid(u).

        Large |u| saturates x to exactly low/high. That is safe because
        ``log_prob`` uses inclusive bounds, so the boundary is finite.
        """
        return self.low + (self.high - self.low) * jax.nn.sigmoid(u)

    def to_unconstrained(self, x):
        """Inverse logit. The clip keeps x exactly on a bound from giving log(0),
        which would seed a gradient sampler with NaN gradients."""
        z = (x - self.low) / (self.high - self.low)
        eps = _logit_eps(z)
        z = jnp.clip(z, eps, 1.0 - eps)
        return jnp.log(z) - jnp.log1p(-z)

    def log_det_jacobian(self, u):
        """log |dx/du| = log(high - low) + log sigmoid(u) + log sigmoid(-u).

        Written with ``log_sigmoid`` rather than ``log(s) + log(1 - s)``: the
        latter underflows to -inf (and NaN gradients) for |u| >~ 37, while this
        form is exact and asymptotically -|u|.
        """
        return (
            jnp.log(self.high - self.low)
            + jax.nn.log_sigmoid(u)
            + jax.nn.log_sigmoid(-u)
        )


class _GaussianDist(_Distribution):
    def __init__(self, mean, std):
        self.mean, self.std = float(mean), float(std)

    def ppf(self, u):
        return self.mean + self.std * jax.scipy.special.ndtri(u)

    def log_prob(self, x):
        return (
            -0.5 * ((x - self.mean) / self.std) ** 2
            - jnp.log(self.std)
            - 0.5 * jnp.log(2 * jnp.pi)
        )

    def sample(self, rng_key, shape):
        return self.mean + self.std * jax.random.normal(rng_key, shape)

    def __str__(self) -> str:
        return f"N(mu={self.mean}, sigma={self.std})"

    def from_unconstrained(self, u):
        """Identity: the support is already all of R."""
        return u

    def to_unconstrained(self, x):
        return x

    def log_det_jacobian(self, u):
        """Zero, since ``from_unconstrained`` is the identity."""
        return jnp.zeros_like(u, dtype=jnp.result_type(float))


_DIST_REGISTRY = {
    "uniform": _UniformDist,
    "gaussian": _GaussianDist,
    "normal": _GaussianDist,
}


def _build_dist(spec):
    spec = dict(spec)
    dist_name = spec.pop("dist").lower()
    if dist_name not in _DIST_REGISTRY:
        raise ValueError(
            f"Unknown prior '{dist_name}'. Available: {sorted(_DIST_REGISTRY)}"
        )
    return _DIST_REGISTRY[dist_name](**spec)


# --- Public interface ---


class Prior:
    """Joint prior over all free coefficients.

    Compatible with ultranest (prior_transform), BlackJax nested sampling
    (log_prob, sample), and gradient-based samplers, which work in the
    unconstrained reparametrisation below rather than in coefficient space.
    """

    def __init__(self, dists, param_names, specs={}):
        self.dists = list(dists)
        self.param_names = list(param_names)
        self.prior_specs = specs

    @jax.jit(static_argnames=("self",))
    def prior_transform(self, unit_cube):
        """Map unit-cube coordinates to parameter values (ultranest interface)."""
        return jnp.array([d.ppf(unit_cube[i]) for i, d in enumerate(self.dists)])

    @jax.jit(static_argnames=("self",))
    def log_prob(self, x):
        """Sum of log-prior densities across all free parameters (BlackJax interface)."""
        return jnp.sum(jnp.array([d.log_prob(x[i]) for i, d in enumerate(self.dists)]))

    def sample(self, rng_key, n_samples):
        """Draw n_samples joint prior samples, shape (n_samples, n_params) (BlackJax interface)."""
        keys = jax.random.split(rng_key, len(self.dists))
        return jnp.stack(
            [d.sample(keys[i], (n_samples,)) for i, d in enumerate(self.dists)], axis=-1
        )

    # --- Unconstrained reparametrisation (gradient-based samplers) ---
    #
    # NUTS and friends need an unbounded, differentiable target: a uniform
    # prior evaluated outside its box returns -inf, which stalls the integrator
    # at the walls and biases the boundary.

    @jax.jit(static_argnames=("self",))
    def from_unconstrained(self, u):
        """Map unconstrained coordinates u to coefficient values."""
        return jnp.array([d.from_unconstrained(u[i]) for i, d in enumerate(self.dists)])

    @jax.jit(static_argnames=("self",))
    def to_unconstrained(self, x):
        """Map coefficient values to unconstrained coordinates."""
        return jnp.array([d.to_unconstrained(x[i]) for i, d in enumerate(self.dists)])

    @jax.jit(static_argnames=("self",))
    def log_det_jacobian(self, u):
        """log|dx/du| of the reparametrisation, summed over parameters."""
        return jnp.sum(
            jnp.array([d.log_det_jacobian(u[i]) for i, d in enumerate(self.dists)])
        )

    @jax.jit(static_argnames=("self",))
    def log_prob_unconstrained(self, u):
        """Log prior density in the unconstrained space."""
        return self.log_prob(self.from_unconstrained(u)) + self.log_det_jacobian(u)

    def sample_unconstrained(self, rng_key, n_samples):
        """Prior draws mapped to u-space, shape (n_samples, n_params)."""
        return jax.vmap(self.to_unconstrained)(self.sample(rng_key, n_samples))


class _WhitenedToPhysicalPrior:
    """Wraps a whitened prior, evaluated at physical-space coordinates.

    When a previous fit used whitening (c = transform.to_physical(c_w) =
    matrix @ c_w + shift), the prior was defined over c_w. To evaluate it at
    physical coords c we invert the transform:

        log p_phys(c) = log p_w(transform.to_whitened(c)) + log|det(matrix⁻¹)|

    The additive ``shift`` only translates coordinates and leaves the
    Jacobian determinant unchanged, so the log-det term depends only on
    ``transform.matrix``.
    """

    def __init__(self, whitened_prior, transform):
        self.param_names = whitened_prior.param_names
        self.prior_specs = whitened_prior.prior_specs
        self._whitened_prior = whitened_prior
        self._transform = transform
        matrix_inv = jnp.linalg.inv(transform.matrix)
        self._log_abs_det_matrix_inv = jnp.log(jnp.abs(jnp.linalg.det(matrix_inv)))

    @jax.jit(static_argnames=("self",))
    def log_prob(self, x_phys):
        x_w = self._transform.to_whitened(x_phys)
        return self._whitened_prior.log_prob(x_w) + self._log_abs_det_matrix_inv


class ExactPosteriorPrior:
    """Exact posterior prior for Bayesian sequential updates.

    Represents P(θ|D1) as the prior for a subsequent fit:

        log_prob(θ) = log_prior_1(θ) + log_likelihood_1(θ)
        sample()    → returns fit1 posterior samples directly.

    Parameters
    ----------
    base_prior : Prior or _WhitenedToPhysicalPrior
        Prior representing P(θ|D1) in physical space. Pass a plain ``Prior``
        when the previous fit had no whitening, or a ``_WhitenedToPhysicalPrior``
        when it did.
    log_likelihood_1 : callable
        Function θ -> scalar: log-likelihood of fit1's data.
    samples_dict : dict[str, array]
        Posterior samples from fit1, keyed by parameter name.
    param_names : list[str]
        Ordered list of free parameter names.
    source_path : str, optional
        Path to the previous fit directory (for bookkeeping).
    """

    def __init__(
        self, base_prior, log_likelihood_1, samples_dict, param_names, source_path=""
    ):
        self.param_names = list(param_names)
        self._base_prior = base_prior
        self._log_likelihood_1 = log_likelihood_1
        # Stack free-parameter samples: shape (n_available, n_params)
        self._posterior_samples = jnp.stack(
            [jnp.array(samples_dict[name]) for name in param_names], axis=-1
        )
        # prior_specs records that this is an exact-posterior prior (for display)
        self.prior_specs = {
            name: {"dist": "exact_posterior", "source": source_path}
            for name in param_names
        }

    @jax.jit(static_argnames=("self",))
    def log_prob(self, x):
        return self._base_prior.log_prob(x) + self._log_likelihood_1(x)

    def sample(self, rng_key, n_samples):
        """Subsample n_samples from fit1's posterior (with replacement if needed)."""
        n_avail = self._posterior_samples.shape[0]
        replace = n_samples > n_avail
        if replace:
            log.warning(
                "ExactPosteriorPrior: requested %d live points but only %d posterior "
                "samples are available. Sampling with replacement.",
                n_samples,
                n_avail,
            )
        indices = jax.random.choice(
            rng_key, n_avail, shape=(n_samples,), replace=replace
        )
        return self._posterior_samples[indices]
