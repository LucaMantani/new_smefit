"""
smefit.priors — Prior distributions for nested sampling.
"""

import logging
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

log = logging.getLogger(__name__)

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

    Compatible with both ultranest (prior_transform) and BlackJax (log_prob, sample).
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


class _WhitenedToPhysicalPrior:
    """Wraps a whitened prior, evaluated at physical-space coordinates.

    When a previous fit used whitening (c = W @ c_w), the prior was defined
    over c_w. To evaluate it at physical coords c we apply W⁻¹:

        log p_phys(c) = log p_w(W⁻¹ c) + log|det(W⁻¹)|
    """

    def __init__(self, whitened_prior, W):
        self.param_names = whitened_prior.param_names
        self.prior_specs = whitened_prior.prior_specs
        self._whitened_prior = whitened_prior
        W_inv = jnp.linalg.inv(W)
        self._W_inv = W_inv
        self._log_abs_det_W_inv = jnp.log(jnp.abs(jnp.linalg.det(W_inv)))

    @jax.jit(static_argnames=("self",))
    def log_prob(self, x_phys):
        x_w = self._W_inv @ x_phys
        return self._whitened_prior.log_prob(x_w) + self._log_abs_det_W_inv


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
