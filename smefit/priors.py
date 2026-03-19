"""
smefit.priors — Prior distributions for nested sampling.
"""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

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
