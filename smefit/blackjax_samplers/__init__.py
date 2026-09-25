"""
smefit.blackjax_samplers

Package entry point: the registry of BlackJAX algorithms selectable with
``blackjax_settings.algorithm``. Each algorithm lives in its own module
exposing exactly two names:

``run(rng_key, prior, log_likelihood, n_samples, settings)``
    the runner, returning a ``SamplerOutput``;
``SETTINGS``
    the frozenset of ``blackjax_settings`` keys it owns.

Adding an algorithm therefore means writing one module and adding one entry to
``_ALGORITHM_MODULES`` below — the runner registry and the settings-ownership
map are both derived from it, so they cannot drift apart. The matching keys
still have to be added to the literal ``known_keys`` set in
``smefitConfig.parse_blackjax_settings``; ``tests/test_config.py`` enforces
that.

Everything the rest of smefit needs is re-exported here, so importing sites
(``smefit.blackjax_fit`` for ``get_sampler``, ``smefit.config`` for the ``BJ_``
constants) are unaffected by how the package is laid out internally. Neither of
those may be imported back from here, and the package is deliberately NOT
registered in ``smefit.app.smefit_providers``: its contents are helpers, not
reportengine nodes.
"""

from smefit.blackjax_samplers import nested_sampling, nuts
from smefit.blackjax_samplers._common import (  # noqa: F401  (public re-export)
    SamplerOutput,
)

#: The algorithms, keyed by the value of ``blackjax_settings.algorithm``.
_ALGORITHM_MODULES = {
    "nested_sampling": nested_sampling,
    "nuts": nuts,
}

_SAMPLER_REGISTRY = {name: mod.run for name, mod in _ALGORITHM_MODULES.items()}

BJ_ALGORITHMS = tuple(_ALGORITHM_MODULES)

#: Keys of ``blackjax_settings`` that every algorithm uses.
BJ_SHARED_SETTINGS = frozenset({"algorithm", "seed", "log_dir"})

#: Keys of ``blackjax_settings`` owned by one algorithm.
BJ_ALGORITHM_SETTINGS = {name: mod.SETTINGS for name, mod in _ALGORITHM_MODULES.items()}


def get_sampler(algorithm):
    """Look up the runner for *algorithm*."""
    if algorithm not in _SAMPLER_REGISTRY:
        raise ValueError(
            f"Unknown BlackJAX algorithm '{algorithm}'. "
            f"Available: {sorted(_SAMPLER_REGISTRY)}"
        )
    return _SAMPLER_REGISTRY[algorithm]
