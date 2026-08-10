---
name: smefit-dev
description: Use this skill when developing smefit itself — adding or changing a runcard key, a reportengine node (parse_*/produce_*), an action, a settings block, a prior distribution, or an external chi2 contract; wiring a new provider module; or working out why reportengine cannot resolve a resource ("Bad configuration encountered", "cannot find a way to compute"). Also covers which tests to extend and when to regenerate the skill references.
---

# Developing smefit

smefit is a **reportengine** application: a runcard is not read top-to-bottom,
it is a set of inputs from which reportengine builds a DAG and executes the
requested `actions_`. Almost every extension is "add a node to that graph".
Keeping this design intact is the one invariant that matters — nodes compose,
ad-hoc plumbing between modules does not.

## The three node kinds

| Kind | Where | Contract |
|---|---|---|
| `parse_<key>` | `smefit/config.py` | Turns the raw YAML value of runcard key `<key>` into a validated object. Called only when `<key>` is present. |
| `@element_of("<keys>")` on `parse_<key>` | `smefit/config.py` | Turns **one entry** of the list-valued runcard key `<keys>` into an object. reportengine generates `parse_<keys>` from it. |
| `produce_<name>` | `smefit/config.py` | Builds a derived resource `<name>` from other resources (its own arguments). Never written in a runcard. |
| provider function | a module listed in `smefit_providers` (`smefit/app.py`) | Its parameters are resource names; reportengine resolves them. Public ones can be used as actions. |

Resolution is by **name**: a provider parameter called `fit_covmat` is
satisfied by `produce_fit_covmat`, a parameter called `use_quad` by the raw
runcard key. This is why renaming a parameter is an API change, and why a
typo'd parameter name surfaces as "cannot find a way to compute" rather than
as a `NameError`.

A `produce_` can also be decorated with `@explicit_node`, which changes what it
returns and lets it pick its *own* dependencies — see below.

## `=None` defaults turn failures silent

`resolve_signature_params` (`reportengine/configparser.py`) resolves each
parameter and **catches `KeyError`** — and `InputNotFoundError` is a `KeyError`
subclass. So when a parameter has a default, *any* failure to resolve it,
including a genuine error deep inside its own `produce_`, is swallowed and the
default substituted. The only trace is `log.debug("Can't satisfy production rule
for X")`.

The rule that follows:

> A `=None` default belongs only on something that may **legitimately be
> absent** — a runcard key the user need not write, or an environment value that
> genuinely may not exist. Never on a derived node.

Push the default down to the single place where "absent" is a valid state, and
let every consumer above it require the resource:

```python
# rge: is optional in a runcard -> the default lives HERE, and only here
def produce_rge_matrix(self, coefficients, theory, rge=None, output_path=None):
    if rge is None:
        return None
    ...


# ...so eft_model can require it, and a broken RGE build raises
def produce_eft_model(self, theory, coefficients, rge_matrix, use_quad=False):
    return EFTModel(theory, coefficients, use_quad, rge_matrix)
```

Written the other way round (`produce_eft_model(..., rge_matrix=None)`), a
runcard *with* an `rge:` block whose matrix failed to build silently yields a
chi2 with no RGE running — a wrong likelihood, no error. That was a real bug.
Guard the invariant with a signature test: `tests/test_config.py` pins
`rge_matrix` as having no default for exactly this reason.

`produce_chi2`'s `eft_model=None` / `data=None` is the deliberate exception: an
external-`chi2`-only runcard has no `datasets:`, so `eft_model` genuinely cannot
resolve. It pays for that with an explicit check in `_build_chi2_impl` that
raises when data *is* present but `eft_model` is `None`.

## `output_path` is an environment value, not a node

`output_path` is a reportengine **`Environment` attribute** — not a runcard key
and not a `produce_`. `smefitApp` sets it from `-o/--output`, but `smefitAPI`
(`smefit/api.py`) constructs `smefitEnvironment()` with no output folder, and
`Environment.ns_dump()` exports only non-`None` attributes, so under the API the
name is **absent from the namespace entirely** and can never be resolved.

Consequences for a `produce_`/`parse_` that takes it:

- Declare it `output_path=None`. Otherwise the node is unresolvable under the
  API and — per the section above — its failure is swallowed by whichever
  consumer has a default.
- If it only drives a **write**, skip the write when it is `None`
  (`produce_rge_matrix` builds the matrix unconditionally and only calls
  `rge_matrix.write(output_path)` when `output_path is not None`).
- If it only supplies a **default for another key**, fall back to `None` so the
  node still resolves, and say in the docstring that the user must then set that
  key explicitly. Both sampler blocks do this for `log_dir`:

  ```python
  reactive_defaults = {
      "log_dir": (
          str(output_path / "ultranest_logs") if output_path is not None else None
      ),
      "resume": False,
      "vectorized": False,
  }
  ```

Actions listed under `actions_:` always run through the CLI, so they may keep
`output_path` required.

## Dynamic dependencies: `@explicit_node`

A plain `produce_X` returns the **value** of `X`, and its dependencies are
fixed: whatever its own signature names, always. An `@explicit_node`-decorated
`produce_X` instead returns a **function**, and reportengine builds the node
from *that* function — resolving its parameters as further graph requirements.
The dependency set is therefore chosen at graph-build time, from the runcard.

The one instance in smefit is `produce_whitening_transformation`
(`smefit/config.py`), dispatching to the workers in `smefit/whitening.py`:

```python
@explicit_node
def produce_whitening_transformation(self, whitening=None):
    if whitening is None:
        return lambda: None  # no deps
    if whitening["shift"] == "gradient_descent":
        return _whitening_gradient_descent_shift  # (chi2, gd_best_fit, whitening)
    return _whitening_baseline_shift  # (chi2, whitening)
```

Only `shift: gradient_descent` pulls `gd_best_fit` into the DAG — and with it
`optimizer` and `gradient_descent_settings`. With a plain `produce_`, every
whitened fit would have to declare a `gradient_descent_settings` block and pay
for a gradient descent it does not use.

Rules:

- **Return a callable, always.** Reportengine calls `inspect.signature` on the
  returned object (`_make_callspec` in `reportengine/resourcebuilder.py`), so
  the "feature disabled" branch must be a zero-argument callable —
  `lambda: None`, never a bare `None`.
- **The workers are plain module-level functions** living next to the feature
  (not in `config.py`), and their parameter names are resource names, resolved
  by the usual rules. Keep them private (`_`-prefixed) so the generated
  reference does not advertise them as actions.
- **It must stay a method on `smefitConfig`.** The dispatch reads parsed
  runcard values (`whitening["shift"]`), which only the config sees; a provider
  function would already be a node and could not rewrite its own edges.
- **Do not call the worker yourself.** Returning `worker(...)` defeats the
  point: the deferral *is* the mechanism.
- **Reach for it only when the alternatives have different dependencies.** Same
  deps, different maths → a plain `produce_` with an `if` is simpler and easier
  to read.
- Document *why* in the method docstring, as the whitening one does — the next
  reader's instinct will be to "simplify" it back into a plain `produce_`.

Machinery, if you need to check behaviour: `explicit_node`/`ExplicitNode` in
`reportengine/configparser.py`, consumed in `ResourceBuilder._process_requirement`
→ `_make_node((name, val.value))` → `_make_callspec`.

## List-valued runcard keys: `element_of`

A runcard key that is a **list of independent things** must never be parsed by
a hand-rolled loop. Write the rule for *one* entry and let reportengine
generate the plural:

```python
@element_of("fits")
def parse_fit(self, fit: (str, Mapping)):
    ...
    return Fit.from_folder(path, label=label)
```

`ElementOfResolver` (`reportengine/configparser.py:197`) then synthesises
`parse_fits`, which loops the list and calls `parse_fit` on each element.
Three things come with it that a loop inside a `parse_`/`produce_` pair cannot
have:

- **`fits` becomes a namespace list** — `NSList(..., nskey="fit")`. A provider
  can take a single `fit` and be collected over `("fits",)`
  (`collect("fit_plot", ("fits",))`), one output per fit, instead of every
  consumer taking the whole list and looping internally.
- **Type checking, free** — the generated plural is annotated `param: list`, so
  `fits: my_fit` (missing dash) raises `BadInputType` instead of iterating the
  characters of the string. The singular's own annotation checks each element.
- **Per-element traps** — an entry can be `{from_: ...}`, and the singular key
  (`fit:`) works on its own.

Rules:

- **It goes on the singular `parse_`, never on a `produce_`.** A `produce_X`
  only runs when `X` is *absent* from the runcard (`_resolve_key`,
  `reportengine/configparser.py:471`), so a `produce_` named after a key the
  user writes is dead code. Needing a second name for it (`fit_objects`) is the
  symptom that the entry-level work belongs in a parser.
- **What the singular returns is what providers receive.** Return the finished
  object when one entry maps to exactly one thing (`fits` → a `Fit`); return a
  validated spec when several derived resources are built from the same entry,
  and add per-namespace `produce_`s on top.
- **The reference generator keys off `_element_of`.** `collect_config_surface`
  (`scripts/generate_skill_reference.py`) skips names with no source-level
  `def`, so the generated `parse_fits` is invisible to it; the documented key
  comes from `getattr(method, "_element_of", ...)`. Without that the key
  vanishes from `runcard-keys.json` and `validate_runcard.py` reports it as
  unknown in every valid runcard.

Not every list-valued key wants it: `parse_coefficients` stays a whole-dict
parser because `CoefficientGroup` is a cohesive object (ordering, `free_names`,
cross-coefficient constraints), not a bag of independent entries.

## Adding a runcard key

1. **A settings block** (`<thing>_settings:`) → add `parse_<thing>_settings`
   to `smefit/config.py`, following `parse_hessian_settings`:
   - a `known_keys = {...}` set literal, then
     `log.warning("Unknown key '%s' in <thing>_settings.", k)` for the rest —
     unknown keys warn, never raise (see the note below), and
   - return a dict built with explicit `settings.get(key, default)` calls.

   Both of those shapes are load-bearing: `scripts/generate_skill_reference.py`
   AST-extracts the `known_keys` literal and the `.get()` defaults to build
   `runcard-keys.md`/`.json`, which is what `validate_runcard.py` checks
   against. Depart from the pattern and the key silently vanishes from the
   documented surface.

2. **A raw scalar** (like `use_quad`) → no parser needed; just take it as a
   provider/`produce_` parameter with a default. It is picked up automatically.

3. **A path** → resolve it through `smefit.paths.resolve_path` so the shareable
   prefix form (`smefit_database/...`, `smefit_results/...`) keeps working; see
   `parse_data_path`. A path the parser only forwards — an `external_chi2` entry
   passes every key but `path` straight to a user-authored constructor — cannot
   be resolved by name at this layer; resolve it in whatever loads the file, as
   `RGEMatrix.read_cache` does for `rg_matrix`.

**Unknown keys warn rather than raise** — that is a deliberate
backwards-compatibility choice, and the reason `validate_runcard.py` exists.
Every new key must reach the generated reference (step above) or users get no
protection against misspelling it.

## Adding an action

1. Write the function in the right provider module — `smefit/fit_actions.py`
   for "run a fit and write it", `smefit/utils_actions.py` for other
   side-effecting actions, `smefit/tables.py` / `smefit/figures.py` for report
   output. Keep actions thin: compute in a `produce_`/provider node, and let
   the action print and write (`run_ultranest_fit` is the model).
2. If the module is new, add it to `smefit_providers` in `smefit/app.py`.
3. Take `output_path` as a parameter if you write files — never construct
   output paths yourself. An action may require it; a `produce_`/`parse_` that
   takes it must default it to `None` (see "`output_path` is an environment
   value, not a node" above).
4. Name it so the generated reference classifies it correctly: the generator
   treats `run_*`, `write_*`, `plot_*`, anything in `smefit.tables` /
   `smefit.figures`, and an explicit allowlist as *runnable actions*, and
   everything else as an internal provider. A new action outside those
   conventions must be added to `EXTRA_ACTION_NAMES` in
   `scripts/generate_skill_reference.py`, or it will be documented as
   "never write this under `actions_:`".
5. **A new fit action** must be named `run_<fit type>_fit`, or
   `run_individual_<fit type>_fits` when it fits one coefficient at a time:
   `_parse_fit_action` (`smefit/fit_result.py`) reads the fit type and
   individual-ness straight off the name. That is how a fit loaded from disk
   knows which sampler produced it and how it was driven — the action in
   `input/runcard.yaml` is the only record — so a name outside the convention
   silently leaves `fit_type` unset.

## Coefficients, priors, external chi2

- **Coefficient sub-keys** are the fields of the `Coefficient` dataclass in
  `smefit/core.py`, passed straight through by `parse_coefficients`. Add a
  field there and it becomes a runcard key automatically — including in the
  generated reference, which introspects the dataclass. Invariants between
  fields belong in `__post_init__`, and must be mirrored in
  `../smefit-runcard/references/coefficients.md` and in
  `validate_runcard.py`'s `check_coefficient`.
- **Priors**: add the distribution class to `_DIST_REGISTRY` in
  `smefit/priors.py`; the constructor signature becomes the required prior
  parameters in the generated `priors.md` and in the validator. Implement all
  of `ppf`, `log_prob`, `sample`, `__str__`, plus the bijector trio
  `to_unconstrained` / `from_unconstrained` / `log_det_jacobian` that gradient
  samplers need — identity is the right answer for any distribution already
  supported on all of R. Keep the log-det in a form that survives the tails
  (`_UniformDist` uses `log_sigmoid(u) + log_sigmoid(-u)` for exactly that
  reason) and make any epsilon dtype-aware: the test suite runs float32, where
  a hardcoded `1e-12` rounds away. The trio is deliberately `@abstractmethod`
  and **must not** gain a default identity implementation: a bounded
  distribution that inherited one would not crash, it would quietly return a
  biased posterior. Add the new distribution to `_CONTRACT_SPECS` in
  `tests/test_priors.py` — a test asserts that dict covers `_DIST_REGISTRY`
  exactly, and the contract tests keyed off it then check round-trip, log-det
  against autodiff, and that the bijector lands inside the support.
  `Prior` then lifts that trio to the joint
  distribution (`from_unconstrained`, `log_prob_unconstrained`,
  `sample_unconstrained`) — these live on `Prior` itself rather than in a
  wrapper class, because they are pure functions of its own `dists`.
  `ExactPosteriorPrior` and `_WhitenedToPhysicalPrior` know only a joint
  `log_prob`, so they have no unconstrained interface. Nothing in the fit path
  checks this: `algorithm: nuts` with `bayesian_update_path` fails inside
  `_run_nuts` with `AttributeError: no attribute 'sample_unconstrained'`. Only
  the `smefit-runcard` validator flags the combination up front.
- **BlackJAX algorithms**: `blackjax_settings.algorithm` dispatches through
  `_SAMPLER_REGISTRY` in `smefit/blackjax_samplers/__init__.py` — the same
  registry pattern as `_DIST_REGISTRY`, since every algorithm has identical DAG
  dependencies (this is why it is *not* an `@explicit_node`). Each algorithm is
  one module in the package, exposing exactly two names:
  `run(rng_key, prior, log_likelihood, n_samples, settings)` returning a
  `SamplerOutput`, and `SETTINGS`, the frozenset of `blackjax_settings` keys it
  owns. `__init__.py` derives both `_SAMPLER_REGISTRY` and
  `BJ_ALGORITHM_SETTINGS` from the `_ALGORITHM_MODULES` map, so a new algorithm
  is one new module plus one entry there — the two cannot drift apart. Its keys
  still have to be added to the literal `known_keys` set in
  `parse_blackjax_settings`; that set must stay an inline set literal (the
  generator AST-extracts it), and
  `tests/test_config.py::test_parse_blackjax_known_keys_cover_all_algorithm_settings`
  is what keeps it in step with `BJ_ALGORITHM_SETTINGS`. Shared pieces
  (`SamplerOutput`, the `_HealthReport` fail/warn/verdict scaffolding,
  `_write_diagnostics`) live in `_common.py`; per-algorithm diagnostics stay
  with their runner, because no statistic transfers between algorithms. Do not
  register the package in `smefit_providers` — it holds helpers, not nodes.
- **Expression constraints**: functions available inside `expr:` are exactly
  `_EXPR_NAMESPACE` in `smefit/core.py` (JAX-backed, so constraints stay
  differentiable). Adding one there widens the runcard language — update
  `coefficients.md` too.
- **External chi2 modules** must accept `coefficients=` and `rge_dict=` plus
  the runcard's extra keys, and expose `compute_chi2`, `num_data`,
  `param_names`. See `parse_external_chi2` and `smefit/external_chi2.py`.

## Keep it JAX-differentiable

`chi2.build_chi2` returns a JAX-differentiable loss, and
`EFTModel.forward_map` is JIT-compiled. Inside anything reachable from them:
no Python branching on traced values, no numpy in the hot path, no in-place
mutation. Precision is set once at startup by `smefitEnvironment`
(`-f32` flag) — never call `jax.config` yourself.

## After the change

1. **Tests** — mirror the module: `smefit/<mod>.py` → `tests/test_<mod>.py`.
   Config surface goes in `tests/test_config.py`; use `tests/fixtures/`
   (`TESTDATA`) rather than the real database, and mark anything that calls a
   real sampler or RGE evolution `@pytest.mark.slow`.
   ```bash
   pytest -m "not slow"
   ```
2. **Regenerate the skill references** whenever the user-facing surface
   changed (a runcard key, an action, a prior, a coefficient field, a
   template):
   ```bash
   python scripts/generate_skill_reference.py
   pytest tests/test_skill_scripts.py -q
   ```
   CI (`.github/workflows/skills.yml`) fails if the committed files are stale.
   Hand-written skill files it cannot generate — `coefficients.md`,
   `recipes.md`, `troubleshooting.md`, `output-layout.md` — are yours to update
   in the same commit.
3. **Formatting**: `pre-commit run --all-files`.

## When reportengine will not resolve something

- *"cannot find a way to compute `X`"* — no `parse_X`, no `produce_X`, and no
  provider named `X`; usually a typo in a parameter name, or a provider module
  missing from `smefit_providers`.
- *An action silently does nothing* — it was resolved as a value, not run;
  check it is in a module listed in `smefit_providers`.
- *A settings block is ignored* — the runcard key and the `parse_` suffix must
  match exactly (`ultranest_settings` ↔ `parse_ultranest_settings`).
- *A resource is demanded that the runcard never asked for* (e.g. a fit without
  `gradient_descent_settings` complaining about it) — follow the
  `@explicit_node` dispatch: the branch taken decides the dependencies.
- *A runcard key appears to be ignored, with no error* — a consumer's `=None`
  default swallowed the resolution failure. Re-run at debug log level and look
  for `Can't satisfy production rule for X`, then fix the signature rather than
  the symptom (see "`=None` defaults turn failures silent").
- Runtime symptoms of a *user's* fit (not the code) belong to the
  `smefit-analysis` skill's `references/troubleshooting.md` and the
  `smefit-fit-doctor` agent.
