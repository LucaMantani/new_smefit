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
| `produce_<name>` | `smefit/config.py` | Builds a derived resource `<name>` from other resources (its own arguments). Never written in a runcard. |
| provider function | a module listed in `smefit_providers` (`smefit/app.py`) | Its parameters are resource names; reportengine resolves them. Public ones can be used as actions. |

Resolution is by **name**: a provider parameter called `fit_covmat` is
satisfied by `produce_fit_covmat`, a parameter called `use_quad` by the raw
runcard key. This is why renaming a parameter is an API change, and why a
typo'd parameter name surfaces as "cannot find a way to compute" rather than
as a `NameError`.

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
   `parse_data_path`.

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
   output paths yourself.
4. Name it so the generated reference classifies it correctly: the generator
   treats `run_*`, `write_*`, `plot_*`, anything in `smefit.tables` /
   `smefit.figures`, and an explicit allowlist as *runnable actions*, and
   everything else as an internal provider. A new action outside those
   conventions must be added to `EXTRA_ACTION_NAMES` in
   `scripts/generate_skill_reference.py`, or it will be documented as
   "never write this under `actions_:`".

## Coefficients, priors, external chi2

- **Coefficient sub-keys** are the fields of the `Coefficient` dataclass in
  `smefit/core.py`, passed straight through by `parse_coefficients`. Add a
  field there and it becomes a runcard key automatically — including in the
  generated reference, which introspects the dataclass. Invariants between
  fields belong in `__post_init__`, and must be mirrored in
  `.claude/skills/smefit-runcard/references/coefficients.md` and in
  `validate_runcard.py`'s `check_coefficient`.
- **Priors**: add the distribution class to `_DIST_REGISTRY` in
  `smefit/priors.py`; the constructor signature becomes the required prior
  parameters in the generated `priors.md` and in the validator. Implement all
  of `ppf`, `log_prob`, `sample`, `__str__`.
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
- Runtime symptoms of a *user's* fit (not the code) belong to the
  `smefit-analysis` skill's `references/troubleshooting.md` and the
  `smefit-fit-doctor` agent.
