# Daughter PR 5: naming pass

Part of the generalization beyond the EFT (issue #116). See
[GENERALIZATION_00_ANALYSIS.md](GENERALIZATION_00_ANALYSIS.md) for the overall
analysis and how the PRs are organised.

- Branch: `generalize-beyond-eft-naming`, created from `generalize-beyond-eft`
  once PRs 2, 3 and 4 are merged into it.
- Target: `generalize-beyond-eft`.
- Depends on PR 2, which adds the `model:` runcard block and `ModelSpec`. It
  only strictly needs PR 2, but it renames names that every other PR touches,
  so it should go last: a mechanical rename is cheap to redo on top of the
  others, while their changes are not cheap to rebase onto it.

## Goal

After PRs 1 to 4 the generic code works for any model, but still calls the
model "EFT". This PR renames what the generic code calls it, plus the two
`RGEMatrix` fields PR 1 left for this pass, and nothing else.

- **The model resources get a model-neutral name.** `eft_model` and its two
  variants are what users request through `smefitAPI` and report templates.
- **`RGEMatrix` drops "operators"** in its two name fields. PR 1 already
  gives the runner neutral names.
- **Old names keep working**, for a long time, wherever someone outside
  smefit can use them.

Results do not change, and no runcard key changes.

## Decisions this plan builds on

1. **The resource is called `physics_model`, not `model`.** After PR 2, `model` is a runcard key: `parse_model` turns
   the `model:` block into a `ModelSpec`. reportengine resolves a parameter by
   name, runcard keys first, so a provider asking for `model` would receive the
   `ModelSpec` whenever the runcard has a `model:` block, and a
   `produce_model` would only run when it has none (the reason PR 2 named its
   own producer `produce_model_spec`).
   - **`physics_model`** says what the object is: the physics model the
     runcard selected (the SMEFT's `EFTModel` today, a σ×BR model later),
     built and ready to map free parameters to predictions. Next to
     `model_spec`, the choice written in the runcard, it is the model built
     from that choice.
   - **Alternatives considered:** `forward_model` (after `forward_map`, but
     jargon from inference rather than physics); `prediction_model` (longer,
     no clearer); renaming the runcard key instead (the `model:` block is what
     users write, and the more important name).
   - The variants follow: `individual_physics_model`,
     `individual_mass_physics_model`.
2. **The old resource names stay as deprecated aliases.** Each is a
   `produce_eft_model(self, physics_model)` that logs a deprecation warning and
   returns its argument, so `smefitAPI.eft_model(**runcard)` and report
   templates keep working.
   - **Logged, not a Python `DeprecationWarning`.** These names are requested
     from runcards, templates and `smefitAPI`, where a `DeprecationWarning` is
     hidden by default. PR 2 logs its `use_quad` deprecation for the same
     reason.
   - **Kept for a long time.** Legacy code and stored results rely on the old
     names, so the master PR sets no removal date, for this PR's aliases or
     for those of PRs 1 to 3.
   - **No default on the argument.** An alias with `physics_model=None` would
     turn a failure to build the model into a silent `None` (see the
     `smefit-dev` skill on `=None` defaults).
3. **Aliases only where callers outside smefit exist.**
   - **Yes:** the three resources (API, templates, users' scripts), and the
     `RGEMatrix` fields, which external chi2 modules may read.
   - **No:** plain helpers that are neither nodes nor documented API, renamed
     outright: `build_chi2`, `build_datasets_chi2`, `_build_chi2_impl` and
     `Projection`'s `eft_model=` keyword. Only smefit and its tests call them.
4. **The runner's names are PR 1's.** `RGERunner` is new, so PR 1 writes it
   with `names`, `matrix` and `matrix_dict` from the start, and keeps `RGE`'s
   old names as deprecated aliases for callers. This PR only renames
   `RGEMatrix`'s fields.
5. **"Coefficients" stays.** The runcard key `coefficients:`, `Coefficient`,
   `CoefficientGroup`, `coeff_list` and `Chi2.param_names` keep their names.
   `coefficients:` is in every runcard and every stored fit, and renaming it to
   POPxf's "parameters" is a runcard migration of its own, not a rename of
   internal names. The analysis's open question on it stays open, past the
   master PR.
6. **Only names whose meaning is generic change.** `EFTModel`, `use_quad`, and
   the `SMEFTRunner`-only names (`RGEevolve`, `RGEbasis`, `map_to_smefit`,
   `ALL_OPS`) are SMEFT and stay. `sm_solution` stays too: it means "start
   from the baseline point", it is a runcard key, and renaming it would need a
   deprecated key path for little gain.

## Current state

`eft_model` and its variants, by place (after PRs 2 to 4, which keep these
names):

| Place | What |
|---|---|
| `smefit/config.py` | The three `produce_*eft_model` producers; the `eft_model=None` / `individual_eft_model=None` / `individual_mass_eft_model=None` parameters of `produce_chi2`, `produce_datasets_chi2`, `produce_individual_chi2`, `produce_individual_mass_chi2` and `produce_pseudodata`; `_build_chi2_impl`'s argument and its error message |
| `smefit/analytic_fit.py` | `analytic_fit(eft_model, …)`, a provider, so the parameter is a resource name |
| `smefit/individual_fit.py` | `individual_analytic_fit(individual_eft_model, …)`, likewise |
| `smefit/chi2.py` | `build_chi2(eft_model, …)`, `build_datasets_chi2(eft_model, …)`: not a provider module, plain arguments |
| `smefit/projections.py` | `Projection(…, eft_model=None)` and `self.eft_model` |
| `smefit/fit_result.py` | Two docstrings, one with `smefitAPI.eft_model(**fit.rebuild_runcard())` |
| Skills | `smefit-dev/SKILL.md` (the `=None` section and the rebuild example); `smefit-analysis/references/output-layout.md` (the same rebuild example); generated: `runcard-keys.md`/`.json` and both `actions.md` |
| Tests | `test_config.py` (12), `test_projections.py` (11), `test_hessian_fit.py` (1) |

The RGE names PR 1 left here are `RGEMatrix.obs_operators` and
`RGEMatrix.init_operators`, in `smefit/rge/matrix.py`. They are read by
`EFTModel`, `build.py`, `config.py` (a log line) and tests, and they exist on
`main`.

`rge_matrix.pkl` stores the settings and one `DataFrame` per scale, with the
names as its index and columns. The field names are not part of the file, so
renaming them does not touch the pickle layout.

## Target names

| Old | New | Kind | Old name after this PR |
|---|---|---|---|
| `eft_model` | `physics_model` | resource | Deprecated alias producer, logged |
| `individual_eft_model` | `individual_physics_model` | resource | Same |
| `individual_mass_eft_model` | `individual_mass_physics_model` | resource | Same |
| `analytic_fit(eft_model, …)`, `individual_analytic_fit(individual_eft_model, …)` | `physics_model`, `individual_physics_model` | provider parameters | Follow the resource |
| `eft_model=None` etc. in the five consumers | `physics_model=None` etc. | `produce_` parameters | The `=None` defaults stay exactly where they are |
| `build_chi2`, `build_datasets_chi2`, `_build_chi2_impl` argument | `model` | plain argument | Removed |
| `Projection(eft_model=…)`, `self.eft_model` | `model=`, `self.model` | plain keyword | Removed |
| `RGEMatrix.obs_operators`, `.init_operators` | `obs_names`, `init_names` | dataclass fields | Properties raising a `DeprecationWarning` |

`build_chi2` and friends can call their argument `model`: they are not nodes,
so their parameter names are never resolved by reportengine.

## Target design

### `config.py`

```python
def produce_physics_model(self, model_spec, theory, coefficients, rge_matrix):
    """The model built from the runcard, mapping free parameters to predictions. …"""
    return model_spec.build(theory, coefficients, rge_matrix)


def produce_eft_model(self, physics_model):
    """Deprecated alias of ``physics_model``; will be removed in <release>."""
    log.warning(
        "The 'eft_model' resource is deprecated; request 'physics_model' instead."
    )
    return physics_model
```

- The same for the two variants. The docstring of each alias starts with
  "Deprecated alias of", which is what the generated reference shows next to
  it.
- `produce_physics_model` keeps `rge_matrix` without a default; the signature
  test in `tests/test_config.py` moves to it.
- The five consumers rename their parameter and keep its `=None`. The comment
  explaining the `produce_chi2` exception moves with it, and
  `_build_chi2_impl`'s error message says "model".
- `_FIT_SETUP_KEYS` and `produce_fitcontext` do not change: no setup key is
  renamed.

### Providers, `chi2.py`, `projections.py`

- `analytic_fit` and `individual_analytic_fit` take `physics_model` and
  `individual_physics_model`; their docstrings say "model".
- `build_chi2`, `build_datasets_chi2` and `Projection` take `model`.

### The RGE layer

- `RGEMatrix` gets `obs_names` and `init_names`, plus the two deprecated
  properties. The pickle is written from the new fields, and its layout does
  not change, so files written before and after read the same way.
- `build.py`, `EFTModel`, `config.py` and the in-repo `external_chi2/` modules
  use the new names. The Drell-Yan and optimal-observables modules read none of
  them today, so they should need no change; the commit checks with `grep`.

### Docstrings

In the modules the analysis classes as general, "EFT", "Wilson coefficients"
and "operators" become "model", "parameters" or "coefficients" where they
describe something generic: `pca.py`, `analytic_fit.py`, `individual_fit.py`,
`projections.py`, `fit_result.py`, `rge/`. A docstring that is about the SMEFT
(the TeV/GeV note, `EFTModel` itself) keeps its wording.

## Commits

Every commit:

- leaves `pytest -m "not slow"` green;
- updates the docs it makes stale in the same commit;
- regenerates the skill references when it changes a node name, a provider
  signature or a `parse_`/`produce_` docstring;
- passes `pre-commit run --all-files`.

### 1. Rename the model resources

- `config.py`: the three producers, the five consumers, the three alias
  producers.
- `analytic_fit.py`, `individual_fit.py`, `chi2.py`, `projections.py`,
  `fit_result.py`.
- Tests: `test_config.py`, `test_projections.py`, `test_hessian_fit.py`, and
  the new tests below.
- Docs:
  - `smefit-dev/SKILL.md`: the `=None` section, whose example becomes
    `produce_physics_model` and whose exception becomes
    `produce_chi2`'s `physics_model=None`; the rebuild example
    (`smefitAPI.physics_model(**runcard)`);
  - `smefit-analysis/references/output-layout.md`: the rebuild example;
  - `AGENTS.md`, wherever PR 2 left the resource's name;
  - regenerate `runcard-keys.md`/`.json` and both `actions.md`.

### 2. Neutral names for `RGEMatrix`'s fields

- `matrix.py`, with the two deprecated properties; `build.py`,
  `smefit/models/smeft/model.py` and `config.py` use the new names.
- Tests: `test_rge.py`, `test_smeft_model.py`, `test_config.py`,
  `test_bayesian_update.py`, `test_external_chi2.py` and `test_fit_result.py`,
  which read the old fields.

### 3. Docstring pass

No code change. The modules listed under "Docstrings", and the matching lines
of the module list in `AGENTS.md`.

## Tests

| File | Change |
|---|---|
| `tests/test_config.py` | The `rge_matrix` signature pin (`test_eft_model_requires_rge_matrix`) moves to `produce_physics_model` and `produce_individual_physics_model`, and gains `produce_individual_mass_physics_model`, whose `individual_mass_rge_matrix` is unpinned today. **Aliases:** each of the three returns the same object as its new name and logs the warning once per call; none has a default. **Graph:** `chi2` on `tests/fixtures/` resolves through `physics_model`, and a runcard whose model fails to build still raises instead of falling back (the `=None` guard under its new name). |
| `tests/test_config.py` (there is no `test_api.py`) | `smefitAPI.eft_model(**runcard)` still works and warns; `smefitAPI.physics_model(**runcard)` does not warn. |
| `tests/test_app.py` | A static scan: no provider or `produce_`/`parse_` parameter is named `eft_model`, `individual_eft_model` or `individual_mass_eft_model`, except the alias producers' own arguments, which are the new names. This keeps the old names from creeping back. |
| `tests/test_projections.py`, `tests/test_hessian_fit.py` | The `model=` keyword and the new resource name |
| `tests/test_rge.py` | The new fields; each alias warns and returns the same value; a `rge_matrix.pkl` written on `generalize-beyond-eft` reads back with the same frames. |

## Verification

1. After every commit, run `pytest -m "not slow"`. After commit 2, also run the
   full `pytest`.
2. **Regression against the integration branch.** Run
   `template_runcards/analytical_fit.yaml`, `hessian_fit.yaml` and `rge.yaml`
   on both branches. `fit_results.json` must be identical, and so must the
   `rge_matrix.pkl` frames.
3. **The old names.** A scratch script calling `smefitAPI.eft_model(...)` on
   the analytic template works and logs the warning once.
4. **Nothing left behind.** `grep -rn "eft_model\|obs_operators\|init_operators" smefit/ external_chi2/ .agents/ AGENTS.md`
   only finds the aliases and their deprecation notes.
5. Regenerating the skill references matches what is committed, and CI's
   skills check passes.

## Not in this PR

- **`coefficients` → `parameters`** (Decision 5).
- **`sm_solution`**, `EFTModel`, `use_quad` and the SMEFT-only RGE names
  (Decision 6).
- **Removing the aliases.** They stay for a long time (Decision 2).
- **The theory's names** (`sm_pred`, `eft_lin_pred`, `operators`, …): PR 3
  already renamed them, with their own aliases.
