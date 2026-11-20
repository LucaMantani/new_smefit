# Daughter PR 4: external chi2 contract

Part of the generalization beyond the EFT (issue #116). See
[GENERALIZATION_00_ANALYSIS.md](GENERALIZATION_00_ANALYSIS.md) for the overall
analysis and how the PRs are organised.

This plan has two parts:

1. **Two bugfix PRs against `main`.** It fixes three bugs in the plugin loader
   that give users wrong results today. It needs nothing from the master PR,
   so it should not wait for it.
2. **Daughter PR 4 proper.** It makes the plugin contract model-aware.

The bugfix PRs fix the two bug reports (#174 and #175) described below.

Daughter PR 4:

- Branch: `generalize-beyond-eft-ext-chi2`, created from
  `generalize-beyond-eft` once PR 2 is merged into it, and once `main` (with
  the bugfix) has been merged into the integration branch.
- Target: `generalize-beyond-eft`.
- Depends on the bugfix PR, PR 1 (`SMEFTRunner`) and PR 2 (`ModelSpec`, the
  `model:` block).
- Touches the same Drell-Yan modules as PR 3. Whichever of the two merges
  second rebases onto the other.

## Goal

An external likelihood ("plugin") should work with any model, or say clearly
that it does not. Today the contract hands plugins the coefficients and the
`rge:` block, and nothing about the model they are used with.

- **Plugin failures are never silent**, and each plugin is loaded from its
  own file (the bugfix PR).
- **Plugins learn the model.** A plugin can receive the runcard's model, as
  the `ModelSpec` of PR 2, and declare which models it supports (PR 4).
- **The written contract matches the code** (PR 4).

Results do not change, with one intended exception. A runcard whose plugin
fails today, and so silently runs without it, fails loudly after the bugfix.
A stored fit with such an entry can no longer be rebuilt, in a report or a
Bayesian update. Its error says which plugin was missing from that fit. The
bugfix's release note says so. Every in-repo plugin keeps its options and its
defaults.

## Decisions this plan builds on

1. **The bug fixes go to `main` first.** The bugs give wrong likelihoods with
   no error, today, to everyone using external chi2. The master PR is
   long-lived and only refactors, and the fixes need nothing from it.
2. **`rge_dict` stays.** After PR 2 the `rge:` block holds the generic keys
   plus the options of the model's runner, so it is no longer SMEFT-specific.
   Removing it would break every plugin. The analysis said plugins would get
   the model's settings "instead of" `rge_dict`. This plan passes them in
   addition to it.
3. **The model reaches a plugin as `model_spec=`, opt-in.** It is passed only
   to a constructor that names a `model_spec` parameter, so existing plugins
   are called exactly as today. A catch-all `**kwargs` does not count, because
   such a constructor may forward its keyword arguments elsewhere.
4. **A plugin declares the models it supports** in a `SUPPORTED_MODELS` class
   attribute. A runcard whose model is not in it fails at load time with a
   clear error. A plugin without the attribute is not checked, for
   compatibility.
5. **In-repo plugins keep their own options.** The Drell-Yan modules keep
   their `use_quad` option and its default of `True`, even though the runcard
   now has `model: {use_quad: ...}`. Making them follow the runcard's model
   would change results for runcards that rely on that default (see the open
   questions).

## Current state

### The contract, as written and as implemented

| Part | Written (`recipes.md`, the `smefit-dev` skill, `parse_external_chi2`) | Implemented (`load_external_chi2`) |
|---|---|---|
| Constructor | `cls(coefficients=..., rge_dict=..., **entry_keys)` | The same. `path` and `group` are not forwarded. |
| `compute_chi2(coeffs)` | Required | Required. It receives the free-parameter values, in `coefficients.free_names` order, and must be a JAX function (fits take its gradient). |
| `num_data` | Required | Required |
| `param_names` | Required | Never read: smefit uses `coefficients.free_names` |
| Supported models | Not stated | Not checked |

### Three bugs, reproduced on `main`

They were reproduced on 2026-11-19 with the fixture dataset `TESTDATA` (3
points), through `smefitAPI.chi2(...)`. Each plugin entry adds 2 points.

| Case | Result | Expected |
|---|---|---|
| Control: a working plugin | `num_data=5`, `has_external=True` | 5 |
| Entry without `path` | `num_data=3`, `has_external=False` | an error |
| Plugin constructor raises `KeyError` | `num_data=3`, `has_external=False` | an error |
| A file `dup.py` in two folders | the second folder loads the first one's class | each its own class |

Nothing reaches stderr in the two silent cases. The CLI shows only the
"Loading external chi2 module: …" line, which is logged before the failure.

1. **An entry without `path`.** `parse_external_chi2` only resolves `path` if
   it is present. `load_external_chi2` then reads `module["path"]` and raises
   a `KeyError`.
2. **A plugin constructor that raises a `KeyError`**, for example from
   `rge_dict["init_scale"]` or a missing settings key.

   In both cases the error escapes `produce_ext_chi2_func`. reportengine
   treats a `KeyError` as "input not available", and `produce_chi2` (and its
   `individual_*` and `individual_mass_*` variants) default `ext_chi2_func` to
   None. The fit runs on the datasets alone, without that likelihood and
   without an error.
   - **The same mechanism as #146.** That issue fixed it for a missing
     theory order in `load_theory`.
   - **Other exceptions are loud.** A plugin's `AttributeError` is reported
     with a full traceback (#54).
   - **`validate_runcard.py` catches a missing `path`**, but smefit itself
     does not.
3. **Plugins with the same file name.** `load_external_chi2` prepends the
   plugin's folder to `sys.path`, once per load, and imports the module by
   its file stem. Python caches modules by name, so a second plugin file with
   the same stem in another folder gets the first one's code.
   - **Where it bites:** in one process that loads two versions of a plugin,
     for example a report rebuilding several fits (`fits:` with
     `fitcontext`).
   - **An example pair:** the `new_smefit` and `smefit_database` copies of
     `optimal_observables/interface_oos_fccee.py`. The `smefit_database` copy
     is written against an old API and fails to import under current smefit.
     But if the `new_smefit` copy is loaded first, the collision hides that
     failure and silently reuses the `new_smefit` code.

**How a user can check past runs** (for the bug reports):

- `smefitAPI.ext_chi2_func(**runcard)` raises the hidden `KeyError` for an
  affected runcard, because nothing there has a default to swallow it. This
  also works on a stored fit's `input/runcard.yaml`.
- In `fit_results.json`, the ultranest, blackjax and hessian fits record a
  `num_data` that includes the plugins' points. An affected fit has fewer
  points than expected.

### In-repo plugins

| Plugin | Contract | Models it makes sense for | Uses the RGE through |
|---|---|---|---|
| `drell_yan/*.py` (9 modules) | Current | SMEFT. It builds its own `EFTModel` on its own commondata and theory, with its own `use_quad` (default `True`). | `build_rge_matrix` |
| `optimal_observables/interface_oos_*.py` (4 modules, 18 classes) | Current | SMEFT. The inverse covariances are given in a fixed SMEFiT operator basis. | `SMEFTRunner.from_rge_dict` and `resolve_rge_matrices` |
| `low_energy/gaussian_constraints.py` | **Old**: reads `coefficients.name`, sets `n_dat`, and indexes `compute_chi2`'s argument as if it were the full coefficient vector | Any model | None |
| `low_energy/superallowed_beta_decay.py` | **Old**, as above, and imports `smefit.rge.rge`, which no longer exists | SMEFT | The SMEFiT → Warsaw translation of the runner, and `rgevolve` |

Both `low_energy` plugins fail on `main` as soon as they are loaded.
`superallowed_beta_decay.py` also needs `rgevolve`, which is not a smefit
dependency.

These bugs have been reported as #174 and #175. Once their fixes are merged into `main`, they should be brought into the master PR and PR 4 by merging `main` into them.

## Daughter PR 4: a model-aware contract

### The contract

```python
class ExternalChi2(ABC):
    """Optional base class for an external likelihood.

    Subclassing documents the contract. smefit checks the same rules on any
    class, subclass or not.
    """

    #: Names of the models (as in the ``model:`` block) this likelihood is valid
    #: for. None means any model, and also means unchecked.
    SUPPORTED_MODELS: ClassVar[tuple[str, ...] | None] = None

    #: Number of data points the likelihood stands for.
    num_data: int

    # The constructor receives:
    #   coefficients: CoefficientGroup   the runcard's coefficients
    #   rge_dict:     dict | None        the parsed rge: block (a deep copy)
    #   model_spec:   ModelSpec          only if the constructor names it
    #   **options                        the entry's other keys, verbatim

    @abstractmethod
    def compute_chi2(self, params: jnp.ndarray) -> jnp.ndarray:
        """chi2 at the free-parameter values params, in coefficients.free_names order."""
```

Rules every plugin follows, written in the base class docstring and checked by
tests:

1. **`compute_chi2` takes the free values.** Use `coefficients.resolve(params)`
   for the full vector, as the optimal-observables plugins do, or a model's
   `forward_map`, as the Drell-Yan ones do.
2. **`compute_chi2` is a JAX function.** The fits take its gradient, so the
   `smefit-dev` rules for differentiable code apply.
3. **`num_data` is an int.** `param_names` is not part of the contract.
   smefit names the parameters itself, from `coefficients.free_names`.
4. **Running goes through the model.** A plugin that needs RGE running
   should use the model's runner, `model_spec.model_class.rge_runner`, with
   `build_rge_matrix` or `resolve_rge_matrices`. Hardcoding `SMEFTRunner` is
   fine for a plugin whose `SUPPORTED_MODELS` is `("smeft",)`.
5. **Sibling modules need unique names.** A plugin that imports a module next
   to it by a bare name should give that module a name no other plugin
   folder uses (see the bugfix's remaining limitation).

### Loading, in `smefit/external_chi2.py`

`load_external_chi2(external_chi2, coefficients, rge_dict, model_spec)` keeps
the bugfix's import, cache and error handling, and for each entry adds:

1. **A check of `SUPPORTED_MODELS`** against `model_spec.name`, before
   construction. A mismatch raises a `ValueError` that names the plugin, the
   runcard's model and the supported models.
2. **`model_spec=model_spec`** in the constructor call, if the constructor
   names that parameter.

The `Chi2` wrapping does not change.

### `config.py`

- **`produce_ext_chi2_func`, `produce_individual_ext_chi2_func` and
  `produce_individual_mass_ext_chi2_func` take `model_spec`** and pass it on.
  `model_spec` always resolves (PR 2 defaults it to SMEFT), so it needs no
  default.
- **Unchanged:** the resource names, the `ext_chi2_func=None` defaults (a
  runcard may have no `external_chi2:`), and the mass scan's `init_scale`
  override.

### In-repo plugins

- **The Drell-Yan and optimal-observables plugins** subclass `ExternalChi2`
  and declare `SUPPORTED_MODELS = ("smeft",)`. Their options, defaults and
  results do not change.
- **`gaussian_constraints.py` is ported to the current contract:**
  `coefficients.names`, `num_data`, and `coefficients.resolve(params)` before
  indexing. It works for any model, so its `SUPPORTED_MODELS` stays None.
- **`superallowed_beta_decay.py` is ported in the same way.** It gets its
  SMEFiT → Warsaw translation from `SMEFTRunner(...).RGEbasis` and declares
  `("smeft",)`. Its tests skip when `rgevolve` is not installed.

### Commits

Every commit:

- leaves `pytest -m "not slow"` green;
- updates the docs it makes stale in the same commit;
- regenerates the skill references whenever it changes a `parse_` docstring or
  the runcard surface;
- passes `pre-commit run --all-files`.

1. **A model-aware contract.**
   - Add the `ExternalChi2` base class and `SUPPORTED_MODELS`.
   - `load_external_chi2` takes `model_spec`, checks `SUPPORTED_MODELS`, and
     passes `model_spec` to constructors that name it.
   - The three `produce_*ext_chi2_func` producers take and pass `model_spec`.
   - Docs:
     - the contract paragraph in `smefit-runcard/references/recipes.md`,
       including the sibling-module rule;
     - the external chi2 bullet of the `smefit-dev` skill, including dropping
       `param_names`;
     - the docstring of `parse_external_chi2`, which feeds the generated
       runcard reference, so regenerate it.
2. **Bring the in-repo plugins up to the contract.**
   - The Drell-Yan and optimal-observables plugins subclass `ExternalChi2`
     with `SUPPORTED_MODELS = ("smeft",)`.
   - Port both `low_energy` plugins.
   - In the test that imports every `external_chi2/` module (from PR 1),
     `superallowed_beta_decay.py` is no longer `xfail`, and is skipped
     without `rgevolve`.

### Tests

| File | Change |
|---|---|
| `tests/test_external_chi2.py` | `model_spec` is passed only to a constructor that names it, and not to a `**kwargs` catch-all. A `SUPPORTED_MODELS` mismatch raises and names both sides. A plugin without the attribute loads under any model. `param_names` is not required. |
| `tests/test_config.py` | The three producers pass `model_spec`. |
| A graph test, on `tests/fixtures/` | Register a toy model (as in PR 2) and a plugin declaring `SUPPORTED_MODELS = ("smeft",)`. A runcard selecting the toy model fails at load time, with the plugin's name in the error. |
| New, `low_energy` plugins | `gaussian_constraints.py`: chi2 at a known point, with fixed coefficients ordered before the constrained one (the case the old indexing got wrong). `superallowed_beta_decay.py`: loads and gives a finite chi2, skipped without `rgevolve`. |

### Verification

1. After every commit, run `pytest -m "not slow"`. After commit 2, also run the
   full `pytest`.
2. **Regression against the integration branch.** Run a runcard with an
   `rge:` block, one Drell-Yan plugin and one optimal-observables plugin, on
   `generalize-beyond-eft` and on this branch. The `fit_results.json` must be
   identical.
3. **Stored fits.** A report over a stored fit that used external chi2
   (`fits:` with `fitcontext`) rebuilds as before.
4. **Two copies of a plugin.** Make two copies of an in-repo plugin in
   scratch folders, with the same file and class names and a different
   `num_data`. A report over two fits, each pointing to one copy, loads each
   fit with its own copy. The `smefit_database` copies cannot be used for this
   check, since they fail to import under current smefit.
5. Regenerating the skill references matches what is committed, and CI's
   skills check passes.

## Not in this PR

- **Making in-repo plugins follow the runcard's model options**, for example
  the Drell-Yan `use_quad` (see the open questions).
- **Turning the Drell-Yan plugins into ordinary datasets.** They load
  commondata and theory with smefit's own loader. Once PR 3's theory container
  and a POPxf loader exist, they may not need to be plugins at all.
- **Documenting each plugin's options** in the generated reference and the
  runcard validator. An entry's extra keys are still free-form.
- **The copies in `smefit_database/external_chi2/`.** They are written
  against an old smefit API (`smefit.rge.rge`, `smefit.coefficients`), and
  that repository updates them separately.

## Open questions

- **Drell-Yan and the runcard's model.** Should the Drell-Yan plugins take
  `use_quad` from the runcard's `model:` block when their entry does not set
  it? That is the less surprising behaviour, but it changes results for
  runcards that rely on today's default of `True`.
- **The legacy `low_energy` plugins.** Port them, as planned, or remove them?
  `superallowed_beta_decay.py` needs `rgevolve`, which smefit does not depend
  on.
- **Requiring the base class.** Should a later release require plugins to
  subclass `ExternalChi2`, so that the contract can be checked with
  `isinstance`?
