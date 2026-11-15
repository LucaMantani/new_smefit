# Daughter PR 1: generic RGE runner

Part of the generalization beyond the EFT (issue #116). See
[GENERALIZATION_00_ANALYSIS.md](GENERALIZATION_00_ANALYSIS.md) for the overall
analysis and how the PRs are organised.

- Branch: `generalize-beyond-eft-rge`, created from `generalize-beyond-eft`.
- Target: `generalize-beyond-eft`.

## Goal

Split `smefit/rge/` into two parts:

- **A generic layer** that computes RGE matrices per scale, caches them, and
  writes and reads `rge_matrix.pkl`. Any model with linear running can use it.
- **The SMEFT runner**, which wraps `wilson`. Together with the WCxf map, the
  SM inputs and the `wilson` patches, it moves into a new `smefit/models/smeft/`
  package.

What changes with the model is only the runner: which parameters are evolved,
into which names, and how. Nothing else changes in behaviour. Every existing
`rge_matrix.pkl` stays valid, and every fit gives the same result.

## Current state

| File | Role | Depends on the model? |
|---|---|---|
| `rge/__init__.py` | Re-exports, and imports `_patches` first | Importing `smefit.rge` imports and patches `wilson` |
| `rge/runner.py` | `RGE`: `wilson` calls, WCxf translation, SM parameter presets | Yes, entirely |
| `rge/_patches.py` | Monkey patches of `wilson` and `ckmutil` | Yes |
| `rge/build.py` | `build_rge_matrix`, `resolve_rge_matrices`, `_resolve_scales` | Only through `RGE.from_rge_dict`, hardcoded in `build_rge_matrix` |
| `rge/matrix.py` | `RGEMatrix`, `RGECache`, the pickle layout, `locate` | No, and it never imports `wilson` |
| `wcxf.py` | SMEFiT ↔ Warsaw map | Yes |
| `constants.py` | SM inputs at MZ | Yes |

The generic interface already exists, only implicitly. `resolve_rge_matrices`
calls nothing but `runner.clone_runner(names)` and `runner.RGEmatrix(scale)`.
`_StubRunner` in `tests/test_rge.py` implements only those two methods, and
the cache tests pass with it.

These callers use the public API:

- `config.py`: `parse_rge`, `produce_rge_matrix`,
  `produce_individual_mass_rge_matrix`.
- `fit_result.py`: `RGECache`, `RGEMatrix.FILENAME`.
- `model.py`: `EFTModel` reads the `RGEMatrix` fields.
- In-repo `external_chi2/`:
  - the 9 Drell-Yan modules use `build_rge_matrix` and `RGEMatrix.write`;
  - the 4 optimal-observables modules use `RGE.from_rge_dict`,
    `RGEMatrix.read_cache` and `resolve_rge_matrices`.
- Tests.

## Target design

### Layout

```
smefit/rge/                generic, never imports wilson
  __init__.py              RGERunner, RGEMatrix, RGECache, build_rge_matrix, resolve_rge_matrices
                           + deprecated lazy aliases: RGE, ALLOWED_YUKAWA, ALLOWED_SMEFT_ACCURACY
  runner.py                RGERunner, the abstract base (new content)
  build.py                 build_rge_matrix(..., runner_cls)
  matrix.py                logic unchanged
smefit/models/             one package per model
  __init__.py              docstring only (PR 2 adds the lookup from runcard name to model)
  smeft/                   SMEFT-specific
    __init__.py            docstring only, no re-exports
    rge.py                 SMEFTRunner(RGERunner): today's RGE, the wilson presets,
                           evolve_gs, _wilson_params
    _patches.py            moved from smefit/rge/
    wcxf.py                moved from smefit/
    constants.py           moved from smefit/
```

`smefit/models/smeft/__init__.py` re-exports nothing, so that importing
`smefit.models.smeft.wcxf` does not load `wilson`.

`constants.py` holds SM inputs that another model may also need. It stays in
`models/smeft/` until one does, and then moves to a shared package (see the
analysis).

`smefit/models/smeft/rge.py` is the only module that imports `wilson`, and it imports
`_patches` before doing anything else. `config.py` imports `SMEFTRunner`, so
the patches are in effect in every smefit run, exactly as today.

`smefit/models/` holds helpers, not DAG nodes, so it is not added to
`smefit_providers` in `app.py`. The same holds for `blackjax_samplers/`.

### The runner interface

```python
class RGERunner(ABC):
    """Linear running of named parameters from init_scale to an observable scale."""

    #: Keys of the runcard ``rge:`` block this runner reads besides the generic
    #: ones (init_scale, obs_scale, scale_variation, rg_matrix), each mapped to
    #: (default, allowed values); None allows any value.
    OPTIONS: ClassVar[Mapping[str, tuple[Any, frozenset | None]]] = {}

    def __init__(self, wc_names: Sequence[str], init_scale: float) -> None:
        self.wc_names = sorted(wc_names)
        self.init_scale = init_scale

    @classmethod
    @abstractmethod
    def from_rge_dict(
        cls, rge_dict: Mapping[str, Any], wc_names: Sequence[str]
    ) -> "RGERunner":
        """Build a runner from the runcard ``rge:`` block."""

    @property
    @abstractmethod
    def settings(self) -> dict[str, Any]:
        """Everything that determines the matrices: the cache key of rge_matrix.pkl."""

    @property
    @abstractmethod
    def all_ops(self) -> tuple[str, ...]:
        """Every name a matrix can have a row for."""

    @abstractmethod
    def RGEmatrix_dict(self, scale: float) -> dict[str, dict[str, float]]:
        """For each name in wc_names, the names it runs into at scale, with their coefficients."""

    def RGEmatrix(self, scale: float) -> pd.DataFrame:
        """Moved unchanged from RGE: the matrix as a DataFrame, all-zero rows dropped."""

    def clone_runner(self, wc_names: Sequence[str]) -> "RGERunner":
        return type(self).from_rge_dict(self.settings, wc_names)
```

Every runner must follow this contract, written in the base-class docstring
and checked by tests:

1. **Round trip.** `from_rge_dict(runner.settings, names)` builds a runner with
   the same settings. This is what lets `clone_runner` live in the base class.
   The SMEFT runner already satisfies it: its four `settings` keys are exactly
   the ones its `from_rge_dict` reads.
2. **Settings identify the runner.** `RGEMatrix.read_cache` only compares
   settings dicts, so two runners must never produce equal settings.
   - A new runner includes a `"runner": "<name>"` entry.
   - The SMEFT runner keeps its four keys (`init_scale`, `smeft_accuracy`,
     `adm_QCD`, `yukawa`). Changing them would invalidate every
     `rge_matrix.pkl` already written, as the `settings` docstring warns.
3. **Names a runner cannot evolve map to `{}`.** The SMEFT runner does this
   today for external couplings.
4. **`all_ops` matches the names in the theory tables.** `EFTModel` contracts
   the rows of the matrix with them.
5. **No file I/O in runners.** Caching stays in `build.py` and `matrix.py`.

These stay on `SMEFTRunner` only: `RGEevolve`, `RGEbasis`, `map_to_smefit`,
`evolve_gs`, `_wilson_params`, the parameter presets, `ALL_OPS`,
`ALLOWED_YUKAWA`, `ALLOWED_SMEFT_ACCURACY` and `_SMALL_VALUE_THRESHOLD`.

`SMEFTRunner.OPTIONS` is:

```python
OPTIONS = {
    "smeft_accuracy": ("integrate", ALLOWED_SMEFT_ACCURACY),
    "yukawa": ("top", ALLOWED_YUKAWA),
    "adm_QCD": (False, None),
}
```

`from_rge_dict` reads its defaults from `OPTIONS`. In this PR, `parse_rge`
keeps its own copy of the defaults (see `config.py` below). PR 2 removes that
duplication, once the reference generator can document `OPTIONS`.

### `build.py`

- `build_rge_matrix(rge_dict, coeff_list, theory_group, runner_cls=None)` builds
  its runner with `runner_cls.from_rge_dict(rge_dict, coeff_list)`. `None`
  falls back to `SMEFTRunner` through a lazy import and raises a
  `DeprecationWarning`, for external modules outside the repo.
- `resolve_rge_matrices` keeps its logic and annotates the runner as
  `RGERunner`.
- `_resolve_scales` is unchanged. It reads `theory_group.scales`, which stays
  part of `Theory`.
- In docstrings, "Wilson coefficients" becomes "parameters".

### `matrix.py`

The logic does not change, and neither do the field names (`obs_operators`,
`init_operators`): `EFTModel`, the external modules and the tests all use them.
The docstrings point to `RGERunner.settings` instead of
`smefit.rge.runner.RGE.settings`.

The pickle layout is unchanged, so files stay readable by both older and newer
versions.

### `config.py`

- **`parse_rge` keeps its validation as it is.** Only the import of
  `ALLOWED_SMEFT_ACCURACY` and `ALLOWED_YUKAWA` changes, to
  `smefit.models.smeft.rge`. Its shape is load-bearing:
  `scripts/generate_skill_reference.py` builds the `rge` section of
  `runcard-keys.md` from three things in it:
  - the `known_keys` literal;
  - the `rge.get("<key>", default)` calls with literal keys;
  - the literal messages of `raise ConfigError(...)`.

  A loop over `SMEFTRunner.OPTIONS` would silently drop from the reference the
  defaults of `smeft_accuracy` and `yukawa` and their "Invalid …" errors. CI
  would not notice, since it only checks that the committed reference matches
  a fresh run of the generator. Validating through `OPTIONS` therefore moves to
  PR 2, together with a generator change that documents each runner's
  `OPTIONS`.
- **`_build_rge_matrix`** (behind `produce_rge_matrix`) and
  **`produce_individual_mass_rge_matrix`** pass `runner_cls=SMEFTRunner`.
- These are the last EFT-specific choices on the RGE path. PR 2 replaces them
  with the model's runner.

### Deprecation shims

| Old usage | Behaviour after this PR |
|---|---|
| `from smefit.rge import RGE` (also `ALLOWED_YUKAWA`, `ALLOWED_SMEFT_ACCURACY`) | A module-level `__getattr__` (PEP 562) imports `smefit.models.smeft.rge` lazily, returns the object and raises a `DeprecationWarning` |
| `build_rge_matrix(...)` without `runner_cls` | Uses `SMEFTRunner` and raises a `DeprecationWarning` |

There are no shims for `smefit.rge.runner` internals, `smefit.rge._patches`,
`smefit.wcxf` or `smefit.constants`, since only tests import them.

## Commits

Every commit:

- leaves `pytest -m "not slow"` green;
- updates the docs that it makes stale, in the same commit, as the
  `smefit-dev` skill requires for hand-written skill files;
- leaves `python scripts/generate_skill_reference.py` producing no diff. This
  PR does not change the runcard surface, so any diff in the generated
  reference is a regression;
- passes `pre-commit run --all-files`.

### 1. Move the SMEFT code into `smefit/models/smeft/`

This commit only moves files and fixes imports. Behaviour does not change.

- `git mv` these files:
  - `smefit/wcxf.py` → `smefit/models/smeft/wcxf.py`
  - `smefit/constants.py` → `smefit/models/smeft/constants.py`
  - `smefit/rge/_patches.py` → `smefit/models/smeft/_patches.py`
  - `smefit/rge/runner.py` → `smefit/models/smeft/rge.py`
- `smefit/models/smeft/rge.py` imports `_patches` at the top.
- For now, `smefit/rge/__init__.py` and `build.py` import `RGE` from
  `smefit.models.smeft.rge`. Commit 2 removes that dependency.
- Tests:
  - Rename the tests of the moved modules, following the convention for
    modules in a subpackage (`test_blackjax_nuts.py` tests
    `blackjax_samplers/nuts.py`): `test_rge_patches.py` →
    `test_smeft_patches.py`, `test_wcxf.py` → `test_smeft_wcxf.py`,
    `test_constants.py` → `test_smeft_constants.py`.
  - Update the import paths and the logger names checked by `caplog`
    (`smefit.rge.runner` → `smefit.models.smeft.rge`, `smefit.wcxf` →
    `smefit.models.smeft.wcxf`) in `test_rge.py` and in the renamed files.
- Docs:
  - In the module list of `AGENTS.md`, replace the `rge/`, `wcxf.py` and
    `constants.py` entries with `rge/` (generic running) and `models/smeft/`
    (SMEFT code).
  - Search the docs and skills for `smefit.rge.runner`, `smefit/rge/runner.py`,
    `smefit.wcxf`, `smefit/wcxf.py` and `smefit.constants`, and update every
    hit.

### 2. Introduce `RGERunner` and make the generic layer independent of `wilson`

- Add `RGERunner` in `smefit/rge/runner.py`.
- Rename `RGE` to `SMEFTRunner(RGERunner)` and add `OPTIONS`. Move `RGEmatrix`
  and `clone_runner` up into the base class.
- `build_rge_matrix` takes `runner_cls`, with the deprecated default.
- `smefit/rge/__init__.py` exports `RGERunner` and the lazy aliases, and no
  longer imports anything from `smefit.models.smeft`.
- Tests: split `test_rge.py` (see the Tests section), and add the new generic
  tests.
- Docs: add a short "Adding an RGE runner" section to the `smefit-dev` skill.
  It covers the runner contract above and the tests a new runner needs.

### 3. Pass the runner to `build_rge_matrix` in `config.py`

- `produce_rge_matrix` and `produce_individual_mass_rge_matrix` pass
  `runner_cls=SMEFTRunner`.
- `parse_rge` is unchanged apart from where it imports `ALLOWED_*` from (see
  `config.py` above).
- Tests:
  - Two `test_config.py` tests assert the arguments `build_rge_matrix` is
    called with. Add `runner_cls=SMEFTRunner` to
    `test_produce_rge_matrix_builds_writes_and_returns` and
    `test_produce_individual_mass_rge_matrix_overrides_init_scale`.
  - Add a test that `parse_rge` accepts every key of `SMEFTRunner.OPTIONS`
    without the "Unknown key" warning, and rejects a value outside each allowed
    set. This keeps the hand-written checks in step with the runner until PR 2
    removes them.

### 4. Update the callers

- `external_chi2/drell_yan/*.py` (9 modules): pass
  `runner_cls=SMEFTRunner` to `build_rge_matrix`.
- `external_chi2/optimal_observables/interface_oos_*.py` (4 modules): use
  `SMEFTRunner.from_rge_dict` from `smefit.models.smeft.rge`.
- `fit_result.py`: the lazy imports of `smefit.rge` were there to avoid
  patching `wilson` when a fit is loaded for a plot. That reason is gone, so
  hoist them to a normal import and drop the comment.
- Tests: add a test that imports every module under `external_chi2/`. Nothing
  imports them in the test suite today, so a wrong import path would go
  unnoticed.

## Tests

| File | Change |
|---|---|
| `tests/test_rge.py` | Keeps the generic tests: `_resolve_scales`, `_find_cached_scale`, `resolve_rge_matrices`, the `RGEMatrix` payload, `locate`, `RGECache`. The two precomputed-cache `build_rge_matrix` tests switch from the SMEFT settings to a toy runner. |
| `tests/test_smeft_rge.py` (new) | Takes the SMEFT tests from `test_rge.py`: Yukawa validation, `evolve_gs`, `_wilson_params`, `RGEbasis`, `map_to_smefit`, `RGEmatrix_dict`, `clone_runner`, `RGEevolve`, and the slow `wilson` tests, with `build_rge_matrix_shape` passing `runner_cls=SMEFTRunner`. |
| `tests/test_rge_patches.py` → `tests/test_smeft_patches.py` | Imports `_patches` from `smefit.models.smeft`, and its docstring says the patches apply on import of `smefit.models.smeft.rge`. |
| `tests/test_wcxf.py` → `tests/test_smeft_wcxf.py`, `tests/test_constants.py` → `tests/test_smeft_constants.py` | New import paths and logger names |
| `tests/test_config.py` | `runner_cls` in the `build_rge_matrix` call assertions, and the test keeping `parse_rge` in step with `SMEFTRunner.OPTIONS` |
| New, generic | **Toy runner, end to end:** a scale-dependent diagonal matrix driving `build_rge_matrix` with a fresh computation and a cached one, without `wilson`. **Contract:** round trip of `from_rge_dict(settings)` and `clone_runner`, for both the toy runner and `SMEFTRunner`. **Isolation:** `import smefit.rge` in a subprocess leaves `wilson` out of `sys.modules`. **Settings guard:** a cache written by one runner is refused by the other. |
| New, shims | `from smefit.rge import RGE` warns and returns `SMEFTRunner`; `build_rge_matrix` without `runner_cls` warns |
| New, external modules | Import every `external_chi2/` module. `low_energy/superallowed_beta_decay.py` is marked `xfail`, since it is already broken on `main` (see below). |

## Verification

1. After every commit, run `pytest -m "not slow"`. After commits 2 and 4, also
   run the full `pytest`, whose slow tests run `wilson`.
2. **Regression against `main`.** Run `template_runcards/rge.yaml` (the
   `rge_matrix` action, with a fixed scale and an external coupling `y`) on
   `main` and on the branch. The frames in the two `rge_matrix.pkl` files must
   be identical. Then do the same with `template_runcards/analytical_fit.yaml`
   and compare `fit_results.json`.
3. **External chi2 smoke run.** Run a scratch runcard with an `rge:` block, one
   Drell-Yan external chi2 and one optimal-observables external chi2.
4. Regenerating the skill references produces no diff, and CI's skills check
   passes.

## Not in this PR

- **`EFTModel._apply_rge`** stays EFT-specific, because it contracts the matrix
  with the linear and quadratic tables. `EFTModel` moves to `smefit/models/smeft/` in
  PR 2, and `BaseModel` to `smefit/models/base.py`.
- **The hardcoded `SMEFTRunner` in `config.py`** is replaced in PR 2, where
  the model declares its runner.
- **Validating the runner options through `OPTIONS`** moves to PR 2, together
  with the generator change that documents them.
- **Neutral names** for `wc_names`, `obs_operators`, `init_operators` and
  `RGEmatrix` go in the naming pass, together with `eft_model` → `model`.
- **The `rge_dict` argument of the external chi2 contract** is unchanged here
  and handled in PR 4.
- **Runcard placement of runner-specific keys.** This only needs deciding once
  a second runner exists.
- **`external_chi2/low_energy/superallowed_beta_decay.py`** imports
  `smefit.rge.rge`, which does not exist on `main`. It is already broken and
  should be fixed separately.

## Open questions

- **Shims.** Keep the deprecation shims for out-of-repo external modules until
  a later release, or make a clean break?
- **Names.** `SMEFTRunner` in `smefit/models/smeft/rge.py`, or something else?
