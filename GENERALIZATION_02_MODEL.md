# Daughter PR 2: model interface and selection

Part of the generalization beyond the EFT (issue #116). See
[GENERALIZATION_00_ANALYSIS.md](GENERALIZATION_00_ANALYSIS.md) for the overall
analysis and how the PRs are organised.

- Branch: `generalize-beyond-eft-model`, created from `generalize-beyond-eft`
  once PR 1 is merged into it.
- Target: `generalize-beyond-eft`.
- Depends on PR 1, which adds `RGERunner` and `SMEFTRunner`.

## Goal

After this PR, adding a model means writing a package under `smefit/models/`
and registering it in one place. Nothing in `config.py`, the fitters or the
reports has to change.

- **The model contract is complete.** `BaseModel` declares everything its
  callers use, and no caller reads anything specific to `EFTModel`.
- **`EFTModel` becomes the SMEFT model.** It moves to `smefit/models/smeft/`
  and declares its own RGE runner and options.
- **A runcard chooses its model** with a `model:` block holding every model
  option.
- **`config.py` has no SMEFT-specific code left.** It no longer imports
  `EFTModel` or `SMEFTRunner`, and has no hand-written SMEFT `rge:` checks or
  TeV→GeV conversion.

Results do not change. Every existing runcard and every stored fit keeps
working.

## Decisions this plan builds on

These were taken while planning and are recorded here so that the review can
check them.

1. **`EFTModel` stays the SMEFT model.** It has the limitations of the SMEFT
   model, among them predictions that are at most quadratic in the
   coefficients. A general model, evaluating any function of the parameters
   that the theory defines, would be a separate and more general model, not a
   rename of `EFTModel`.
2. **The model declares its RGE runner.** A runcard does not pick a runner. It
   picks a model, and the model brings its runner, or none.
3. **The `model:` block holds every model option**, for example
   `model: {name: smeft, use_quad: true}`. Its `name` defaults to `smeft`. A
   top-level `use_quad` is still accepted as a deprecated shorthand for the
   SMEFT model, because every stored fit's runcard uses it. The deprecation
   warning is logged once per fit, so a report that rebuilds several old fits
   warns once for each of them.
4. **The t0 covmat uses the theory's reference prediction**, the theory's
   predictions at a reference point the theory defines. That point is zero
   for every parameter unless the theory file names another, so t0 depends
   neither on the model nor on the runcard.
   - In this PR the reference prediction is still `theory.sm_pred`.
   - PR 3 renames it `reference_prediction` and adds the optional field that
     names a reference point. It also covers observables built from several
     polynomials, such as σ_SM × BR_SM for a σ×BR model.
   - Switching to the model's prediction at its baseline would change
     existing fits whenever a fixed coefficient is non-zero.
5. **Mass scans stay in `config.py`.** The one SMEFT convention in them, scan
   points in TeV against an `init_scale` in GeV, moves into the model. SMEFT
   needs no reportengine nodes of its own.

Two reportengine constraints shape the design:

- **The model cannot be an `@explicit_node`.** `produce_chi2` and
  `produce_datasets_chi2` build their results from `eft_model` while the config
  is still being read. An explicit node only has a value once the graph runs,
  so these producers would receive the wrapper, not the model. Model-specific
  options therefore have to be read while the config is read, from the `model:`
  block.
- **A `parse_` function can depend on other resources.** reportengine resolves
  the parameters after the first one like those of a `produce_`
  (`resolve_signature_params` with `start_from=1` in
  `reportengine/configparser.py`). This lets `parse_rge` validate the runner
  options against the model.

## Current state

What callers use from the model:

| Caller | Uses |
|---|---|
| `chi2.py` (`build_chi2`, `build_datasets_chi2`) | `forward_map`; `coefficients.free_names` and `coefficients.baseline_free` |
| `config.py` (`_build_chi2_impl`) | `coefficients.free_names` |
| `analytic_fit.py` | `forward_map` and its Jacobian; `coefficients`; `use_quad` |
| `projections.py` | `forward_map` |
| `external_chi2/drell_yan/*.py` (9 modules) | Build `EFTModel(theory, coefficients, use_quad, rge_matrix=...)` and call `forward_map` |

`BaseModel` in `smefit/model.py` only requires `forward_map`.

Where `config.py` is EFT-specific (after PR 1):

- `produce_eft_model`, `produce_individual_eft_model` and
  `produce_individual_mass_eft_model` build `EFTModel` and take `use_quad`.
- `produce_rge_matrix` and `produce_individual_mass_rge_matrix` pass
  `runner_cls=SMEFTRunner`.
- `parse_rge` checks `smeft_accuracy` and `yukawa` by hand. PR 1 kept this,
  because the reference generator reads those literals.
- `produce_individual_mass_rge_matrix` and
  `produce_individual_mass_ext_chi2_func` convert the scan point with
  `1e3 * mass`.
- `_FIT_SETUP_KEYS` lists `use_quad` and `rge`, but no model key.

Other places that read `use_quad`:

- `Fit.use_quad` in `fit_result.py`, read by tests only;
- `validate_runcard.py` in the `smefit-runcard` skill;
- the 10 template runcards that set it at the top level, and their copies in
  the skill. Most of them also set `use_quad` inside an `external_chi2` entry.
  That one is the Drell-Yan module's own constructor option, and it does not
  change;
- the skills, `AGENTS.md` and the `smefit-fit-doctor` agent.

## Target design

### Layout

```
smefit/models/
  __init__.py      the model lookup: name -> class (lazy), ModelSpec, legacy_spec
  base.py          BaseModel, the full contract (moved from smefit/model.py)
  smeft/
    model.py       EFTModel(BaseModel), moved from smefit/model.py
    rge.py, wcxf.py, constants.py, _patches.py   from PR 1
smefit/model.py    deprecated re-export of BaseModel and EFTModel
```

### The model contract

```python
class BaseModel(ABC):
    """Maps the free parameters to the predictions for the data of a TheoryGroup."""

    #: Options of the runcard ``model:`` block: name -> (default, allowed values);
    #: None allows any value. The constructor accepts every one as a keyword.
    OPTIONS: ClassVar[Mapping[str, tuple[Any, frozenset | None]]] = {}

    #: The runner an ``rge:`` block uses with this model; None if it has no running.
    rge_runner: ClassVar[type[RGERunner] | None] = None

    #: The parameters; forward_map takes values for coefficients.free_names.
    coefficients: CoefficientGroup

    @abstractmethod
    def forward_map(self, params: jnp.ndarray) -> jnp.ndarray:
        """Predictions, shape (n_data,), for the free-parameter values params."""

    @property
    def is_linear(self) -> bool:
        """True when forward_map is linear in the parameters it is given."""
        return False

    @classmethod
    def mass_to_init_scale(cls, mass: float) -> float:
        """The rge init_scale, in GeV, for a mass-scan point."""
        raise NotImplementedError(f"{cls.__name__} does not support mass scans")
```

Every model follows this contract, which is written in the `BaseModel`
docstring and checked by tests:

1. **Constructor.** `config.py` builds every model as
   `cls(theory, coefficients, rge_matrix=..., **options)`. Here `options` is
   the validated `model:` block with defaults filled in. When `rge_runner` is
   None, `rge_matrix` is always None, because `parse_rge` refuses an `rge:`
   block for such a model.
2. **`coefficients`.** The `CoefficientGroup` the model was built with.
   `forward_map` takes values for its `free_names`, in that order.
3. **JIT.** This requirement already exists: if `forward_map` is
   `jax.jit`-compiled with a static `self`, then `self` is hashable and its
   arrays are fixed at construction.
4. **`is_linear` is conservative.** Return True only when that is guaranteed,
   because `analytic_fit` relies on it.
5. **A model checks the theory it is given.** It raises a clear error for a
   theory it cannot use, rather than reading only part of it. Today's theory
   always fits `EFTModel`. PR 3 lets the theory hold terms of any degree, and
   its interface allows several polynomials per data point. From then on,
   `EFTModel` rejects degrees above 2 and more than one polynomial per data
   point. The check lands in PR 3, together with those cases.
6. **Errors raise `ValueError`, never `KeyError`.** This covers the check in
   item 5 and anything else that fails while a model is built.
   - **Why.** The model is built while the config is read, and reportengine
     treats a `KeyError` raised there as "input not available".
   - **What goes wrong otherwise.** A consumer with a default then swallows
     the error. `produce_pseudodata` has `eft_model=None`, so it would build
     SM-only pseudodata without any message.
   - **Precedent.** `load_theory` converts its `KeyError` for the same
     reason.

### `EFTModel`, the SMEFT model

- It moves to `smefit/models/smeft/model.py`. The maths is unchanged, and so is
  the constructor signature, `(theory, coefficients, use_quad=False,
  rge_matrix=None)`, which external modules and tests call positionally.
- It declares the following:

  ```python
  OPTIONS = {"use_quad": (False, frozenset({True, False}))}
  rge_runner = SMEFTRunner
  ```

- `is_linear` returns `not self.use_quad`. This keeps today's behaviour, in
  which non-linear `expr:` constraints are linearised at the expansion point.
  `validate_runcard.py` already warns about those.
- `mass_to_init_scale(mass)` returns `1e3 * mass`. This is where the
  TeV-against-GeV convention now lives: scan points are masses in TeV, matching
  the TeV^-2 coefficients, while `init_scale` is in GeV.
- `__repr__` shows the options, for example `EFTModel(use_quad=True)`, so that
  generic error messages can name them.

### The model lookup, `smefit/models/__init__.py`

```python
_MODELS = {"smeft": "smefit.models.smeft.model:EFTModel"}
DEFAULT_MODEL = "smeft"


def get_model_class(name: str) -> type[BaseModel]: ...  # imports lazily
def available_models() -> list[str]: ...


@dataclass(frozen=True)
class ModelSpec:
    name: str
    options: Mapping[str, Any]  # validated, defaults filled in

    @property
    def model_class(self) -> type[BaseModel]: ...

    def build(self, theory, coefficients, rge_matrix) -> BaseModel:
        return self.model_class(
            theory, coefficients, rge_matrix=rge_matrix, **self.options
        )


def legacy_spec(use_quad) -> tuple[str, dict]:
    """(DEFAULT_MODEL, {"use_quad": use_quad}) for a runcard without model:."""
```

- **The lookup is lazy.** `_MODELS` maps names to import paths, so looking up
  one model never imports another model's dependencies. `blackjax_samplers/`
  imports every algorithm up front, so it is not the pattern to copy here.
- **`legacy_spec` holds the deprecated shorthand in one place.** `config.py`
  and `Fit` both use it, and deleting it later is a one-place change.
- **The package is not registered in `smefit_providers`.** It holds helpers,
  not DAG nodes.

### The runcard

```yaml
model:
  name: smeft         # optional, default smeft: a name from the model lookup
  use_quad: true      # the model's options, from its OPTIONS
```

- **The block is optional, and so is its `name`.** Both default to SMEFT, so
  `model: {use_quad: true}` is the SMEFT model with quadratic terms.
- **Without the block**, `use_quad` is taken from the top-level key, or
  defaults to False. When the top-level key is used, a deprecation warning is
  logged, once for each fit whose model is built from it.
- **The block and a top-level `use_quad` together** raise a `ConfigError`,
  because they are two answers to the same question.
- **Unknown options in the block log a warning,** like every other block.
  A value outside the allowed set raises a `ConfigError`.
- **The templates switch to the block.** The 10 template runcards with a
  top-level `use_quad` move it into `model:`, and their skill copies are
  regenerated from them. The `use_quad` inside `external_chi2` entries stays
  where it is.

### `config.py`

| Node | After this PR |
|---|---|
| `parse_model(self, model: Mapping)` | **New.** Reads `name`, defaulting to `DEFAULT_MODEL` (`smeft`), and checks it against the lookup. Checks the options against the model's `OPTIONS`, then returns a `ModelSpec`. `known_keys = {"name"}`, since the options are documented per model (see the generator section below). |
| `produce_model_spec(self, model=None, use_quad=None)` | **New.** Returns the parsed block, or `legacy_spec(use_quad)` validated like a block. Raises `ConfigError` when both are given. Both defaults are runcard keys that may be absent, which is what the `=None` rule allows. A `produce_model` would never run when the key is present, hence the different name. |
| `produce_eft_model(self, model_spec, theory, coefficients, rge_matrix)` | Calls `model_spec.build(...)`. `rge_matrix` still has no default, and its signature test stays. The resource keeps its name until PR 5. |
| `produce_individual_eft_model`, `produce_individual_mass_eft_model` | Same, with their own coefficients and RGE matrix. `use_quad` leaves their signatures. |
| `parse_rge(self, rge, model_spec)` | Checks the generic keys, as today. If the model has no runner, it raises `ConfigError`. It then checks the runner's options against `runner.OPTIONS`, replacing the hand-written `smeft_accuracy` and `yukawa` checks. `known_keys` keeps only the generic keys. |
| `produce_rge_matrix`, `produce_individual_mass_rge_matrix` | Take `model_spec` and pass `runner_cls=model_spec.model_class.rge_runner`. The cache key of `rge_matrix` gains the model name. |
| `produce_individual_mass_rge_matrix`, `produce_individual_mass_ext_chi2_func` | Use `model_spec.model_class.mass_to_init_scale(mass)` instead of `1e3 * mass`. A `NotImplementedError` becomes a `ConfigError` that names the model. |
| `_FIT_SETUP_KEYS` | Gains `model`. `use_quad` stays, because stored fits carry it. |

`parse_model`, `produce_model_spec` and `parse_rge` share one private helper,
`_check_options(block, options_spec, given, generic=())`. It logs a warning for
unknown keys, raises `ConfigError` for values that are not allowed, and returns
the options with defaults filled in. `ConfigError` therefore stays in
`config.py`, and the models and runners do not depend on reportengine.

`produce_fitcontext` needs no change. Under `fits::fitcontext`, an old fit
rebuilds with its top-level `use_quad`, through the legacy path, and a new fit
rebuilds with its `model:` block. A report runcard that sets `model:` while
rebuilding a fit that has no such block is refused, by the existing check for
setup keys.

### `analytic_fit.py`

- **The check uses the contract.** `if eft_model.use_quad` becomes
  `if not eft_model.is_linear`.
- **The error message becomes generic.** It says that `analytic_fit` needs a
  model linear in its parameters, and names the model through its `__repr__`
  (`EFTModel(use_quad=True)`).

### `fit_result.py`

- **New properties.** `Fit.model_name` and `Fit.model_options` are read from
  the stored runcard, from its `model:` block or through `legacy_spec`. Loading
  a fit never imports a model class.
- **`Fit.use_quad` keeps working.** It returns
  `self.model_options.get("use_quad", False)`, for both old and new runcards.

### Reference generator and runcard validator

The generator only documents literals in `config.py`, so options declared on
model and runner classes would disappear from the reference. A new collector
in `scripts/generate_skill_reference.py`, `collect_models()`, reads the model
lookup the way `collect_priors` reads `_DIST_REGISTRY`. For each model it
records:

- its name, its class and the first paragraph of its docstring;
- its `OPTIONS`, with defaults and allowed values;
- its RGE runner, or none, and the runner's `OPTIONS`.

It also records `DEFAULT_MODEL`, so that the reference says `name` defaults to
`smeft`. From `parse_model`'s `.get("name", DEFAULT_MODEL)`, the generator
would only see the constant's name.

It writes them to `runcard-keys.json` (under `models`) and to a "Models"
section of `runcard-keys.md`. The allowed values listed there replace the
"Invalid 'smeft_accuracy' …" and "Invalid 'yukawa' …" entries the reference
loses.

`validate_runcard.py` reads `models` from the JSON. It checks:

- the `model:` block: that the name, or the default `smeft`, is known, and
  that the option names and values are allowed;
- the `rge:` sub-keys, against the generic `known_keys` plus the chosen
  model's runner options;
- the `run_analytic_fit` linearity check, which reads `use_quad` from the
  block or from the top-level key.

### Deprecations

| Old usage | After this PR |
|---|---|
| Top-level `use_quad` | Accepted for the SMEFT model, with a logged deprecation warning |
| `from smefit.model import EFTModel` (or `BaseModel`) | A module-level `__getattr__` imports the class lazily and raises a `DeprecationWarning` |

## Commits

Every commit:

- leaves `pytest -m "not slow"` green;
- updates the docs it makes stale in the same commit;
- regenerates the skill references whenever it changes the runcard surface;
- passes `pre-commit run --all-files`.

### 1. Complete the contract and move `EFTModel`

There is no runcard change in this commit.

- Add `smefit/models/base.py` with the full `BaseModel`.
- Move `EFTModel` to `smefit/models/smeft/model.py`, with `OPTIONS`,
  `rge_runner`, `is_linear`, `mass_to_init_scale` and `__repr__`.
- Turn `smefit/model.py` into the deprecated re-export.
- `analytic_fit.py` uses `is_linear`.
- Update the in-repo imports: `config.py`, the 9 Drell-Yan modules and the
  tests.
- Tests: split `test_model.py` into `test_models.py` (the contract, run on a
  toy model) and `test_smeft_model.py` (`EFTModel`).
- Docs: `AGENTS.md` (the module list, and the `model.py` entry under "Key
  modules"), and the paths in the `smefit-dev` skill.

### 2. Add the model lookup

- Add `smefit/models/__init__.py` with `_MODELS`, `get_model_class`,
  `available_models`, `ModelSpec` and `legacy_spec`.
- Tests:
  - lazy import: looking up `smeft` imports `wilson`, while importing
    `smefit.models` alone does not;
  - an unknown name raises an error listing the available models;
  - `ModelSpec.build` works on a toy model.

### 3. Add the `model:` runcard block

- In `config.py`: add `parse_model`, `produce_model_spec` and
  `_check_options`; build the three model producers through `model_spec`; add
  `model` to `_FIT_SETUP_KEYS`.
- In `fit_result.py`: add `model_name` and `model_options`, and base
  `use_quad` on them.
- In the generator: add `collect_models()`, with model options only at this
  stage. In `validate_runcard.py`: check the `model:` block and read
  `use_quad` from it.
- Switch the template runcards to the `model:` block, and regenerate the skill
  references.
- Docs:
  - in the `smefit-runcard` skill: `SKILL.md`, `references/recipes.md` and
    `references/coefficients.md`;
  - in the `smefit-datasets` skill: `references/database-layout.md`;
  - in the `smefit-analysis` skill: `SKILL.md`, `references/troubleshooting.md`
    and `references/output-layout.md`;
  - the runcard example in `AGENTS.md`;
  - the `smefit-fit-doctor` agent;
  - the `=None` example in the `smefit-dev` skill, which shows
    `produce_eft_model` building `EFTModel`.

### 4. Take the RGE runner and the mass-scan scale from the model

- In `config.py`:
  - `parse_rge(rge, model_spec)` validates through `runner.OPTIONS`;
  - `produce_rge_matrix` and the mass producers use `model_spec`;
  - the `rge_matrix` cache key includes the model name.
- In `config.py`, the following go: the `SMEFTRunner` import, the
  `ALLOWED_*` imports, the hand-written `rge` checks and the literal `1e3`.
- In the generator, `collect_models()` adds the runner options.
  `validate_runcard.py` checks the `rge:` sub-keys for the chosen model.
- Regenerate the skill references. In the `rge` section, `smeft_accuracy`,
  `yukawa` and `adm_QCD` move from the hand-written list to the SMEFT model's
  runner options, with their defaults and allowed values.

## Tests

| File | Change |
|---|---|
| `tests/test_models.py` (new) | **Contract, on a toy linear model and a toy non-linear model:** `forward_map` shape, `coefficients`, `is_linear`, and `mass_to_init_scale` raising by default. **Lookup:** laziness and unknown names. **`ModelSpec.build`.** |
| `tests/test_smeft_model.py` (from `test_model.py`) | Today's `EFTModel` tests, plus `OPTIONS`, `rge_runner is SMEFTRunner`, `is_linear` and `mass_to_init_scale` |
| `tests/test_config.py` | **`parse_model`:** valid block, unknown name, a block without `name` is the SMEFT model, an unknown option warns, a bad value raises. **`produce_model_spec`:** default, legacy `use_quad` warns, both together raise. **Warning frequency:** a report over two old fits logs the deprecation warning once for each. **`parse_rge`:** refused for a model without a runner; options checked against that model's runner. **Mass producers:** use `mass_to_init_scale`. **Other:** the signature pin on `rge_matrix` stays; `_FIT_SETUP_KEYS` contains `model`; `test_produce_individual_mass_eft_model_builds_model` follows the new signature. **Errors (contract item 6):** a toy model whose constructor raises `ValueError` makes `produce_pseudodata` fail, instead of falling back to SM-only pseudodata. |
| New graph test | **The test that proves the goal.** Register a toy model in `_MODELS` with `monkeypatch`. A runcard with `model: {name: toy}` on `tests/fixtures/` then resolves `chi2` and runs an analytic fit, with no change to `config.py`. |
| `tests/test_analytic_fit.py` | The error is raised for a non-linear model and matches the new message |
| `tests/test_fit_result.py` | `model_name`, `model_options` and `use_quad`, from a new runcard and from an old one |
| `tests/test_fitcontext.py` | An old fit (top-level `use_quad`) and a new fit (`model:` block) both rebuild |
| `tests/test_skill_scripts.py` | `validate_runcard.py`: the `model:` block, the `rge:` sub-keys for the chosen model, and the analytic-fit check with the block |
| `tests/test_app.py` | No change. It still checks that every new `parse_` and provider uses plain annotations: `Mapping`, `ModelSpec`, `bool \| None`. |

## Verification

1. After every commit, run `pytest -m "not slow"`. After commits 1 and 4, also
   run the full `pytest`.
2. **Regression against the integration branch.**
   - `template_runcards/analytical_fit.yaml` sets no `use_quad`, so it does
     not change. Run it on `generalize-beyond-eft` and on this branch: the
     `fit_results.json` must be identical.
   - Do the same with `template_runcards/rge.yaml`, whose `rge_matrix.pkl`
     frames must be identical.
   - `template_runcards/hessian_fit.yaml` sets a top-level `use_quad: True`.
     Run its old form on `generalize-beyond-eft`. On this branch, run both its
     old form and its new form with the `model:` block. All three
     `fit_results.json` must be identical, and only the old form on this
     branch logs the deprecation warning.
3. **Stored fits.** A report runcard over existing fits (`fits:` with
   `fitcontext`) runs as before.
4. **External chi2.** A test runcard with a Drell-Yan external chi2 and an
   `rge:` block runs without errors.
5. Regenerating the skill references after commit 4 matches what is
   committed, and CI's skills check passes.

## Not in this PR

- **Renaming the `eft_model` resource** and its variants to `model` is PR 5,
  with deprecated aliases or an announced break.
- **The external chi2 contract** (`coefficients=`, `rge_dict=`) is PR 4. The
  Drell-Yan modules only change their import here.
- **The theory container and the loader interface** are PR 3.
- **A general model**, evaluating any function of the parameters that the
  theory defines, is later work, outside the master PR, which only refactors
  (see Decision 1).
- **The σ×BR model** of the long-term goals is later work too, and the
  contract here already fits it:
  - its options, for example whether to expand the product again, go in its
    `model:` block;
  - it reports `is_linear` False;
  - its `forward_map` combines the production and decay polynomials of the
    theory;
  - it receives one RGE matrix per polynomial once PR 3 gives each polynomial
    its own scale.
- **DAG nodes contributed by a model package.** SMEFT needs none, so a
  mechanism waits for the first model that does.
- **Analytic fits with non-linear `expr:` constraints** keep being linearised,
  as today.

## Open questions

- **Size.** Commits 1–2 (contract, move, lookup) could be their own PR before
  commits 3–4 (runcard block, runner from the model), if a smaller review is
  preferred.
