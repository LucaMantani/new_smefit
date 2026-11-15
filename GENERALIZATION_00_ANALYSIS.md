# Generalizing smefit beyond the EFT: analysis

Tracking issue: #116, *Do not assume EFT throughout*.

## Goal

smefit should assume as little as possible about the model that produces the
theory predictions. Changing the physical model used to fit the data should then
need no changes in the fitting, sampling and reporting code.

## How the work is organised

- **Master PR.** An integration branch, `generalize-beyond-eft`, is opened as a
  PR into `main`. It holds this analysis and one plan file per daughter
  PR (`GENERALIZATION_NN_<topic>.md`).
- **Daughter PRs.** Each part is done on its own branch, created from the
  integration branch, and merged back into it. A daughter's plan is written and
  reviewed on the integration branch.
- **Keeping up with `main`.** `main` is merged into the integration branch, not
  rebased onto it, since the daughter branches are created from it.
- **Finishing.** The master PR merges into `main` once every daughter is in and
  the whole refactorization is validated.

## Scope of this analysis

Only the `smefit/` package. Tests, the in-repo `external_chi2/` modules, the
skills and the docs are updated by the daughter PR.

The theory and data files are out of scope. They live in each model's
database, whose location the runcard gives, so a different model brings a
different database and nothing in smefit decides where it is.

## Classification of `smefit/`

### 1. Already general: nothing to do

Files with no EFT vocabulary:
`app.py`, `api.py`, `environment.py`, `utils.py`, `data_utils.py`,
`input_manifest.py`, `whitening.py`, `priors.py`, `bayesian_update.py`,
`gradient_descent.py`, `hessian_fit.py`, `ultranest_fit.py`, `blackjax_fit.py`,
`blackjax_samplers/`, `credible_intervals.py`, `fit_actions.py`, `fisher.py`,
`dataset_info.py`, `figures.py`, `plot_utils.py`, `latex_preview.py`,
`utils_actions.py`.

Files that mention EFT concepts but work unchanged for any model:

| File | Why it is already general |
|---|---|
| `chi2_scan.py` | Evaluates a chi2 along one parameter. What a mass scan means for the EFT lives in `config.py`. |
| `tables.py` | `mass_scan_table` only puts the scan points in a table. |
| `pca.py` | Only its docstring says "Wilson coefficients". |
| `latex_labels.py` | The built-in table holds SMEFiT labels for (SM)EFT parameters, but any other name falls back to the runcard's label or is typeset as written. |
| `fit_result.py` | `use_quad` returns False and `rge_cache` None for a fit without them. Nothing in `smefit/` reads either. |
| `paths.py`, `server_utils.py`, `scripts/` | They treat `rge_matrix.pkl` as an optional file of a fit. |

The `sm_solution` key of `gradient_descent_settings` (read by
`gradient_descent.py`, documented in `fisher.py`) only means "start from the
baseline point". Renaming it is optional.

### 2. Inherently EFT: stays model-specific

This code moves into `smefit/models/smeft/`, the package of the SMEFT model,
which also takes the EFT model class in a later PR. Each further model gets
its own package under `smefit/models/`.

| Code | What it is |
|---|---|
| `wcxf.py` | Map between the SMEFiT basis and the Warsaw basis of WCxf |
| `constants.py` | SM inputs at MZ, used only by the RGE runner and `wcxf.py` |
| `rge/runner.py` (the `wilson` part) | The SMEFT RGE runner built on `wilson` |
| `rge/_patches.py` | Monkey patches of `wilson` and `ckmutil` |
| `model.py` (`EFTModel`) | `sm + lin·c + quad·c⊗c`, with the RGE contraction |

### 3. To be generalized

| File | What assumes the EFT | Change | PR |
|---|---|---|---|
| `rge/runner.py` | The only runner is the `wilson` one | Generic runner interface; the `wilson` runner becomes the SMEFT one | 1 |
| `rge/build.py` | `build_rge_matrix` hardcodes the `wilson` runner | Takes the runner class | 1 |
| `rge/matrix.py` | Nothing beyond naming | Docstrings only; names kept | 1 |
| `config.py` | Always builds `EFTModel`. `use_quad` and `rge` are top-level keys. The t0 covmat uses `theory.sm_pred`. Mass scans set the scan point as the RGE `init_scale` (TeV→GeV). | Model and runner chosen in one place | 1 (RGE part), 2 |
| `model.py` | `BaseModel` only requires `forward_map`, but callers also use `.coefficients` and `.use_quad` | Extend the `BaseModel` contract | 2 |
| `analytic_fit.py` | Checks linearity through `eft_model.use_quad` | Ask the model whether it is linear | 2 |
| `core.py` | `Theory`/`TheoryGroup` names: `sm_pred`, `eft_lin_pred`, `eft_quad_pred`, `operators` | Neutral names; the polynomial layout stays and is generalized to higher orders | 3 |
| `loader.py` | `load_theory` names the parsed predictions `eft_pred` | Neutral names; the JSON layout stays | 3 |
| `external_chi2.py` | Plugins are built as `cls(coefficients=…, rge_dict=…)` | Pass the model's settings, staying compatible with existing modules | 4 |
| `chi2.py`, `projections.py`, `individual_fit.py` | `eft_model` argument and resource names | Rename to `model` | 5 |

## Decisions so far

- **The polynomial layout of the theory tables is generic.** The layout is an
  SM (zeroth-order) term plus linear and quadratic terms in the parameters,
  with quadratic keys written `a*b`. Any model with that layout reuses
  `load_theory` and `Theory`, so those only get neutral names. It can be further
  generalized to a linear expansion in a basis of elementary functions, where the
  linear, quadratic and other orders in the polynomial are just elements in the basis.
- **The RGE machinery is generic.** Computing an RGE matrix per scale, caching
  it and reading it back do not depend on the model. Only the runner does: it
  decides which parameters are evolved, into which names, and how. The
  `wilson` runner becomes the SMEFT runner.
- **Model-specific code goes in `smefit/models/<model>/`.** There is one
  package per model, starting with `smefit/models/smeft/`. Generic code stays
  outside `smefit/models/`, so the top level of the package does not grow with
  the number of models.
- **Code shared between models moves only when it is shared.** Code another
  model might also need, such as the SM inputs in `constants.py`, stays in its
  first model's package. It moves to a shared package, such as
  `smefit/models/common/`, once a second model actually needs it.
- **The master PR only refactors.** It changes how the existing code is
  organised and adds no new model. Newer models (such as a general polynomial model) is later work.
- **`EFTModel` is the SMEFT model.** Its limitations, among them predictions at
  most quadratic in the coefficients, are those of the SMEFT. A general
  polynomial model, with higher orders or a basis of functions, would be a
  separate and more general model.
- **A model declares its RGE runner.** A runcard picks a model, and the model
  brings its runner, or none. The `rge:` block therefore holds the generic keys
  plus the options of that model's runner.
- **The runcard chooses the model in a `model:` block** that holds every model
  option, for example `model: {name: smeft, use_quad: true}`. Its `name`
  defaults to `smeft`. A top-level `use_quad` is still accepted as a
  deprecated shorthand, because stored fits use it, and the deprecation
  warning is logged once per fit.
- **The t0 covmat keeps using `theory.sm_pred`.** It is the zeroth-order term
  of the shared theory layout, so it does not depend on the model.

## Daughter PRs

PRs 1 and 2 have detailed plans. The scope of the others is preliminary, and
each gets its own plan file before work on it starts.

| # | Topic | Main files | Depends on | Plan |
|---|---|---|---|---|
| 1 | Generic RGE runner, SMEFT runner | `rge/`; `wcxf.py` and `constants.py` (moved to `models/smeft/`); `config.py` (RGE part); `external_chi2/` callers | none | [GENERALIZATION_01_RGE.md](GENERALIZATION_01_RGE.md) |
| 2 | Model interface and selection | `models/` (from `model.py`), `config.py`, `analytic_fit.py`, `fit_result.py`, the reference generator and `validate_runcard.py`, the template runcards | 1 | [GENERALIZATION_02_MODEL.md](GENERALIZATION_02_MODEL.md) |
| 3 | Neutral names in `Theory` and `load_theory` | `core.py`, `loader.py`, `projections.py`, `model.py` | 2 | to be written |
| 4 | External chi2 contract | `external_chi2.py`, `external_chi2/` | 1, 2 | to be written |
| 5 | `eft_model` → `model` naming pass | `config.py`, `chi2.py`, `projections.py`, `individual_fit.py`, skills, docs | 2 | to be written |

**PR 2, model interface and selection.** See
[GENERALIZATION_02_MODEL.md](GENERALIZATION_02_MODEL.md). In short:

- `BaseModel` gets the full contract that callers rely on: `coefficients`,
  `is_linear`, the model's options and RGE runner, and the mass-scan scale.
- `EFTModel` moves to `smefit/models/smeft/` as the SMEFT model.
- A lazy model lookup in `smefit/models/__init__.py` serves the new `model:`
  runcard block.
- `config.py` builds the model, its RGE runner and its options through the
  lookup, and keeps no SMEFT-specific code.
- The reference generator documents each model's and runner's options from
  the lookup.

**PR 3, neutral names in `Theory` and `load_theory`.** The polynomial layout
and the database keys (`best_sm`, the order keys, `a*b`) stay. Only the Python
names change.

**PR 4, external chi2 contract.** Plugins receive the model's settings instead
of a dedicated `rge_dict`, and existing modules keep working. The `smefit-dev`
skill, which states the current contract (`coefficients=`, `rge_dict=`), is
updated in the same PR.

**PR 5, naming pass.**

- Rename the `eft_model` resource and its `individual_*` variants to `model`.
  `smefitAPI.eft_model(...)` appears in the docs and skills.
- This is an API change: resources are requested by name, by
  `smefitAPI.eft_model(...)`, report templates and users' scripts, and an
  unknown name only fails with "cannot find a way to compute". Keep `eft_model`
  and its variants as deprecated aliases for a while, or announce the break.
- Update the argument names in `chi2.py`, `projections.py` and
  `individual_fit.py`.
- Regenerate the skill references.
- PR 2 keeps the resource names, so this stays a separate PR.

## Open questions

- **Reportengine nodes from a model package.** SMEFT needs none after PR 2.
  How a model would contribute its own nodes is decided with the first model
  that needs one.
- **Parameter naming.** The issue mentions fitted parameters that are not
  Wilson coefficients. Should "coefficients" also become "parameters" in the
  code and the runcard? (PR 5 or later)
- **These files after the merge.** Do the `GENERALIZATION_*.md` files stay in
  `main` once the master PR merges, move to `docs/`, or get removed?
