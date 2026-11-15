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

## Daughter PRs

Only PR 1 has a detailed plan. The scope of the others is preliminary, and each
gets its own plan file before work on it starts.

| # | Topic | Main files | Depends on | Plan |
|---|---|---|---|---|
| 1 | Generic RGE runner, SMEFT runner | `rge/`; `wcxf.py` and `constants.py` (moved to `models/smeft/`); `config.py` (RGE part); `external_chi2/` callers | none | [GENERALIZATION_01_RGE.md](GENERALIZATION_01_RGE.md) |
| 2 | Model interface and selection | `model.py`, `config.py`, `analytic_fit.py`, `fit_result.py` | 1 | to be written |
| 3 | Neutral names in `Theory` and `load_theory` | `core.py`, `loader.py`, `projections.py`, `model.py` | 2 | to be written |
| 4 | External chi2 contract | `external_chi2.py`, `external_chi2/` | 1, 2 | to be written |
| 5 | `eft_model` → `model` naming pass | `config.py`, `chi2.py`, `projections.py`, `individual_fit.py`, skills, docs | 2 | to be written |

**PR 2, model interface and selection.**

- `BaseModel` declares everything callers use:
  - the parameters, today `.coefficients`, used by `chi2.py`,
    `analytic_fit.py` and `config.py`;
  - whether the model is linear, which `analytic_fit.py` needs;
  - the RGE runner class it needs, or None if it has no running.
- `BaseModel` moves to `smefit/models/base.py`, and `EFTModel` to
  `smefit/models/smeft/`. `smefit/model.py` is then removed, or kept as a thin
  re-export for existing imports such as those in the Drell-Yan external chi2
  modules.
- `smefit/models/__init__.py` maps runcard names to model classes. The lookup
  must be lazy, mapping names to module paths, so that loading one model does
  not import another's dependencies (for SMEFT, `wilson` and its patches).
  `blackjax_samplers/` imports every algorithm up front, so it is not the
  pattern to copy here.
- `config.py` takes the model and its runner from one place. This removes the
  `SMEFTRunner` that PR 1 leaves hardcoded in `config.py`.
- The mass-scan producers, which turn the scan point into the RGE
  `init_scale`, move behind the model.
- Model-specific runcard keys and reportengine nodes cannot simply live in the
  model's package:
  - runcard parsing is one `smefitConfig` class;
  - the skill-reference generator reads the `known_keys` literals in it;
  - providers are registered from the fixed list in `app.py`.

  PR 2 decides how a model contributes them. For SMEFT these are the `rge`
  options and the mass-scan producers.
- Validating the runner options through the runner's `OPTIONS` moves here
  from PR 1. It needs a generator change: the generator only documents literal
  `known_keys`, literal `.get()` defaults and literal `ConfigError` messages in
  `config.py`, so options declared on a model or runner class would vanish from
  the reference. The generator has to document each model's and runner's
  options from the registry, as `collect_priors` does with `_DIST_REGISTRY`.
- The `smefit-dev` skill is updated. Its main example for the `=None` rule is
  `produce_eft_model` building `EFTModel(...)`.

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
- This could be folded into PR 2 if that PR stays small.

## Open questions

- **Model selection.** How does a runcard select the model, and with it the
  RGE runner? (PR 2)
- **Model-specific keys and nodes.** How does a model package contribute its
  own runcard keys and reportengine nodes? (PR 2)
- **t0 covmat.** Should it keep using `theory.sm_pred` or use the model's
  prediction at its baseline? The two differ when a fixed coefficient is
  non-zero, so switching changes existing fits. (PR 2)
- **Runner-specific keys.** Today the SMEFT keys (`smeft_accuracy`, `yukawa`,
  `adm_QCD`) sit directly in `rge:`. Where do another runner's keys go? This
  only needs an answer once a second runner exists.
- **Parameter naming.** The issue mentions fitted parameters that are not
  Wilson coefficients. Should "coefficients" also become "parameters" in the
  code and the runcard? (PR 5 or later)
- **These files after the merge.** Do the `GENERALIZATION_*.md` files stay in
  `main` once the master PR merges, move to `docs/`, or get removed?
