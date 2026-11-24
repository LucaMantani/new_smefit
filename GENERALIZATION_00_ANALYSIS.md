# Generalizing smefit beyond the EFT: analysis

Tracking issue: #116, *Do not assume EFT throughout*.

## Goal

smefit should assume as little as possible about the model that produces the
theory predictions. Changing the physical model used to fit the data should then
need no changes in the fitting, sampling and reporting code.

## Long-term goals

The master PR adds no feature. It should, however, make three later goals
easier to reach.

1. **Predictions that are any function of the theory parameters**, not only
   polynomials in them. An example is
   `a1 sin(x) + a2 cos(x - y) + a3 sin(x)**2 + a4 exp(x + y)`, where `x` and
   `y` are theory parameters, each fitted or fixed, and the `a_i` come from
   the theory tables. POPxf encodes it in its function-of-polynomials mode,
   with the polynomials `x`, `y`, `x + y` and `x - y` combined by the
   expression above. A first example model that will make use of this is the $\sigma \times$BR model mentioned in point 3.
2. **POPxf as a theory input format.** POPxf
   ([arXiv:2511.17348](https://arxiv.org/abs/2511.17348),
   [github.com/pop-xf](https://github.com/pop-xf)) is the LHC EFT and Higgs
   working groups' exchange format for predictions that are polynomials in the
   model parameters, or functions of several such polynomials. The parts that
   matter here:
   - **Terms.** Each observable is a polynomial of any degree, up to 5 in the
     current schema. Its terms are keyed by monomials, such as `"('', 'C1')"`
     or `"('C1', 'C2')"`, and an `R`/`I` tag selects the real or imaginary part
     of a complex parameter.
   - **Two modes.** In the single-polynomial mode each observable is one
     polynomial. In the function-of-polynomials mode, a file defines named
     polynomials, and each observable has an expression combining them, for
     example `num / den`.
   - **Scale.** It is one number for the whole file, or one per observable or
     per polynomial.
   - **Basis.** The parameters' basis is named, through WCxf or a custom
     definition.
   - **Uncertainties.** They are given per source and may depend on the
     parameters. Correlations, including across files, live in separate
     correlation files.
3. **A σ×BR model.** This is an EFT-based model whose theory gives production
   cross sections and branching ratios as separate predictions. For each
   observable the model multiplies the matching cross section and branching
   ratio, then either keeps the full product or expands it again to the EFT
   order. Production and decay may be given at different scales. This will be probably the first model to benefit from the generalizations in points 1 and 2.

[How the PRs prepare for the long-term goals](#how-the-prs-prepare-for-the-long-term-goals)
lists what each PR does so that all three can be added later without
reshaping the code.

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
| `core.py` | `Theory`/`TheoryGroup` store exactly an SM, a linear and a quadratic term, under EFT names (`sm_pred`, `eft_lin_pred`, `eft_quad_pred`, `operators`). They have one scale per data point, and the theory covmat is fixed as block-diagonal when the group is built. | A format-neutral theory container, covering both the current SMEFiT layout and POPxf: polynomials of any degree in named parameters, combined per observable by an expression (designed; only the identity is implemented). Each file format is a loader translating into it. Today's arrays become views of it. The stored data and the results stay the same. | 3 |
| `loader.py` | `load_theory` only reads the SMEFiT JSON layout, and names the parsed predictions `eft_pred` | A lookup of theory formats, with the SMEFiT JSON layout as its only entry. The JSON layout stays. | 3 |
| `external_chi2.py` | Plugins are built as `cls(coefficients=…, rge_dict=…)` and know nothing of the model | Pass the model (`model_spec=`) to plugins that ask for it, and let plugins declare the models they support. Existing plugins keep working. | 4 |
| `chi2.py`, `projections.py`, `individual_fit.py` | `eft_model` argument and resource names | Rename to `physics_model` (`model` is the runcard key of PR 2) | 5 |

## Decisions so far

- **A theory prediction is a function of the theory parameters.**
  - **Fitted or fixed.** Each theory parameter is either fitted, or fixed to a
    value or to an expression of other parameters (`expr:`). That choice
    belongs to the runcard, not to the theory, as it does for coefficients
    today.
  - **Today the function is a quadratic polynomial.** The tables hold an SM
    (zeroth-order) term plus linear and quadratic terms in the parameters,
    with quadratic keys written `a*b`.
  - **The generalization lets the theory define the function**, as POPxf
    does. The tables hold polynomials of any degree in the parameters, with
    terms keyed by monomials. Each observable is an expression of one or more
    of them.
  - **Both are JAX functions,** the polynomials and the expressions, so the
    chi2 stays differentiable. Any model whose theory has this layout reuses
    `load_theory` and `Theory`.
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
  organised and adds no new model. New models are later work: a general model
  that evaluates any function of the parameters the theory defines, and the
  σ×BR model of the long-term goals.
- **`EFTModel` is the SMEFT model.** Its limitations, among them predictions at
  most quadratic in the coefficients, are those of the SMEFT. A general model
  would evaluate any function of the parameters that the theory defines, and
  would be separate from `EFTModel`.
- **A model declares its RGE runner.** A runcard picks a model, and the model
  brings its runner, or none. The `rge:` block therefore holds the generic keys
  plus the options of that model's runner.
- **The runcard chooses the model in a `model:` block** that holds every model
  option, for example `model: {name: smeft, use_quad: true}`. Its `name`
  defaults to `smeft`. A top-level `use_quad` is still accepted as a
  deprecated shorthand, because stored fits use it, and the deprecation
  warning is logged once per fit.
- **The t0 covmat uses the theory's reference prediction**: the theory's
  predictions at its reference point, which gives a value to every theory
  parameter, fitted or fixed. The theory defines the reference point, so t0
  depends neither on the model nor on the runcard.
  - **Default: zero.** The reference point is zero for every parameter
    unless the theory file names another. Today's files name none, so the
    reference prediction is `theory.sm_pred`, the zeroth-order term, and
    nothing changes.
  - **A model-independent name.** PR 3 calls it `reference_prediction`, which
    also covers observables built from several polynomials. For a σ×BR model
    at the default point it is σ_SM × BR_SM.
  - **Naming another point.** A theory file can name another reference point
    through a new optional field (PR 3). POPxf 1.0 has no such field, so a
    POPxf theory uses zero unless a later loader reads one, for example from
    `misc`.
- **PR 3 designs the framework to support data points whose predictions are
  combinations of polynomials and elementary functions of polynomials.** A
  theory prediction holds a stack of polynomials (rows), shareable between
  data points as POPxf's are, and each data point is either a row or an
  expression combining rows through named variables.
  Only data points whose theory predictions map one-to-one to a row will be
  implemented by the moment this PR is merged. The first loader or model that
  produces such theories, POPxf's function-of-polynomials mode or the σ×BR
  model, implements the rest.
- **PR 3 moves the JAX expression compiler into its own module.** The code
  that turns a coefficient's `expr:` into a differentiable JAX function moves
  out of `Coefficient`, so that POPxf's `observable_expressions` can reuse it
  later.
- **Deprecated aliases are kept for a long time.** Legacy code and stored
  results rely on the old names.

## Daughter PRs

Every daughter PR has a detailed plan. PR 5 should merge last, since it
renames names that every other PR touches.

| # | Topic | Main files | Depends on | Plan |
|---|---|---|---|---|
| 1 | Generic RGE runner, SMEFT runner | `rge/`; `wcxf.py` and `constants.py` (moved to `models/smeft/`); `config.py` (RGE part); `external_chi2/` callers | none | [GENERALIZATION_01_RGE.md](GENERALIZATION_01_RGE.md) |
| 2 | Model interface and selection | `models/` (from `model.py`), `config.py`, `analytic_fit.py`, `fit_result.py`, the reference generator and `validate_runcard.py`, the template runcards | 1 | [GENERALIZATION_02_MODEL.md](GENERALIZATION_02_MODEL.md) |
| 3 | Theory container and loader interface | `core.py`, `loader.py`, `projections.py`, `config.py` (t0), `models/smeft/model.py`, the new expression module, `external_chi2/drell_yan/` | 2 | [GENERALIZATION_03_THEORY.md](GENERALIZATION_03_THEORY.md) |
| 4 | External chi2 contract | `external_chi2.py`, `config.py` (the `*ext_chi2_func` producers, `parse_external_chi2`), `external_chi2/` | 1, 2 | [GENERALIZATION_04_EXTERNAL_CHI2.md](GENERALIZATION_04_EXTERNAL_CHI2.md) |
| 5 | `eft_model` → `physics_model` naming pass, and `RGEMatrix`'s field names | `config.py`, `chi2.py`, `projections.py`, `individual_fit.py`, `analytic_fit.py`, `rge/`, skills, docs | 2 (merges last) | [GENERALIZATION_05_NAMING.md](GENERALIZATION_05_NAMING.md) |

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

**PR 3, theory container and loader interface.** See
[GENERALIZATION_03_THEORY.md](GENERALIZATION_03_THEORY.md). The database files and their
keys (`best_sm`, the order keys, `a*b`) stay, and so do the results. What
changes is how the theory is held in memory and how it is loaded.

- **Polynomials of any degree in the theory parameters**, with terms keyed by
  monomials as in POPxf. The parameters are matched by name to the runcard's
  coefficients, each fitted or fixed. Together with the expression per
  observable (designed below), a prediction can be any function of the
  parameters that the theory defines. The SMEFT model reads its linear and
  quadratic arrays as views of these polynomials. It raises a clear error for
  a theory it cannot use, such as one of degree above 2.
- **One scale per polynomial.** Today each data point is one polynomial, so
  this is still one scale per data point. `build_rge_matrix` only reads
  `scales` from the theory, so it computes one RGE matrix per polynomial with
  no change, also when data points combine several.
- **A `reference_prediction`**, the predictions at the theory's reference
  point. It replaces `sm_pred` in the t0 covmat and in the projections.
  - The reference point is zero for every parameter unless the theory file
    names another.
  - The SMEFiT JSON layout gains an optional field for it, for example
    `reference_point: {<parameter>: <value>}`. Existing files lack it, so their
    results do not change.
  - The `smefit-datasets` skill's `database-layout.md` documents the field.
- **The theory covmat comes from a method** of the container instead of being
  fixed as block-diagonal when the group is built. A later loader can then
  bring correlations across datasets.
- **The parameters' names and basis are stored on the container**, as POPxf's
  `parameters` and `basis` are. A model can then check that it receives tables
  in its own basis.
- **A lookup of theory formats** dispatches on the file content (a POPxf file
  declares its `$schema`). The SMEFiT JSON layout is its only entry.
- **Designed for data points that combine polynomials.** The container holds
  a stack of polynomials and, optionally, per data point an expression and the
  variables mapping its names to polynomials, as in POPxf's
  function-of-polynomials mode. Such theories can be stored, but not
  evaluated, and no loader builds them yet.
- **The JAX expression compiler gets its own module.** Today `Coefficient`
  compiles its `expr:` with `eval` over `_EXPR_NAMESPACE` in `core.py`. That
  compiler moves to its own module and takes the namespace as an argument.
  Coefficient `expr:` keeps exactly its current functions, so the runcard
  language does not change. A later POPxf loader can pass POPxf's function set
  for the observable expressions. The same compiler then serves coefficient
  `expr:` and the theory's observable expressions.
- **Expressions are checked before they are compiled.**
  - **The check.** The compiler parses each expression with `ast`, and
    accepts only names, numbers, arithmetic operators and calls to the
    functions of its namespace. Only then does it compile.
  - **Why.** `eval` with empty builtins is not a sandbox, and POPxf
    expressions will come from files shared by others or downloaded from the
    server.
  - **Coefficient `expr:` gets the same check.** It accepts everything the
    current namespace allows, and a test runs it on every `expr:` in the
    template runcards.
- **The docs naming the compiler change in the same commit.** `AGENTS.md`
  ("Coefficient constraints") and the `smefit-dev` skill ("Expression
  constraints") both say the `expr:` language is `_EXPR_NAMESPACE` in
  `core.py`.
  - `validate_runcard.py` keeps its own list of non-linear functions,
    `_NONLINEAR_FUNCS`, which must stay in step with the namespace.
  - The new module holds helpers, not DAG nodes, so it stays out of
    `smefit_providers`.
- **Errors raise `ValueError`, never `KeyError`.**
  - **Why.** The container, its views and `EFTModel`'s check of the theory
    run inside `produce_theory` and `produce_eft_model`, while the config is
    read. reportengine treats a `KeyError` raised there as "input not
    available", and a consumer with a default then swallows it.
    `produce_pseudodata` has `eft_model=None`, so it would build SM-only
    pseudodata without any message.
  - **Precedent.** `load_theory` already turns its `KeyError` into a
    `ValueError` for this reason.
  - **Test.** A theory the model cannot use, or a missing monomial, raises
    `ValueError` both through `produce_eft_model` and through
    `produce_pseudodata`.
- **Callers of the renamed names are updated, and the old names stay as
  deprecated aliases.**
  - **In-repo callers.** The 9 Drell-Yan external chi2 modules call
    `load_theory`, build `TheoryGroup`, and read `theory_group.sm_pred` and
    `theory_group.sm_covmat`. They move to the new names.
  - **Aliases.** `sm_pred`, `sm_covmat`, `eft_lin_pred`, `eft_quad_pred` and
    `operators` stay available with a `DeprecationWarning` that clarifies they're
    specific to the EFT model.
    `load_theory(theory_path, dataset_dict)` keeps its signature, with the
    format lookup behind it, so it needs no alias.
- **Neutral names** replace `sm_pred`, `eft_lin_pred`, `eft_quad_pred` and
  `operators`.

**PR 4, external chi2 contract.** See
[GENERALIZATION_04_EXTERNAL_CHI2.md](GENERALIZATION_04_EXTERNAL_CHI2.md). In
short:

- **Plugins can learn the model.** They receive the runcard's model as
  `model_spec=`, but only when their constructor names it, in addition to
  `rge_dict`. After PR 2 `rge_dict` is no longer SMEFT-specific.
- **Plugins declare the models they support** in `SUPPORTED_MODELS`.
  smefit checks it when it loads them.
- **Three bugs in the plugin loader are fixed first, in a bugfix PR against
  `main`.**:
  - an entry without `path`, or a plugin constructor raising a `KeyError`,
    makes the fit quietly run without that likelihood;
  - two plugin files with the same name in different folders are confused.

  The fix to those bugs will be incorporated into this master PR via a merge main after they have been applied on main.
- **The in-repo plugins conform to the contract**, including the two legacy
  `low_energy` ones, and their results do not change.
- **The docs match the code.** The `smefit-dev` skill and `recipes.md` drop
  `param_names`, which smefit never reads.

**PR 5, naming pass.** See
[GENERALIZATION_05_NAMING.md](GENERALIZATION_05_NAMING.md). In short:

- Rename the `eft_model` resource and its `individual_*` variants to
  `physics_model`. They cannot become `model`, which after PR 2 is the runcard
  key of the `model:` block: reportengine would hand the `ModelSpec` to
  anything asking for `model`.
- This is an API change: resources are requested by name, by
  `smefitAPI.eft_model(...)`, report templates and users' scripts, and an
  unknown name only fails with "cannot find a way to compute". The old names
  stay as deprecated alias producers, which log a warning.
- Rename the arguments in `chi2.py`, `projections.py`, `analytic_fit.py` and
  `individual_fit.py`, and the `RGEMatrix` fields PR 1 left for this pass
  (`obs_operators`, `init_operators`). PR 1 gives the new runner neutral
  names (`names`, `matrix`, `matrix_dict`) from the start.
- "Coefficients" keeps its name.
- PR 2 keeps the resource names, so this stays a separate PR.

## How the PRs prepare for the long-term goals

| Need | Prepared by | Left for later |
|---|---|---|
| Polynomials of any degree in the theory parameters, as POPxf has | PR 3: the theory container keyed by monomials | The POPxf loader, and a model that evaluates degrees above 2 |
| Predictions that are any function of the theory parameters, such as `a1 sin(x) + a2 cos(x - y) + a3 sin(x)**2 + a4 exp(x + y)` | PR 3: the container holds the polynomials (`x`, `y`, `x + y`, `x - y`) and is designed for an expression per observable, so the theory defines the function, as in POPxf. The shared JAX compiler compiles the expression. | Evaluating observable expressions, a model that uses them, and reading them from POPxf |
| POPxf function-of-polynomials mode, and σ×BR products | PR 3: the container holds polynomials shared between data points, and per data point an expression combining them | Evaluating the expressions, and the σ×BR model itself |
| Expressions compiled to JAX, needed for POPxf `observable_expressions` | Exists already: coefficient `expr:` strings are compiled with a JAX namespace (`_EXPR_NAMESPACE` in `core.py`). PR 3 moves that compiler to its own module, taking the namespace as an argument. | The POPxf function namespace |
| Production and decay at different scales, and POPxf per-polynomial scales | PR 1: one RGE matrix per entry of `scales`. PR 3: one scale per polynomial. | Contracting the matrices, in the model that combines the polynomials |
| Tables in the Warsaw basis (POPxf `basis.wcxf`) | PR 1: the runner's rows follow the basis of the theory tables. The SMEFT runner already computes in the Warsaw basis before mapping to SMEFiT. PR 3: the basis is stored on the container. | Warsaw-basis output from the SMEFT runner, or translating the tables with `wcxf.py` |
| SM reference of a combined observable (σ_SM × BR_SM), for t0 and the projections | PR 3: `reference_prediction`, at the theory's reference point, which is zero by default | Nothing |
| Theory files supplied by others | PR 3: expressions checked with `ast` before they are compiled | Nothing |
| External likelihoods that only make sense for some models | PR 4: plugins declare `SUPPORTED_MODELS`, and can receive the runcard's model as `model_spec` | Nothing |
| Model options such as "expand the product again" | PR 2: options of the `model:` block | Nothing |
| Predictions that are not linear | PR 2: `is_linear` stops the analytic fit from being used on them | Nothing |
| POPxf uncertainties and correlation files | PR 3: the theory covmat comes from the container | Uncertainties that depend on the parameters, which make the chi2 covariance depend on the parameters too |
| Complex parameters in POPxf | Nothing | The POPxf loader, which maps real and imaginary parts to real fit parameters |
| Database layout. A SMEFiT theory file holds several perturbative orders, picked by each dataset's `order` key, while a POPxf file holds one prediction. | PR 3: the format lookup keeps `theory_path/<dataset>.json` and the `order` key working for the SMEFiT layout | How a database lays out POPxf files (for example one per dataset and order), and how `order` selects among them |

## Open questions


- **Reportengine nodes from a model package.** SMEFT needs none after PR 2.
  How a model would contribute its own nodes is decided with the first model
  that needs one.
- **Parameter naming.** The issue mentions theory parameters that are not
  Wilson coefficients, and POPxf calls them `parameters`. Should
  "coefficients" become "parameters" in the code and the runcard? Each one
  stays fitted or fixed, as today. (After the master PR: PR 5 keeps
  "coefficients", since renaming the runcard key is a migration of its own.)
- **These files after the merge.** Do the `GENERALIZATION_*.md` files stay in
  `main` once the master PR merges, move to `docs/`, or get removed?
