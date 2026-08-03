# AGENTS.md

This file provides guidance to coding agents working in this repository. It is
read as project context by Codex CLI and Gemini CLI directly, and by Claude Code
through the `CLAUDE.md` symlink that points here — edit this file, not the
symlink.

## Skills

The skills under `.agents/skills/` carry the detail this file deliberately does
not — see `.agents/skills/README.md`. They follow the Agent Skills open
standard, so Claude Code, Codex CLI and Gemini CLI all discover them (Claude
Code via the `.claude/skills` → `../.agents/skills` symlink). Reach for them by
name:

- `smefit-runcard`, `smefit-datasets`, `smefit-analysis` — user-facing:
  authoring runcards, finding datasets/operators, running and reading fits.
- **`smefit-dev` — extending smefit itself**: adding a reportengine node
  (`parse_*`/`produce_*`), a runcard key, an action, a prior, a coefficient
  field; the conventions the generator depends on; which tests to extend.
- `smefit-server` — the server/registry infrastructure.

Files marked AUTO-GENERATED there are produced by
`python scripts/generate_skill_reference.py`; regenerate and commit them whenever
user-facing surface changes (runcard keys, actions, priors, template_runcards/).
CI (`.github/workflows/skills.yml`) fails if they are stale; the scripts the
skills bundle are covered by `tests/test_skill_scripts.py` in the normal suite.

## Agents (Claude Code only)

Custom subagents live under `.claude/agents/` — see `.claude/agents/README.md`.
Currently just `smefit-fit-doctor`, for diagnosing failing/hanging smefit runs.
Subagents are not part of the Agent Skills standard and have no cross-tool
equivalent, so other agent CLIs simply do not see them; everything they rely on
is in the skills, which are portable.

## Commands

**Install**:
```bash
pip install -e .
pip install -e ".[docs]"  # with docs dependencies
```

## Activate conda environment
Always activate the environment before running anything.
```
conda activate new_smefit
```

**Run the CLI**:
```bash
smefit <runcard.yaml>
```

## Test new functionalities
Test functionalities by copying a runcard from `template_runcards/` (e.g.
`template_runcards/analytical_fit.yaml`, fast and exact) into a scratch file and
modifying it to the needs, rather than editing the templates in place.
```bash
smefit my_test_runcard.yaml
```

**Run the test suite**:
```bash
pytest                    # full suite
pytest -m "not slow"      # skip tests that call real external samplers or RGE evolution
```

**Code formatting/linting**:
```bash
pre-commit run --all-files
```

## Architecture

SMEFiT is a SMEFT (Standard Model Effective Field Theory) analysis framework built on top of **reportengine** and **JAX**.

This tool has the objective of building an analysis framework in the SMEFT, implementing different analysis features such as Nested Sampling and plotting routines.

The tool is built on reportengine and its core feature of building a Directed Acyclic Graph (DAG).
In particular, the fundamental components of the code are nodes of this graph and it is very important that this design choice stays consistent so that extending features and functionalities of each node does not break the code.

### Key modules

- **`app.py`**: CLI entry point. `smefitApp` extends `reportengine.app.App`. Adds `-o/--output` and `-f32/--float32` flags.
- **`config.py`**: `smefitConfig` extends `reportengine.configparser.Config`. Implements `produce_*` methods that reportengine calls to build objects from runcard YAML keys.
- **`core.py`**: Domain dataclasses — `Dataset`, `Theory`, `Coefficient`, and their group variants (`DataGroup`, `TheoryGroup`, `CoefficientGroup`).
- **`model.py`**: `EFTModel` — maps free coefficient values to theory predictions. `forward_map` is JAX JIT-compiled.
- **`loader.py`**: Reads datasets from YAML and theories from JSON files in an external `smefit_database`.
- **`data_utils.py`**: Covariance matrix construction (handles correlated/uncorrelated systematics).
- **`chi2.py`**: `build_chi2` returns a JAX-differentiable loss function.
- **`fit_actions.py`**: The reportengine actions listed under `actions_:` in a runcard (`run_analytic_fit`, `run_ultranest_fit`, `run_blackjax_fit`, `run_hessian_fit`, and their `run_individual_*_fits` counterparts) — each takes its produced fit object plus `output_path` and executes/writes it.
- **`utils.py`**: Whitening (`apply_whitening`), posterior helpers (`resolve_posterior`, `build_exact_posterior_prior`), benchmarking (`chi2_timing`), and the `run_test`/`run_prior_test` reportengine actions.
- **`environment.py`**: `smefitEnvironment` sets JAX float32/float64 precision at startup.

Other modules not detailed here (see file docstrings): `analytic_fit.py`, `ultranest_fit.py`,
`blackjax_fit.py`, `hessian_fit.py`, `individual_fit.py`, `gradient_descent.py`, `projections.py`,
`external_chi2.py`, `rge/`, `priors.py`, `paths.py`, `fit_result.py`, `fisher.py`, `figures.py`,
`tables.py`, `wcxf.py`, `op_to_latex.py`, `utils_actions.py`, `constants.py`, `api.py` (the
`reportengine` programmatic API).

### reportengine integration

The framework uses reportengine's dependency injection pattern. All `produce_*` methods in `smefitConfig` are automatically called by reportengine when the corresponding key is needed. Actions listed under `actions_:` in the runcard are executed as the analysis steps.

A `produce_*` decorated with `@explicit_node` returns a *function* instead of a
value, and reportengine resolves that function's parameters as the node's
dependencies — so the dependency set is chosen at graph-build time from the
runcard. `produce_whitening_transformation` (`config.py`) is the example: only
`whitening.shift: gradient_descent` makes the graph depend on `gd_best_fit`.
Never flatten one back into a plain `produce_`; see the `smefit-dev` skill.

**Adding to the graph** — a node, a runcard key, an action, a prior — has
conventions that the reference generator and validator depend on: use the
`smefit-dev` skill rather than pattern-matching on an existing method.

### Coefficient constraints

`Coefficient` objects can be free (with a prior distribution), fixed to a value, or expression-constrained (e.g., `expr: "y**2 + 0.5*OpWB**2"`). Expressions reference other coefficient names and are evaluated as a lambda built with `eval` over `_EXPR_NAMESPACE` (empty builtins plus a few JAX functions, so constraints stay differentiable), cached on first use and applied in `constrain(*args)`.

### EFT predictions

`Theory` stores SM predictions, linear EFT corrections (`eft_lin_pred`: shape `[ndata, n_ops]`), and quadratic corrections (`eft_quad_pred`: shape `[ndata, n_ops, n_ops]`, upper-triangular). `EFTModel.forward_map(coeffs)` computes: `sm + lin @ c + quad @ c ⊗ c`.

### Covariance matrices

Three variants are used:
- **Experimental**: stat errors + correlated/uncorrelated systematics (from `data_utils.py`)
- **Theory**: block-diagonal across datasets, loaded from JSON
- **t0**: replaces multiplicative systematics using theory predictions as the central value

The runcard flags `use_theory_covmat` and `use_t0` control which are included in `fit_covmat`.

## Server infrastructure

Fits and reports are shared through a WebDAV server described by a `registry.json`.
All the logic lives in `smefit/server_utils.py`, driven by the `smefit_*` CLI
scripts in `smefit/scripts/` (`smefit_ls`, `smefit_upload`, `smefit_get`, …).

**Working on it? Use the `smefit-server` skill** — it carries the registry and
remote-layout schemas, the CLI command table, the design rules, the WebDAV
client cheatsheet, and the recipes for adding a metadata field or a new command.
Deliberately not duplicated here: this file is loaded into every session, the
skill only when it's relevant.

User-facing docs: `SERVER.md` (server usage, credentials, registry) and
`LOCAL_SETUP.md` (local path/database setup).

---

## Runcard structure

```yaml
data_path: smefit_database/commondata    # prefix resolved via .config/paths.yaml (smefit_setup_local)
theory_path: smefit_database/theory
use_theory_covmat: False
use_t0: False
use_quad: True
datasets:
  - {name: DATASET_NAME, order: LO}  # order: LO/NLO/NNLO
coefficients:
  OpName: {free: True, prior: {dist: uniform, low: -1.0, high: 1.0}, baseline_value: 0.0}
  OpFixed: {free: False, value: 1.0}
  OpExpr: {free: False, vars: [other_coeff], expr: "other_coeff**2"}
actions_:
  - run_test
```

`baseline_value` (optional, defaults to `0.0`) sets the "default" value of a free
coefficient. It is the point the gradient descent starts from, and the vector
returned directly when `gradient_descent_settings.sm_solution: true`.
It is a property of the coefficients dictionary, so it also
applies to external-`chi2`-only fits. It is ignored for non-free coefficients.

An optional `whitening:` block reparametrises free coefficients into a
better-conditioned space for sampling: `whitening: {sigma_prior: 5.0, eps: 1e-8,
shift: baseline}`. The transform is always centred on a point; `shift` selects
which one: `baseline` (default) centres on the coefficients' baseline point
(`baseline_value`, zero by default), while `gradient_descent` centres on the
`gd_best_fit` point instead, which additionally requires a
`gradient_descent_settings` block in the runcard.
