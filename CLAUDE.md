# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

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
Test functionalities by running the prototype_runcard.yaml in the root of the repository, after
modifying it to the needs.
```bash
smefit prototype_runcard.yaml
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
- **`utils.py`**: `build_chi2` returns a JAX-differentiable loss function; `run_test` is a reportengine action.
- **`environment.py`**: `smefitEnvironment` sets JAX float32/float64 precision at startup.

### reportengine integration

The framework uses reportengine's dependency injection pattern. All `produce_*` methods in `smefitConfig` are automatically called by reportengine when the corresponding key is needed. Actions listed under `actions_:` in the runcard are executed as the analysis steps.

### Coefficient constraints

`Coefficient` objects can be free (with a prior distribution), fixed to a value, or expression-constrained (e.g., `expr: "y**2 + 0.5*OpWB**2"`). Expressions reference other coefficient names and are compiled via `compile()` with empty builtins for safety. Evaluated in `constrain(*args)` which is cached after first compilation.

### EFT predictions

`Theory` stores SM predictions, linear EFT corrections (`eft_lin_pred`: shape `[ndata, n_ops]`), and quadratic corrections (`eft_quad_pred`: shape `[ndata, n_ops, n_ops]`, upper-triangular). `EFTModel.forward_map(coeffs)` computes: `sm + lin @ c + quad @ c ⊗ c`.

### Covariance matrices

Three variants are used:
- **Experimental**: stat errors + correlated/uncorrelated systematics (from `data_utils.py`)
- **Theory**: block-diagonal across datasets, loaded from JSON
- **t0**: replaces multiplicative systematics using theory predictions as the central value

The runcard flags `use_theory_covmat` and `use_t0` control which are included in `fit_covmat`.

## Server infrastructure

All server logic lives in `smefit/server_utils.py`. The registry is a JSON file (`registry.json`) stored on the remote WebDAV server with the structure:

```json
{
  "fits":     { "<name>": { "created_at", "uploaded_by", "has_rge", "rge_path", "runcard_path", "comment", "project" } },
  "reports":  { "<name>": { "created_at", "uploaded_by", "comment", "project" } },
  "projects": ["project_a", "project_b"]
}
```

Key design points:
- `_read_registry` / `_write_registry` download/upload the JSON atomically via a temp file.
- `_empty_registry()` always includes the `"projects"` key — migration from old format happens in `_read_registry`.
- Two servers: `public` (bundled read-only creds, team members can also have write creds) and `private`.
- `Uploader.upload()` accepts an optional `project` kwarg that is stored in the registry entry.
- `sync_registry` rebuilds the registry from scratch but preserves `projects` via `_empty_registry` merge.

### CLI scripts (`smefit/scripts/`)

| Command | Description |
|---|---|
| `smefit_ls` | List fits/reports/rge/misc from the registry |
| `smefit_upload` | Upload a resource; prompts for comment then project |
| `smefit_get` | Download a resource |
| `smefit_mv` | Rename a resource |
| `smefit_rm` | Move a resource to `bin/` on the server (soft delete) |
| `smefit_manage_project` | Manage the project list (add/rename/remove/list) |
| `smefit_server` | Server management: setup credentials, check storage, sync registry, tutorial |
| `smefit_mkdir` | Create a directory under `misc/` |
| `view_report` | Download + open a report in the browser |

### Adding new metadata fields

1. Add the field to the `entry` dict in `Uploader.upload()` (and `sync_registry` if it should survive a resync).
2. If it needs a separate managed list (like `projects`), add helpers following the pattern of `add_project` / `rename_project` / `remove_project` in `server_utils.py`.
3. Update `_fit_rows` / `_report_rows` in `smefit_ls.py` to display the field conditionally (only when at least one resource has it set).
4. Document in `SERVER.md`.

---

## Runcard structure

```yaml
data_path: /path/to/smefit_database/commondata
theory_path: /path/to/smefit_database/theory
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
