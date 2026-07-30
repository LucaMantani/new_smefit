# Coefficient specification rules

Each entry under `coefficients:` is `Name: {…}` and must be exactly one of
three kinds. The invariants below are enforced at parse time
(`smefit.core.Coefficient.__post_init__`) — violating them raises immediately.

## 1. Free (fitted)

```yaml
OpWB: {free: True, prior: {dist: uniform, low: -0.5, high: 0.5}}
```

- `prior` is **required** (allowed distributions: see `priors.md`).
- `value`, `expr`, `vars` are **forbidden**.
- `baseline_value` (optional, default `0.0`) sets the coefficient's "default"
  point: where gradient descent starts, and the value returned directly under
  `gradient_descent_settings: {sm_solution: True}`. Because it belongs to the
  coefficients block rather than to any dataset, it applies to
  external-`chi2`-only fits too. Ignored for non-free coefficients.

```yaml
OpWB: {free: True, prior: {dist: uniform, low: -0.5, high: 0.5}, baseline_value: 0.0}
```

## 2. Fixed constant

```yaml
OpBox: {free: False, value: 0.1}
```

- `value` is required; `prior`, `expr`, `vars` are forbidden.

## 3. Constrained (expression)

```yaml
OtG:
  free: False
  vars: [y, OpWB]
  expr: "y**2 + 0.5*OpWB**2"
```

- `expr` is a Python expression over the names listed in `vars` (non-empty, no
  duplicates); `prior` and `value` are forbidden.
- `vars` entries reference **other coefficients defined in the same runcard**
  (free or fixed), including auxiliary parameters that are not operators at all
  — e.g. a free `y: {free: True, prior: …}` used only inside expressions.
- The expression is compiled once with empty builtins. Available functions/
  constants (all JAX, so constraints stay differentiable):
  `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `pi`.
  Nothing else — no numpy, no arbitrary Python.

## Practical notes

- Operator names must match the keys of the theory files exactly (check with
  the smefit-datasets skill: `smefit_db.py operators <pattern>`; the master
  list is `operators_implemented.yaml` in smefit_database).
- Only coefficients listed in the runcard enter the model; theory-file
  operators you omit are simply not varied.
- Constrained coefficients make the model nonlinear even at `use_quad: False`
  if the expression is nonlinear — `run_analytic_fit` is then invalid.
- Individual-fit actions (`run_individual_*_fits`) loop over the **free**
  coefficients one at a time; fixed/constrained entries are kept as-is in each
  single-parameter fit.
