# Daughter PR 3: theory container and loader interface

Part of the generalization beyond the EFT (issue #116). See
[GENERALIZATION_00_ANALYSIS.md](GENERALIZATION_00_ANALYSIS.md) for the overall
analysis and how the PRs are organised.

- Branch: `generalize-beyond-eft-theory`, created from `generalize-beyond-eft`
  once PR 2 is merged into it.
- Target: `generalize-beyond-eft`.
- Depends on PR 2: `EFTModel` in `smefit/models/smeft/model.py`, and the
  `BaseModel` contract, whose item 5 ("a model checks the theory it is given")
  lands here.
- Touches the same Drell-Yan modules as PR 4. Whichever of the two merges
  second rebases onto the other.

## Goal

Today the in-memory theory *is* the SMEFiT JSON layout: an SM term plus linear
and quadratic terms in Wilson coefficients, under EFT names. After this PR it is
a format-neutral description of a theory prediction, and each file format is a
loader that translates into it.

- **A format-neutral theory container.** Per dataset it holds what any
  polynomial-based format has to say:
  - a stack of polynomials in named parameters (its *rows*), each term keyed
    by its monomial, of any degree;
  - the scale of each row, at which its parameters are defined;
  - how the data points are built from the rows: either data point *i* is
    row *i* as it is, or each data point has an expression combining rows
    through named variables;
  - the theory covmat, a reference point, and the parameters' basis.

  Combined data points can be stored and bookkept, but not evaluated: only
  data points that are a row as it is are implemented.

The container is built to cover two formats, and others of the same kind:

| Concept in the container | SMEFiT JSON layout (implemented) | POPxf (a later loader) |
|---|---|---|
| The rows' terms, keyed by monomial | `best_sm` (the constant), `"a"`, `"a*b"` in the order dict, one value per data point | `"('', 'C1')"`-style keys in `observable_central` (one value per observable) or `polynomial_central` (one per polynomial) |
| Rows and data points | one row per data point, used as it is | single-polynomial mode: the same. Function-of-polynomials mode: K rows (`polynomial_names`) shared by the M observables |
| Combining expression | none | `observable_expressions`: per observable, an `expression` and the `variables` mapping its names to polynomials |
| Scale of each row | `scales`, one per data point | one per file (every row), per polynomial, or per observable in single-polynomial mode |
| Theory covmat | `theory_cov_<type>`, picked by the dataset's `theory_cov` key | per-source uncertainties and correlation files, assembled by the loader |
| Reference point | the new optional `reference_point`; zero otherwise | none in POPxf 1.0, so zero |
| Basis | not stated, so `None` | `basis` |
| Choosing a prediction in the file | the dataset's `order` key | one prediction per file |

- **It is compatible with the SMEFiT layout**: every current file loads into
  it, and the views give back exactly today's arrays.
- **It is compatible with POPxf**: every concept a POPxf loader needs has a
  place in it.
- **It is not POPxf's schema in memory.** Format-specific details are left to
  each loader and do not reach the container. For POPxf these are the `R`/`I`
  tags of complex parameters (mapped to real parameters), per-source
  uncertainties (combined into a covmat) and metadata.
- **A new format of the same kind** (polynomials in named parameters, maybe
  combined) is one more loader, with no change to the container or the
  models.
- **A reference prediction**, at a point the theory defines (zero unless the
  file names another), used by the t0 covmat and the projections.
- **The theory covmat comes from a method**, so a later loader can correlate
  datasets.
- **A lookup of theory formats** behind `load_theory`, with the SMEFiT JSON
  layout as its only entry.
- **The JAX expression compiler in its own module**, checking every expression
  with `ast` before compiling it.
- **`EFTModel` reads its arrays as views** of the polynomial, and refuses a
  theory it cannot use.

Results do not change. The database files do not change either; the SMEFiT
JSON layout only gains one optional field, `reference_point`.

## Decisions this plan builds on

1. **The constant term and the reference prediction are two things.** The
   constant term is the prediction with every parameter at zero, and it is what
   `EFTModel` adds its corrections to. The reference prediction is the
   prediction at the theory's reference point, and it is what t0 and the
   projections use. They are equal for every existing file, since none names a
   reference point. They differ as soon as one does, so the code must not use
   one for the other. The deprecated `sm_pred` alias returns the constant term,
   which is what it has always been (`best_sm`). A combined theory has no
   constant term per data point: its prediction at zero is its expressions
   evaluated there, which is the reference prediction at the default point.
2. **The polynomials are rows, not data points.** A theory's polynomials are a
   stack of K scalar polynomials, as POPxf's `polynomial_names` are: global to
   the dataset, shareable between data points, each with its own scale. A data
   point either *is* a row (the SMEFiT layout, and POPxf's single-polynomial
   mode, with K = n_data), or is a `Combination`: an expression over local
   variable names, each mapped to a row. That mapping is POPxf's `variables`;
   it keeps expressions free of polynomial names, which need not be Python
   identifiers (`"polynomial 1"` in POPxf's own example).
3. **Combinations: storable, not evaluable.** A `Theory` accepts combinations,
   and everything that is bookkeeping works with them: the rows, their scales,
   the group's row offsets. Everything that needs an expression evaluated (the
   reference prediction, the deprecated aliases, `EFTModel`) raises. That keeps
   the interface the analysis designed, makes `EFTModel`'s check testable, and
   implements nothing no caller uses.
4. **`load_theory(theory_path, dataset_dict)` keeps its signature.** The format
   lookup sits behind it, so it needs no deprecation. *This departs from the
   analysis*, which listed today's `load_theory` among the deprecated aliases.
   Nothing about its arguments is SMEFiT-specific, and its return type keeps
   the old attribute names as aliases.
5. **Scales belong to the rows.** `TheoryGroup.scales` is the scale of every
   row, in row order, and it is all `build_rge_matrix` reads, so it computes one
   RGE matrix per row with no change. For today's theories a row is a data
   point, so the matrices are exactly today's. With combinations there is still
   one matrix per polynomial, as POPxf's per-polynomial scales and a σ×BR
   model's separate production and decay scales need. Contracting them is the
   job of the model that combines the rows.
6. **Views are computed, not stored.** `Polynomial.linear(names)` and
   `Polynomial.quadratic(names)` take any list of names and return zero columns
   for names the polynomial does not contain. Today `Theory` builds a dense
   `(n_data, n_ops, n_ops)` tensor per dataset and `TheoryGroup` another for the
   group, even for a linear fit; after this PR only a model asking for it pays.
   The same accessor also replaces the hand-written alignment loops of
   `EFTModel._apply_rge`.
7. **The expression check is a whitelist.** It accepts numbers, names,
   `+ - * / **`, unary `+ -`, and calls to the namespace's functions with
   positional arguments. Comparisons, conditionals and boolean operators are
   rejected too: today they compile, but they fail as soon as `resolve` runs
   under `jax.jit` (Python branching on a traced value), so rejecting them loses
   nothing that works.
8. **Errors raise `ValueError`, never `KeyError`** (analysis, and PR 2's
   contract item 6). This includes the loader's missing-key cases that raise a
   bare `KeyError` today (`best_sm`, `scales`).
9. **Where the code goes.** `Polynomial`, `Theory` and `TheoryGroup` stay in
   `smefit/core.py`, next to the other domain classes, so every import of them
   keeps working. The compiler goes to a new `smefit/expressions.py`, which
   holds no DAG nodes and is not added to `smefit_providers`.
10. **Duplicate monomials warn.** If a file has the same monomial under two keys
       (`a*b` and `b*a`), the loader logs a warning naming the file and the
    keys, and keeps the last one, as today. Results do not change.

## Current state

### What the loader reads

`load_theory` reads `theory_path/<dataset>.json`:

- `best_sm` is the constant term. The order dict's own `"SM"` entry is never
  read, although `database-layout.md` describes it.
- The orders are the top-level keys whose value is a dict.
- In an order dict, a key `a` is a linear term and `a*b` a quadratic one,
  normalised so that `a <= b`. If both `a*b` and `b*a` are present, the last
  one silently wins. A key with three factors fails with an unpacking
  `ValueError` that names nothing.
- A missing `best_sm` or `scales` raises a bare `KeyError`. `produce_chi2` has
  `eft_model=None`, so it swallows it and then reports "Data provided but
  eft_model or fit_covmat is None", which points elsewhere.

### Who reads the theory

| Consumer | Reads |
|---|---|
| `EFTModel._setup_direct` | `operators`, `eft_lin_pred`, `eft_quad_pred`, `names` |
| `EFTModel._apply_rge` | `operators`, `eft_lin_pred`, `eft_quad_pred`, `n_data`, with its own alignment loops |
| `EFTModel.forward_map` | `self.theory.sm_pred`, inside the JIT-compiled function |
| `config._build_fit_covmat` | `names`, `n_data`, `sm_pred` (t0), `sm_covmat` |
| `Projection` | `sm_pred` (when there is no model), `sm_covmat` |
| `build_rge_matrix` (`_resolve_scales`) | `scales` |
| `external_chi2/drell_yan/*.py` (9 modules) | `load_theory`, `TheoryGroup`, `sm_pred` (t0), `sm_covmat` |
| `InputManifest.from_runcard`, `write_pseudodata` | build `theory_path/<dataset>.json` themselves |
| `validate_runcard.py`, `smefit_db.py` (skills) | the orders, as every top-level key except `best_sm`, `scales` and `theory_cov_*` |
| tests | `conftest.py` builds `Theory(sm_pred=…, eft_pred=…, operators=…)`; `test_core.py`, `test_loader.py`, `test_model.py`, `test_projections.py`, `test_fisher.py`, … read the old attributes |

### How expressions are compiled

`Coefficient._expr_fn` builds `lambda <vars>: (<expr>)` and `eval`s it over
`_EXPR_NAMESPACE` in `core.py`, lazily, on the first `constrain`. A name that
is neither in `vars` nor in the namespace is only noticed when the lambda is
called, as a `NameError` inside the JIT trace of `forward_map`.

## Target design

### Layout

```
smefit/expressions.py   COEFFICIENT_FUNCTIONS, check_expression, compile_expression
smefit/core.py          Dataset, Coefficient, the groups; Monomial, Polynomial, Combination, and the new Theory, TheoryGroup
smefit/loader.py        load_dataset; load_theory and the format lookup
```

### `Polynomial`, a stack of rows

```python
Monomial = tuple[str, ...]
"""Sorted parameter names, repeated for powers: () is the constant term,
("a",) is a, ("a", "a") is a**2, ("a", "b") is a*b."""


@dataclass(frozen=True, eq=False)
class Polynomial:
    """K scalar polynomials: row k is the sum over m of terms[m][k] * prod(x[p] for p in m)."""

    terms: Mapping[Monomial, jnp.ndarray]  # each of shape (K,)
    scales: jnp.ndarray  # (K,): the scale at which each row's parameters are defined

    n_rows: int  # property: K
    parameters: tuple[str, ...]  # property: sorted names in any monomial
    degree: int  # property: the longest monomial, 0 if only the constant

    @property
    def constant(self) -> jnp.ndarray: ...  # (K,): terms[()], or zeros

    def linear(self, names: Sequence[str]) -> jnp.ndarray: ...

    # (K, len(names)): column i is terms[(names[i],)], or zeros

    def quadratic(self, names: Sequence[str]) -> jnp.ndarray: ...

    # (K, n, n): [:, i, j] for i <= j is the term of names[i]*names[j],
    # zero below the diagonal and for monomials the polynomial lacks

    def evaluate(self, point: Mapping[str, float]) -> jnp.ndarray: ...

    # (K,): any degree; parameters missing from point are zero

    @classmethod
    def concatenate(cls, polynomials: Sequence["Polynomial"]) -> "Polynomial": ...

    # stacks the rows; a monomial missing from one block is zero there
```

- `__post_init__` checks that every monomial is a sorted tuple of non-empty
  names and that every array has K entries, raising `ValueError`.
- `quadratic(names)` reproduces both of today's layouts: with sorted `names` it
  is `eft_quad_pred` sliced to them, and with the RGE matrix's
  `obs_operators` it is `_apply_rge`'s `quad_aligned`. The values are copied
  from the same floats into the same positions, so the results are identical.
- `evaluate` is what makes the polynomial usable by more than the SMEFT views:
  it gives the reference prediction, and tests use it to check that the views
  and `EFTModel.forward_map` agree with it.
- Rows carry no names. A loader maps a format's polynomial names to row
  indices, and the SMEFiT layout, whose rows are its data points, needs no
  names at all.

### `Combination`, one combined data point

```python
@dataclass(frozen=True)
class Combination:
    expression: str  # over the names of `variables`, e.g. "num / den"
    variables: Mapping[str, int]  # variable name -> row of the theory's polynomial
```

- It is POPxf's `observable_expressions` entry, with the polynomial names
  already resolved to rows.
- `__post_init__` checks, as `ValueError`, that every variable name is a
  Python identifier. The expression itself is stored as given: checking it
  needs the namespace of the format it came from, so it is checked by
  `check_expression` when a model compiles it (see "Not in this PR").

### `Theory`, one dataset

```python
@dataclass
class Theory:
    name: str
    polynomial: Polynomial  # K rows
    covmat: jnp.ndarray  # (n_data, n_data)
    order: str | None = None  # what the SMEFiT layout selected; None for others
    combinations: tuple[Combination, ...] | None = (
        None  # one per data point; None: point i is row i
    )
    reference_point: Mapping[str, float] = field(default_factory=dict)
    basis: str | None = None  # the parameters' basis, if the file names one
```

| Member | Meaning |
|---|---|
| `n_data` | K without combinations, otherwise their number |
| `combined` | True when `combinations` is set |
| `parameters` | `polynomial.parameters` |
| `scales` | `polynomial.scales`: one per row, defined in both cases |
| `reference_prediction` | Without combinations: with no `reference_point`, `polynomial.constant`, returned as it is; otherwise `polynomial.evaluate(reference_point)`. With combinations: `NotImplementedError`. |

- **The empty reference point short-circuits.** `evaluate` at zero would
  compute `constant + term * 0`, which turns a NaN anywhere in the tables into
  a NaN in t0. Returning the constant keeps t0 bit-identical.
- **Invariants, in `__post_init__`, as `ValueError`:** `covmat` is
  `(n_data, n_data)`; every row a combination names exists; every name in
  `reference_point` is a parameter of the theory, so a typo is caught rather
  than silently ignored.
- **The SMEFiT layout** gives one row per data point and no combinations.

### `TheoryGroup`, the fit's datasets

| Member | Meaning |
|---|---|
| `theories`, `names`, `n_data` | As today; theories sorted by name |
| `parameters` | Sorted union over the theories; replaces `operators` |
| `polynomial` | `Polynomial.concatenate` of the theories' polynomials: every row of every dataset, in dataset order |
| `scales` | `polynomial.scales`, so one per row, always defined; for today's theories the per-data-point scales |
| `combined` | True when any theory has combinations |
| `combinations` | None when no theory is combined. Otherwise one `Combination` per data point of the group, with rows offset to the group's polynomial, and each data point that is a row written as `Combination("p", {"p": row})`. Bookkeeping only: nothing evaluates it in this PR. |
| `reference_prediction` | The concatenation of each theory's own, so `NotImplementedError` if any is combined |
| `covmat()` | Block-diagonal of the theories' covmats, computed once. A method, so a later loader can add cross-dataset correlations. |
| `basis` | The theories' common basis, `None` if none names one, `ValueError` if two disagree |

### Deprecated aliases

On both `Theory` and `TheoryGroup`, each raising a `DeprecationWarning`, as
PR 1 and PR 2 do for their renames. The four array aliases describe data
points, so they raise a `ValueError` for a combined theory:

| Old | Returns |
|---|---|
| `sm_pred` | `polynomial.constant` (see Decision 1) |
| `sm_covmat` | `covmat` (`covmat()` on the group) |
| `eft_lin_pred` | `polynomial.linear(parameters)` |
| `eft_quad_pred` | `polynomial.quadratic(parameters)` |
| `operators` | `parameters` |

`Theory.eft_pred`, `n_ops` and `op_index` are dropped without an alias: only
tests read them. The old `Theory(...)` constructor arguments go too; only
`conftest.py` and `load_theory` build a `Theory`.

### `EFTModel`

- **It checks the theory first** (PR 2's contract item 5), raising
  `ValueError`:
  - no dataset is combined, naming those that are;
  - the degree is at most 2, naming a dataset and one offending monomial.

  With `use_quad: False` the quadratic terms are ignored, as today. That is the
  user's choice.
- **It reads views.** `p = theory.polynomial`, whose rows are then the data
  points. `_setup_direct` takes
  `p.linear(active_names)` and, with `use_quad`, `p.quadratic(active_names)`.
  `_apply_rge` takes `p.linear(rge_obs_ops)` and `p.quadratic(rge_obs_ops)`,
  and its alignment loops go.
- **`forward_map` adds `self.constant`**, stored at construction, instead of
  reading `self.theory.sm_pred` inside the JIT trace.
- The constructor signature and the warnings about inactive coefficients do
  not change; the warnings read `theory.parameters`.

### The loader, `smefit/loader.py`

```python
class TheoryFormat(NamedTuple):
    matches: Callable[[Mapping], bool]  # does this file content use the format?
    load: Callable[
        [Mapping, Mapping, pathlib.Path], Theory
    ]  # (content, dataset_dict, file)


_THEORY_FORMATS = {"smefit_json": TheoryFormat(_is_smefit_json, _load_smefit_json)}


def load_theory(theory_path, dataset_dict) -> Theory: ...
```

- **Dispatch on content.** `load_theory` reads `theory_path/<dataset>.json`,
  then picks the one format whose `matches` accepts it. A file declaring a
  `$schema` that no format matches, such as a POPxf file, raises a
  `ValueError` naming the schema and the formats smefit reads. A file no format
  matches, or more than one, raises too.
- **The SMEFiT JSON format** matches a file with `best_sm`. It keeps every
  current behaviour, except the silent ones:
  - `best_sm` is the constant term; the order dict's `"SM"` is still skipped;
  - `a` and `a*b*…` are monomials of any degree, factors sorted;
  - an empty factor (`a*`) raises a `ValueError` naming the file and the key;
  - the same monomial twice (`a*b` and `b*a`) logs a warning and keeps the
    last, as today (Decision 10);
  - `scales` becomes the scale of each row;
  - `theory_cov` and `order` are handled as today, missing `best_sm` or
    `scales` raise a `ValueError` naming the file;
  - the optional `reference_point` is a mapping from parameter to number;
  - `basis` is `None`: the layout has no field for it.
- **Reserved keys.** The orders are the dict-valued keys outside
  `_SMEFIT_JSON_RESERVED = {"best_sm", "scales", "reference_point"}` and the
  `theory_cov_*` prefix. Without this, `reference_point` would be listed as an
  available order. `validate_runcard.py` and `smefit_db.py` detect the orders
  the same way and get the same exclusion.
- **The file location is unchanged**: `theory_path/<dataset>.json` for every
  format, and `InputManifest.from_runcard` and `write_pseudodata` keep building
  that path themselves. `input_manifest.py` deliberately imports nothing heavier
  than `smefit.paths`, so it should not import the loader. The first format with
  a different layout moves the path into a light shared helper.

### The expression compiler, `smefit/expressions.py`

```python
COEFFICIENT_FUNCTIONS: Mapping[str, Any] = MappingProxyType(
    {
        "abs": jnp.abs,
        "sqrt": jnp.sqrt,
        "exp": jnp.exp,
        "log": jnp.log,
        "sin": jnp.sin,
        "cos": jnp.cos,
        "tan": jnp.tan,
        "pi": float(jnp.pi),
    }
)


def check_expression(
    expr: str, variables: Sequence[str], namespace: Mapping[str, Any]
) -> None:
    """Raise ValueError unless expr uses only the accepted grammar (Decision 7)."""


def compile_expression(expr, variables, namespace) -> Callable[..., jnp.ndarray]:
    """check_expression, then eval of a lambda over {"__builtins__": {}, **namespace}."""
```

- **The grammar.** Accepted: integer and float literals; names that are in
  `variables` or are non-callable constants of the namespace (`pi`); calls whose
  function is a callable of the namespace, positional arguments only; binary
  `+ - * / **`; unary `+ -`. Anything else is a `ValueError` naming the
  construct and the expression: attributes, subscripts, strings, keywords,
  `%`, `//`, comparisons, `and`/`or`, conditionals, lambdas, comprehensions.
- **`Coefficient` checks in `__post_init__`**, with `COEFFICIENT_FUNCTIONS`. A
  name outside `vars` is now a `ValueError` from `parse_coefficients` instead
  of a `NameError` inside the JIT trace. `_expr_fn` calls
  `compile_expression`.
- **`_EXPR_NAMESPACE` goes.** It is private, and no test imports it.
- **The coefficient `expr:` language does not change**: the same functions,
  and every form the templates and docs use (`1/m**2`, `100*y`,
  `y**2 + 0.5*OpWB**2`) is accepted.
- **POPxf later passes its own namespace** for the expressions of
  `Combination`, with a `Combination`'s variable names as `variables`. Nothing
  in this PR compiles them.

### `config.py` and `projections.py`

- `_build_fit_covmat`: t0 uses `theory.reference_prediction`, and the theory
  covmat is `theory.covmat()`.
- `Projection`: `reference_prediction` when there is no model, and
  `theory.covmat()`.
- `produce_theory` does not change.

### The Drell-Yan plugins

The 9 modules use `reference_prediction` for their t0 and `covmat()`. Their
imports do not change. Their results do not change, since
both equal the old attributes for their files.

## Commits

Every commit:

- leaves `pytest -m "not slow"` green;
- updates the docs it makes stale in the same commit;
- leaves `python scripts/generate_skill_reference.py` producing no diff, unless
  it changes a `parse_` docstring, in which case the regenerated files are
  committed with it;
- passes `pre-commit run --all-files`.

### 1. Move the expression compiler into `smefit/expressions.py`

It is independent of the theory work, so it goes first and can be reviewed on
its own.

- Add the module. `Coefficient` checks in `__post_init__` and compiles through
  it; `_EXPR_NAMESPACE` goes.
- Tests: the grammar (see the Tests section), and every `expr:` of the template
  runcards passes the check.
- Docs:
  - `AGENTS.md`, "Coefficient constraints", and the module list;
  - the "Expression constraints" bullet of the `smefit-dev` skill: the language
    is `COEFFICIENT_FUNCTIONS` in `smefit/expressions.py`, checked by
    `check_expression`;
  - `smefit-runcard/references/coefficients.md`: the accepted grammar, and that
    a name outside `vars` is now refused when the runcard is read;
  - `validate_runcard.py`: `_NONLINEAR_FUNCS` stays, and a test now pins it to
    the callables of `COEFFICIENT_FUNCTIONS`.

### 2. Add the theory container and read the arrays as views

- In `smefit/core.py`, add `Polynomial` and rewrite `Theory` and `TheoryGroup`,
  with the deprecated aliases.
- `_load_smefit_json` builds the new `Theory`, with the new key checks (empty
  factor, missing `best_sm`/`scales`) and the duplicate-monomial warning. It stays called
  directly by `load_theory` in this commit.
- Move every consumer to the new names: `EFTModel` (views, `self.constant`, no
  alignment loops), `_build_fit_covmat` and `Projection` (still on
  `polynomial.constant` in this commit, which is `sm_pred`), and the Drell-Yan
  modules (`covmat()`; `polynomial.constant` for their t0, until commit 4).
- Tests: `conftest.py` builds the fixtures with `Polynomial`; the tests of the
  old attributes move to the new ones; new container tests.
- Docs: the "EFT predictions" section of `AGENTS.md` becomes a "Theory
  predictions" section describing the polynomial and the views; the `core.py`
  and `loader.py` entries of the module list.

### 3. `EFTModel` checks the theory it is given

- The two checks, with `ValueError`.
- Tests: through `produce_eft_model` and through `produce_pseudodata` (the
  consumer whose `eft_model=None` would swallow a `KeyError`), a degree-3
  theory and a combined theory both raise `ValueError`.
- Docs: the `BaseModel` docstring's item 5 no longer says "lands in PR 3".

### 4. Add the reference point and the reference prediction

- `reference_point` in the SMEFiT JSON layout, the reserved keys, and
  `Theory.reference_prediction`.
- `_build_fit_covmat`, `Projection` and the Drell-Yan t0 switch from
  `polynomial.constant` to `reference_prediction`. This is the commit where the
  two concepts separate, so it is the one to review for Decision 1.
- `validate_runcard.py` and `smefit_db.py` exclude `reference_point` from the
  orders; a test pins their reserved set to `_SMEFIT_JSON_RESERVED`.
- Docs: `smefit-datasets/references/database-layout.md` documents
  `reference_point`, says that `best_sm` (not the order's `"SM"`) is the
  constant term, and that keys may have any number of factors, while the SMEFT
  model takes at most two.

### 5. Add the format lookup

- `TheoryFormat`, `_THEORY_FORMATS`, the dispatch in `load_theory`, and the
  `$schema` error.
- Tests: dispatch, the `$schema` error, a file no format matches.
- Docs: a short "Adding a theory format" paragraph in the `smefit-dev` skill:
  one `TheoryFormat` entry, `ValueError` never `KeyError`, and the file
  location note above.

## Tests

| File | Change |
|---|---|
| `tests/test_expressions.py` (new) | **Accepted:** every template `expr:`, `pi`, nested calls, unary minus, powers. **Refused, each with the construct in the message:** `().__class__`, `x[0]`, `"a"`, `f(x=1)`, `x % 2`, `x if x > 0 else -x`, `x and y`, `lambda: 1`, a name outside `variables`, calling a variable, calling `pi`. **Compiled:** values and `jax.grad` match `jnp` written by hand; a variable named like a namespace function shadows it, as in Python. |
| `tests/test_core.py` | `Coefficient` with an `expr:` naming something outside `vars` raises in `__post_init__`. **`Polynomial`:** monomial validation, `degree`, `constant`, `linear` and `quadratic` with absent names and with unsorted names (the `_apply_rge` layout), `evaluate` at degree 3, `concatenate` with disjoint monomials. **`Combination`:** a variable that is not an identifier raises. **`Theory`:** each invariant raises `ValueError`, including a combination naming a missing row; `n_data` with and without combinations; `reference_prediction` with and without a point, and bit-identical to `constant` with none, even with a NaN term; `NotImplementedError` when combined. **`TheoryGroup`:** today's union and block-diagonal tests on the new names; a group mixing a plain and a combined theory (two data points sharing one of three rows): `polynomial` stacks the rows, `scales` has one entry per row, `combinations` offsets the rows and writes the plain points as identity, `reference_prediction` raises; `basis` conflicts. **Aliases:** each warns and equals what it replaces, and the array aliases raise when combined. |
| `tests/test_loader.py` | Monomials of degree 3 load; an empty factor raises; a duplicate monomial warns and keeps the last; missing `best_sm`/`scales` raise `ValueError`; `reference_point` is not an order and an unknown name in it raises; format dispatch and the `$schema` error. |
| `tests/test_model.py` / `test_smeft_model.py` (PR 2's name) | `forward_map` equals `polynomial.evaluate` at random points, with and without `use_quad`; degree 3 and a combined theory raise; the RGE path gives the same `lin_corr`/`quad_corr` as before on a stub matrix whose `obs_operators` include a name the theory lacks. |
| `tests/test_config.py` | t0 uses `reference_prediction`: a fixture with a `reference_point` changes t0 and nothing else. The two `ValueError` cases through `produce_pseudodata`. |
| `tests/test_projections.py` | The SM-only projection uses `reference_prediction`. |
| `tests/test_skill_scripts.py` | `_NONLINEAR_FUNCS` matches `COEFFICIENT_FUNCTIONS`; the reserved keys of both scripts match the loader's; `validate_runcard.py` does not list `reference_point` as an order. |
| Every other test reading the old attributes (`test_fisher.py`, `test_chi2.py`, `test_analytic_fit.py`, `test_hessian_fit.py`, `test_pca.py`) | Moves to the new names, so the suite runs with no `DeprecationWarning` of ours. |

## Verification

1. After every commit, run `pytest -m "not slow"`. After commits 2 and 4, also
   run the full `pytest`.
2. **Sweep the database.** Before commit 2 is merged, load every theory file of
   the local `smefit_database`, at every order, with the new loader. None may
   raise; list the files that log the duplicate-monomial warning. For each, `sm_pred`, `eft_lin_pred` and `eft_quad_pred` computed on
   `generalize-beyond-eft` (dumped with a scratch script) must equal the new
   views exactly.
3. **Regression against the integration branch.** Run, on both branches, and
   compare `fit_results.json`, which must be identical:
   - `template_runcards/analytical_fit.yaml` (linear, no RGE);
   - `template_runcards/hessian_fit.yaml` (`use_quad`);
   - `template_runcards/rge.yaml` (compare the `rge_matrix.pkl` frames too);
   - a scratch copy of `analytical_fit.yaml` with `use_t0: True` and
     `use_theory_covmat: True`;
   - a pseudodata template; the written pseudodata must be identical.
4. **External chi2.** A runcard with one Drell-Yan plugin and `use_t0` inside
   its entry gives the same `fit_results.json` on both branches.
5. **Stored fits.** A report over stored fits (`fits:` with `fitcontext`)
   rebuilds as before. The theory files and their location do not change, so
   the input manifests still match.
6. Regenerating the skill references matches what is committed, and CI's
   skills check passes.

## Not in this PR

- **A POPxf loader**, and with it complex parameters, the POPxf function
  namespace, and how a database lays out POPxf files.
- **Evaluating combinations**: compiling their expressions with a format's
  namespace, the reference prediction of a combined observable, and a model
  that contracts one RGE matrix per row (Decision 5).
- **A model evaluating degrees above 2.** `EFTModel` refuses them.
- **Cross-dataset theory correlations, and uncertainties that depend on the
  parameters.** `covmat()` is where the first would go.
- **A `basis` field in the SMEFiT JSON layout, and a model checking it.** The
  container stores a basis; no current file names one.
- **Neutral names outside the theory**: `eft_model` and the RGE field names
  are PR 5.
- **Turning the Drell-Yan plugins into ordinary datasets.**
