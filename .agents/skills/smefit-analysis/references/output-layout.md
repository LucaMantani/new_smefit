# smefit output layout

`smefit <runcard.yaml> [-o DIR]` writes everything under one output directory
(default: the runcard filename stem, created in the working directory).

```
<output>/
  input/
    runcard.yaml          # copy of the runcard (reportengine bookkeeping;
                          #  required later by bayesian_update_path)
    lockfile.yaml         # reportengine lockfile
  figures/                # created on every run; populated by report actions
    <name>.pdf, .png      #  every figure, in both formats
  tables/                 # likewise
    <name>.csv            #  every table, tab separated
    <name>.tex            #  LaTeX of it, for the tables that offer one
                          #  (coefficient_bounds_table); needs tabularx,
                          #  multirow and amssymb, as its comment says
  fit_results.json        # written by every run_*_fit action
  individual_fits/        # only for run_individual_*_fits actions
    <coefficient>/
      fit_results.json    # single-coefficient result, same schema
  ultranest_logs/         # UltraNest working dir (resume data, diagnostics)
  blackjax_logs/          # BlackJAX log dir
  rge_matrix.pkl          # RGE matrices per scale, saved when an rge block ran;
                          #  reusable via rge.rg_matrix in later runcards
  pseudodata/             # written by the write_pseudodata action
    <dataset>.yaml        # projected commondata (+ theory JSON copies for _proj sets)
  index.html              # report action output, alongside the index.md it was
  index.md                #  rendered from, plus report.css / report.template
```

## fit_results.json schema (joint fit)

Written by `FitResult.write` (`smefit/fit_result.py`):

| Field | Meaning |
|---|---|
| `free_parameters` | names of the fitted coefficients |
| `num_data`, `n_free`, `ndof` | data points, free parameters, `num_data - n_free` |
| `max_loglikelihood` | log-likelihood at the best-fit point |
| `chi2`, `chi2_ndof` | chi2 at best fit and per degree of freedom |
| `logz` | log-evidence (nested sampling only, else null) |
| `best_fit_point` | `{coefficient: value}` — includes derived (constrained) coefficients |
| `std` | `{coefficient: standard deviation}` |
| `bic`, `aic` | information criteria |
| `samples` | `{coefficient: [posterior samples]}` — null for pure best-fit runs |
| `prior_specs` | the prior each free coefficient used |
| `whitening_transformation`, `whitening_active` | set when the fit ran in whitened space |

Load a fit back in Python with `Fit.from_folder("<output>")` (takes the
directory, not the file): it reads this file *and* `input/runcard.yaml`, so the
fit comes back with how it was configured — `use_quad`, `fit_type`,
`individual_fit` — alongside its numbers, under `.fit_results`.
`FitResult.from_json("<output>")` reads this file alone, and plain `json.load`
does for quick lookups.

## fit_results.json schema (individual fits)

`run_individual_*_fits` writes one subdirectory per free coefficient under
`individual_fits/`, each a single-coefficient result with the full schema
above, plus a merged top-level `fit_results.json` written by
`FitResultGroup.write_summary`:

| Field | Meaning |
|---|---|
| `free_parameters` | the coefficients fitted, each on its own |
| `num_data`, `n_free` | data points, and how many coefficients were fitted |
| `best_fit_point` | `{coefficient: value}` |
| `std` | `{coefficient: standard deviation}` |
| `chi2`, `chi2_ndof`, `logz` | per-coefficient dictionaries here, not scalars |
| `samples` | `{coefficient: [posterior samples]}` — null for pure best-fit runs |
| `prior_specs` | the prior each coefficient used |
| `whitening_active` | whether the fits ran in whitened space |

Nothing in the payload records that the fit was run one coefficient at a time —
that is read off the `run_individual_*_fits` action in `input/runcard.yaml`,
which is why `Fit.from_folder` needs the whole directory. It hands back a
`FitResultGroup`, one genuine single-parameter `FitResult` per coefficient.

## Reading constraints from samples

```python
import json, numpy as np

d = json.load(open("<output>/fit_results.json"))
s = np.array(d["samples"]["OpWB"])
lo, hi = np.percentile(s, [2.5, 97.5])  # 95% interval
```
