# smefit output layout

`smefit <runcard.yaml> [-o DIR]` writes everything under one output directory
(default: the runcard filename stem, created in the working directory).

```
<output>/
  input/
    runcard.yaml          # copy of the runcard (reportengine bookkeeping;
                          #  required later by bayesian_update_path)
    lockfile.yaml         # reportengine lockfile
  figures/  tables/       # created on every run; populated by report actions
  fit_results.json        # written by every run_*_fit action
  individual_fits/        # only for run_individual_*_fits actions
    <coefficient>/
      fit_results.json    # single-coefficient result, same schema
  ultranest_logs/         # UltraNest working dir (resume data, diagnostics)
  blackjax_logs/          # BlackJAX log dir
    nested_samples.csv    #  algorithm: nested_sampling — anesthetic NestedSamples
    nuts_samples.csv      #  algorithm: nuts — thinned draws, one column per parameter
    nuts_diagnostics.json #  algorithm: nuts — R-hat, ESS, divergences, step sizes
  rge_matrix.pkl          # RGE matrices per scale, saved when an rge block ran;
                          #  reusable via rge.rg_matrix in later runcards
  pseudodata/             # written by the write_pseudodata action
    <dataset>.yaml        # projected commondata (+ theory JSON copies for _proj sets)
  index.html              # report action output
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
| `whitening_matrix`, `whitening_active` | set when the fit ran in whitened space |

Load it back in Python with `FitResult.from_json("<output>")` (takes the
directory, not the file), or plain `json.load` for quick lookups.

## Individual fits

`run_individual_*_fits` writes one subdirectory per free coefficient under
`individual_fits/`, each with the full schema above, plus a merged top-level
`fit_results.json` marked `"individual_fit": true` where `chi2`, `chi2_ndof`,
and `logz` become per-coefficient dictionaries.

## Reading constraints from samples

```python
import json, numpy as np

d = json.load(open("<output>/fit_results.json"))
s = np.array(d["samples"]["OpWB"])
lo, hi = np.percentile(s, [2.5, 97.5])  # 95% interval
```
