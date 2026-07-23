# Toggling experimental uncertainties per dataset

Two optional keys, `stat_unc` and `syst_unc`, can be set on any dataset entry
in a runcard to opt out of statistical or systematic uncertainties for that
dataset. This mirrors the existing `theory_cov: zero` opt-out for the theory
covariance matrix.

```yaml
datasets:
  - {name: DATASET_A, order: LO, stat_unc: zero}                                    # stat off
  - {name: DATASET_B, order: LO, syst_unc: zero}                                    # syst off
  - {name: DATASET_C, order: LO, stat_unc: zero, syst_unc: zero, theory_cov: zero}  # everything off
```

- Default (key absent, or explicit `current`): the uncertainty is used as loaded
  from the dataset file — unchanged behavior.
- `zero`: the corresponding uncertainty is set to zero for that dataset only.
  A warning is logged whenever this happens.
- Any other value raises a `ValueError`.

## Singular covariance matrix guard

Zeroing out both `stat_unc` and `syst_unc` (and `theory_cov`) for a dataset
makes its diagonal block of the fit covariance matrix all zeros, i.e.
singular — which would kill the fit. `produce_fit_covmat` checks every
dataset's diagonal block individually and raises a single `ValueError`
listing **all** datasets with a singular block, rather than stopping at the
first one found.
