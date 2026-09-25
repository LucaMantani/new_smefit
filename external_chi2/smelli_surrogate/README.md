# `smelli_surrogate` — smelli's likelihoods as a smefit external chi2

`smelli_surrogate.py` provides four external-chi2 classes, each reproducing one
smelli likelihood from a precomputed surrogate:

| runcard key / class | smelli block | observables | SM chi2 | tables |
| --- | --- | --- | --- | --- |
| `fast_likelihood_quarks` | `fast_likelihood_quarks.yaml` | 352 | 501.468 | `S2` |
| `likelihood_lfu_fcnc` | `likelihood_lfu_fcnc.yaml` (q2 <= 6) | 17 | 23.514 | `S2L` |
| `likelihood_ewpt` | `likelihood_ewpt.yaml` | 28 | 36.772 | `EW1` |
| `likelihood_eell` | `likelihood_eell.yaml` (LEP2) | 148 | 149.816 | `EW1` |

Input is the **NFU Z' model** of `Zp_model.yaml` (parameters `g`, `tsb`, `tdb`,
`tds`, `delta`, `phi1`..`phi5`); a fit floats whichever of them the runcard's
`coefficients:` names.

```yaml
external_chi2:
  fast_likelihood_quarks:
    path: <...>/external_chi2/smelli_surrogate/smelli_surrogate.py
  likelihood_lfu_fcnc:
    path: <...>/external_chi2/smelli_surrogate/smelli_surrogate.py
  # likelihood_ewpt, likelihood_eell: one line each, the same
```

Optional per block: `model`, `tables` (both default to the copies here) and, for
the flavour blocks, `penalty`, `dg_tol`.

## Self-contained

Everything is in this directory and nothing outside it is read:

    smelli_surrogate.py   the four external-chi2 classes
    surrogate/            the evaluator (numpy + jax only, no flavio/smelli)
    tables/               S2, S2L, EW1 — 39 MB
    Zp_model.yaml         the NFU Z' model card the tables were built for

A fit needs only this plus jax; `wilson` is imported once per new table set to
build the Warsaw -> JMS matrix, which is then cached in `tables/`.

The tables are a frozen product of the zprime_NFU analysis
(`code/python_scripts/left_surrogate`, `PLAN.md` §12) — rebuild or extend them
there and copy the result over.

## Accuracy

Against smelli's own `chi2_dict` at the SM and at 8 Z' points with g in
[0.02, 0.29] and random angles/phases:

| block | max difference |
| --- | --- |
| quarks | <= 7e-3 for g <= 0.11, 1.9e-2 at g = 0.29; 0.13 at one point whose chi2 is ~105 above the SM |
| LFU | 1e-4 |
| EWPT | 1.6e-3 |
| LEP2 | 1.2e-5 |

SM point: 711.5696 here against smelli's 711.5695.

Two things a reference computation must get right, both measured:

* run the SMEFT with `smeft_accuracy='integrate'` (wilson's default,
  `leadinglog`, moves EWPT by 0.08 and the quark block by ~0.05 at g ~ 0.2-0.3);
* compare in the JMS basis or with the SMEFT Wilson object — a flavio-basis WET
  dictionary is a different point once smelli re-extracts the CKM.
