# Plan: smelli surrogate with smefit operators, jointly with Drell-Yan

Status: notes only, nothing implemented. Module:
`external_chi2/smelli_surrogate/smelli_surrogate.py` (+ `surrogate/`, `tables/`).

## Where it stands

- Input is the NFU Z' model only. `_ZprimeBlock.__init__` reads the 10
  `uv.UV_PARAMS` (`g, tsb, tdb, tds, delta, phi1..phi5`) from `coefficients:`;
  any other coefficient is **silently ignored** (a runcard with `OpD`, `Oeu`, …
  runs, but the surrogate chi2 does not move with them).
- Downstream is already SMEFT-shaped: `_warsaw_vector` gives a Warsaw (wilson
  names, Re/Im slots) vector at `M = 3000 GeV`, then
  - flavour blocks (`S2`, `S2L`): `_W2J` = Warsaw(M) → JMS(M_Z), a linear matrix
    built with `wilson.match_run` (`smeft_accuracy='integrate'`), cached as
    `tables/<SUPPORT>/w2j_<hash>.npy`; the tables live in JMS at `M_Z`.
  - EW blocks (`EW1`): pick 36 real flavour-diagonal Warsaw coordinates at
    3 TeV; the observables are exact quadratic forms in them (running 3 TeV →
    EW scale is inside the tables). `_EWBlock.__init__` raises if table scale ≠ M.
- `rge_dict` is accepted everywhere and never used.

## 1. smefit operators as input

Replace the UV → Warsaw step (`_uv` + `uv.smeft_wcs`) with smefit coefficients →
Warsaw via `smefit/wcxf.py::wcxf_translate`. Everything after `_warsaw_vector`
stays. Keep the Z' path as a mode (e.g. `input: zprime | smeft`).

Should also **fail loudly** on a free coefficient the block does not consume,
instead of the current silent drop.

## 2. Operator coverage (user will extend the tables)

Tables span only the Z' directions: 84 Warsaw four-fermions; EW coords are
those plus `phiD, phiBox, phil1_11/22, phie_11/22, phiq1_33, phiu_33, phid_33`.
Missing what DY needs (`phiWB`, `phil3`, light-quark currents, most
flavour-universal four-fermions). Rebuild in zprime_NFU `left_surrogate`.

Once extended: a smefit operator whose Warsaw image leaves the table span must
error, not be projected away.

## 3. Scale toggle from smefit — no table rebuild needed

The tables never have to be recomputed; only the linear maps in front of them.

- **Flavour blocks**: keep `_W2J` (wilson run+match 3 TeV → `M_Z`) fixed at
  the table scale and pre-multiply by R(Λ → 3 TeV). (Simplest alternative:
  start `_W2J` at Λ — `2 × n_warsaw` wilson calls, ~1 min per Λ; superseded
  by the speed bullet below.)
- **EW blocks**: pre-multiply by the same Warsaw(Λ) → Warsaw(3 TeV) running
  matrix R (from rgevolve, see below). Exact: dim-6 one-loop running is linear in C and
  the observables are quadratic in the 3 TeV coordinates, so the composition is
  still quadratic. Works for Λ < 3 TeV too (runs upward). Drop the
  `scale ≠ M` check in SMEFT mode.
- **Truncation**: running from Λ mixes into Warsaw (EW) / JMS (flavour)
  directions outside the table coordinates; those are dropped (the W2J loop
  already only keeps `jms_names`). Small, log-suppressed for Λ near 3 TeV;
  measure it (compare against wilson+smelli at a few Λ) and warn above a
  threshold. Shrinks as the coverage in §2 grows.
- **smefit side**: take Λ from `rge_dict["init_scale"]` (already passed to the
  class). Do *not* reuse smefit's `build_rge_matrix`: its output is projected
  on the smefit basis at `obs_scale` and would lose the flavour-specific Warsaw
  entries the surrogate needs. So the surrogate runs itself, from Λ, whatever
  `obs_scale` is.
- **Open**: meaning of "no `rge:` block" in SMEFT mode. The flavour blocks
  cannot skip running to M_Z (it is part of the matching), so options are
  Λ = 3 TeV default, or require `rge:`. Decide.
- **Mass scans**: the RG matrices are rebuilt per scan point (see
  `individual_mass_ext_chi2_func`) → one W2J + one R per point; fine thanks to
  the cache after the first pass, slow on the first.
- **Speed: rgevolve for the running, wilson for the matching** (benchmarked
  2026-09-30 on the Z' Warsaw set, 120 real/imag inputs, scripts were in the
  session scratchpad):
  - SMEFT→SMEFT Warsaw Λ→3 TeV: rgevolve `run_and_match` 0.1–0.2 s vs wilson
    25 s; max |R_rgevolve − R_wilson| = 3e-4 against a running effect of
    6–7e-2 (0.5 %), for Λ = 1 and 10 TeV. Use rgevolve for R.
  - SMEFT(3 TeV)→JMS(M_Z) (`_W2J`): rgevolve 0.3 s vs wilson ~60 s, but they
    **disagree**: direct tree-level entries by 1–6 %, loop-induced ones by
    ×1.5–4.5, some in a different direction. Likely the CKM override the
    surrogate passes to wilson (rgevolve's SM inputs are frozen) plus
    below-EW running/matching conventions. The tables come from smelli, i.e.
    from wilson, so keep the wilson `_W2J` at the fixed table scale (computed
    once, cached).
  - Structure: chi2 input = `W2J(3 TeV→M_Z) · R(Λ→3 TeV) · c` for the flavour
    blocks, `R(Λ→3 TeV) · c` for EW; one R shared by both block types.
  - rgevolve rejects Im parts of hermiticity-real coefficients
    (`dd_3333_I`, …): drop those columns (they are exactly zero in W2J).
    Needs `import importlib.resources` before the rgevolve import on 3.14
    (as in `beta_decays_ext_LL`'s `superallowed_beta_decay.py`).
- Consistency with DY: DY modules run with smefit's RGE from the same
  `init_scale`; check both use `integrate`-equivalent accuracy (README: 
  `leadinglog` moves EWPT by 0.08, quarks by ~0.05 at g ~ 0.2–0.3).

## 4. Flavour conventions (open, keep in mind)

- smefit flavour-universal / U(2) operators expand to several flavour-indexed
  Warsaw entries; `wcxf_translate` gives the expansion — check every entry it
  produces is in the table span (§2).
- smefit coefficients are real: only the Re slots of the Warsaw vector get
  filled; Im slots stay zero. Fine unless CP-odd directions are wanted.
- wilson symmetrisation / index-ordering conventions (e.g. `qq1_1221` vs
  `qq1_2112`, `ll_1122` normalisation) vs smefit's definitions — verify the
  factors, ideally with a round-trip test at a few points against smelli.
- Mismatch with DY: the ATLAS DY theory tables currently use inconsistent
  operator names (see PR note: `ATLAS_DYMmm` carries electron-flavour names,
  `ATLAS_DYMee` has `*_big`/`*_small` names). The joint fit needs one naming
  convention across DY and the surrogate.

## Tests (ship with each step)

- SMEFT mode reproduces Z' mode when the smefit coefficients are set to the
  Warsaw point a given Z' parameter set generates.
- Λ = 3 TeV reproduces the current tables to machine precision; Λ ≠ 3 TeV
  agrees with wilson+smelli within the README tolerances.
- Unconsumed / out-of-span coefficient raises.
