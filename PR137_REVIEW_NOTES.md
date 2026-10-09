# PR #137 (HDI credible intervals): Luca's review, state of work

Status as of 2026-10-07. Branch `hdi_credible_intervals`, head `7d31a6c`, pushed and
in sync with `origin`. **Nothing in the repository has changed since that commit,
and nothing is committed.** All of the work below is prototypes and measurements
outside the repository.

## Luca's review: what he asked for, and status

Review state: CHANGES_REQUESTED (2026-10-06).

| # | Item | Status |
|---|---|---|
| 1 | **Blocking:** the multimodal HDI path reports far more mass than the level | redesigned and prototyped, see below. One decision is open (*Open decision 1*). |
| 1a | Spurious mode count in heavily resampled unimodal posteriors | depends on *Open decision 1* |
| 1b | Gluing does not conserve mass: replace it with per-mode smooth densities thresholded at one common level | prototyped (`proposed_v2`), fixes the mass |
| 1c | Tests: bound the mass from both sides; add a resampled unimodal posterior with a narrow dip as a regression | not done; the benchmark cases below are the material |
| 2 | Import arviz and getdist at module level | not done (trivial) |
| 3 | Bayesian updates: `ExactPosteriorPrior` stores `dist: exact_posterior`, so the bound at 0 is lost | *Open decision 2* |
| 4 | Add `hdi` next to `eti` in `template_runcards/coefficient_bounds_table.yaml` | not done (trivial) |
| 5 | Inline, `fit_result.py:189`: move `_INTERVAL_TYPES` into `credible_intervals.py` | not done (mechanical, see "Remaining mechanical changes") |

No reply has been posted to Luca on GitHub.

### His failing case

He ran `template_runcards/coefficient_bounds_table.yaml` with `interval_types: [eti, hdi]`
on the `ultranest_fit` template output:

| Coefficient | Level | HDI reported | Mass inside | Expected |
|---|---|---|---|---|
| `OpWB` | 68% | `[-0.0137, 0.004] ∪ [0.0041, 0.0115]` | 95.1% | `[-0.0068, 0.0055]`, 68.2% |
| `y` | 95% | `[-0.0659, -0.0551] ∪ [-0.0548, 0.0225]` | 99.5% | `[-0.0543, 0.013]`, 95.0% |

Only 1241 of his 3461 draws were distinct. In his diagnosis, ISJ returned about 150
one-bin spikes, and the min/max gluing filled every gap between them.

## Reproduction

- **The template fit run here passes.** Re-running the `ultranest_fit` template (17 s)
  gave 1223 distinct draws out of 3456. With the current code every coefficient gets
  one interval, holding 0.677–0.681 (68%) and 0.948–0.953 (95%) of the draws. The
  failure depends on the particular sample, and his triggered it.
- **On simulated draws at his level of repetition it fails often.** The draws are
  N(0, 1) targets importance-resampled with replacement from wide proposals, n = 3460:

  | repetition | level | current code: wrong piece count | mass in pieces |
  |---|---|---|---|
  | 54% distinct, 200 samples | 68% | 26 | up to 0.949 |
  | | 95% | 11 | up to 0.993 |
  | 35% distinct, 200 samples | 68% | 66 | up to 0.969 |
  | | 95% | 19 | up to 0.999 |

  This is the limitation of the gluing, and on resampled samples it is not rare. Our
  earlier tests and real fits (about 65% distinct draws) simply didn't reach it.

## Redesign of the multimodal path (prototype `proposed_v2`)

This is Luca's suggestion, and the standard construction of a highest-density region
(Hyndman 1996, "Computing and graphing highest density regions", *Am. Stat.* 50,
120–126):

1. **Cuts.** The `az.hdi(method="multimodal")` pieces give candidate modes, with a cut
   at the midpoint of each gap. One piece means getdist's single interval, as today.
2. **Per-mode densities.** Split the draws at the cuts, and estimate each mode's
   density with getdist from its own draws. Each cut is a hard edge in getdist's
   `ranges`, and so is the coefficient's 0 bound where there is one. Weight each
   density by the mode's share of the draws.
3. **One common threshold.** t is the (1 − p) quantile of the weighted densities
   evaluated at the draws, so a fraction p of the draws lies above it.
4. **Pieces.** For each mode, the stretch where its weighted density is ≥ t, with the
   edges linearly interpolated between grid points. A mode never reaching t gives no
   piece.
5. **Merging at cuts.** Two neighbouring pieces that are both still above t at their
   shared cut are one piece. A spurious cut through one mode then disappears, while a
   real dip stays below t and keeps them apart.

This removes ISJ, the gluing and the 2048-point grid entirely. Each mode's density
gets getdist's bandwidth for that mode alone, which fixes the blurring of narrow,
well-separated modes that ISJ had been brought in to fix.

Why the cuts are hard edges: without `ranges`, getdist's estimate of a mode cut at
0.3 drops to 0.000 at the cut. With `ranges=[None, 0.3]` it gives 0.608, against the
true 0.612.

An earlier variant, `proposed` (v1), merged cuts whose gap is narrower than the
`experimental` bandwidth, instead of steps 2 and 5. It wrongly merged the real 80/20
narrow mode at 95% (31/50), because at a low threshold a real gap can be narrower
than the bandwidth. That is why v2 replaced it.

### Results of v2

Mass is now on target. Two cases from the benchmarks below:
- heavily resampled two-mode posterior: the 68% pieces hold 0.68–0.73, against
  0.85–0.92 with the current code;
- 35%-distinct unimodal: at 95%, the worst case holds 0.970, against 0.999.

On the real fits, everything is single-interval except Obp, and the mass is on target.
v2 and the Silverman-count variant D (see *Open decision 1*) give identical results on
all three:

| fit | coefficient | 68% | 95% |
|---|---|---|---|
| `obp_bimodal_ns_fit` | Obp | [-0.3722, -0.3628] ∪ [-0.0042, 0.0042], mass 0.680 | [-0.3761, -0.3587] ∪ [-0.0092, 0.0083], mass 0.950 |
| `hdi_test_ns_fit` | OpBox | [0, 0.0234] | [0, 0.0458] |
| | OpG | [-0.0006, 0.0006] | [-0.0011, 0.0012] |
| | OpW | [-0.0022, 0.0014] | [-0.0040, 0.0032] |
| Luca's template, our run | OpD | [-0.1015, -0.0284] | [-0.1407, 0.0101] |
| | OpWB | [-0.0070, 0.0057] | [-0.0136, 0.0110] |
| | y | [-0.0353, -0.0031] | [-0.0534, 0.0134] |

The remaining issue is the **piece count** for heavily resampled unimodal
posteriors. The spurious pieces are slivers at the edge of the region (about ±1σ at
68%, ±2σ at 95%) holding 0.5–6% of the mass. Near the region's edge the
`experimental` KDE sits close to the threshold, and the repeated draws make it cross
back and forth. In two cases the sliver is a clump of tail draws behind a real empty
stretch. Merging at cuts doesn't catch these, because the main piece already ends
before the cut.

## Open decision 1: which bandwidth counts the modes

Each cell: wrong piece count / number of samples. In every cell of every variant, at
most 10% of the samples have a mass more than 0.02 from the level. The worst is
unimodal at 35% distinct, 68%: 4 to 10 out of 100.

| case (level) | v2 as is (`experimental` count) | A: + merge gaps < bandwidth | B: + absorb pieces < 3% mass | A + B | D: Silverman count |
|---|---|---|---|---|---|
| unimodal, 54% distinct (68%) | 14/100 | 2/100 | 9/100 | 1/100 | **0/100** |
| unimodal, 54% distinct (95%) | 8/100 | 3/100 | 0/100 | 0/100 | **0/100** |
| unimodal, 35% distinct (68%) | 28/100 | 14/100 | 20/100 | 10/100 | **0/100** |
| unimodal, 35% distinct (95%) | 9/100 | 4/100 | 0/100 | 0/100 | **0/100** |
| bimodal ±2 (sd 0.3), 32% distinct (68%, 95%) | 0/30, 0/30 | 0, 0 | 0, 0 | 0, 0 | 0, 0 |
| 80/20 with a narrow mode, iid (68%) | 0/30 | 0 | 0 | 0 | 0 |
| 80/20 with a narrow mode, iid (95%) | **2/30** | 6/30 | 2/30 | 6/30 | 24/30 |
| Obp-like narrow modes, iid (68%, 95%) | 0/20, 0/20 | 0, 0 | 0, 0 | 0, 0 | 0, 0 |

**Recommendation: D, count with Silverman.**
- It never split a unimodal posterior in 400 resampled samples.
- It resolves Obp-like and well-separated two-mode posteriors.
- It needs no extra thresholds.
- Its cost: a small narrow mode whose 95% region is separated from the main one only
  by a shallow dip gets reported as one interval covering both. That errs on the wide
  side.

The alternative, A + B, still leaves spurious "∪" pieces in up to 10% of heavily
resampled unimodal fits, which is exactly Luca's complaint. It also needs a 3% mass
threshold that has no principled justification.

**This reverses the reason we picked `experimental`.** Its advantage over Silverman was
resolving that narrow mode. Now that the edges no longer depend on the counting
bandwidth, its fragility under heavy resampling dominates.

If D is chosen:
- `test_hdi_resolves_a_small_narrow_second_mode` (80/20 at 95%) pins the opposite
  behaviour and must be changed: either to the 68% level, where all variants resolve
  it, or to assert the conservative merge.
- The module header of `credible_intervals.py` and the PR paragraph on bandwidths must
  be rewritten. The paragraph still describes `experimental` + ISJ gluing, and its
  "up to 0.81" figure is obsolete.

## Open decision 2: the bound at 0 in Bayesian updates

`ExactPosteriorPrior.__init__` (`smefit/priors.py`) stores
`{"dist": "exact_posterior", "source": ...}` as every coefficient's spec, so
`_prior_bounds` (`smefit/fit_result.py`) finds no bound at 0 after an update.

- **Document it (recommended for this PR):** a note in the `_prior_bounds` docstring
  and in the `smefit-analysis` skill text about `hdi`.
- **Carry the bound through:**
  - `ExactPosteriorPrior` gets its `base_prior` from `build_exact_posterior_prior`
    (`smefit/utils.py`), which can nest `base_prior.prior_specs[name]` under a key such
    as `"base"`.
  - It must not do so when the base is a `WhitenedToPhysicalPrior`: its specs are the
    whitened `uniform [-sigma_prior, sigma_prior]`, not physical bounds.
  - For chains of updates, `_prior_bounds` must recurse through the nested `"base"`.
  - This changes what `fit_results.json` stores, and needs tests in `test_priors.py`,
    `test_utils.py` and `test_fit_result.py`.

## Remaining mechanical changes (no decision needed)

- **Module-level imports:** move `import arviz as az` and the `getdist` imports to the
  top of `smefit/credible_intervals.py`, and drop the "only an hdi report needs it"
  comment.
- **Move `_INTERVAL_TYPES`** from `smefit/fit_result.py` to
  `smefit/credible_intervals.py`. Places to update:
  - `smefit/config.py:28`: `from smefit.fit_result import _INTERVAL_TYPES, Fit`;
  - `smefit/fit_result.py`: the import and its use in `confidence_bounds`;
  - the docstring of `credible_intervals.py`, which refers to
    `smefit.fit_result._INTERVAL_TYPES`;
  - the tests that monkeypatch `fit_result._INTERVAL_TYPES`: `tests/test_fit_result.py`
    lines 771, 785 and 841, `tests/test_tables.py` lines 183, 200 and 224, and
    `tests/test_config.py` line 1350. Patch `credible_intervals._INTERVAL_TYPES`
    instead; it's the same dict object either way, but patch it where it's defined.
- **Template runcard:** `template_runcards/coefficient_bounds_table.yaml` should read
  `interval_types: [eti, hdi]`. Then run `python scripts/generate_skill_reference.py`;
  the template is copied into the skill, and CI's `check-skill-references` fails if
  the copy is stale.
- **Tests (Luca 1c):**
  - Bound `test_hdi_of_a_heavily_resampled_bimodal_posterior_keeps_every_mode`
    from both sides: |mass − level| ≤ 0.02.
  - Add a resampled unimodal regression: `resampled(norm.pdf, rng, 3460, 2500, 3.0)`
    gives about 35% distinct draws, like Luca's sample. Seeds that the current code
    fails on (54% distinct, n_prop = 5000) include 11, 20, 24, 25 and 28 at 68%, and
    20, 24 and 27 at 95%. Assert one piece and |mass − level| ≤ 0.02.
- **Before committing:** run `pytest -m "not slow"` and `pre-commit run --all-files`,
  and check black 26.10.0 as well. CI's lint job uses `psf/black@stable`, which differs
  from the pinned pre-commit version (26.1.0).

## Other open items

- **Issue #138** (filed): `run_blackjax_fit` crashes on `main` because of type
  annotations from `e61d9c7`. Until it's fixed, BlackJAX fits can be run with
  `review_pr137/run_smefit_patched.py <runcard>`. That script relaxes the two
  annotations in memory only.
- **Reply to Luca** on the PR once the design is settled. Nothing has been posted.

## Files

All of these are under `~/smefit_results/hdi_test/`.

| path | what |
|---|---|
| `review_pr137/hdi_prototype.py` | `proposed` (v1), `proposed_v2`, the `post_merge` and `silverman_count` variants, and the `resampled`/`mass`/`benchmark` helpers. Run it from the repo root with the `new_smefit` env. Its `__main__` block benchmarks v1 only; for v2, import the module, set `proposed = proposed_v2`, and call `benchmark`. |
| `review_pr137/ultranest_template_fit/` | our run of `template_runcards/ultranest_fit.yaml`, with absolute paths, and its output |
| `review_pr137/run_smefit_patched.py` | runs `smefit` with the #138 annotations relaxed in memory |
| `hdi_test_ns_fit.yaml`, `obp_bimodal_ns_fit.yaml` and their outputs | the HL-LHC + FCC-ee test fit (OpBox with a positivity prior, OpG, OpW) and the Obp bimodal fit |
| `hdi_test_bounds.yaml` | the ETI-vs-HDI report runcard for both test fits |
| `bounds_table_isj.csv` | the bounds table from before the `experimental` change, kept for comparison |
| `plot_posterior_histograms.py`, `posterior_histograms.png` | OpBox and OpW histograms |
