r"""The electroweak-scale blocks of smelli: EWPT, Higgs, LEP2 e+e- -> ll and WW.

Why this is a separate module from the rest of the package
----------------------------------------------------------
Everything else here surrogates ``fast_likelihood_quarks``, whose observables are
low-energy and take **WET** coefficients.  These four blocks live at or above the
electroweak scale and take **SMEFT** coefficients directly, so no amount of
widening the WET support reaches them (PLAN 10).  Three things are different, and
all three make this the easy half:

* **No recipes.**  Measured, not assumed: all 257 observables are exact quadratic
  forms in the Warsaw coefficients (residual 5e-15 over a random box), so the
  observables *are* the elementary quantities and one design serves the lot.
  ``likelihood_eeww`` is exactly *linear* (2.9e-15) and could take a degree-1
  design; it rides along quadratically because it is cheap either way.
* **No CKM axis.**  These observables are leptonic or nearly so; ``check_ckm``
  below measures how much they move under smelli's re-extraction, and the answer
  is what justifies freezing it.
* **A different chi2.**  These are plain ``flavio.Likelihood`` blocks, not
  ``FastLikelihood`` ones: there is no pseudo-measurement and no SM covariance
  folded in, just a sum over measurement constraints.  Of the 115 constraints,
  113 are (multivariate) normal and 2 are asymmetric-normal, so the whole thing
  is a sum of Gaussian terms plus one ``where`` -- traceable under JAX.

Building requires the ``flavour_recent`` env with
``PYTHONPATH=~/Software/flavio:~/Software/smelli``.  **Evaluating does not**:
like ``surrogate/evaluate.py``, the evaluator imports neither flavio nor smelli,
so it runs in ``new_smefit`` alongside jax and smefit.  Every heavy import below
is therefore inside the function that needs it.
"""
import os
import warnings

import numpy as np

warnings.filterwarnings('ignore')

BLOCKS = ('likelihood_ewpt.yaml', 'likelihood_higgs.yaml',
          'likelihood_eell.yaml', 'likelihood_eeww.yaml')
MATCHING_SCALE = 3000.0          # the Z' matching scale; coordinates live here

_G = {}


def yaml_dir():
    import smelli
    return os.path.join(os.path.dirname(smelli.__file__), 'data', 'yaml')


def load(blocks=BLOCKS):
    """``{block: flavio.Likelihood}``, cached."""
    import flavio
    from flavio.statistics.likelihood import Likelihood
    key = tuple(blocks)
    if key not in _G:
        out = {}
        for fn in blocks:
            with open(os.path.join(yaml_dir(), fn)) as f:
                out[fn] = Likelihood.load_dict(flavio.io.yaml.load_include(f))
        _G[key] = out
    return _G[key]


def observables(blocks=BLOCKS):
    """The ordered observable list, block by block."""
    L = load(blocks)
    return [o for fn in blocks for o in L[fn].observables]


def block_sizes(blocks=BLOCKS):
    """``[(block, n_observables)]`` in table order -- what `EWSurrogate` needs to
    evaluate a subset of the blocks without flavio."""
    L = load(blocks)
    return [(fn, len(L[fn].observables)) for fn in blocks]


def obs_name(o):
    if isinstance(o, str):
        return o
    return o[0] + ' ' + ' '.join(str(x) for x in o[1:])


# --------------------------------------------------------------------------
# the chi2, transcribed from what flavio's Likelihood actually sums
# --------------------------------------------------------------------------

def constraints(blocks=BLOCKS, obs=None):
    r"""Flatten every measurement constraint into array data.

    Returns ``(terms, n_obs)``; each term is a dict with

    * ``idx``    -- indices into the observable vector
    * ``mu``     -- central values
    * ``kind``   -- 'normal' | 'multivariate' | 'asymmetric' | 'numerical' | 'upper'
    * ``sigma``  -- standard deviations (normal, multivariate, upper), or
      ``(sigma_left, sigma_right)`` for 'asymmetric'
    * ``inv``    -- inverse *correlation* matrix (multivariate only)
    * ``x``, ``y``, ``logc`` -- 'numerical' only: the tabulated grid, the pdf on it
      normalised as flavio normalises it, and ln pdf at the central value

    Every term contributes ``-2 [ln p(pred) - ln p(central value)]``, which is what
    flavio sums with ``delta=True`` and what smelli's ``chi2_dict`` reports.

    `obs` is the observable order the indices refer to; it defaults to the
    concatenation of `blocks`.  The rules are flavio's own
    (``Measurement.get_logprobability_all``): a constraint touching none of the
    observables is skipped; a multivariate normal touching only some of them is
    *marginalised*, which for a Gaussian is the covariance sub-block; and within
    one measurement only the constraint registered last for a parameter counts.
    flavio's ``log_likelihood_pred`` is the reference; ``check()`` and
    ``lfu_check`` compare against it rather than trusting this transcription.
    """
    import flavio
    from flavio.statistics.probability import (NormalDistribution,
                                               MultivariateNormalDistribution,
                                               AsymmetricNormalDistribution,
                                               NumericalDistribution,
                                               HalfNormalDistribution)
    L = load(blocks)
    obs = observables(blocks) if obs is None else obs
    pos = {str(o): i for i, o in enumerate(obs)}
    terms, seen = [], set()
    for fn in blocks:
        for mname in L[fn].measurement_likelihood.get_measurements:
            if mname in seen:          # a measurement can appear in two blocks
                continue
            seen.add(mname)
            M = flavio.Measurement[mname]
            for constraint, pars in M._constraints:
                keep = [k for k, p in enumerate(pars) if str(p) in pos
                        and M._parameters.get(p) == (k, constraint)]
                if not keep:
                    continue           # constrains nothing in these blocks
                idx = np.array([pos[str(pars[k])] for k in keep], dtype=int)
                if isinstance(constraint, MultivariateNormalDistribution):
                    cov = np.asarray(constraint.covariance, dtype=float)[np.ix_(keep, keep)]
                    mu = np.asarray(constraint.central_value, float)[keep]
                    err = np.sqrt(np.diag(cov))
                    if len(keep) == 1:
                        terms.append(dict(idx=idx, kind='normal', mu=mu, sigma=err))
                    else:
                        terms.append(dict(idx=idx, kind='multivariate', mu=mu, sigma=err,
                                          inv=np.linalg.inv(cov / np.outer(err, err))))
                elif len(pars) != 1:
                    raise NotImplementedError(
                        '%s in %s is a %d-dimensional %s; only multivariate normals '
                        'are handled beyond 1D' % (pars, mname, len(pars),
                                                   type(constraint).__name__))
                elif isinstance(constraint, AsymmetricNormalDistribution):
                    terms.append(dict(idx=idx, kind='asymmetric',
                                      mu=np.atleast_1d(constraint.central_value).astype(float),
                                      sigma=(float(constraint.left_deviation),
                                             float(constraint.right_deviation))))
                elif isinstance(constraint, NormalDistribution):
                    terms.append(dict(idx=idx, kind='normal',
                                      mu=np.atleast_1d(constraint.central_value).astype(float),
                                      sigma=np.atleast_1d(constraint.standard_deviation).astype(float)))
                elif isinstance(constraint, HalfNormalDistribution):
                    # includes GaussianUpperLimit (central value 0).  The sign of
                    # sigma says which side is allowed; the forbidden side is -inf
                    # in flavio and is not reproduced (see `chi2`).
                    terms.append(dict(idx=idx, kind='upper',
                                      mu=np.atleast_1d(constraint.central_value).astype(float),
                                      sigma=np.atleast_1d(constraint.standard_deviation).astype(float)))
                elif isinstance(constraint, NumericalDistribution):
                    # flavio interpolates the *pdf* linearly (interp1d, 0 outside)
                    # and takes the log afterwards -- so must we
                    x = np.asarray(constraint.x, dtype=float)
                    y = np.asarray(constraint.y_norm, dtype=float)
                    c = float(constraint.central_value)
                    terms.append(dict(idx=idx, kind='numerical', x=x, y=y,
                                      mu=np.array([c]),
                                      logc=float(np.log(np.interp(c, x, y)))))
                else:
                    raise NotImplementedError(
                        '%s in %s is a %s; ew.chi2 covers normal, multivariate-normal, '
                        'asymmetric-normal, half-normal and numerical'
                        % (pars, mname, type(constraint).__name__))
    return terms, len(obs)


def pack(terms):
    """Group the constraint terms so the chi2 is a handful of array operations.

    All 102 univariate normal terms collapse into one vector; the 8 multivariate
    ones and the 2 asymmetric ones stay as they are.  Without this the chi2 is a
    112-block Python loop, which is correct but takes JAX ~6 minutes to trace and
    compile -- measured, on the first call of a fit.
    """
    idx, mu, sig = [], [], []
    rest = []
    for t in terms:
        if t['kind'] == 'normal':
            idx.append(np.atleast_1d(t['idx']))
            mu.append(np.atleast_1d(t['mu']))
            sig.append(np.atleast_1d(t['sigma']))
        else:
            rest.append(t)
    return dict(idx=np.concatenate(idx) if idx else np.zeros(0, dtype=int),
                mu=np.concatenate(mu) if mu else np.zeros(0),
                sigma=np.concatenate(sig) if sig else np.zeros(0),
                rest=rest)


def chi2(pred, terms, xp=np):
    """``-2 log L`` up to the constant that does not depend on the prediction.

    `terms` is either the list from `constraints` or the packed form from `pack`;
    the two agree exactly, and `pack` is what the evaluator uses.
    """
    pred = xp.asarray(pred)
    if isinstance(terms, dict):
        p = terms
        d = pred[p['idx']] - xp.asarray(p['mu'])
        total = xp.sum((d / xp.asarray(p['sigma'])) ** 2)
        rest = p['rest']
    else:
        total = xp.zeros(()) * 1.0
        rest = terms
    for t in rest:
        d = pred[np.asarray(t['idx'])] - xp.asarray(t['mu'])
        if t['kind'] == 'multivariate':
            z = d / xp.asarray(t['sigma'])
            total = total + z @ xp.asarray(t['inv']) @ z
        elif t['kind'] == 'asymmetric':
            lo, hi = t['sigma']
            s = xp.where(d < 0, lo, hi)
            total = total + xp.sum((d / s) ** 2)
        elif t['kind'] == 'numerical':
            # linear interpolation of the pdf, as flavio's interp1d.  Outside the
            # tabulated support flavio returns ln 0 = -inf; here the pdf is floored
            # at 1e-300 instead, a steep but finite wall a sampler can live with.
            p_ = xp.interp(pred[np.asarray(t['idx'])], xp.asarray(t['x']), xp.asarray(t['y']))
            total = total - 2 * xp.sum(xp.log(xp.maximum(p_, 1e-300)) - t['logc'])
        elif t['kind'] == 'upper':
            # half-normal: Gaussian on the allowed side.  The forbidden side (a
            # negative branching ratio for an upper limit) is -inf in flavio; no
            # prediction here can reach it, so the Gaussian form is kept there.
            total = total + xp.sum((d / xp.asarray(t['sigma'])) ** 2)
        else:
            total = total + xp.sum((d / xp.asarray(t['sigma'])) ** 2)
    return total


# --------------------------------------------------------------------------
# predictions
# --------------------------------------------------------------------------

def par_sm():
    import flavio
    return flavio.default_parameters.get_central_all()


def make_w(wc_dict, scale=MATCHING_SCALE):
    """``wilson.Wilson`` from Warsaw-basis SMEFT coefficients at ``scale``."""
    import wilson
    w = wilson.Wilson(dict(wc_dict) or {'ll_1111': 0.0}, scale=scale,
                      eft='SMEFT', basis='Warsaw')
    w.set_option('smeft_accuracy', 'integrate')
    return w


def predictions(par, w, blocks=BLOCKS):
    import flavio
    out = []
    for o in observables(blocks):
        od = flavio.classes.Observable.argument_format(o, 'dict')
        name = od.pop('name')
        out.append(flavio.classes.Observable[name].prediction_par(par, w, **od))
    return np.array(out, dtype=float)


def errors(blocks=BLOCKS, obs=None):
    """Per-observable sigma: the tightest constraint that touches it.

    Only used to put residuals and the sensitivity screen in physical units --
    the chi2 itself goes through `constraints`, correlations and all.  A
    tabulated pdf contributes its standard deviation.
    """
    terms, n = constraints(blocks, obs)
    err = np.full(n, np.inf)
    for t in terms:
        if t['kind'] == 'asymmetric':
            s = np.array([0.5 * (t['sigma'][0] + t['sigma'][1])])
        elif t['kind'] == 'numerical':
            x, y = t['x'], t['y']
            m = np.trapezoid(x * y, x)
            s = np.array([np.sqrt(np.trapezoid((x - m) ** 2 * y, x))])
        else:
            s = np.abs(np.asarray(t['sigma'], dtype=float))
        s = np.broadcast_to(s, t['idx'].shape)
        err[t['idx']] = np.minimum(err[t['idx']], s)
    return err


def check(blocks=BLOCKS, wc=None):
    """``ew.chi2`` against flavio's own ``log_likelihood_pred``, at one point.

    Returns ``(chi2, -2 log L)``.  The two differ by the Gaussian normalisation,
    which does not depend on the prediction, so what must hold is that the
    *offset is the same at every point*: measured -842.737526 at three points
    spanning the SM and two Z' points with the Higgs charge on.  Run this after
    touching `constraints`.
    """
    L = load(blocks)
    par, w = par_sm(), make_w(wc or {})
    pred = predictions(par, w, blocks)
    obs = observables(blocks)
    terms, _ = constraints(blocks, obs)
    mine = float(chi2(pred, terms))
    theirs = 0.0
    for fn in blocks:
        d = dict(zip([str(o) for o in L[fn].observables],
                     [pred[i] for i, o in enumerate(obs)
                      if str(o) in {str(x) for x in L[fn].observables}]))
        theirs += -2 * L[fn].measurement_likelihood.log_likelihood_pred(
            {o: pred[obs.index(o)] for o in L[fn].observables})
    return mine, theirs


# --------------------------------------------------------------------------
# support, build, evaluate
# --------------------------------------------------------------------------

class EWCoords:
    """The ordered real coordinate vector of an EW support file."""

    def __init__(self, spec):
        self.spec = spec
        self.name = spec['name']
        self.scale = float(spec['scale'])
        self.box = float(spec.get('box', 1.2))
        self.blocks = tuple(spec.get('blocks', BLOCKS))
        self.names = list(spec['coefficients'])
        self.units = np.array([float(spec['coefficients'][n]) for n in self.names])

    def __len__(self):
        return len(self.names)

    def x_to_wc(self, x):
        """Internal-unit vector -> Warsaw coefficients (physical, real)."""
        v = np.asarray(x, dtype=float) * self.units
        return {n: float(t) for n, t in zip(self.names, v)}

    def to_x(self, wc):
        """Warsaw dict -> internal vector, dropping whatever is outside support."""
        return np.array([complex(wc.get(n, 0.0)).real / u
                         for n, u in zip(self.names, self.units)])

    def random(self, rng, n=None, box=None):
        box = self.box if box is None else box
        shape = (len(self),) if n is None else (n, len(self))
        return rng.uniform(-box, box, size=shape)


def load_coords(name='EW1'):
    import yaml
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = name if os.path.sep in name else os.path.join(root, 'coords', name + '.yaml')
    with open(path) as f:
        return EWCoords(yaml.safe_load(f))


def check_ckm(co=None, blocks=BLOCKS):
    """How much do these observables care about smelli's CKM re-extraction?

    The LEFT surrogate carries delta gamma as an explicit axis because flavour
    observables move with it.  These are leptonic or nearly so, and this is the
    measurement that says whether freezing the CKM is allowed: it returns the
    max |dP|/sigma over the block for a 1 % shift of each CKM input.
    """
    par = par_sm()
    err = errors(blocks)
    base = predictions(par, make_w({}), blocks)
    out = {}
    for p in ('Vus', 'Vcb', 'Vub', 'gamma'):
        q = dict(par); q[p] = q[p] * 1.01
        out[p] = float(np.max(np.abs(predictions(q, make_w({}), blocks) - base) / err))
    return out


def build(coords_name='EW1', workers=6, n_random=8, out=None, verbose=True):
    """Fit the quadratic form and write the tables.

        OMP_NUM_THREADS=1 PYTHONPATH=~/Software/flavio:~/Software/smelli \
          python -m surrogate.ew --coords EW1 --workers 6

    One design for everything: the observables *are* the elementary quantities
    (they are exact quadratic forms, measured to 5e-15), there is no gamma axis
    and no CKM axis (`check_ckm` returns 1e-14), so this is 1 + N + N(N+1)/2
    evaluations and nothing else.
    """
    import pickle
    import time
    from multiprocessing import get_context
    from . import design

    co = load_coords(coords_name)
    obs = observables(co.blocks)
    terms, n = constraints(co.blocks)
    X, n_exact = design.quad_design(len(co), n_random=n_random, box=co.box, seed=7)
    if verbose:
        print('%s: %d coordinates, %d observables, %d design points (%d exact)'
              % (co.name, len(co), n, len(X), n_exact), flush=True)

    ctx = get_context('fork')
    t0 = time.time()
    with ctx.Pool(workers, initializer=_build_init, initargs=(coords_name,)) as pool:
        Y = []
        for i, y in enumerate(pool.imap(_build_task, [row for row in X])):
            Y.append(y)
            if verbose and (i % 50 == 0 or i == len(X) - 1):
                print('  %4d/%d  (%.1f min)' % (i + 1, len(X), (time.time() - t0) / 60),
                      flush=True)
    Y = np.array(Y)
    C = np.linalg.lstsq(design.quad_features(X), Y, rcond=None)[0]
    res = np.max(np.abs(design.quad_features(X) @ C - Y), axis=0) / (np.abs(Y).max(axis=0) + 1e-300)
    if verbose:
        k = int(np.argmax(res))
        print('max relative design residual %.2e  (worst %s)'
              % (res.max(), obs_name(obs[k])), flush=True)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = out or os.path.join(root, 'tables', coords_name)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'tables.pkl')
    with open(path, 'wb') as fh:
        pickle.dump(dict(meta=dict(coords=coords_name, coords_spec=co.spec,
                                   observables=[obs_name(o) for o in obs],
                                   block_sizes=block_sizes(co.blocks),
                                   design=X, n_exact=n_exact),
                         coef=C, terms=terms), fh)
    if verbose:
        print('wrote %s' % path, flush=True)
    return path


_B = {}


def _build_init(coords_name):
    _B['co'] = load_coords(coords_name)
    _B['par'] = par_sm()


def _build_task(row):
    co = _B['co']
    return predictions(_B['par'], make_w(co.x_to_wc(row), co.scale), co.blocks)


class EWSurrogate:
    """Evaluator: internal coordinate vector -> 257 predictions and the chi2.

    `blocks` restricts the chi2 to a subset of the tabulated blocks, e.g.
    ``('likelihood_ewpt.yaml', 'likelihood_eell.yaml')`` for the four-block
    benchmark of PLAN 12, which has neither Higgs nor WW.  Predictions are
    always the full vector (they are one matrix product); only the constraint
    terms are selected, and a term may not straddle a selected and an
    unselected block.  `self.obs` follows the selection; `self.all_obs` is the
    prediction vector's order.
    """

    def __init__(self, path, blocks=None):
        import pickle
        with open(path, 'rb') as fh:
            t = pickle.load(fh)
        self.t = t
        self.meta = t['meta']
        self.coords = EWCoords(self.meta['coords_spec'])
        self.all_obs = self.meta['observables']
        self.coef = t['coef']
        terms = t['terms']
        self.blocks = tuple(b for b, _ in self.meta.get('block_sizes', ())) or None
        if blocks is not None:
            sizes = self.meta.get('block_sizes')
            if sizes is None:
                raise RuntimeError('%s predates block selection; run '
                               '`python -m surrogate.ew --add-block-sizes`' % path)
            unknown = set(blocks) - {b for b, _ in sizes}
            if unknown:
                raise ValueError('blocks %s are not in these tables, which hold %s'
                                 % (sorted(unknown), [b for b, _ in sizes]))
            keep = np.zeros(len(self.all_obs), dtype=bool)
            start = 0
            for b, n in sizes:
                keep[start:start + n] = b in blocks
                start += n
            sel = []
            for tm in terms:
                k = keep[np.asarray(tm['idx'])]
                if k.all():
                    sel.append(tm)
                elif k.any():
                    raise ValueError('a constraint straddles the block selection')
            terms = sel
            self.blocks = tuple(b for b, _ in sizes if b in blocks)
            self.obs = [o for o, k in zip(self.all_obs, keep) if k]
        else:
            self.obs = self.all_obs
        self.terms = pack(terms)

    def predict(self, x, xp=np):
        from . import design
        q = design.quad_features_1d(xp.asarray(x), len(self.coords), xp)
        return q @ xp.asarray(self.coef)

    def chi2(self, x, xp=np):
        return chi2(self.predict(x, xp), self.terms, xp)

    def from_warsaw(self, wc):
        """Warsaw dict at the matching scale -> internal coordinates."""
        return self.coords.to_x(wc)

    def unsupported(self, wc, rtol=0.0):
        """Coefficients present in `wc` but outside the support."""
        keep = set(self.coords.names)
        return sorted(k for k, v in wc.items()
                      if k not in keep and abs(complex(v)) > rtol)


def validate(coords_name='EW1', n=20, seed=11, box=None, tables=None, verbose=True):
    """Random points in the box: surrogate vs flavio, in chi2 and in sigma."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    S = EWSurrogate(tables or os.path.join(root, 'tables', coords_name, 'tables.pkl'))
    co, par = S.coords, par_sm()
    err = errors(co.blocks)
    rng = np.random.default_rng(seed)
    if verbose:
        print('%-6s %12s %12s %12s   %s'
              % ('point', 'chi2 flavio', 'chi2 surr', 'Delta chi2',
                 'max |dP|/sigma  (worst observable)'))
    obs = observables(co.blocks)
    d2 = []
    for i in range(n + 1):
        x = np.zeros(len(co)) if i == 0 else co.random(rng, box=box)
        ref = predictions(par, make_w(co.x_to_wc(x), co.scale), co.blocks)
        sur = S.predict(x)
        r = (sur - ref) / err
        j = int(np.argmax(np.abs(r)))
        c_ref, c_sur = float(chi2(ref, S.terms)), float(S.chi2(x))
        d2.append(c_sur - c_ref)
        if verbose:
            print('%-6d %12.5f %12.5f %12.2e   %8.2e  %s'
                  % (i, c_ref, c_sur, c_sur - c_ref, abs(r[j]), obs_name(obs[j])))
    d2 = np.array(d2)
    if verbose:
        print('\n|Delta chi2|: max %.2e, rms %.2e' % (np.abs(d2).max(), np.sqrt((d2**2).mean())))
    return d2


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--coords', default='EW1')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--n-random', type=int, default=8)
    ap.add_argument('--out', default=None)
    ap.add_argument('--validate', type=int, default=0,
                    help='validate against flavio at N random box points instead of building')
    ap.add_argument('--box', type=float, default=None)
    ap.add_argument('--add-block-sizes', action='store_true',
                    help='record the per-block observable counts in existing tables '
                         '(needed for EWSurrogate(blocks=...)); no rebuild')
    a = ap.parse_args(argv)
    if a.add_block_sizes:
        import pickle
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = a.out or os.path.join(root, 'tables', a.coords, 'tables.pkl')
        with open(path, 'rb') as fh:
            t = pickle.load(fh)
        blocks = tuple(t['meta']['coords_spec'].get('blocks', BLOCKS))
        sizes = block_sizes(blocks)
        assert sum(n for _, n in sizes) == len(t['meta']['observables'])
        assert [obs_name(o) for o in observables(blocks)] == t['meta']['observables']
        t['meta']['block_sizes'] = sizes
        with open(path, 'wb') as fh:
            pickle.dump(t, fh)
        print('%s: block sizes %s' % (path, sizes))
    elif a.validate:
        validate(a.coords, n=a.validate, box=a.box)
    else:
        build(a.coords, workers=a.workers, n_random=a.n_random, out=a.out)


if __name__ == '__main__':
    main()
