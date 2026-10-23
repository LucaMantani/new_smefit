r"""Observable families ("class adapters").

Every family exposes

* ``sectors``     : which coordinate sectors its quadratic forms live in
* ``obs``         : the observable keys it is responsible for
* ``elem_names``  : the elementary quantities that are fitted
* ``elementary(par, w)``  : build side, calls flavio
* ``rebuild(elem, ctx)``  : evaluation side, pure array algebra

``ctx`` is the dict returned by ``df2.Df2.context`` -- the analytic Delta F=2
sector: mixing phases, q/p per meson, and the Delta F=2 observables.

The elementary quantities are chosen so that they are *exact* quadratic forms
in the coordinates (flavio's predictions are algebraic in the WCs), and so that
``rebuild`` contains no parameter values at all -- every parameter dependence,
including the CKM phase, is absorbed into the fitted coefficients.
"""

import math
import re

import numpy as np

# --------------------------------------------------------------------------
# observable bookkeeping
# --------------------------------------------------------------------------

BVLL_PROCS = {
    "B0->K*mumu": ("B0", "K*0", "mu"),
    "B+->K*mumu": ("B+", "K*+", "mu"),
    "B0->K*ee": ("B0", "K*0", "e"),
}
BSVLL_PROCS = {"Bs->phimumu": ("Bs", "phi", "mu"), "Bs->phiee": ("Bs", "phi", "e")}
BSVLL_BD_PROCS = {"Bs->K*0mumu": ("Bs", "K*0", "mu")}
# whole-range Bs branching ratios: kind -> (process, (q2min, q2max), extra factor).
# `BR_LHCb(Bs->K*0mumu)` is flavio's `bsvll_dbrdq2_19_func`: the q2-integrated
# rate over [0.1, 19] times a normalisation nuisance parameter.
BSVLL_WHOLE = {
    "BR_LHCb(Bs->K*0mumu)": (
        "Bs->K*0mumu",
        (0.1, 19.0),
        lambda par: 1 + par["delta_BsKstarmumu"],
    )
}
DIRECT_PROCS = {
    "B0->K*mumu": "mu",
    "B+->K*mumu": "mu",
    "B0->K*ee": "e",
    "B+->Kmumu": "mu",
    "B0->Kmumu": "mu",
    "B->Xsmumu": "mu",
    "B->Xsee": "e",
    "Lambdab->Lambdamumu": "mu",
}
DIRECT_STRINGS = {"BR(Bs->mumu)": "mu", "BR(Bs->ee)": "e"}
SECTOR = {"mu": "bsmumu", "e": "bsee"}
# QED running between M_Z and the b scale mixes the two lepton flavours, so every
# b->s l l family sees *both* semileptonic sectors -- they do not factorise.
SEMILEPTONIC = ("bsmumu", "bsee")

_BINNED = re.compile(r"^<(.+)>\((.+)\)$")


def obs_str(o):
    """The observable's printable key, as ``harness.obs_name`` writes it.

    flavio keys are either plain strings or ``(name, *arguments)`` tuples; the
    pattern-selected families below match on this form, which is also what
    ``surrogate.sensitivity`` records.
    """
    if isinstance(o, str):
        return o
    return o[0] + " " + " ".join(str(x) for x in o[1:])


def parse(o):
    """(kind, process, q2min, q2max) for a binned observable, else (name, None, None, None)."""
    if isinstance(o, str):
        return o, None, None, None
    m = _BINNED.match(o[0])
    if m is None or len(o) != 3:
        return o[0], None, None, None
    return m.group(1), m.group(2), float(o[1]), float(o[2])


class Family:
    name = ""
    group = None  # key into the coordinate file's `groups:` block
    sectors = ()  # fallback when the file declares no group of that name
    uses_phis = False
    phi_design = False
    degree = 2  # polynomial degree in the coordinates (see design.py)
    aux = False  # True: feeds the CKM solver, not the prediction vector
    # How the response to Vus/Vcb/Vub is tabulated (build.py, df2.apply_dlnv):
    #   False -- the *relative* log-derivative at the design origin.  Exact
    #            wherever the CKM enters as an overall factor, which is every
    #            b->s / b->d family: flavio normalises those Wilson coefficients
    #            *by* lambda_t, so the SM and NP pieces carry the same CKM power
    #            and d ln P / d ln V does not depend on the coordinates at all
    #            (measured: 2.066 -> 2.120 across the box).
    #   True  -- the *absolute* derivative, fitted over the family's own design.
    #            Needed where the CKM does not factor out: the K-sector Wilson
    #            coefficients carry no lambda_t, so the SM piece scales with the
    #            CKM and the NP piece does not, and the relative derivative runs
    #            from 0.39 at the origin to 0.06 at a box point.  The absolute
    #            derivative is itself an exact polynomial of the same degree, so
    #            this is exact -- at 6 extra evaluations per design point.
    dlnv_full = False

    def __init__(self, gl, coords, obs_all):
        self.coords = coords
        # Which coordinates a family must carry is *measured* (PLAN 8.2,
        # `surrogate.sensitivity`), and the answer is recorded in the coordinate
        # file rather than here -- see `coords.Coords.group`.
        # A coordinate file may give a single family its own sector list, keyed
        # by the *family* name, when a sector is relevant to one member of a
        # group and not the others -- the B+->pi+ subleading parameters reach
        # `bpll_bd` and `direct_bd` but not `bsvll_bd_mu`, and putting them in
        # `bd_ll` would inflate the third family's design for nothing.  The
        # family name wins; `self.group` is the fallback.  Every pre-existing
        # group whose name equals a family name (`epsp`, `cc_*`, `ckm_inputs`)
        # is already that family's own group, so this changes nothing for S1/S2.
        if self.name in coords.groups:
            self.sectors = coords.groups[self.name]
        elif self.group is not None:
            self.sectors = coords.group(self.group, self.sectors)
        self.obs = []
        self._select(gl, obs_all)

    def _select(self, gl, obs_all):
        raise NotImplementedError

    @property
    def n_out(self):
        return 3 * len(self.elem_names) if self.phi_design else len(self.elem_names)

    @property
    def n_var(self):
        return len(self.coords.sector_index(*self.sectors))

    def x_sub(self, x):
        return np.asarray(x, dtype=float)[self.coords.sector_index(*self.sectors)]


# --------------------------------------------------------------------------
# 1. observables that are themselves exact quadratic forms
# --------------------------------------------------------------------------


class Direct(Family):
    """BR-type observables: the observable *is* the elementary quantity."""

    def __init__(self, gl, coords, obs_all, lepton):
        self.lepton = lepton
        self.name = "direct_" + lepton
        self.group, self.sectors = "bs_ll", SEMILEPTONIC
        super().__init__(gl, coords, obs_all)

    def _select(self, gl, obs_all):
        for o in obs_all:
            kind, proc, _, _ = parse(o)
            if proc is None:
                if DIRECT_STRINGS.get(kind) == self.lepton:
                    self.obs.append(o)
            elif kind in ("dBR/dq2", "BR") and DIRECT_PROCS.get(proc) == self.lepton:
                self.obs.append(o)
        self.elem_names = [str(o) for o in self.obs]

    def elementary(self, par, w):
        import flavio

        out = []
        for o in self.obs:
            od = flavio.classes.Observable.argument_format(o, "dict")
            name = od.pop("name")
            out.append(flavio.classes.Observable[name].prediction_par(par, w, **od))
        return np.array(out, dtype=float)

    def rebuild(self, elem, ctx=None, xp=np):
        return elem


# --------------------------------------------------------------------------
# 2. B -> K* l l angular observables (ratios of bin-integrated quadratic forms)
# --------------------------------------------------------------------------

_BVLL_KEYS = {
    "N2s": "2s",
    "N2c": "2c",
    "N3": 3,
    "N4": 4,
    "N5": 5,
    "N6s": "6s",
    "N7": 7,
    "N8": 8,
    "N9": 9,
}
_PP_KEY = {"P4p": 4, "P5p": 5, "P6p": 7, "P8p": 8}
_S_KEY = {"S3": 3, "S4": 4, "S5": 5, "S6c": "6c", "S7": 7, "S8": 8, "S9": 9}


class BVll(Family):
    """B0/B+ -> K* l l: FL, AFB, S_i, P_i, P'_i, A_T^Im."""

    def __init__(self, gl, coords, obs_all, lepton):
        self.lepton = lepton
        self.name = "bvll_" + lepton
        self.group, self.sectors = "bs_ll", SEMILEPTONIC
        super().__init__(gl, coords, obs_all)

    def _select(self, gl, obs_all):
        self.recipes, bins, elems = [], [], {}
        for o in obs_all:
            kind, proc, q2min, q2max = parse(o)
            if proc not in BVLL_PROCS or BVLL_PROCS[proc][2] != self.lepton:
                continue
            need = None
            if kind == "FL":
                need = ["N2c", "D"]
            elif kind == "AFB":
                need = ["N6s", "D"]
            elif kind == "ATIm":
                need = ["NA9", "N2s"]
            elif kind in ("P1", "P2", "P3"):
                need = [{"P1": "N3", "P2": "N6s", "P3": "N9"}[kind], "N2s"]
            elif kind in _PP_KEY:
                need = ["N%s" % _PP_KEY[kind], "N2s", "N2c"]
            elif kind in _S_KEY:
                need = ["N%s" % _S_KEY[kind], "D"]
            if need is None:
                continue
            b = BVLL_PROCS[proc] + (q2min, q2max)
            if b not in bins:
                bins.append(b)
                elems[b] = set()
            elems[b].update(need)
            self.obs.append(o)
            self.recipes.append((kind, b))
        self.bins = bins
        self.bin_elems = {b: sorted(elems[b]) for b in bins}
        self.elem_names = [
            "{}|{}".format(b, e) for b in bins for e in self.bin_elems[b]
        ]
        self._index = {}
        k = 0
        for b in bins:
            for e in self.bin_elems[b]:
                self._index[(b, e)] = k
                k += 1

    def elementary(self, par, w):
        import flavio
        from flavio.physics.bdecays.bvll import observables as O

        wc = flavio.WilsonCoefficients.from_wilson(w, par)
        out = np.zeros(len(self.elem_names))
        for b in self.bins:
            B, V, lep, q2min, q2max = b
            for e in self.bin_elems[b]:
                if e == "D":
                    f = O.SA_den
                elif e == "NA9":
                    f = lambda J, Jb: O.A_experiment_num(J, Jb, 9)
                else:
                    key = _BVLL_KEYS[e]
                    f = lambda J, Jb, k=key: O.S_experiment_num(J, Jb, k)
                out[self._index[(b, e)]] = O.BVll_obs_int(
                    f, q2min, q2max, B, V, lep, wc, par
                )()
        return out

    def rebuild(self, elem, ctx=None, xp=np):
        out = []
        for kind, b in self.recipes:
            g = lambda e: elem[self._index[(b, e)]]
            if kind == "FL":
                v = -g("N2c") / g("D")
            elif kind == "AFB":
                v = 0.75 * g("N6s") / g("D")
            elif kind == "ATIm":
                v = 0.5 * g("NA9") / g("N2s")
            elif kind == "P1":
                v = 0.5 * g("N3") / g("N2s")
            elif kind == "P2":
                v = 0.125 * g("N6s") / g("N2s")
            elif kind == "P3":
                v = -0.25 * g("N9") / g("N2s")
            elif kind in _PP_KEY:
                v = g("N%s" % _PP_KEY[kind]) / (2 * xp.sqrt(-g("N2s") * g("N2c")))
            else:
                v = g("N%s" % _S_KEY[kind]) / g("D")
            out.append(v)
        return xp.stack(out)


# --------------------------------------------------------------------------
# 3. Bs -> phi l l  (same structure, plus the Bs mixing phase)
# --------------------------------------------------------------------------


def _bsvll_obs_phi(function, q2, wc_obj, par, B, V, lep, phi):
    """flavio's ``observables_bs.bsvll_obs`` with the mixing phase passed in."""
    import flavio

    ml = par["m_" + lep]
    mB = par["m_" + B]
    mV = par["m_" + V]
    y = par["DeltaGamma/Gamma_" + B] / 2.0
    if q2 < 4 * ml**2 or q2 > (mB - mV) ** 2:
        return 0
    scale = flavio.config["renormalization scale"]["bvll"]
    mb = flavio.physics.running.running.get_mb(par, scale)
    A = flavio.physics.bdecays.bvll.amplitudes
    ang = flavio.physics.bdecays.angular
    ff = A.get_ff(q2, par, B, V)
    h = A.helicity_amps(q2, ff, wc_obj, par, B, V, lep)
    h_bar = A.helicity_amps_bar(q2, ff, wc_obj, par, B, V, lep)
    J = ang.angularcoeffs_general_v(h, q2, mB, mV, mb, 0, ml, ml)
    J_bar = ang.angularcoeffs_general_v(h_bar, q2, mB, mV, mb, 0, ml, ml)
    h_tilde = h_bar.copy()
    h_tilde[("pl", "V")] = h_bar[("mi", "V")]
    h_tilde[("pl", "A")] = h_bar[("mi", "A")]
    h_tilde[("mi", "V")] = h_bar[("pl", "V")]
    h_tilde[("mi", "A")] = h_bar[("pl", "A")]
    h_tilde["S"] = -h_bar["S"]
    J_h = ang.angularcoeffs_h_v(phi, h, h_tilde, q2, mB, mV, mb, 0, ml, ml)
    return function(y, J, J_bar, J_h)


def _bsvll_int_phi(function, q2min, q2max, wc_obj, par, B, V, lep, phi, epsrel=0.005):
    import flavio

    nint = flavio.physics.bdecays.bvll.observables.nintegrate_pole
    return nint(
        lambda q2: _bsvll_obs_phi(function, q2, wc_obj, par, B, V, lep, phi),
        q2min,
        q2max,
        epsrel=epsrel,
    )


_BS_PHI_PARTS = ("U", "C", "S")  # phi-independent, cos(phi), sin(phi)


class BsVll(Family):
    r"""Bs -> phi l l.

    Every bin-integrated quantity is linear in :math:`e^{\pm i\phi_s}`, i.e.
    ``Q(phi) = U + cos(phi) C + sin(phi) S``.  ``U`` is obtained as
    ``(Q(0)+Q(pi))/2`` and so on; here we use the three evaluations at
    ``phi = 0, pi/2, pi``.
    """

    uses_phis = True
    phi_key = "phi_Bs"

    def __init__(
        self,
        gl,
        coords,
        obs_all,
        lepton,
        name=None,
        group="bs_ll",
        sectors=SEMILEPTONIC,
        procs=None,
    ):
        self.lepton = lepton
        self.name = name or ("bsvll_" + lepton)
        self.group, self.sectors = group, sectors
        self.procs = BSVLL_PROCS if procs is None else procs
        super().__init__(gl, coords, obs_all)

    def _select(self, gl, obs_all):
        self.recipes, bins, elems = [], [], {}
        for o in obs_all:
            kind, proc, q2min, q2max = parse(o)
            # the whole-range LHCb branching ratio is written as a single bin
            if kind in BSVLL_WHOLE and BSVLL_WHOLE[kind][0] in self.procs:
                proc, (q2min, q2max) = BSVLL_WHOLE[kind][0], BSVLL_WHOLE[kind][1]
            if proc not in self.procs or self.procs[proc][2] != self.lepton:
                continue
            if kind == "dBR/dq2":
                need = ["BR"]
            elif kind in BSVLL_WHOLE:
                need = [kind]
            elif kind == "FL":
                need = ["N2c", "D"]
            elif kind == "AFB":
                need = ["N6s", "D"]
            elif kind in _S_KEY:
                need = ["N%s" % _S_KEY[kind], "D"]
            else:
                continue
            b = self.procs[proc] + (q2min, q2max)
            if b not in bins:
                bins.append(b)
                elems[b] = set()
            elems[b].update(need)
            self.obs.append(o)
            self.recipes.append((kind, b))
        self.bins = bins
        self.bin_elems = {b: sorted(elems[b]) for b in bins}
        self.elem_names, self._index = [], {}
        for b in bins:
            for e in self.bin_elems[b]:
                for p in _BS_PHI_PARTS:
                    self._index[(b, e, p)] = len(self.elem_names)
                    self.elem_names.append("{}|{}|{}".format(b, e, p))

    def elementary(self, par, w):
        import flavio
        from flavio.physics.bdecays.bvll import observables_bs as OB

        wc = flavio.WilsonCoefficients.from_wilson(w, par)
        out = np.zeros(len(self.elem_names))
        for b in self.bins:
            B, V, lep, q2min, q2max = b
            for e in self.bin_elems[b]:
                if e == "D":
                    f = OB.SA_den_Bs
                elif e == "BR" or e in BSVLL_WHOLE:
                    f = None
                else:
                    key = _BVLL_KEYS[e]
                    f = lambda y, J, Jb, Jh, k=key: OB.S_experiment_num_Bs(
                        y, J, Jb, Jh, k
                    )
                vals = []
                for phi in (0.0, 0.5 * math.pi, math.pi):
                    if f is None:
                        # `e` is either 'BR' (a binned <dBR/dq2>) or a whole-range
                        # branching ratio with its own normalisation
                        tau = par["tau_" + B]
                        v = tau * _bsvll_int_phi(
                            OB.dGdq2_ave_Bs, q2min, q2max, wc, par, B, V, lep, phi
                        )
                        v = (
                            v * BSVLL_WHOLE[e][2](par)
                            if e in BSVLL_WHOLE
                            else v / (q2max - q2min)
                        )
                    else:
                        v = _bsvll_int_phi(f, q2min, q2max, wc, par, B, V, lep, phi)
                    vals.append(v)
                q0, q90, q180 = vals
                out[self._index[(b, e, "U")]] = 0.5 * (q0 + q180)
                out[self._index[(b, e, "C")]] = 0.5 * (q0 - q180)
                out[self._index[(b, e, "S")]] = q90 - 0.5 * (q0 + q180)
        return out

    def rebuild(self, elem, ctx, xp=np):
        phi_s = ctx[self.phi_key]
        c, s = xp.cos(phi_s), xp.sin(phi_s)
        out = []
        for kind, b in self.recipes:

            def g(e):
                k = self._index
                return (
                    elem[k[(b, e, "U")]]
                    + c * elem[k[(b, e, "C")]]
                    + s * elem[k[(b, e, "S")]]
                )

            if kind == "dBR/dq2":
                v = g("BR")
            elif kind in BSVLL_WHOLE:
                v = g(kind)
            elif kind == "FL":
                v = -g("N2c") / g("D")
            elif kind == "AFB":
                v = 0.75 * g("N6s") / g("D")
            else:
                v = g("N%s" % _S_KEY[kind]) / g("D")
            out.append(v)
        return xp.stack(out)


# --------------------------------------------------------------------------
# 4. Lambda_b -> Lambda mu mu forward-backward asymmetries
# --------------------------------------------------------------------------


class LambdabLambdall(Family):
    name = "lambdab"
    group, sectors = "bs_ll", SEMILEPTONIC
    _NUM = ("AFBl", "AFBh", "AFBlh")

    def _select(self, gl, obs_all):
        self.recipes, bins = [], []
        for o in obs_all:
            kind, proc, q2min, q2max = parse(o)
            if proc != "Lambdab->Lambdamumu" or kind not in self._NUM:
                continue
            b = (q2min, q2max)
            if b not in bins:
                bins.append(b)
            self.obs.append(o)
            self.recipes.append((kind, b))
        self.bins = bins
        self.elem_names, self._index = [], {}
        for b in bins:
            for e in ("D",) + self._NUM:
                self._index[(b, e)] = len(self.elem_names)
                self.elem_names.append("{}|{}".format(b, e))

    def elementary(self, par, w):
        import flavio
        from flavio.physics.bdecays import lambdablambdall as L

        wc = flavio.WilsonCoefficients.from_wilson(w, par)
        out = np.zeros(len(self.elem_names))
        funcs = {
            "D": L.dGdq2,
            "AFBl": L.AFBl_num,
            "AFBh": L.AFBh_num,
            "AFBlh": L.AFBlh_num,
        }
        for b in self.bins:
            for e, f in funcs.items():
                out[self._index[(b, e)]] = L.obs_int(f, b[0], b[1], wc, par, "mu")
        return out

    def rebuild(self, elem, ctx=None, xp=np):
        return xp.stack(
            [
                elem[self._index[(b, kind)]] / elem[self._index[(b, "D")]]
                for kind, b in self.recipes
            ]
        )


# --------------------------------------------------------------------------
# 5. Bs -> mu mu effective lifetime
# --------------------------------------------------------------------------


class BqllLifetime(Family):
    name = "bqll_tau"
    group, sectors = "bs_ll", SEMILEPTONIC
    elem_names = ["tau_num", "tau_den"]

    def _select(self, gl, obs_all):
        self.obs = [o for o in obs_all if o == "tau_mumu"]

    def elementary(self, par, w):
        import flavio
        from flavio.physics.bdecays import bll
        from flavio.physics.bdecays.wilsoncoefficients import wctot_dict

        wc_obj = flavio.WilsonCoefficients.from_wilson(w, par)
        scale = flavio.config["renormalization scale"]["bll"]
        wc = wctot_dict(wc_obj, "bsmumu", scale, par)
        P, S = bll.amplitudes(par, wc, "Bs", "mu", "mu")
        A = abs(P) ** 2 + abs(S) ** 2
        Bq = (P**2).real - (S**2).real
        y = 0.5 * par["DeltaGamma/Gamma_Bs"]
        tau = par["tau_Bs"]
        # tau_ll = -(1+y^2+2 y ADG) tau / ((y^2-1)(1+y ADG)) with ADG = Bq/A
        return np.array(
            [-tau * ((1 + y**2) * A + 2 * y * Bq), (y**2 - 1) * (A + y * Bq)]
        )

    def rebuild(self, elem, ctx=None, xp=np):
        return xp.stack([elem[0] / elem[1]]) if self.obs else xp.zeros(0)


# --------------------------------------------------------------------------
# 6. observables that are exact quadratic forms, listed explicitly
# --------------------------------------------------------------------------


class Quadratic(Family):
    """A family whose observables *are* exact polynomial forms in its coordinates.

    Everything flavio computes as |amplitude|^2 with amplitudes linear in the
    Wilson coefficients lands here, so no recipe is needed: the observable is
    its own elementary quantity.  Whether the assumption holds is checked, not
    assumed -- ``build.py`` prints the design residual over `n_random` points
    beyond the exact design, which is ~1e-10 iff the form really is polynomial
    of the declared degree.

    Two variations, both *measured* (see PLAN 8.8.3):

    * ``inverse`` -- names whose **reciprocal** is the quadratic form.  A width
      is a quadratic form in the Wilson coefficients, so a lifetime or an ``Ft``
      value is one over it: ``Ft(26mAl)`` fits to 3e-15 inverted against 1.7e-9
      taken directly.  The reciprocal is the fitted elementary and ``rebuild``
      inverts it back.
    * ``degree`` -- 1 where the observable is exactly linear (``epsp/eps``), so
      that a family with 38 coordinates costs 39 evaluations instead of 780.
    """

    def __init__(
        self,
        gl,
        coords,
        obs_all,
        name,
        group,
        names=(),
        pattern=None,
        inverse=(),
        inverse_pattern=None,
        degree=2,
        dlnv_full=False,
    ):
        self.name = name
        self.group = group
        self.dlnv_full = dlnv_full
        self._names = set(names)
        self._pattern = re.compile(pattern) if pattern else None
        self._inverse_names = set(inverse)
        self._inverse_pattern = re.compile(inverse_pattern) if inverse_pattern else None
        self.degree = degree
        super().__init__(gl, coords, obs_all)

    def _match(self, o):
        k = obs_str(o)
        return (
            str(o) in self._names
            or k in self._names
            or (self._pattern is not None and self._pattern.search(k) is not None)
        )

    def _is_inverse(self, o):
        k = obs_str(o)
        return k in self._inverse_names or (
            self._inverse_pattern is not None
            and self._inverse_pattern.search(k) is not None
        )

    def _select(self, gl, obs_all):
        self.obs = [o for o in obs_all if self._match(o)]
        self.elem_names = [obs_str(o) for o in self.obs]
        # +1 where the observable itself is the quadratic form, -1 where its
        # reciprocal is; kept as an array so `rebuild` stays pure array algebra.
        self._pow = np.array([-1.0 if self._is_inverse(o) else 1.0 for o in self.obs])

    def elementary(self, par, w):
        v = _predict_all(self.obs, par, w)
        return np.where(self._pow < 0, 1.0 / v, v)

    def rebuild(self, elem, ctx=None, xp=np):
        if not np.any(self._pow < 0):
            return elem
        inv = xp.asarray(self._pow < 0)
        return xp.where(inv, 1.0 / elem, elem)


def _predict_all(obs, par, w):
    import flavio

    out = []
    for o in obs:
        od = flavio.classes.Observable.argument_format(o, "dict")
        name = od.pop("name")
        out.append(flavio.classes.Observable[name].prediction_par(par, w, **od))
    return np.array(out, dtype=float)


# --------------------------------------------------------------------------
# 7. B -> P l l  (b -> d): binned ACP and FH are ratios of bin integrals
# --------------------------------------------------------------------------

BPLL_PROCS = {"B+->pimumu": ("B+", "pi+", "mu")}
_BPLL_NUM = {"ACP": "Ndiff", "FH": "NFH", "AFB": "NAFB"}


class BPll(Family):
    r"""B -> P l l: the binned branching ratio and the ratios ACP, FH, AFB.

    All four are built from bin integrals of ``bpll.dGdq2_cpaverage`` and of the
    numerator functions, each an exact quadratic form in the Wilson coefficients
    (``J`` and ``J_bar`` are bilinear in helicity amplitudes that are linear in
    the WCs; conjugating the coefficients keeps the form real-quadratic in
    Re/Im).  The lifetime and bin width are folded into the fitted ``BR``
    elementary so that ``rebuild`` carries no parameter values.
    """

    def __init__(self, gl, coords, obs_all, name="bpll_bd", group="bd_ll", procs=None):
        self.name = name
        self.group = group
        self.procs = BPLL_PROCS if procs is None else procs
        super().__init__(gl, coords, obs_all)

    def _select(self, gl, obs_all):
        self.recipes, bins, elems = [], [], {}
        for o in obs_all:
            kind, proc, q2min, q2max = parse(o)
            if proc not in self.procs:
                continue
            if kind == "dBR/dq2":
                need = ["BR"]
            elif kind in _BPLL_NUM:
                need = [_BPLL_NUM[kind], "D"]
            else:
                continue
            b = self.procs[proc] + (q2min, q2max)
            if b not in bins:
                bins.append(b)
                elems[b] = set()
            elems[b].update(need)
            self.obs.append(o)
            self.recipes.append((kind, b))
        self.bins = bins
        self.bin_elems = {b: sorted(elems[b]) for b in bins}
        self.elem_names, self._index = [], {}
        for b in bins:
            for e in self.bin_elems[b]:
                self._index[(b, e)] = len(self.elem_names)
                self.elem_names.append("{}|{}".format(b, e))

    def elementary(self, par, w):
        import flavio
        from flavio.physics.bdecays import bpll as BP

        wc = flavio.WilsonCoefficients.from_wilson(w, par)
        out = np.zeros(len(self.elem_names))
        funcs = {
            "D": BP.dGdq2_cpaverage,
            "Ndiff": BP.dGdq2_cpdiff,
            "NAFB": BP.AFB_cpaverage_num,
            "NFH": BP.FH_cpaverage_num,
        }
        for b in self.bins:
            B, P, lep, q2min, q2max = b
            for e in self.bin_elems[b]:
                f = funcs["D" if e == "BR" else e]
                v = BP.bpll_obs_int(f, q2min, q2max, wc, par, B, P, lep)
                if e == "BR":
                    v = par["tau_" + B] * v / (q2max - q2min)
                    if P == "pi0":  # pi0 = (uu - dd)/sqrt(2)
                        v = v / 2.0
                out[self._index[(b, e)]] = v
        return out

    def rebuild(self, elem, ctx=None, xp=np):
        out = []
        for kind, b in self.recipes:
            if kind == "dBR/dq2":
                v = elem[self._index[(b, "BR")]]
            else:
                v = (
                    elem[self._index[(b, _BPLL_NUM[kind])]]
                    / elem[self._index[(b, "D")]]
                )
            out.append(v)
        return xp.stack(out)


# --------------------------------------------------------------------------
# 8. B -> V gamma
# --------------------------------------------------------------------------

BVGAMMA_PROCS = {
    "B0->K*gamma": ("B0", "K*0"),
    "B+->K*gamma": ("B+", "K*+"),
    "Bs->phigamma": ("Bs", "phi"),
}
# observable -> (process, recipe).  `S`/`ADeltaGamma` need q/p of the decaying
# meson, which df2 supplies analytically; `BRt` is the time-integrated BR.
BVGAMMA_OBS = {
    "BR(B0->K*gamma)": ("B0->K*gamma", "BR"),
    "BR(B+->K*gamma)": ("B+->K*gamma", "BR"),
    "BR(Bs->phigamma)": ("Bs->phigamma", "BRt"),
    "ACP(B->Xgamma)": None,  # handled by BXgamma
    "S_K*gamma": ("B0->K*gamma", "S"),
    "S_phigamma": ("Bs->phigamma", "S"),
    "ADeltaGamma(Bs->phigamma)": ("Bs->phigamma", "ADG"),
}
BVGAMMA_RATIO = {
    "BR(B0->K*gamma)/BR(Bs->phigamma)": (("B0->K*gamma", "BR"), ("Bs->phigamma", "BRt"))
}
_BVG_MESON = {"B0->K*gamma": "B0", "B+->K*gamma": "B+", "Bs->phigamma": "Bs"}


class BVgamma(Family):
    r"""B -> V gamma: BR, the time-integrated BR, S and A_{Delta Gamma}.

    The two helicity amplitudes ``a['L']``, ``a['R']`` (and their CP conjugates)
    are linear in the Wilson coefficients, so the four real bilinears

        Gam   = (|a_L|^2 + |a_R|^2 + |abar_L|^2 + |abar_R|^2)/2
        XR/XI = Re/Im of  a_L abar_L^* + a_R abar_R^*

    are exact quadratic forms.  flavio then builds every observable of this
    class from them together with ``q/p``, which the surrogate computes
    analytically instead of asking flavio for -- that is the only reason this
    family needs a recipe at all.
    """

    uses_phis = True
    _ELEM = ("Gam", "XR", "XI", "BR", "TIa", "TIb")

    def __init__(
        self, gl, coords, obs_all, name="bvgamma", group="bs_dipole", sectors=("bs",)
    ):
        self.name = name
        self.group, self.sectors = group, sectors
        super().__init__(gl, coords, obs_all)

    def _select(self, gl, obs_all):
        self.recipes, procs = [], []
        for o in obs_all:
            key = str(o)
            if key in BVGAMMA_OBS and BVGAMMA_OBS[key] is not None:
                proc, kind = BVGAMMA_OBS[key]
                self.obs.append(o)
                self.recipes.append((kind, proc, None))
                procs.append(proc)
            elif key in BVGAMMA_RATIO:
                (pn, kn), (pd, kd) = BVGAMMA_RATIO[key]
                self.obs.append(o)
                self.recipes.append((kn, pn, (kd, pd)))
                procs += [pn, pd]
        self.procs = [p for p in BVGAMMA_PROCS if p in set(procs)]
        self.elem_names, self._index = [], {}
        for p in self.procs:
            for e in self._ELEM:
                self._index[(p, e)] = len(self.elem_names)
                self.elem_names.append("{}|{}".format(p, e))

    def elementary(self, par, w):
        import flavio
        from flavio.physics.bdecays import bvgamma as BG

        wc = flavio.WilsonCoefficients.from_wilson(w, par)
        out = np.zeros(len(self.elem_names))
        for p in self.procs:
            B, V = BVGAMMA_PROCS[p]
            a, abar = BG.get_a_abar(wc, par, B, V)
            gam = 0.5 * (
                abs(a["L"]) ** 2
                + abs(a["R"]) ** 2
                + abs(abar["L"]) ** 2
                + abs(abar["R"]) ** 2
            )
            X = a["L"] * abar["L"].conjugate() + a["R"] * abar["R"].conjugate()
            br = par["tau_" + B] * gam
            y = (
                par["DeltaGamma/Gamma_" + B] / 2.0
                if "DeltaGamma/Gamma_" + B in par
                else 0.0
            )
            vals = dict(
                Gam=gam,
                XR=X.real,
                XI=X.imag,
                BR=br,
                TIa=br / (1 - y**2),
                TIb=-y * br / (1 - y**2),
            )
            for e in self._ELEM:
                out[self._index[(p, e)]] = vals[e]
        return out

    def _piece(self, elem, kind, proc, ctx, xp):
        g = lambda e: elem[self._index[(proc, e)]]
        if kind == "BR":
            return g("BR")
        # -q/p * (XR + i XI) / Gam  -- flavio's S_A_complex
        z = -ctx["qp_" + _BVG_MESON[proc]] * (g("XR") + 1j * g("XI")) / g("Gam")
        if kind == "S":
            return z.imag
        if kind == "ADG":
            return z.real
        if kind == "BRt":
            return g("TIa") + g("TIb") * z.real
        raise KeyError(kind)

    def rebuild(self, elem, ctx, xp=np):
        out = []
        for kind, proc, den in self.recipes:
            v = self._piece(elem, kind, proc, ctx, xp)
            if den is not None:
                v = v / self._piece(elem, den[0], den[1], ctx, xp)
            out.append(v)
        return xp.stack(out)


# --------------------------------------------------------------------------
# 9. ACP(B -> X gamma): the one observable that mixes the b->s and b->d dipoles
# --------------------------------------------------------------------------


class BXgammaACP(Family):
    r"""Direct CP asymmetry of $B\to X_{s+d}\gamma$.

    ``(a_s + a_d)/(br_s + br_d)`` with both sums exact quadratic forms; the
    b->s and b->d dipoles enter together, which is why this is the only family
    carrying the union of the two dipole sectors.

    The **four** elementary quantities are the b->s and b->d pieces separately,
    not their sums.  Each piece carries one CKM structure, ``|xi_t^bq|^2``, so
    its response to a shift of Vus/Vcb/Vub is exactly multiplicative -- which is
    what the ``dlnv`` correction assumes (df2.apply_dlnv).  The *sums* are not:
    ``a_s`` and ``a_d`` cancel to about a part in 150, so the relative
    log-derivative of the sum came out at 150 and the first-order correction
    overshot by up to 0.55 sigma, which was the whole Delta chi2 budget.
    """

    name = "bxgamma_acp"
    group, sectors = "bsbd_dipole", ("bs", "bd")
    elem_names = ["num_s", "num_d", "den_s", "den_d"]
    E0 = 1.6

    def _select(self, gl, obs_all):
        self.obs = [o for o in obs_all if str(o) == "ACP(B->Xgamma)"]

    def elementary(self, par, w):
        import flavio
        from flavio.physics.bdecays import bxgamma as BX
        from flavio.physics.bdecays.wilsoncoefficients import wctot_dict

        wc_obj = flavio.WilsonCoefficients.from_wilson(w, par)
        scale = flavio.config["renormalization scale"]["bxgamma"]
        num, den = [], []
        for q in ("s", "d"):
            xi = abs(flavio.physics.ckm.xi("t", "b" + q)(par)) ** 2
            wc = wctot_dict(wc_obj, "b%see" % q, scale, par, nf_out=5)
            den.append(xi * BX.PE0_BR_BXgamma(wc, par, q, self.E0))
            num.append(xi * BX.PE0_ACP_BXgamma(wc, par, q, self.E0))
        return np.array(num + den)

    def rebuild(self, elem, ctx=None, xp=np):
        if not self.obs:
            return xp.zeros(0)
        return xp.stack([(elem[0] + elem[1]) / (elem[2] + elem[3])])


# --------------------------------------------------------------------------
# 10. everything else: depends only on (delta gamma, phi_s)
# --------------------------------------------------------------------------


class PhisConst(Family):
    """Observables inert in the semileptonic coordinates.

    Their value is ``a(dg) + b(dg) cos(phi_s) + c(dg) sin(phi_s)`` -- this covers
    both the genuinely inert observables (b = c = 0) and the Bs -> phi gamma
    ones, which depend on the mixing phase but not on C9/C10.
    """

    name = "phis_const"
    sectors = ()
    uses_phis = True
    phi_design = True  # built on a grid of phi_s instead of a coordinate design

    def __init__(self, gl, coords, obs_all, covered):
        self._covered = set(map(str, covered))
        super().__init__(gl, coords, obs_all)

    def _select(self, gl, obs_all):
        self.obs = [o for o in obs_all if str(o) not in self._covered]
        self.elem_names = [str(o) for o in self.obs]

    def elementary(self, par, w):
        import flavio

        out = []
        for o in self.obs:
            od = flavio.classes.Observable.argument_format(o, "dict")
            name = od.pop("name")
            out.append(flavio.classes.Observable[name].prediction_par(par, w, **od))
        return np.array(out, dtype=float)

    @staticmethod
    def combine_phi(vals, phis):
        """Solve value(phi) = a + b cos(phi) + c sin(phi) from >=3 (phi, value) pairs.

        `vals` is (n_phi, n_out); returns (3 * n_out,) with the a/b/c blocks stacked.
        """
        A = np.stack([np.ones(len(phis)), np.cos(phis), np.sin(phis)], axis=1)
        abc, *_ = np.linalg.lstsq(A, np.asarray(vals), rcond=None)
        return abc.reshape(-1)

    def rebuild(self, elem, ctx=None, xp=np):
        a, b, c = xp.asarray(elem).reshape(3, -1)
        phi_s = ctx["phi_s"]
        return a + xp.cos(phi_s) * b + xp.sin(phi_s) * c


# --------------------------------------------------------------------------
# 11. the CKM input observables -- not part of the 352, but part of the answer
# --------------------------------------------------------------------------

# smelli's CKM scheme (`CKMSchemeRmuBtaunuBxlnuDeltaM`) fixes Vus, Vcb, Vub and
# gamma from four observables.  The fourth, DeltaM_d/DeltaM_s, is handled
# analytically by df2.py; these are the other three.
CKM_NOCKM_OBS = ("RKpi(P+->munu)", "BR(B->Xcenu)", "BR(B+->taunu)")


class CkmInput(Family):
    r"""smelli's three non-Delta F=2 CKM inputs, with the CKM prefactor divided out.

    This is the **structural** change Phase B needs.  With only Phase A
    coordinates, Vus/Vcb/Vub are inert -- nothing moves the three observables
    that fix them -- so ``df2.Df2.solve_gamma`` collapses smelli's four-vector
    fixed point to a scalar iteration in gamma.  Charged-current coordinates end
    that: ``RKpi`` responds to ``CVL_sumunumu``/``CVL_dumunumu``, ``BR(B->Xcenu)``
    to ``CVL_bcenue`` and ``BR(B+->taunu)`` to ``CVL_butaunutau``.

    What keeps the fixed point scalar anyway is a measured fact (PLAN 8.8.3):
    the three ``nockm`` predictions are **exactly CKM-independent** -- moving
    Vus, Vcb, Vub by 1 % or gamma by 5 % changes them by <= 1e-16 relative,
    which is what dividing by ``ckm_fac`` is for.  So ``f1``, ``f2``, ``f3`` are
    functions of the coordinates *alone*: this family tabulates them once, the
    first three CKM parameters follow in closed form, and only gamma still has
    to be iterated (because ``f4`` does depend on it).

    The elementary quantities are the ``nockm`` predictions themselves, each an
    exact quadratic form (flavio's charged-current rates go as |1 + C_VL|^2).
    ``aux = True``: they feed the CKM solver, not the prediction vector.
    """

    name = "ckm_inputs"
    group, sectors = "ckm_inputs", ()
    aux = True
    elem_names = list(CKM_NOCKM_OBS)

    def _select(self, gl, obs_all):
        self.obs = []

    def elementary(self, par, w):
        from . import harness as H

        scheme = H.load_gl()._ckm_scheme
        ckm = {p: par[p] for p in H.CKM_PARS}
        return np.array(scheme.np_predictions_nockm(w=w, **ckm)[:3], dtype=float)

    def rebuild(self, elem, ctx=None, xp=np):
        return elem


# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 9. lepton-flavour ratios and differences -- `likelihood_lfu_fcnc` (PLAN 12)
# --------------------------------------------------------------------------

# flavio's `*_obs_int_ratio_leptonflavour` integrate numerator and denominator
# with epsrel = 5e-4 (not the 5e-3 default); the elementary integrals below use
# the same, so the surrogate fits exactly the numbers flavio divides.
EPS_LFU = 0.0005
RMUE_BPLL = {"B+->Kll": ("B+", "K+"), "B0->Kll": ("B0", "K0")}
RMUE_BVLL = {"B0->K*ll": ("B0", "K*0"), "B+->K*ll": ("B+", "K*+")}
RMUE_BSVLL = {"Bs->phill": ("Bs", "phi")}
_DMUE_KEY = {"Dmue_P4p": 4, "Dmue_P5p": 5}
LEPTONS = ("mu", "e")


class _LfuFamily(Family):
    """Shared bookkeeping: per-lepton bin integrals, keyed (B, M, lep, q2min, q2max, e).

    A family lists, per observable kind, the elementary integrals one lepton
    needs (`_need`); the ratio or difference is formed in `rebuild`.  Both
    leptons of every bin are always tabulated, so muon and electron integrals
    share one design -- they must, since QED running mixes the two sectors.
    """

    group, sectors = "bs_ll_lfu", SEMILEPTONIC
    PROCS = {}
    PARTS = ("",)  # BsVll-style phi decomposition where needed

    def _need(self, kind):
        raise NotImplementedError

    def _select(self, gl, obs_all):
        self.recipes, self._index, self._elems = [], {}, []
        for o in obs_all:
            kind, proc, q2min, q2max = parse(o)
            if proc not in self.PROCS:
                continue
            need = self._need(kind)
            if need is None:
                continue
            B, M = self.PROCS[proc]
            for lep in LEPTONS:
                for e in need:
                    for part in self.PARTS:
                        k = (B, M, lep, q2min, q2max, e, part)
                        if k not in self._index:
                            self._index[k] = len(self._elems)
                            self._elems.append(k)
            self.obs.append(o)
            self.recipes.append((kind, (B, M, q2min, q2max)))
        self.elem_names = ["|".join(map(str, k)) for k in self._elems]

    def _g(self, elem, b, lep, e, ctx=None, xp=np):
        B, M, q2min, q2max = b
        return elem[self._index[(B, M, lep, q2min, q2max, e, "")]]


class RmueBPll(_LfuFamily):
    """<Rmue>(B -> K ll): ratio of the CP-averaged bin-integrated rates."""

    name = "rmue_bpll"
    PROCS = RMUE_BPLL

    def _need(self, kind):
        return ["G"] if kind == "Rmue" else None

    def elementary(self, par, w):
        import flavio
        from flavio.physics.bdecays import bpll as BP

        wc = flavio.WilsonCoefficients.from_wilson(w, par)
        out = np.zeros(len(self._elems))
        for i, (B, P, lep, q2min, q2max, e, _) in enumerate(self._elems):
            out[i] = BP.bpll_obs_int(
                BP.dGdq2_cpaverage, q2min, q2max, wc, par, B, P, lep, epsrel=EPS_LFU
            )
        return out

    def rebuild(self, elem, ctx=None, xp=np):
        return xp.stack(
            [
                self._g(elem, b, "mu", "G") / self._g(elem, b, "e", "G")
                for _, b in self.recipes
            ]
        )


class RmueBVll(_LfuFamily):
    """B -> K* ll: <Rmue> (rate ratio) and <Dmue_P4p/P5p> = <P'>_mu - <P'>_e.

    The P' of each lepton is flavio's binned P' -- N_i / (2 sqrt(-N_2s N_2c)) of
    bin-integrated `S_experiment_num`, at flavio's default tolerance, exactly
    as `BVll` does it for the quark block.
    """

    name = "rmue_bvll"
    PROCS = RMUE_BVLL

    def _need(self, kind):
        if kind == "Rmue":
            return ["G"]
        if kind in _DMUE_KEY:
            return ["N%d" % _DMUE_KEY[kind], "N2s", "N2c"]
        return None

    def elementary(self, par, w):
        import flavio
        from flavio.physics.bdecays.bvll import observables as O

        wc = flavio.WilsonCoefficients.from_wilson(w, par)
        out = np.zeros(len(self._elems))
        for i, (B, V, lep, q2min, q2max, e, _) in enumerate(self._elems):
            if e == "G":
                obj = O.BVll_obs_int(O.dGdq2_ave, q2min, q2max, B, V, lep, wc, par)
                obj.epsrel = EPS_LFU
            else:
                key = _BVLL_KEYS[e]
                obj = O.BVll_obs_int(
                    lambda J, Jb, k=key: O.S_experiment_num(J, Jb, k),
                    q2min,
                    q2max,
                    B,
                    V,
                    lep,
                    wc,
                    par,
                )
            out[i] = obj()
        return out

    def rebuild(self, elem, ctx=None, xp=np):
        out = []
        for kind, b in self.recipes:
            g = lambda lep, e: self._g(elem, b, lep, e)
            if kind == "Rmue":
                out.append(g("mu", "G") / g("e", "G"))
            else:
                n = "N%d" % _DMUE_KEY[kind]
                pp = [
                    g(l, n) / (2 * xp.sqrt(-g(l, "N2s") * g(l, "N2c"))) for l in LEPTONS
                ]
                out.append(pp[0] - pp[1])
        return xp.stack(out)


class RmueBsVll(_LfuFamily):
    """<Rmue>(Bs -> phi ll): time-integrated rates, so the Bs mixing phase enters.

    Each rate is ``U + cos(phi_s) C + sin(phi_s) S`` exactly as in `BsVll`; the
    three parts are tabulated per lepton and recombined with df2's phi_s.
    """

    name = "rmue_bsvll"
    PROCS = RMUE_BSVLL
    PARTS = _BS_PHI_PARTS
    uses_phis = True
    phi_key = "phi_Bs"

    def _need(self, kind):
        return ["G"] if kind == "Rmue" else None

    def elementary(self, par, w):
        import flavio
        from flavio.physics.bdecays.bvll import observables_bs as OB

        wc = flavio.WilsonCoefficients.from_wilson(w, par)
        out = np.zeros(len(self._elems))
        done = set()
        for B, V, lep, q2min, q2max, e, _ in self._elems:
            key = (B, V, lep, q2min, q2max, e)
            if key in done:
                continue
            done.add(key)
            q0, q90, q180 = (
                _bsvll_int_phi(
                    OB.dGdq2_ave_Bs,
                    q2min,
                    q2max,
                    wc,
                    par,
                    B,
                    V,
                    lep,
                    phi,
                    epsrel=EPS_LFU,
                )
                for phi in (0.0, 0.5 * math.pi, math.pi)
            )
            out[self._index[key + ("U",)]] = 0.5 * (q0 + q180)
            out[self._index[key + ("C",)]] = 0.5 * (q0 - q180)
            out[self._index[key + ("S",)]] = q90 - 0.5 * (q0 + q180)
        return out

    def _g(self, elem, b, lep, e, ctx=None, xp=np):
        B, M, q2min, q2max = b
        k = lambda part: elem[self._index[(B, M, lep, q2min, q2max, e, part)]]
        return (
            k("U")
            + xp.cos(ctx[self.phi_key]) * k("C")
            + xp.sin(ctx[self.phi_key]) * k("S")
        )

    def rebuild(self, elem, ctx, xp=np):
        return xp.stack(
            [
                self._g(elem, b, "mu", "G", ctx, xp)
                / self._g(elem, b, "e", "G", ctx, xp)
                for _, b in self.recipes
            ]
        )


def df2_observables(coords):
    """Observables ``df2.py`` computes in closed form for *this* support.

    Not a constant: ``eps_K`` and ``x12Im_D`` are only analytic when the K0 / D0
    Delta F=2 sectors are carried.  Where they are not, they are inert and belong
    to ``phis_const`` -- which is what S1-min does.
    """
    from . import df2

    mesons = set(df2.mesons_for(coords))
    return tuple(o for o, m in df2.OBS_MESON.items() if m in mesons)


# Quadratic families: (name, group, observable names).  See ``Quadratic`` --
# membership is measured (study_results/sens_phaseA.txt), and the "is it really
# a quadratic form?" question is answered by the build's design residual.
QUADRATIC_FAMILIES = [
    dict(
        name="direct_bd",
        group="bd_ll",
        names=["BR(B0->mumu)", "BR(B0->ee)", "BR_Belle(B+->piee)"],
    ),
    # the three K-sector families need `dlnv_full` -- see Family.dlnv_full
    dict(
        name="kll",
        group="sd_ll",
        dlnv_full=True,
        names=["BR(KL->ee)", "BR(KL->mumu)", "BR(KS->ee)", "BR(KS->mumu)"],
    ),
    dict(
        name="kpinunu",
        group="sd_nunu",
        dlnv_full=True,
        names=["BR(K+->pinunu)", "BR(KL->pinunu)"],
    ),
    dict(name="bxgamma_br", group="bs_dipole", names=["BR(B->Xsgamma)"]),
    # ---- Phase B -------------------------------------------------------
    # epsp/eps is the only observable the four-quark s->d coefficients reach,
    # and it is *linear* in all 38 of its coordinates (measured: 1e-15 over a
    # random box, PLAN 8.8.3), so it gets its own degree-1 family: 39 flavio
    # calls per gamma node instead of 780.  It used to ride along in `kll`.
    dict(name="epsp", group="epsp", names=["epsp/eps"], degree=1, dlnv_full=True),
    # ---- charged current.  Membership is the measured one
    # (study_results/sens_phaseB.npz, max |dP|/sigma > 1e-5), expressed as a
    # pattern so that a new bin of the same measurement is picked up too.
    # The superallowed `Ft` values and the neutron lifetime are the *inverse*
    # of a quadratic form -- a width is quadratic, a lifetime is one over it.
    dict(
        name="cc_du",
        group="cc_du",
        pattern=r"^(Ft\(|tau_n\b|Gamma\(pi\+->munu\))",
        inverse_pattern=r"^(Ft\(|tau_n\b)",
    ),
    dict(
        name="cc_su", group="cc_su", pattern=r"^BR\((K\+|KL|KS)->(munu|pienu|pimunu)\)$"
    ),
    dict(
        name="cc_sc",
        group="cc_sc",
        pattern=r"^(BR|<BR>)\((D0|D\+)->K(e|mu)nu\)|^BR\(Ds->(mu|tau)nu\)$",
    ),
    dict(
        name="cc_dc",
        group="cc_dc",
        pattern=r"^(BR|<BR>)\((D0|D\+)->pi(e|mu)nu\)|^BR\(D\+->(mu|tau)nu\)$",
    ),
    dict(name="cc_bu", group="cc_bu", pattern=r"^BR\((B\+->munu|B0->pitaunu)\)$"),
    # ---- likelihood_lfu_fcnc (PLAN 12): the three b -> q tau tau branching
    # ratios.  |amplitude|^2, so plain quadratic forms; limits this weak make
    # them nearly inert, but they are part of the block's 17.
    dict(
        name="tautau_bs", group="bs_tautau", names=["BR(B+->Ktautau)", "BR(Bs->tautau)"]
    ),
    dict(name="tautau_bd", group="bd_ll", names=["BR(B0->tautau)"]),
]


def build_families(gl, coords, obs_all):
    """Every family the coordinate file supports, in build order.

    A family whose group is empty in this coordinate file (S1-min declares no
    `groups:` at all, and S1 declares only some) still gets built -- with no
    coordinates it degenerates to a constant, which is exactly right for a
    sector that support does not carry.  Families that match no observable are
    dropped.
    """
    fams = [
        Direct(gl, coords, obs_all, "mu"),
        Direct(gl, coords, obs_all, "e"),
        BVll(gl, coords, obs_all, "mu"),
        BVll(gl, coords, obs_all, "e"),
        BsVll(gl, coords, obs_all, "mu"),
        BsVll(gl, coords, obs_all, "e"),
        LambdabLambdall(gl, coords, obs_all),
        BqllLifetime(gl, coords, obs_all),
    ]
    # ---- Phase A: b->d, s->d, dipoles.  Only built when the support has them.
    if coords.group("bd_ll"):
        fams.append(BPll(gl, coords, obs_all))
        fams.append(
            BsVll(
                gl,
                coords,
                obs_all,
                "mu",
                name="bsvll_bd_mu",
                group="bd_ll",
                sectors=(),
                procs=BSVLL_BD_PROCS,
            )
        )
    if coords.group("bs_dipole"):
        fams.append(BVgamma(gl, coords, obs_all))
    if coords.group("bsbd_dipole"):
        fams.append(BXgammaACP(gl, coords, obs_all))
    # ---- PLAN 12: the LFU block.  Only a support that declares `bs_ll_lfu`
    # builds them; on the quark block's observables they would match nothing.
    if coords.group("bs_ll_lfu"):
        fams += [
            RmueBPll(gl, coords, obs_all),
            RmueBVll(gl, coords, obs_all),
            RmueBsVll(gl, coords, obs_all),
        ]
    for spec in QUADRATIC_FAMILIES:
        if not coords.group(spec["group"]):
            continue
        spec = dict(spec)
        # Supports without the Phase B four-quark s->d coordinates have no
        # `epsp` group, and there epsp/eps is a plain quadratic form in the
        # s->d semileptonic and dipole coordinates -- which is where S1 and
        # S1-min carry it.  Keeping that intact is what lets their tables still
        # be read by this code.
        if spec["name"] == "kll" and not coords.group("epsp"):
            spec["names"] = list(spec["names"]) + ["epsp/eps"]
        fams.append(Quadratic(gl, coords, obs_all, **spec))
    fams = [f for f in fams if f.obs]
    # ---- Phase B: the CKM inputs.  Auxiliary (no observables of its own), so
    # it is added after the `f.obs` filter and before `phis_const` is told what
    # is already covered.
    if coords.group("ckm_inputs"):
        fams.append(CkmInput(gl, coords, obs_all))
    covered = [o for f in fams for o in f.obs] + list(df2_observables(coords))
    rest = PhisConst(gl, coords, obs_all, covered)
    # the LFU block is covered entirely by its own families; an empty phi-design
    # family would only cost build time
    if rest.obs:
        fams.append(rest)
    return fams
