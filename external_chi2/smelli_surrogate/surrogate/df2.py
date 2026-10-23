r"""Delta F = 2 sectors and smelli's CKM re-extraction, handled analytically.

``flavio.physics.mesonmixing.amplitude.M12`` is **exactly linear** in the Delta F=2
Wilson coefficients (the matrix elements are parameter-only), and Gamma_12 carries
no new physics at all.  So nothing in this sector is ever fitted: for every meson
whose Delta F=2 sector appears in the coordinate file we tabulate

* ``M12sm(dg)``  -- the SM amplitude,
* ``k_j(dg)``    -- the response to each Delta F=2 coefficient of that sector,
* ``G12(dg)``    -- the (NP-free) width difference amplitude,

on the ``gamma_nodes`` grid, and interpolate exactly by a polynomial in
:math:`\delta\gamma = \gamma - \gamma_{\rm SM}`.  Everything downstream --
:math:`\Delta M_q`, :math:`q/p`, :math:`\phi_s`, :math:`S_{\psi\phi}`,
:math:`S_{\psi K}`, :math:`\epsilon_K` -- is then closed-form.

Of smelli's four CKM input observables, ``DeltaM_d/DeltaM_s`` is the one that
responds through this sector.  The other three (``RKpi``, ``BR(B->Xcenu)``,
``BR(B+->taunu)``) respond only to charged-current coefficients, and their
``nockm`` predictions are exactly CKM-independent (measured: <= 1e-16 relative
for a 1 % shift of Vus/Vcb/Vub or 5 % of gamma).  So ``f1``, ``f2``, ``f3`` are
functions of the coordinates alone -- supplied by ``families.CkmInput`` when the
support carries charged-current coordinates, and frozen at their SM values
otherwise -- ``Vus``/``Vcb``/``Vub`` follow in closed form, and the 10-step fixed
point of ``smelli.ckm`` still collapses to a scalar iteration in
:math:`\gamma`.
"""

import numpy as np

from . import design

# The three CKM inputs that are not gamma.  Everything in this module, and every
# family table, is built at their SM-extracted values with delta gamma as the only
# CKM axis; charged-current coordinates move them too (see `families.CkmInput`), so
# the *relative* response to each is tabulated alongside and applied as a first-order
# factor.  d ln V is at most ~1e-3 over the support box, so the second-order term is
# below 1e-6 -- measured in PLAN 8.8.4.
DLNV = ("Vus", "Vcb", "Vub")
EPS_LNV = 1.0e-2  # step for the central log-derivative


def dlnv_of(ckm, meta, xp=np):
    """``[d ln Vus, d ln Vcb, d ln Vub]`` of an extracted CKM vs the tabulated one."""
    return xp.stack([xp.log(ckm[i] / meta[DLNV[i]]) for i in range(3)])


def apply_dlnv(q0, coef, dlnv, dg, xp=np):
    """``q0 * (1 + sum_k a_k(dg) d ln V_k)``; a no-op when either input is None."""
    if coef is None or dlnv is None:
        return q0
    a = design.poly_eval(coef, dg, xp)  # (3, ...)
    return q0 * (1 + xp.tensordot(dlnv, a, axes=(0, 0)))


# meson -> (Delta F=2 coordinate sector, flavio's quark label)
MESON_SECTOR = {"Bs": "bsbs", "B0": "bdbd", "K0": "sdsd", "D0": "ucuc"}

# the two mixing-induced CP asymmetries this block measures
S_OBS = {"S_psiphi": ("Bs", +1), "S_psiK": ("B0", -1)}

# Delta F=2 observable -> the meson whose mixing produces it
OBS_MESON = {
    "DeltaM_s": "Bs",
    "S_psiphi": "Bs",
    "S_psiK": "B0",
    "eps_K": "K0",
    "x12Im_D": "D0",
}


def mesons_for(coords):
    """Which mesons this support handles analytically.

    Bs and B0 always: smelli's CKM extraction needs DeltaM_d/DeltaM_s whether or
    not the Delta F=2 coefficients are coordinates.  K0 and D0 only when their
    sector is carried -- otherwise eps_K and x12Im_D are inert and belong to
    ``phis_const`` instead.
    """
    return [
        m
        for m, sector in MESON_SECTOR.items()
        if sector in coords.sectors or m in ("Bs", "B0")
    ]


# --------------------------------------------------------------------------
# smelli.ckm.CKMSchemeRmuBtaunuBxlnuDeltaM, transcribed
# --------------------------------------------------------------------------


def ckm_fac(Vus, Vcb, Vub, gamma, xp=np):
    cos, sqrt = xp.cos, xp.sqrt
    return [
        (Vus**2 / (1 - Vub**2 - Vus**2)),
        Vcb**2,
        Vub**2,
        -(
            (
                Vcb**2 * Vus**2
                + Vub**2 * (-1 + Vcb**2 + Vub**2) * (-1 + Vub**2 + Vus**2)
                - 2
                * Vcb
                * Vub
                * Vus
                * ((-1 + Vcb**2 + Vub**2) ** 2 * (-1 + Vub**2 + Vus**2) ** 2) ** 0.25
                * cos(gamma)
            )
            / (
                Vcb**2 * (-1 + Vub**2)
                + (Vcb**2 + (-1 + Vcb**2) * Vub**2 + Vub**4) * Vus**2
                - 2
                * Vcb
                * Vub
                * Vus
                * sqrt((-1 + Vcb**2 + Vub**2) * (-1 + Vub**2 + Vus**2))
                * cos(gamma)
            )
        ),
    ]


def gamma_from_fac(f1, f2, f3, f4, xp=np):
    """The gamma component of ``CKMSchemeRmuBtaunuBxlnuDeltaM.get_ckm``.

    Returns ``(gamma, excess)``; ``excess`` is how far the arccos argument left
    [-1, 1].  smelli raises there ("extraction of CKM elements failed"); under
    JAX we cannot raise, so the argument is clipped and the caller turns the
    excess into a chi2 penalty.
    """
    sqrt = xp.sqrt
    arg = -(
        (-1 + f3)
        * (f3 - f3 * (f2 + f3) - f2 * f4 + f1 * (f2 + f3 * (-1 + f2 + f3) * f4))
    ) / (
        2.0
        * sqrt(1 + f1)
        * sqrt(f2)
        * sqrt(f3)
        * sqrt(f1 - f1 * f3)
        * (
            (((-1 + f3) ** 2 * (-1 + f2 + f3) ** 2) / (1 + f1) ** 2) ** 0.25
            + sqrt(((-1 + f3) * (-1 + f2 + f3)) / (1 + f1)) * f4
        )
    )
    excess = xp.maximum(xp.abs(arg) - 1.0, 0.0)
    return xp.arccos(xp.clip(arg, -1.0, 1.0)), excess


# --------------------------------------------------------------------------
# mesonmixing.common, transcribed (numpy/complex, no branch surprises)
# --------------------------------------------------------------------------


def q_over_p(M12, G12, xp=np):
    z = 2 * M12 - 1j * G12
    # note: flavio conjugates M12 and G12 separately, *not* the combination z
    zc = 2 * xp.conj(M12) - 1j * xp.conj(G12)
    return -xp.sqrt(zc / z)


def delta_m(M12, G12, xp=np):
    return -2 * (q_over_p(M12, G12, xp) * (M12 - 0.5j * G12)).real


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def build(gl, coords, verbose=True):
    """Tabulate the parameter-only Delta F=2 quantities on the delta-gamma grid.

    One block per meson whose sector appears in ``coords``.  ``B0`` is always
    included even if its sector is not a coordinate, because smelli's CKM
    extraction needs ``DeltaM_d``.
    """
    import flavio
    from flavio.physics.common import conjugate_par
    from flavio.physics.mesonmixing import observables as mmobs

    from . import harness as H

    nodes = coords.gamma_nodes
    g_sm = H.par_sm(gl)["gamma"]
    w0 = H.make_w({}, scale=coords.scale)

    mesons = {}
    for meson in mesons_for(coords):
        sector = MESON_SECTOR[meson]
        idx = coords.sectors.get(sector, [])
        wcs = [coords.wcs[i][0] for i in idx[::2]]  # one name per complex WC
        units = [coords.units[i] for i in idx[::2]]
        mesons[meson] = dict(
            sector=sector,
            idx=np.array(idx, dtype=int),
            wcs=wcs,
            units=np.array(units, dtype=float),
        )

    # a unit probe per Delta F=2 coefficient; M12 is linear, so one call each
    probes = {}
    for meson, m in mesons.items():
        for wc, unit in zip(m["wcs"], m["units"]):
            probes[wc] = H.make_w({wc: unit}, scale=coords.scale)

    ckm_sm = H.ckm_sm(gl)

    def node_row(dg, ckm_shift=None):
        par = H.par_with_ckm(gl, gamma=g_sm + dg, **(ckm_shift or {}))
        wc0 = H.wc_obj(w0, par)
        out = {}
        for meson, m in mesons.items():
            M0, G0 = mmobs.get_M12_G12(wc0, par, meson)
            k = []
            for wc, unit in zip(m["wcs"], m["units"]):
                M1, _ = mmobs.get_M12_G12(H.wc_obj(probes[wc], par), par, meson)
                k.append((M1 - M0) / unit)
            out[meson] = dict(
                M12sm=M0,
                G12=G0,
                k=np.array(k, dtype=complex) if k else np.zeros(0, dtype=complex),
            )
        return out

    rows = {meson: [] for meson in mesons}
    # rows_v[meson][key] gets shape (n_nodes, 3) + shape(key): the relative
    # log-derivative with respect to Vus, Vcb, Vub
    rows_v = {meson: [] for meson in mesons}
    for dg in nodes:
        r0 = node_row(dg)
        for meson in mesons:
            rows[meson].append(r0[meson])
        drv = {meson: {key: [] for key in ("M12sm", "G12", "k")} for meson in mesons}
        for p in DLNV:
            rp = node_row(dg, {p: ckm_sm[p] * (1 + EPS_LNV)})
            rm = node_row(dg, {p: ckm_sm[p] * (1 - EPS_LNV)})
            for meson in mesons:
                for key in ("M12sm", "G12", "k"):
                    q0 = r0[meson][key]
                    d = (rp[meson][key] - rm[meson][key]) / (2 * EPS_LNV)
                    drv[meson][key].append(
                        np.where(
                            np.abs(q0) > 0, d / np.where(np.abs(q0) > 0, q0, 1.0), 0.0
                        )
                    )
        for meson in mesons:
            rows_v[meson].append(
                {key: np.array(drv[meson][key]) for key in ("M12sm", "G12", "k")}
            )
        if verbose:
            print(
                "  df2 node dg=%+.4f  " % dg
                + "  ".join(
                    "|M12_{}|={:.4e}".format(m, abs(rows[m][-1]["M12sm"]))
                    for m in mesons
                ),
                flush=True,
            )

    tab = {}
    for meson, m in mesons.items():
        block = dict(m)
        for key in ("M12sm", "G12", "k"):
            block[key] = design.interp_nodes(
                nodes, np.array([r[key] for r in rows[meson]])
            )
            block[key + "_dlnv"] = design.interp_nodes(
                nodes, np.array([r[key] for r in rows_v[meson]])
            )
        tab[meson] = block

    # amplitude ratios of the mixing-induced CP asymmetries (parameter-only)
    amps = {"Bs": mmobs.amplitude_Bspsiphi, "B0": mmobs.amplitude_BJpsiK}
    for obs, (meson, etaCP) in S_OBS.items():
        if meson not in tab:
            continue

        def _ratio(dg, shift=None):
            par = H.par_with_ckm(gl, gamma=g_sm + dg, **(shift or {}))
            return amps[meson](par) / amps[meson](conjugate_par(par))

        ratios = [_ratio(dg) for dg in nodes]
        drv = [
            [
                (
                    _ratio(dg, {p: ckm_sm[p] * (1 + EPS_LNV)})
                    - _ratio(dg, {p: ckm_sm[p] * (1 - EPS_LNV)})
                )
                / (2 * EPS_LNV)
                / _ratio(dg)
                for p in DLNV
            ]
            for dg in nodes
        ]
        tab[meson]["Aratio"] = design.interp_nodes(nodes, np.array(ratios))
        tab[meson]["Aratio_dlnv"] = design.interp_nodes(nodes, np.array(drv))
        tab[meson]["etaCP"] = float(etaCP)

    # a few more parameter-only numbers: eps_K's normalisation and the
    # lifetimes that turn |M12| into x12
    par = H.par_with_ckm(gl)
    for meson in tab:
        # K0 is not a mass eigenstate and has no lifetime parameter; nothing
        # here needs one (x12 is only used for D0).
        if "tau_" + meson in par:
            tab[meson]["tau"] = float(par["tau_" + meson])
    if "K0" in tab:
        tab["K0"]["keps"] = float(par["kappa_epsilon"])
        tab["K0"]["DeltaM_K0"] = float(par["DeltaM_K0"])

    # CKM scheme constants, frozen at the SM extraction
    ckm = H.ckm_sm(gl)
    fac_sm = ckm_fac(**ckm)
    scheme = gl._ckm_scheme
    exp = scheme.exp_measurements()
    exp_cen = [float(exp[o].central_value) for o in scheme.observables]
    # The first three CKM inputs enter only through exp_cen_i / nockm_i(x).  At
    # the SM point that ratio *is* ckm_fac(SM)[i] -- that is how smelli extracted
    # the SM CKM in the first place -- so this is a free consistency check.
    nockm_sm = np.array(scheme.np_predictions_nockm(w=None, **ckm))
    rel = np.array([exp_cen[i] / nockm_sm[i] / fac_sm[i] - 1 for i in range(3)])
    if verbose:
        print(
            "  df2 CKM inputs: exp/nockm vs ckm_fac(SM), relative: %s"
            % np.array2string(rel, precision=2),
            flush=True,
        )
    tab["meta"] = dict(
        gamma_sm=g_sm,
        Vus=ckm["Vus"],
        Vcb=ckm["Vcb"],
        Vub=ckm["Vub"],
        f1=fac_sm[0],
        f2=fac_sm[1],
        f3=fac_sm[2],
        exp_cen4=exp_cen[3],
        exp_cen123=np.array(exp_cen[:3]),
        nockm_sm=nockm_sm[:3],
        nodes=nodes,
        mesons=list(mesons),
    )
    return tab


# --------------------------------------------------------------------------
# evaluation (numpy or jax.numpy)
# --------------------------------------------------------------------------


class Df2:
    """Analytic Delta F=2 sectors + smelli's gamma re-extraction."""

    N_ITER = 10  # smelli's fixed trip count -- keeps this traceable

    def __init__(self, tab, xp=np):
        self.tab = tab
        self.m = tab["meta"]
        self.mesons = self.m["mesons"]
        self.xp = xp

    def _p(self, meson, key, dg, dlnv=None):
        q0 = design.poly_eval(self.tab[meson][key], dg, self.xp)
        return apply_dlnv(q0, self.tab[meson].get(key + "_dlnv"), dlnv, dg, self.xp)

    def V123(self, nockm=None):
        """``(Vus, Vcb, Vub)`` -- closed form, no iteration (see `fac123`)."""
        xp = self.xp
        f1, f2, f3 = self.fac123(nockm)
        return (xp.sqrt(f1 - f1 * f3) / xp.sqrt(1 + f1), xp.sqrt(f2), xp.sqrt(f3))

    def dlnv(self, nockm=None):
        """``d ln V`` of this point vs the tabulated CKM, or None if inert."""
        if nockm is None:
            return None
        return dlnv_of(self.V123(nockm), self.m, self.xp)

    # ----------------------------------------------------------------
    def M12(self, meson, x, dg, dlnv=None):
        """M12 of `meson`; exactly linear in that sector's Delta F=2 coefficients."""
        xp = self.xp
        b = self.tab[meson]
        out = self._p(meson, "M12sm", dg, dlnv)
        idx = b["idx"]
        if len(idx):
            k = self._p(meson, "k", dg, dlnv)
            c = xp.asarray(x)[idx[0::2]] + 1j * xp.asarray(x)[idx[1::2]]
            out = out + xp.sum(k * c * xp.asarray(b["units"]))
        return out

    def mixing(self, meson, x, dg, dlnv=None):
        """(|DeltaM|, phi, q/p, signed DeltaM) of `meson`."""
        xp = self.xp
        M12 = self.M12(meson, x, dg, dlnv)
        G12 = self._p(meson, "G12", dg, dlnv)
        qp = q_over_p(M12, G12, xp)
        DM = -2 * (qp * (M12 - 0.5j * G12)).real
        return xp.abs(DM), xp.angle(-qp), qp, DM

    # ----------------------------------------------------------------
    def fac123(self, nockm=None):
        """``(f1, f2, f3)`` -- the CKM prefactors of the three non-mixing inputs.

        ``nockm`` is what ``families.CkmInput`` returns at this point; without it
        (Phase A supports, which cannot move those observables) the SM values are
        used, which is what the frozen ``m['f1'..'f3']`` are.
        """
        m = self.m
        if nockm is None:
            return m["f1"], m["f2"], m["f3"]
        xp = self.xp
        cen = xp.asarray(m["exp_cen123"])
        f = cen / xp.asarray(nockm)
        return f[0], f[1], f[2]

    def ckm(self, x, nockm=None):
        """``(Vus, Vcb, Vub, gamma)`` -- smelli's extraction at this point.

        Vus/Vcb/Vub follow from ``f1, f2, f3`` in closed form (``smelli.ckm``'s
        ``get_ckm``); only gamma has to be iterated.
        """
        Vus, Vcb, Vub = self.V123(nockm)
        return Vus, Vcb, Vub, self.m["gamma_sm"] + self.solve_gamma(x, nockm=nockm)

    def solve_gamma(self, x, with_excess=False, nockm=None):
        """smelli's CKM fixed point, collapsed to gamma.

        The first three inputs fix Vus/Vcb/Vub through ``f1, f2, f3``, which do
        not depend on the CKM at all (``fac123``), so the 10-step vector
        iteration reduces to a scalar one in gamma even when charged-current
        coordinates move them.  The trip count is fixed, so this traces under
        JAX.

        ``f4`` uses the extracted Vus/Vcb/Vub, consistently with the tabulated
        ``DeltaM_d``/``DeltaM_s``, which ``apply_dlnv`` has corrected to the same
        values.
        """
        xp, m = self.xp, self.m
        f1, f2, f3 = self.fac123(nockm)
        dlnv = self.dlnv(nockm)
        dg = xp.zeros(()) * 1.0
        excess = xp.zeros(()) * 1.0
        for _ in range(self.N_ITER):
            DMs = self.mixing("Bs", x, dg, dlnv)[0]
            DMd = self.mixing("B0", x, dg, dlnv)[0]
            Vus, Vcb, Vub = self.V123(nockm)
            f4 = (
                m["exp_cen4"]
                * ckm_fac(Vus, Vcb, Vub, m["gamma_sm"] + dg, xp)[3]
                / (DMd / DMs)
            )
            gamma, exc = gamma_from_fac(f1, f2, f3, f4, xp)
            excess = excess + exc
            dg = gamma - m["gamma_sm"]
        return (dg, excess) if with_excess else dg

    # ----------------------------------------------------------------
    def context(self, x, dg, dlnv=None):
        """Everything the families and the observable slots need.

        Keys: ``DeltaM_<q>``, ``phi_<meson>``, ``qp_<meson>``, plus the
        Delta F=2 observables of this block (``DeltaM_s``, ``S_psiphi``,
        ``S_psiK``, ``eps_K``).  ``phi_s`` is kept as an alias for the
        S1-min families.
        """
        xp = self.xp
        out = {}
        for meson in self.mesons:
            DM, phi, qp, DM_signed = self.mixing(meson, x, dg, dlnv)
            out["DeltaM_" + meson] = DM
            out["phi_" + meson] = phi
            # `qp_<meson>` is flavio's `mesonmixing.observables.q_over_p`, used
            # as-is by B->V gamma.  `S()` -- and only `S()` -- flips its sign
            # when DeltaM comes out negative, so that variant is kept separate.
            out["qp_" + meson] = qp
            out["qpS_" + meson] = xp.where(DM_signed < 0, -qp, qp)
            out["M12_" + meson] = self.M12(meson, x, dg, dlnv)
        out["phi_s"] = out["phi_Bs"]
        out["DeltaM_s"] = out["DeltaM_Bs"]
        out["DeltaM_d"] = out["DeltaM_B0"]
        for obs, (meson, _eta) in S_OBS.items():
            if meson in self.mesons:
                xi = (
                    self.tab[meson]["etaCP"]
                    * out["qpS_" + meson]
                    * self._p(meson, "Aratio", dg, dlnv)
                )
                out[obs] = -2 * xi.imag / (1 + xp.abs(xi) ** 2)
        if "K0" in self.mesons:
            b = self.tab["K0"]
            out["eps_K"] = (
                b["keps"] * out["M12_K0"].imag / b["DeltaM_K0"] / xp.sqrt(2.0)
            )
        if "D0" in self.mesons:
            # x12Im = x12 sin(phi12) = 2 tau Im(M12 conj(G12)) / |G12|
            G12 = self._p("D0", "G12", dg, dlnv)
            out["x12Im_D"] = (
                2
                * self.tab["D0"]["tau"]
                * (out["M12_D0"] * xp.conj(G12)).imag
                / xp.abs(G12)
            )
        return out

    # kept for the S1-min call sites
    def observables(self, x, dg, dlnv=None):
        return self.context(x, dg, dlnv)
