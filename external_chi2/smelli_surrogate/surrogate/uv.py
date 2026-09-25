r"""Z' UV -> SMEFT matching, JAX-traceable.

A faithful mirror of ``code/local_modules/physics_functions.calcSMEFTwcs_gentype``
(and the ``fill_qq1`` / ``fill_in_end`` helpers in ``code/local_modules/utils.py``),
rewritten so that the mixing angles and phases can be traced parameters.

The algebra is unchanged; only the array module is abstracted.  ``check()``
compares this implementation against the project's own at random points and is
the thing to run after any edit -- it must agree to machine precision.

Downstream, SMEFT(M) -> WET is done with **wilson**, which is what
``calc_likelihoods_gentype`` gets when it calls
``gl.parameter_point(SMEFT_coeff, scale=M)``.  ``wilson``'s
``smeft_accuracy`` is set to ``'integrate'``, matching ``utils.maybe_write``.
"""

import numpy as np

# Parameters the matching actually reads.  Note `tme`, `tet`, `tmt` are parsed by
# extract_uv_parameters but are NOT used by calcSMEFTwcs_gentype -- lepton mixing
# is not implemented in the matching, so floating them does nothing.
UV_PARAMS = ("g", "tsb", "tdb", "tds", "delta", "phi1", "phi2", "phi3", "phi4", "phi5")
CHARGES = ("xl1", "xl2", "xl3", "xe1", "xe2", "xe3", "xq3", "xu3", "xd3", "xh")


def _eye_like(a, xp):
    return xp.asarray(a)


def vdl_matrix(p, xp=np):
    """VdL = phase_L . rot_23 . rot_13 . rot_12 . phase_R (physics_functions L206-244)."""
    one = xp.ones(()) * 1.0
    zero = xp.zeros(()) * 1.0
    s23, c23 = xp.sin(p["tsb"]), xp.cos(p["tsb"])
    s13, c13 = xp.sin(p["tdb"]), xp.cos(p["tdb"])
    s12, c12 = xp.sin(p["tds"]), xp.cos(p["tds"])
    e = lambda a: xp.exp(1j * a)

    def M(rows):
        return xp.stack([xp.stack([r * (1 + 0j) for r in row]) for row in rows])

    rot_23 = M([[one, zero, zero], [zero, c23, s23], [zero, -s23, c23]])
    rot_13 = M(
        [
            [c13, zero, s13 * e(-p["delta"])],
            [zero, one, zero],
            [-s13 * e(p["delta"]), zero, c13],
        ]
    )
    rot_12 = M([[c12, s12, zero], [-s12, c12, zero], [zero, zero, one]])
    phase_L = M(
        [[one, zero, zero], [zero, e(p["phi1"]), zero], [zero, zero, e(p["phi2"])]]
    )
    phase_R = M(
        [
            [e(-p["phi3"]), zero, zero],
            [zero, e(-p["phi4"]), zero],
            [zero, zero, e(-p["phi5"])],
        ]
    )
    return phase_L @ rot_23 @ rot_13 @ rot_12 @ phase_R


def xi_matrix(VdL, xp=np):
    """l_xi[i][k] = conj(VdL[2, i]) * VdL[2, k] -- the b-row couplings."""
    row = VdL[2]
    return [[xp.conj(row[i]) * row[k] for k in range(3)] for i in range(3)]


def _fill_in_end(init_str, num, mix, end_str=""):
    """utils.fill_in_end, verbatim."""
    out = {}
    for i in range(1, 4):
        for j in range(i, 4):
            out[init_str + str(i) + str(j) + end_str] = num * mix[i - 1][j - 1]
    return out


def _fill_qq1(alpha, mix, xq3, qq1_indices):
    """utils.fill_qq1, verbatim (static index bookkeeping, traced values)."""
    base = -0.5 * alpha * xq3**2
    out = {}
    for i in range(1, 4):
        for j in range(1, 4):
            for k in range(1, 4):
                for l in range(1, 4):
                    idx = 1000 * i + 100 * j + 10 * k + l
                    perm = 1000 * k + 100 * l + 10 * i + j
                    hc1 = 1000 * l + 100 * k + 10 * j + i
                    hc2 = 1000 * j + 100 * i + 10 * l + k
                    perms = [idx, perm, hc1, hc2]
                    norm = ["qq1_%d" % idx, "qq1_%d" % perm]
                    match = [m for m in qq1_indices if m in perms][0]
                    name = "qq1_%d" % match
                    if name in norm:
                        term = base * mix[i - 1][j - 1] * mix[k - 1][l - 1]
                        out[name] = out[name] + term if name in out else term
    return out


def smeft_wcs(p, charges, qq1_indices, xp=np):
    """SMEFT Warsaw coefficients at the matching scale M, as a dict of complex."""
    x = charges
    alpha = p["g"] ** 2 / p["M"] ** 2
    mix = xi_matrix(vdl_matrix(p, xp), xp)
    a = alpha
    out = {
        "ll_1111": -(1 / 2) * x["xl1"] ** 2 * a,
        "ll_1122": -x["xl1"] * x["xl2"] * a,
        "ll_1133": -x["xl1"] * x["xl3"] * a,
        "ll_2222": -(1 / 2) * x["xl2"] ** 2 * a,
        "ll_2233": -x["xl2"] * x["xl3"] * a,
        "ll_3333": -(1 / 2) * x["xl3"] ** 2 * a,
        **_fill_in_end("lq1_33", -x["xq3"] * x["xl3"] * a, mix),
        **_fill_in_end("lq1_22", -x["xq3"] * x["xl2"] * a, mix),
        **_fill_in_end("lq1_11", -x["xq3"] * x["xl1"] * a, mix),
        "ee_1111": -(1 / 2) * x["xe1"] ** 2 * a,
        "ee_1122": -x["xe1"] * x["xe2"] * a,
        "ee_1133": -x["xe1"] * x["xe3"] * a,
        "ee_2222": -(1 / 2) * x["xe2"] ** 2 * a,
        "ee_2233": -x["xe2"] * x["xe3"] * a,
        "ee_3333": -(1 / 2) * x["xe3"] ** 2 * a,
        "uu_3333": -(1 / 2) * x["xu3"] ** 2 * a,
        "dd_3333": -(1 / 2) * x["xd3"] ** 2 * a,
        "eu_1133": -x["xe1"] * x["xu3"] * a,
        "eu_2233": -x["xe2"] * x["xu3"] * a,
        "eu_3333": -x["xe3"] * x["xu3"] * a,
        "ed_1133": -x["xe1"] * x["xd3"] * a,
        "ed_2233": -x["xe2"] * x["xd3"] * a,
        "ed_3333": -x["xe3"] * x["xd3"] * a,
        "ud1_3333": -x["xu3"] * x["xd3"] * a,
        "le_1111": -x["xl1"] * x["xe1"] * a,
        "le_1122": -x["xl1"] * x["xe2"] * a,
        "le_1133": -x["xl1"] * x["xe3"] * a,
        "le_2211": -x["xl2"] * x["xe1"] * a,
        "le_2222": -x["xl2"] * x["xe2"] * a,
        "le_2233": -x["xl2"] * x["xe3"] * a,
        "le_3311": -x["xl3"] * x["xe1"] * a,
        "le_3322": -x["xl3"] * x["xe2"] * a,
        "le_3333": -x["xl3"] * x["xe3"] * a,
        "lu_1133": -x["xl1"] * x["xu3"] * a,
        "lu_2233": -x["xl2"] * x["xu3"] * a,
        "lu_3333": -x["xl3"] * x["xu3"] * a,
        "ld_1133": -x["xl1"] * x["xd3"] * a,
        "ld_2233": -x["xl2"] * x["xd3"] * a,
        "ld_3333": -x["xl3"] * x["xd3"] * a,
        **_fill_in_end("qe_", -x["xq3"] * x["xe1"] * a, mix, "11"),
        **_fill_in_end("qe_", -x["xq3"] * x["xe2"] * a, mix, "22"),
        **_fill_in_end("qe_", -x["xq3"] * x["xe3"] * a, mix, "33"),
        **_fill_in_end("qu1_", -x["xq3"] * x["xu3"] * a, mix, "33"),
        **_fill_in_end("qd1_", -x["xq3"] * x["xd3"] * a, mix, "33"),
        "phil1_11": -x["xh"] * x["xl1"] * a,
        "phil1_22": -x["xh"] * x["xl2"] * a,
        "phil1_33": -x["xh"] * x["xl3"] * a,
        **_fill_in_end("phiq1_", -x["xh"] * x["xq3"] * a, mix),
        "phie_11": -x["xh"] * x["xe1"] * a,
        "phie_22": -x["xh"] * x["xe2"] * a,
        "phie_33": -x["xh"] * x["xe3"] * a,
        "phiu_33": -x["xh"] * x["xu3"] * a,
        "phid_33": -x["xh"] * x["xd3"] * a,
        "phiD": -2 * x["xh"] ** 2 * a,
        "phiBox": -(1 / 2) * x["xh"] ** 2 * a,
    }
    out.update(_fill_qq1(alpha, mix, x["xq3"], qq1_indices))
    return out


# --------------------------------------------------------------------------


def load_model(path):
    """Read the project's Zp_model.yaml -> (charges, qq1 indices, M, defaults)."""
    import yaml

    with open(path) as f:
        m = yaml.safe_load(f)
    charges = {}
    charges.update(m.get("lepton_charges", {}))
    charges.update(m.get("quark_charges", {}))
    charges["xh"] = m.get("higgs_charge", {}).get("xh", 0)
    defaults = {
        "M": float(m["Zprime_properties"]["M"]),
        "g": float(m["Zprime_properties"]["g"]),
    }
    for block, keys in (
        ("quark_mixing_angles", ("tsb", "tdb", "tds")),
        ("lepton_mixing_angles", ("tme", "tet", "tmt")),
        ("complex_phases", ("delta", "phi1", "phi2", "phi3", "phi4", "phi5")),
    ):
        for k in keys:
            defaults[k] = float(m.get(block, {}).get(k, 0.0))
    return charges, list(m["nonredundant_indices"]["qq1"]), defaults, m


def check(model_path, n=8, seed=0):
    """Compare against physics_functions.calcSMEFTwcs_gentype at random points."""
    import copy
    import os
    import sys

    sys.path.insert(0, os.path.expanduser("~/Projects/zprime_NFU/code/local_modules"))
    from physics_functions import calcSMEFTwcs_gentype

    charges, qq1_idx, defaults, raw = load_model(model_path)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for t in range(n):
        p = dict(defaults)
        if t:
            p["g"] = rng.uniform(0.05, 0.4)
            for k in ("tsb", "tdb", "tds"):
                p[k] = rng.uniform(-0.2, 0.2)
            for k in ("delta", "phi1", "phi2", "phi3", "phi4", "phi5"):
                p[k] = rng.uniform(-np.pi, np.pi)
        mine = smeft_wcs(p, charges, qq1_idx, np)
        card = copy.deepcopy(raw)
        card["Zprime_properties"]["g"] = p["g"]
        card["Zprime_properties"]["M"] = p["M"]
        card["quark_mixing_angles"] = {k: p[k] for k in ("tsb", "tdb", "tds")}
        card["complex_phases"] = {
            k: p[k] for k in ("delta", "phi1", "phi2", "phi3", "phi4", "phi5")
        }
        ref = calcSMEFTwcs_gentype(card, [])
        assert set(ref) == set(mine), set(ref) ^ set(mine)
        scale = max(abs(np.array(list(ref.values()))).max(), 1e-300)
        d = max(abs(complex(mine[k]) - complex(ref[k])) for k in ref) / scale
        worst = max(worst, d)
    return worst
