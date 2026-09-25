r"""The external contract: WET coefficients in the **JMS basis at mu = M_Z**.

PLAN.md §2 fixes the surrogate's input as WCxf JMS names + ``_R``/``_I`` part,
i.e. what ``wilson.Wilson(...).match_run(91.1876, 'WET', 'JMS').dict`` or
rgevolve's ``run_and_match(..., 'JMS', M, 91.1876)`` produce.  Internally the
surrogate works in the flavio basis, so we need the JMS -> flavio rotation.

That rotation is **not** CKM-independent: flavio normalises its Delta F=1
coefficients by ``4 G_F/sqrt(2) * lambda_t``, so the map carries the CKM phase
and therefore moves with smelli's re-extracted gamma (measured: 4.7 % over
delta gamma = 0.09, i.e. ~1e-3 on C9 -- enough to matter at the target
accuracy).  The map is therefore tabulated on the same ``gamma_nodes`` grid as
everything else.

The Delta F=2 block is dimensionful and carries no CKM factor, so ``CVLL_bsbs``
is gamma-independent; that is what keeps the gamma fixed point from coupling
back into the input map.
"""
import numpy as np

from . import design
from .df2 import DLNV, EPS_LNV


def jms_coordinates(sectors):
    """Ordered (name, part) list of the real JMS coordinates of `sectors`."""
    from wilson import wcxf
    basis = wcxf.Basis['WET', 'JMS']
    out = []
    for s in sectors:
        if s not in basis.sectors:
            raise KeyError('unknown JMS sector %r; available: %s'
                           % (s, sorted(basis.sectors)))
        for name, prop in basis.sectors[s].items():
            out.append((name, 'R'))
            if prop is None or prop.get('real') is not True:
                out.append((name, 'I'))
    return out


def _project(coords, fl):
    col = np.zeros(len(coords))
    for i, (wcname, p) in enumerate(coords.wcs):
        v = fl.get(wcname, 0.0)
        col[i] = (np.real(v) if p == 'R' else np.imag(v)) / coords.units[i]
    return col


def _column_wcxf(coords, name, part):
    """Cheap wcxf translation -- used only to discover which columns are nonzero."""
    from wilson import wcxf
    val = 1.0 if part == 'R' else 1j
    wc = wcxf.WC(eft='WET', basis='JMS', scale=coords.scale,
                 values={name: {'Re': val.real, 'Im': val.imag}})
    return _project(coords, wc.translate('flavio').dict)


def _column(coords, name, part, ckm):
    """The rotation flavio actually applies: wilson, with smelli's CKM stamped on.

    Going through ``wilson.Wilson.match_run`` rather than a bare
    ``wcxf.WC.translate`` matters: the two disagree by a uniform 4e-4 in the
    normalisation, because wcxf's default translator parameters are not
    wilson's.
    """
    import wilson as _w
    val = 1.0 if part == 'R' else 1j
    w = _w.Wilson({name: val}, scale=coords.scale, eft='WET', basis='JMS')
    # `get_option` returns the *class-level* default dict by reference when the
    # instance has no override, so updating it in place would rewrite the CKM
    # defaults of every Wilson object created afterwards -- including the forked
    # build workers.  Copy first.
    opt = dict(w.get_option('parameters'))
    opt.update(ckm)
    w.set_option('parameters', opt)
    return _project(coords, w.match_run(coords.scale, 'WET', 'flavio').dict)


def build_map(coords, sectors, ckm_at, tol=0.0, verbose=True, with_dlnv=True):
    """Tabulate the JMS -> internal-coordinate map on the delta-gamma grid.

    ``ckm_at(dg)`` must return ``{'Vus':..., 'Vcb':..., 'Vub':..., 'gamma':...}``
    at that node -- the same four values smelli stamps onto the input Wilson.  Returns ``(jms_names, T)`` where
    ``T`` has shape ``(deg+1, len(coords), n_kept)`` -- polynomial coefficients
    in delta gamma -- and ``jms_names`` lists only the JMS coordinates that
    actually reach the support (everything else is identically zero and would
    be dropped by the surrogate anyway).
    """
    jms = jms_coordinates(sectors)
    nodes = coords.gamma_nodes

    # pass 1: find the columns that reach the support (normalisation irrelevant here)
    kept = []
    for name, part in jms:
        if np.any(np.abs(_column_wcxf(coords, name, part)) > tol):
            kept.append((name, part))
    if verbose:
        print('  jms: %d of %d JMS coordinates in %s reach the support'
              % (len(kept), len(jms), list(sectors)), flush=True)

    # pass 2: the kept columns at every node, plus their response to the three
    # CKM inputs that are not gamma.  The map carries a 1/lambda_t normalisation,
    # so a charged-current coordinate that moves Vcb by 1e-3 moves every b->s
    # coordinate by the same 1e-3 -- worth Delta chi2 ~ 5e-3 at the Z' benchmarks
    # and more further out.  Same first-order treatment as the family tables.
    def _cols(ckm):
        return np.stack([_column(coords, n, p, ckm) for n, p in kept], axis=1)

    per_node, per_node_d = [], []
    for dg in nodes:
        ckm = ckm_at(float(dg))
        per_node.append(_cols(ckm))
        if with_dlnv:
            drv = []
            for prm in DLNV:
                cp = dict(ckm, **{prm: ckm[prm] * (1 + EPS_LNV)})
                cm = dict(ckm, **{prm: ckm[prm] * (1 - EPS_LNV)})
                drv.append((_cols(cp) - _cols(cm)) / (2 * EPS_LNV))
            per_node_d.append(np.stack(drv))
        if verbose:
            print('  jms: node dg=%+.4f done' % dg, flush=True)
    T = design.interp_nodes(nodes, np.array(per_node))
    Td = design.interp_nodes(nodes, np.array(per_node_d)) if with_dlnv else None
    return ['%s_%s' % (n, p) for n, p in kept], T, Td


def apply_map(T, names, values, dg, T_dlnv=None, dlnv=None):
    """values: dict {'<JMS name>_R': v, ...} or a vector ordered like `names`.

    ``T_dlnv``/``dlnv`` add the first-order response to Vus/Vcb/Vub (see
    ``build_map``); without them the map is evaluated at the tabulated CKM.
    """
    if isinstance(values, dict):
        y = np.array([float(values.get(n, 0.0)) for n in names])
    else:
        y = np.asarray(values, dtype=float)
    x = design.poly_eval(T, dg) @ y
    if T_dlnv is not None and dlnv is not None:
        d = design.poly_eval(T_dlnv, dg)          # (3, n_coord, n_kept)
        x = x + np.tensordot(np.asarray(dlnv), d @ y, axes=(0, 0))
    return x


def jms_dict_to_vector(names, d):
    """wilson's ``match_run(...).dict`` (complex, JMS names) -> the R/I vector."""
    y = np.zeros(len(names))
    for i, n in enumerate(names):
        base, part = n.rsplit('_', 1)
        v = d.get(base, 0.0)
        y[i] = np.real(v) if part == 'R' else np.imag(v)
    return y


def unsupported(sectors, names, d, rtol=0.0):
    """JMS coefficients present in `d` and in `sectors` but outside the support."""
    kept = {n.rsplit('_', 1)[0] for n in names}
    inside = {n for n, _ in jms_coordinates(sectors)}
    return sorted(k for k, v in d.items()
                  if k in inside and k not in kept and abs(v) > rtol)
