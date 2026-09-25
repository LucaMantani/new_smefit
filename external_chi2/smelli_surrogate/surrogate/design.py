"""Exact designs and fits for polynomial forms, with a polynomial delta-gamma axis.

Every elementary quantity of the surrogate has the form

    Q(x, dg) = sum_{a} p_a(dg) * q_a(x)

with `q_a` running over the monomials of the coordinates the quantity depends
on -- {1, x_i, x_i x_j} at ``degree=2``, {1, x_i} at ``degree=1`` -- and `p_a` a
polynomial in dg of degree len(gamma_nodes)-1.  The build fits the form at each
dg node (exact least squares) and then interpolates the coefficients across the
nodes exactly.

``degree`` is a per-family property, and it is measured, not assumed: a family
declares ``degree = 1`` only where the design residual says the observable
really is linear (``epsp/eps`` is, to 1e-15 in all 38 of its coordinates), and
the build prints the residual either way.
"""

import numpy as np


def quad_design(n, n_random=None, box=1.2, seed=0, degree=2):
    """Exact design for a polynomial form in `n` variables + random redundancy.

    Returns (X, n_exact): the first `n_exact` rows determine the form uniquely.
    At ``degree=1`` that is the origin plus the `n` unit vectors; at ``degree=2``
    also the -unit vectors and the (i, j) corners.
    """
    pts = [np.zeros(n)]
    for i in range(n):
        for s in ((1.0, -1.0) if degree >= 2 else (1.0,)):
            e = np.zeros(n)
            e[i] = s
            pts.append(e)
    if degree >= 2:
        for i in range(n):
            for j in range(i + 1, n):
                e = np.zeros(n)
                e[i] = e[j] = 1.0
                pts.append(e)
    n_exact = len(pts)
    if n_random is None:
        n_random = max(6, n)
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        pts.append(rng.uniform(-box, box, size=n))
    return np.array(pts), n_exact


def quad_features(X, degree=2):
    """(n_pts, n_var) -> (n_pts, n_quad_features(n_var, degree)) monomial matrix."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, nv = X.shape
    cols = [np.ones(n)]
    cols += [X[:, i] for i in range(nv)]
    if degree >= 2:
        for i in range(nv):
            for j in range(i, nv):
                cols.append(X[:, i] * X[:, j])
    return np.stack(cols, axis=1)


def quad_features_1d(x, nv, xp=np, degree=2):
    """Monomials of a single point, in the same order as `quad_features`.

    Kept separate from `quad_features` so that the evaluation path never needs
    `atleast_2d`/indexing tricks and traces cleanly under JAX.
    """
    cols = [xp.ones(()) * 1.0]
    cols += [x[i] for i in range(nv)]
    if degree >= 2:
        for i in range(nv):
            for j in range(i, nv):
                cols.append(x[i] * x[j])
    return xp.stack(cols)


def n_quad_features(nv, degree=2):
    return 1 + nv + (nv * (nv + 1) // 2 if degree >= 2 else 0)


def gamma_features(dg, nodes):
    """Monomials 1, dg, dg^2, ... up to degree len(nodes)-1."""
    dg = np.atleast_1d(np.asarray(dg, dtype=float))
    deg = len(nodes) - 1
    return np.stack([dg**k for k in range(deg + 1)], axis=1)


def gamma_features_1d(dg, deg, xp=np):
    """Monomials 1, dg, ..., dg^deg for a single (possibly traced) dg."""
    return xp.stack([dg**k for k in range(deg + 1)])


def fit(X, nodes, Y, degree=2):
    """Fit Q(x, dg).

    Parameters
    ----------
    X : (n_pts, n_var) design in coordinate space
    nodes : (n_nodes,) delta-gamma nodes
    Y : (n_nodes, n_pts, n_out) evaluated elementary quantities

    Returns
    -------
    C : (n_gam_feat, n_quad_feat, n_out) coefficient tensor, such that
        Q = einsum('g,q,gqo->o', gamma_features(dg), quad_features(x), C)
    """
    Y = np.asarray(Y)
    F = quad_features(X, degree)
    per_node = np.stack(
        [np.linalg.lstsq(F, Y[k], rcond=None)[0] for k in range(len(nodes))], axis=0
    )  # (n_nodes, nq, nout)
    V = np.vander(np.asarray(nodes, dtype=float), N=len(nodes), increasing=True)
    return np.linalg.solve(V, per_node.reshape(len(nodes), -1)).reshape(per_node.shape)


def evaluate(C, x, dg):
    g = gamma_features(dg, range(C.shape[0]))[0]
    q = quad_features(np.atleast_2d(x))[0]
    return np.einsum("g,q,gqo->o", g, q, C)


def fit_residual(C, X, nodes, Y, degree=2):
    """Max |fit - data| / scale over the design, per output."""
    Y = np.asarray(Y)
    F = quad_features(X, degree)
    G = np.vander(np.asarray(nodes, dtype=float), N=C.shape[0], increasing=True)
    pred = np.einsum("kg,pq,gqo->kpo", G, F, C)
    scale = np.max(np.abs(Y), axis=(0, 1)) + 1e-300
    return np.max(np.abs(pred - Y), axis=(0, 1)) / scale


def interp_nodes(nodes, values):
    """Exact polynomial interpolation of `values` (n_nodes, ...) across `nodes`."""
    values = np.asarray(values)
    V = np.vander(np.asarray(nodes, dtype=float), N=len(nodes), increasing=True)
    flat = values.reshape(len(nodes), -1)
    return np.linalg.solve(V, flat).reshape(values.shape)


def poly_eval(coef, dg, xp=np):
    """coef (deg+1, ...) evaluated at scalar dg."""
    coef = xp.asarray(coef)
    p = xp.stack([dg**k for k in range(coef.shape[0])])
    return xp.tensordot(p, coef, axes=(0, 0))
