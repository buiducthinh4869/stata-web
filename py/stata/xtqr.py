"""xtqreg: Machado & Santos Silva (2019) method-of-moments quantile regression with fixed effects,
ported from the SSC xtqreg.ado (version 14)."""
import math

import numpy as np
from scipy import stats as sps

from .design import coef_table, Col
from .errors import StataError
from .interp import command
from .syntax import parse_command, split_top


def _within(v, gid, G):
    c = np.bincount(gid, minlength=G)
    return v - (np.bincount(gid, v, minlength=G) / c)[gid]


def _fe(y, X, gid, G):
    """xtreg, fe: slopes b, constant, u_i (fixed effects), e_it (idiosyncratic)."""
    Xw = np.column_stack([_within(X[:, j], gid, G) for j in range(X.shape[1])])
    yw = _within(y, gid, G)
    b = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    cons = y.mean() - X.mean(0) @ b
    xb = X @ b + cons
    cnt = np.bincount(gid, minlength=G)
    ui = (np.bincount(gid, y - xb, minlength=G) / cnt)[gid]
    e = y - xb - ui
    return b, cons, ui, e, xb


def _qfit(us, tau):
    """qreg us, q(tau) vce(iid) on a constant: sample quantile and density (fitted, Hall-Sheather)."""
    n = us.size
    s = np.sort(us)

    def Q(t):
        k = int(math.ceil(n * t - 1e-12))
        return s[min(max(k, 1), n) - 1]
    q = Q(tau)
    a = 0.05
    za = sps.norm.ppf(1 - a / 2)
    zt = sps.norm.ppf(tau)
    h = n ** (-1 / 3) * za ** (2 / 3) * ((1.5 * sps.norm.pdf(zt) ** 2) / (2 * zt ** 2 + 1)) ** (1 / 3)
    lo, hi = max(tau - h, 1 / n), min(tau + h, 1 - 1 / n)
    f = (hi - lo) / (Q(hi) - Q(lo)) if Q(hi) > Q(lo) else np.nan
    return q, f


@command('xtqreg')
def cmd_xtqreg(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    from .xtmg import _expand, _vals
    vl = _expand(ip, split_top(p.body))
    if len(vl) < 2:
        raise StataError('too few variables specified', 102)
    yv, rhs = vl[0], vl[1:]
    idv = p.opts.get('id', 1)
    if not idv:
        if not d.ts or not d.ts.get('panel'):
            raise StataError('Must specify panelvar; use xtset or the id option', 459)
        idv = d.ts['panel']
    idv = d.resolve(idv)
    quants = [float(v) for v in (p.opts.get('quantile', 1) or '.5').split()]
    for qq in quants:
        if qq <= 0 or qq >= 1:
            raise StataError('quantiles must be between 0 and 1', 198)
    y = _vals(ip, yv)
    Xd = {v: _vals(ip, v) for v in rhs}
    touse = ip.mark(p, d) & ~np.isnan(y) & ~np.isnan(d.cols[idv].astype(float))
    for v in rhs:
        touse &= ~np.isnan(Xd[v])
    _, gid = np.unique(d.cols[idv].astype(float)[touse], return_inverse=True)
    G = int(gid.max()) + 1
    yy = y[touse]
    X = np.column_stack([Xd[v][touse] for v in rhs])
    n, k = X.shape
    if np.bincount(gid).max() < 3:
        raise StataError('It is not possible to identify the conditional quantiles with T<3; the estimator is valid '
                         'only for large T', 459)
    ip.txt('\n\n\n' + ' ' * 30 + 'MM-QR regression results\n')
    ip.txt('Number of obs = ')
    ip.res(f'{n}\n')
    be, bc, alpha_i, u, _ = _fe(yy, X, gid, G)
    up = (u >= 0).astype(float)
    up -= up.mean()
    au = u * up * 2
    g, gc, gam_i, _, xbs = _fe(au, X, gid, G)
    auhat = xbs + gam_i
    if auhat.min() <= 0:
        ip.txt('WARNING: some fitted values of the scale function are negative\n')
    us = u / auhat
    si = (gam_i - gam_i.mean() * 0) + (X.mean(0) @ g + gc)
    # deltas = predict u (fixed effect of the scale regression); xtreg's u_i are centred so that _cons absorbs mean
    si = gam_i + (X.mean(0) @ g + gc)
    Xw = np.column_stack([_within(X[:, j], gid, G) for j in range(k)])
    Qxx = Xw.T @ Xw
    Pxx = (Xw * (auhat ** 2)[:, None]).T @ Xw
    Px = (auhat ** 2) @ Xw
    iQ = np.linalg.pinv(Qxx)
    O = np.zeros((2 * k + 1, 2 * k + 1))
    XI = np.zeros((k, 2 * k + 1))
    XI[:, 2 * k] = g / si.sum()
    XI[:, :k] = iQ
    O[:k, :k] = np.mean(us ** 2) * Pxx
    vp = (us >= 0).astype(float)
    vp -= vp.mean()
    av = us * vp * 2
    O[:k, k:2 * k] = O[k:2 * k, :k] = np.mean(us * av) * Pxx
    O[k:2 * k, k:2 * k] = np.mean(av ** 2) * Pxx
    msi2 = float(np.mean(si ** 2))
    res = []
    savefe = p.opts.get('savefe', 4)
    pred = p.opts.get('predict', 1)
    for qu in quants:
        Q, fuq = _qfit(us, qu)
        w = (qu - (us <= Q)) / fuq - us - Q * av
        b = be + g * Q
        O[2 * k, :k] = np.mean(us * w) * Px
        O[:k, 2 * k] = O[2 * k, :k]
        O[2 * k, k:2 * k] = np.mean(av * w) * Px
        O[k:2 * k, 2 * k] = O[2 * k, k:2 * k]
        O[2 * k, 2 * k] = float(np.sum(w ** 2)) * msi2
        XI[:, k:2 * k] = iQ * Q
        V = XI @ O @ XI.T
        res.append((qu, b, V, Q))
        qs_ = f'{qu:g}'
        qs_ = qs_[1:] if qs_.startswith('0.') else qs_
        tag = qs_.replace('.', '_')
        if savefe:
            nm = f'{savefe}_{tag}'
            out = np.full(d.N, np.nan)
            out[touse] = (alpha_i + bc) + Q * (gam_i + gc)
            d.add_var(nm, out, 'float')
        if pred:
            nm = f'{pred}_{tag}'
            out = np.full(d.N, np.nan)
            out[touse] = X @ b + (alpha_i + bc) + Q * (gam_i + gc)
            d.add_var(nm, out, 'float')
        ip.txt(f'{qs_} Quantile regression\n')
        sub = {'names': rhs, 'b': b, 'V': V, 'colinfo': [_c(v) for v in rhs], 'df_r': None}
        coef_table(ip, sub, 'z', depvar=' ')
        ip.txt('\n')
    qu, b, V, Q = res[-1]
    from .expr import Matrix
    est = {'nomargins': True, 'names': rhs, 'b': b, 'V': V, 'colinfo': [_c(v) for v in rhs], 'df_r': None,
           'depvar': yv, 'sample': touse, 'cmd': 'xtqreg', 'stat': 'z',
           'scalars': {'N': n}, 'macros': {'cmd': 'xtqreg', 'depvar': yv}}
    est['replay'] = lambda ip_: coef_table(ip_, est, 'z', depvar=' ')
    est['predict'] = None
    ip.set_estimation(est)
    E = ip.e.matrices
    E['b_location'] = Matrix(be.reshape(1, -1), ['y1'], rhs)
    E['b_scale'] = Matrix(g.reshape(1, -1), ['y1'], rhs)
    E['q'] = Matrix(np.array([[Q]]), ['y1'], ['_cons'])
    if len(res) > 1:
        for qu, b_, V_, Q_ in res:
            t = f'{qu:g}'
            t = (t[1:] if t.startswith('0.') else t).replace('.', '_')
            E[f'b{t}'] = Matrix(b_.reshape(1, -1), ['y1'], rhs)
            E[f'V{t}'] = Matrix(V_, rhs, rhs)
            E[f'q{t}'] = Matrix(np.array([[Q_]]), ['y1'], ['_cons'])
    if p.opts.has('ls'):
        pass


def _c(nm):
    from .xtmg import _col
    return _col(nm)
