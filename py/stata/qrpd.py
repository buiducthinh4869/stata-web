"""qregpd: Powell (2016) quantile regression for panel data (QRPD), ported from the SSC qregpd.ado
(Nelder-Mead and grid-search optimizers; analytic variance qrpd_var)."""
import math

import numpy as np
from scipy import optimize, stats as sps

from .design import coef_table
from .errors import StataError
from .interp import command
from .syntax import parse_command, split_top


def _wquantile(u, w, tau):
    """moremata mm_quantile(), default definition 2 (inverse EDF, averaging at discontinuities)."""
    o = np.argsort(u, kind='stable')
    u, w = u[o], w[o]
    cw = np.cumsum(w)
    W = cw[-1]
    t = tau * W
    i = int(np.searchsorted(cw, t - 1e-12 * W, side='left'))
    if abs(cw[i] - t) <= 1e-12 * W and i + 1 < u.size:
        return 0.5 * (u[i] + u[i + 1])
    return u[i]


def _qreg(y, X, tau, w):
    """Weighted linear quantile regression by LP (HiGHS)."""
    n, k = X.shape
    b = _qreg_b(y, X, tau, w)
    # vce(iid): sparsity by the fitted method with Hall-Sheather bandwidth
    a = .05
    zt = sps.norm.ppf(tau)
    h = n ** (-1 / 3) * sps.norm.ppf(1 - a / 2) ** (2 / 3) * (
        (1.5 * sps.norm.pdf(zt) ** 2) / (2 * zt ** 2 + 1)) ** (1 / 3)
    lo, hi = max(tau - h, 1e-4), min(tau + h, 1 - 1e-4)
    xbar = (w @ X) / w.sum()
    try:
        bl = _qreg_b(y, X, lo, w)
        bh = _qreg_b(y, X, hi, w)
        s = (xbar @ bh - xbar @ bl) / (hi - lo)
    except Exception:
        s = np.nan
    V = tau * (1 - tau) * s ** 2 * np.linalg.pinv((X * w[:, None]).T @ X)
    return b, V


def _qreg_b(y, X, tau, w):
    from scipy import sparse
    n, k = X.shape
    c = np.r_[np.zeros(2 * k), tau * w, (1 - tau) * w]
    Xs = sparse.csr_matrix(X)
    I = sparse.identity(n, format='csr')
    A = sparse.hstack([Xs, -Xs, I, -I], format='csr')
    res = optimize.linprog(c, A_eq=A, b_eq=y, bounds=(0, None), method='highs')
    if not res.success:
        raise StataError('qreg starting values could not be computed', 430)
    return res.x[:k] - res.x[k:2 * k]


@command('qregpd')
def cmd_qregpd(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    from .xtmg import _expand, _vals, _col
    vl = _expand(ip, split_top(p.body))
    lhs, rhs = vl[0], vl[1:]
    q = float(p.opts.get('quantile', 1) or 0.5)
    tau = q / 100 if q >= 1 else q
    if tau <= 0 or tau >= 1:
        raise StataError(f'quantiles({q:g}) out of range', 198)
    idv = p.opts.get('identifier', 2)
    fixv = p.opts.get('fix', 3)
    if not idv:
        raise StataError('Error: You must specify a panel identifier', 198)
    if not fixv:
        raise StataError('Error: You must specify a fixed effect', 198)
    idv, fixv = d.resolve(idv), d.resolve(fixv)
    method = (p.opts.get('optimize', 8) or 'nm').lower()
    if method == 'mcmc':
        raise StataError('optimize(mcmc) is not available in Stata Web; use the default Nelder-Mead or optimize(grid)',
                         198)
    inst = _expand(ip, split_top(p.opts.get('instruments', 5) or '')) or list(rhs)
    if len(inst) < len(rhs):
        raise StataError('Error: # of instruments must be >= # of right-hand side variables.', 198)
    excluded = [v for v in inst if v not in rhs]
    yv = _vals(ip, lhs)
    D = {v: _vals(ip, v) for v in set(rhs + inst)}
    touse = ip.mark(p, d) & ~np.isnan(yv)
    for v in rhs:
        touse &= ~np.isnan(D[v])
    wt_all = np.ones(d.N)
    if p.weight:
        from .estim import get_weights
        w_, _ = get_weights(ip, p, d)
        wt_all = w_
        touse &= ~np.isnan(w_)
    idx = np.flatnonzero(touse)
    o = np.lexsort((d.cols[fixv].astype(float)[idx], d.cols[idv].astype(float)[idx]))
    idx = idx[o]
    y = yv[idx]
    di = np.column_stack([D[v][idx] for v in rhs])
    zi = np.column_stack([D[v][idx] for v in inst])
    wt = wt_all[idx].astype(float)
    pid = d.cols[idv].astype(float)[idx]
    fix = d.cols[fixv].astype(float)[idx]
    ids, gid = np.unique(pid, return_inverse=True)
    Np = ids.size
    wt = wt / wt.sum() * Np
    cnt = np.bincount(gid)
    # starting values: qreg y rhs [aw=wt], q(tau)
    Xq = np.column_stack([di, np.ones(y.size)])
    b0, V0 = _qreg(y, Xq, tau, wt)
    k = len(rhs)
    beta_start = b0[:k]
    se0 = np.sqrt(np.maximum(np.diag(V0)[:k], 0))
    zmean = np.column_stack([(np.bincount(gid, wt * zi[:, j]) / np.bincount(gid, wt))[gid]
                             for j in range(zi.shape[1])])
    Znew = zi - zmean
    fixes = np.unique(fix)
    fsel = [fix == f for f in fixes]

    def gamma(b):
        u = y - di @ b
        g = np.zeros(y.size)
        for s in fsel:
            g[s] = _wquantile(u[s], wt[s], tau)
        return g

    def gi_mat(b):
        gm = gamma(b)
        below = wt * (y <= gm + di @ b)
        Gi = np.zeros((Np, Znew.shape[1]))
        np.add.at(Gi, gid, below[:, None] * Znew)
        return Gi
    Gi = gi_mat(beta_start)
    A = np.linalg.pinv(Gi.T @ Gi / y.size)

    def obj(b):
        gm = gamma(b)
        below = wt * (y <= gm + di @ b)
        g = (below @ Znew) / math.sqrt(Np)
        return -0.5 * float(g @ A @ g)
    solutions = None
    if method == 'grid':
        if k > 2:
            raise StataError('grid search allows at most two right-hand side variables', 198)
        grids = []
        for j in range(k):
            gs = p.opts.get(f'grid{j + 1}', 5)
            if gs:
                grids.append(np.array([float(x) for x in gs.replace(',', ' ').split()], float))
            else:
                lo, hi = beta_start[j] - 1.96 * se0[j], beta_start[j] + 1.96 * se0[j]
                grids.append(np.linspace(lo, hi, 81))
        ip.txt('Number of grid points: ' + str(int(np.prod([g.size for g in grids]))) + '\n')
        pts = np.array(np.meshgrid(*grids, indexing='ij')).reshape(k, -1).T
        vals = np.array([obj(b) for b in pts])
        mx = vals.max()
        best = pts[vals >= mx - 1e-15]
        b = np.array([(best[:, j].min() + best[:, j].max()) / 2 for j in range(k)])
        fval = mx
        solutions = best
        ip.txt('Grid search optimization\n')
    else:
        ip.txt('Nelder-Mead optimization\n')
        simplex = np.vstack([beta_start] + [beta_start + 10 * np.eye(k)[j] for j in range(k)])
        r = optimize.minimize(lambda bb: -obj(bb), beta_start, method='Nelder-Mead',
                              options={'initial_simplex': simplex, 'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 5000})
        b = r.x
        fval = -r.fun
    # analytic variance (qrpd_var)
    deriv = np.where(se0 > 0, se0, 1e-3)
    gm = gamma(b)
    uu = y - di @ b - gm
    q75, q25 = _wquantile(uu, wt, .75), _wquantile(uu, wt, .25)
    meanuu = float((uu * wt).sum() / wt.sum())
    meansqr = math.sqrt(float(np.sum(((uu * wt) - meanuu) ** 2)) / (wt.sum() - 1))
    bdw = 1.0589 * min(meansqr, (q75 - q25) / 1.349) * y.size ** (-.2)
    yd = np.zeros_like(di)
    for l in range(k):
        bp, bm = b.copy(), b.copy()
        bp[l] += deriv[l]
        bm[l] -= deriv[l]
        yd[:, l] = (gamma(bp) - gamma(bm)) / (2 * deriv[l])
    xd = di + yd
    absb = (np.abs(uu) <= bdw).astype(float)
    Gi2 = gi_mat(b)
    A2 = np.linalg.pinv(Gi2.T @ Gi2 / y.size)
    wZ = wt[:, None] * Znew
    Gm = (wZ * absb[:, None]).T @ xd * (0.5 / bdw) / Np
    below = (y <= gm + di @ b).astype(float)
    S = np.zeros((Np, Znew.shape[1]))
    np.add.at(S, gid, below[:, None] * wZ)
    Sigma = S.T @ S / Np
    GAG = np.linalg.pinv(Gm.T @ A2 @ Gm)
    V = GAG @ (Gm.T @ A2 @ Sigma @ A2 @ Gm) @ GAG / Np
    est = {'nomargins': True, 'names': rhs, 'b': b, 'V': V, 'colinfo': [_col(v) for v in rhs], 'df_r': None,
           'depvar': lhs, 'sample': touse, 'cmd': 'qregpd', 'stat': 'z',
           'scalars': {'N': int(y.size), 'N_g': Np, 'g_min': int(cnt.min()), 'g_max': int(cnt.max())},
           'macros': {'cmd': 'qregpd', 'depvar': lhs, 'indepvars': ' '.join(rhs),
                      'title': 'Quantile Regression for Panel Data (QRPD)'}}

    def replay(ip_):
        ip_.txt('\n\nQuantile Regression for Panel Data (QRPD)\n')
        for lab, v in (('Number of obs:        ', y.size), ('Number of groups:     ', Np),
                       ('Min obs per group:    ', int(cnt.min())), ('Max obs per group:    ', int(cnt.max()))):
            ip_.txt('     ' + lab)
            ip_.res(f'{v:10.0f}\n')
        coef_table(ip_, est, 'z', depvar=' ')
        if not excluded:
            ip_.txt('No excluded instruments - standard QRPD estimation.\n')
        else:
            ip_.txt('Excluded instruments: ' + ' '.join(excluded) + '\n')
        if method == 'grid':
            if solutions is not None and solutions.shape[0] > 1:
                ip_.txt('Note: Alternative solutions exist. See e(solutions).\n')
            ip_.txt('Value of objective function: ')
            ip_.res(f'{fval:12.8f}\n')
    est['replay'] = replay
    est['predict'] = None
    ip.set_estimation(est)
    from .expr import Matrix
    ug = np.unique(np.column_stack([fix, gm]), axis=0)
    ip.e.matrices['gamma'] = Matrix(ug, [str(i + 1) for i in range(ug.shape[0])], [fixv, 'estimate'])
    ip.e.matrices['b_init'] = Matrix(b0[:k].reshape(1, -1), ['y1'], rhs)
    replay(ip)
