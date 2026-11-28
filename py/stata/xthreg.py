"""xthreg: Hansen (1999) fixed-effect panel threshold model (Wang 2015, Stata Journal 15(1)).

The original computations live in a compiled Mata library; this follows Hansen (1999): within transformation,
grid search over trimmed quantiles of the threshold variable, LR-based confidence intervals with
c(a) = -2 ln(1 - sqrt(a)), sequential (refined) estimation for two and three thresholds, and a residual
bootstrap by panel for the threshold-effect F tests."""
import math

import numpy as np

from .errors import StataError
from .interp import command
from .syntax import parse_command, split_top


def _within(M, gid, G, Ti):
    if M.ndim == 1:
        return M - (np.bincount(gid, M, minlength=G) / Ti)[gid]
    return np.column_stack([_within(M[:, j], gid, G, Ti) for j in range(M.shape[1])])


class _TH:
    def __init__(self, y, X, R, q, gid, G, Ti, grid, trims):
        self.gid, self.G, self.Ti = gid, G, Ti
        self.R, self.q = R, q
        self.Xw = _within(X, gid, G, Ti) if X.shape[1] else X
        self.n = y.size
        self.grid = grid
        self.trims = trims
        # project out X once (Frisch-Waugh)
        if self.Xw.shape[1]:
            self.Q, _ = np.linalg.qr(self.Xw)
        else:
            self.Q = np.zeros((self.n, 0))
        qs = np.sort(q)
        self.qs = qs

    def resid(self, v):
        return v - self.Q @ (self.Q.T @ v) if self.Q.shape[1] else v

    def cands(self, trim):
        n = self.n
        qs = self.qs
        lo, hi = int(math.floor(trim * n)), int(math.ceil((1 - trim) * n))
        c = np.unique(qs[lo:hi])
        if c.size > self.grid:
            idx = np.round(np.linspace(0, c.size - 1, self.grid)).astype(int)
            c = c[np.unique(idx)]
        return c

    def regimes(self, gammas):
        g = np.sort(np.asarray(gammas, float))
        reg = np.searchsorted(g, self.q, side='left')        # 0: q<=g1, 1: g1<q<=g2, ...
        cols = []
        for k in range(g.size + 1):
            I = (reg == k).astype(float)
            cols.append(self.R * I[:, None])
        Z = np.column_stack(cols)
        return _within(Z, self.gid, self.G, self.Ti)

    def rss(self, yr, gammas):
        Z = self.resid(self.regimes(gammas))
        b = np.linalg.lstsq(Z, yr, rcond=None)[0]
        e = yr - Z @ b
        return float(e @ e), e

    def search(self, yr, fixed, trim):
        best, gbest, path = np.inf, np.nan, []
        for c in self.cands(trim):
            if any(abs(c - f) < 1e-12 for f in fixed):
                path.append((c, np.nan))
                continue
            s, _ = self.rss(yr, list(fixed) + [c])
            path.append((c, s))
            if s < best:
                best, gbest = s, c
        return gbest, best, np.array(path)


def _ci(path, s_hat, sig2, level):
    crit = -2 * math.log(1 - math.sqrt(level))
    ok = path[:, 1] == path[:, 1]
    lr = (path[ok, 1] - s_hat) / sig2
    g = path[ok, 0]
    inside = g[lr <= crit]
    return (float(inside.min()), float(inside.max())) if inside.size else (np.nan, np.nan), np.column_stack([lr, g])


@command('xthreg')
def cmd_xthreg(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    if not d.ts or not d.ts.get('panel'):
        raise StataError('panel variable not set; use xtset varname ...', 459)
    ivar = d.ts['panel']
    from .xtmg import _expand, _vals
    vl = _expand(ip, split_top(p.body))
    dep, ix = vl[0], vl[1:]
    rx = _expand(ip, split_top(p.opts.get('rx', 2) or ''))
    qx = _expand(ip, split_top(p.opts.get('qx', 2) or ''))
    if not rx or len(qx) != 1:
        raise StataError('options rx() and qx() required', 198)
    qx = qx[0]
    thnum = int(p.opts.get('thnum', 2) or 1)
    if thnum <= 0 or thnum > 3:
        raise StataError('number of threshold must be positive integer and less than 3!', 198)
    trims = [float(v) for v in (p.opts.get('trim', 1) or '').split()] or [0.01] * thnum
    if len(trims) != thnum:
        raise StataError('number of parameters in trim() is not equal to thnum()!', 198)
    bss = [int(v) for v in (p.opts.get('bs', 2) or '').split()] or [0] * thnum
    if len(bss) != thnum:
        raise StataError('number of parameters in bs() is not equal to thnum()!', 198)
    grid = int(p.opts.get('grid', 1) or 300)
    level = float(p.opts.get('thlevel', 3) or 95) / 100
    gen = p.opts.get('gen', 3) or '_cat'
    noreg = p.opts.has('noreg', 5)
    bslog = not p.opts.has('nobslog', 5)
    y = _vals(ip, dep)
    Xv = {v: _vals(ip, v) for v in ix}
    Rv = {v: _vals(ip, v) for v in rx}
    qv = _vals(ip, qx)
    touse = ip.mark(p, d) & ~np.isnan(y) & ~np.isnan(qv)
    for a in list(Xv.values()) + list(Rv.values()):
        touse &= ~np.isnan(a)
    pid = d.cols[ivar].astype(float)
    ids, gid = np.unique(pid[touse], return_inverse=True)
    G = ids.size
    Ti = np.bincount(gid).astype(float)
    if Ti.min() != Ti.max():
        raise StataError('Panel threshold model need balanced panel, check your data!', 459)
    T = int(Ti[0])
    for nm, a in [(dep, y)] + list(Xv.items()) + list(Rv.items()) + [(qx, qv)]:
        v = a[touse]
        if np.all(np.abs(v - (np.bincount(gid, v) / Ti)[gid]) < 1e-12):
            raise StataError(f'There exist time-invaying variable: {nm}', 459)
    order = np.lexsort((d.cols[d.ts['time']].astype(float)[touse], gid))
    yy = y[touse]
    X = np.column_stack([Xv[v][touse] for v in ix]) if ix else np.zeros((yy.size, 0))
    R = np.column_stack([Rv[v][touse] for v in rx])
    q = qv[touse]
    M = _TH(yy, X, R, q, gid, G, Ti, grid, trims)
    yr = M.resid(_within(yy, gid, G, Ti))
    nT1 = G * (T - 1)
    S0 = float(yr @ (yr - M.resid(np.zeros_like(yr))))  # placeholder for no-threshold model
    Z0 = M.resid(_within(R, gid, G, Ti))
    e0 = yr - Z0 @ np.linalg.lstsq(Z0, yr, rcond=None)[0]
    S0 = float(e0 @ e0)
    rows_th, rows_F = [], []
    LRs = {}
    # ---- single
    g1, S1, path1 = M.search(yr, [], trims[0])
    sig1 = S1 / nT1
    (lo1, hi1), LRs['LR'] = _ci(path1, S1, sig1, level)
    rows_th.append(('Th-1', g1, S1, lo1, hi1))
    F1 = (S0 - S1) / sig1
    _, e1 = M.rss(yr, [g1])
    fits = [(S1, sig1, F1, e1)]
    ths = [g1]
    if thnum >= 2:
        g2, S2, path2 = M.search(yr, [g1], trims[1])
        g1r, S2r, path1r = M.search(yr, [g2], trims[1])
        sig2 = S2r / nT1
        (lo1r, hi1r), LRs['LR21'] = _ci(path1r, S2r, sig2, level)
        (lo2, hi2), LRs['LR22'] = _ci(path2, S2, S2 / nT1, level)
        rows_th.append(('Th-21', g1r, S2r, lo1r, hi1r))
        rows_th.append(('Th-22', g2, S2, lo2, hi2))
        S2f, e2 = M.rss(yr, [g1r, g2])
        F2 = (S1 - S2f) / (S2f / nT1)
        fits.append((S2f, S2f / nT1, F2, e2))
        ths = [g1r, g2]
    if thnum == 3:
        g3, S3, path3 = M.search(yr, ths, trims[2])
        (lo3, hi3), LRs['LR3'] = _ci(path3, S3, S3 / nT1, level)
        rows_th.append(('Th-3', g3, S3, lo3, hi3))
        S3f, e3 = M.rss(yr, ths + [g3])
        F3 = (fits[-1][0] - S3f) / (S3f / nT1)
        fits.append((S3f, S3f / nT1, F3, e3))
        ths = ths + [g3]
    # ---- bootstrap
    rng = ip.rng
    panel_rows = [np.flatnonzero(gid == g) for g in range(G)]
    Frows = []
    for k in range(thnum):
        B = bss[k]
        S_, mse_, F_, e_ = fits[k]
        if B <= 0:
            Frows.append((S_, mse_, F_, np.nan, np.nan, np.nan, np.nan))
            continue
        if bslog:
            ip.txt(f'Bootstrap for {["single", "double", "triple"][k]} threshold\n')
        Fb = np.empty(B)
        for bi in range(B):
            draw = rng.integers(0, G, G)
            ystar = np.empty_like(yr)
            for g in range(G):
                ystar[panel_rows[g]] = e_[panel_rows[draw[g]]]
            ystar = M.resid(_within(ystar, gid, G, Ti))
            # null model: k thresholds
            if k == 0:
                e0b = ystar - Z0 @ np.linalg.lstsq(Z0, ystar, rcond=None)[0]
                Snull = float(e0b @ e0b)
                gb, Sa, _ = M.search(ystar, [], trims[0])
            elif k == 1:
                gb1, Snull, _ = M.search(ystar, [], trims[0])
                gb2, Sa, _ = M.search(ystar, [gb1], trims[1])
            else:
                gb1, _, _ = M.search(ystar, [], trims[0])
                gb2, _, _ = M.search(ystar, [gb1], trims[1])
                gb1r, Snull, _ = M.search(ystar, [gb2], trims[1])
                gb3, Sa, _ = M.search(ystar, [gb1r, gb2], trims[2])
            Fb[bi] = (Snull - Sa) / (Sa / nT1)
            if bslog and (bi + 1) % 50 == 0:
                ip.txt('.' * 50 + f' {bi + 1:5d}\n')
        if bslog and B % 50:
            ip.txt('.' * (B % 50) + '\n')
        pval = float(np.mean(Fb > F_))
        c10, c5, c1 = np.quantile(Fb, [.90, .95, .99])
        Frows.append((S_, mse_, F_, pval, c10, c5, c1))
    # ---- display
    ip.txt(f'\nThreshold estimator (level = {level * 100:g}):\n')
    w = 1 + 10 + 3 + 13 * 3
    ip.txt('-' * w + '\n')
    ip.txt(f'{"model":>10} | {"Threshold":>12} {"Lower":>12} {"Upper":>12}\n')
    ip.txt('-' * 11 + '+' + '-' * (w - 12) + '\n')
    for nm, gth, S, lo, hi in rows_th:
        ip.txt(f'{nm:>10} | ')
        ip.res(f'{gth:12.4f} {lo:12.4f} {hi:12.4f}\n')
    ip.txt('-' * w + '\n')
    if sum(bss) > 0:
        ip.txt(f'\nThreshold effect test (bootstrap = {" ".join(str(b) for b in bss)}):\n')
        ip.txt('-' * 79 + '\n')
        ip.txt(f'{"Threshold":>10} | {"RSS":>10} {"MSE":>9} {"Fstat":>9} {"Prob":>7} {"Crit10":>8} {"Crit5":>8} '
               f'{"Crit1":>8}\n')
        ip.txt('-' * 11 + '+' + '-' * 67 + '\n')
        for nm, r in zip(('Single', 'Double', 'Triple'), Frows):
            ip.txt(f'{nm:>10} | ')
            def f(v, w, dd):
                return f'{v:{w}.{dd}f}' if v == v else f'{".":>{w}}'
            ip.res(f'{f(r[0], 10, 4)} {f(r[1], 9, 4)} {f(r[2], 9, 2)} {f(r[3], 7, 4)} {f(r[4], 8, 4)} '
                   f'{f(r[5], 8, 4)} {f(r[6], 8, 4)}\n')
        ip.txt('-' * 79 + '\n')
    from .expr import Matrix
    thm = np.array([[g_, S_, np.nan, lo, hi] for _, g_, S_, lo, hi in rows_th])
    fm = np.array(Frows)
    mats = {'Thrss': Matrix(thm, [r[0] for r in rows_th], ['Threshold', 'RSS', 'Location', 'Lower', 'Upper']),
            'Fstat': Matrix(fm, ['Single', 'Double', 'Triple'][:thnum],
                            ['RSS', 'MSE', 'Fstat', 'Prob', 'Crit10', 'Crit5', 'Crit1'])}
    for k_, v_ in LRs.items():
        mats[k_] = Matrix(v_, [str(i + 1) for i in range(v_.shape[0])], ['LR', 'Threshold'])
    if not noreg:
        cut = sorted(ths)
        if d.has(gen):
            d.drop_vars([gen])
        cat = np.full(d.N, np.nan)
        cat[touse] = np.searchsorted(np.array(cut), q, side='left')
        d.add_var(gen, cat, 'float')
        inter = ' '.join(f'i.{gen}#c.{v}' for v in rx)
        extra = ' '.join(o for o in [p.opts.get('vce', 1) and f'vce({p.opts.get("vce", 1)})'] if o)
        tv = '__xth_touse'
        d.add_var(tv, touse.astype(float), 'byte')
        try:
            ip.dispatch(f'xtreg {dep} {" ".join(ix)} {inter} if {tv}, fe {extra}'.strip())
        finally:
            if d.has(tv):
                d.drop_vars([tv])
        if ip.est is not None:
            ip.est['sample'] = touse
    e = ip.e
    for k_, v_ in mats.items():
        e.matrices[k_] = v_
    e.scalars['grid'] = float(grid)
    e.scalars['thnum'] = float(thnum)
    e.macros.update({'ix': ' '.join(ix), 'rx': ' '.join(rx), 'qx': qx, 'ivar': ivar})
