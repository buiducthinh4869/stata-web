"""xtcointtest kao (Kao 1999), following Stata's xtcointtest.ado. The Pedroni and Westerlund tests rely on
Stata's compiled Mata library (and simulated adjustment constants) and are not available."""
import math

import numpy as np
from scipy import stats as sps

from .errors import StataError
from .fmt import fmt_num
from .interp import command
from .syntax import parse_command, split_top


def _nw_lags(w, kernel):
    T = w.size
    if kernel == 'bartlett':
        q, cy, intb, n = 1, 1.1447, 1, int(4 * (T / 100) ** (2 / 9))
    elif kernel == 'parzen':
        q, cy, intb, n = 2, 2.6614, 1, int(4 * (T / 100) ** (4 / 25))
    else:
        q, cy, intb, n = 2, 1.3221, 0, int(4 * (T / 100) ** (2 / 25))
    s0 = float(w @ w)
    sq, sz = 0.0, s0
    for j in range(1, n + 1):
        sj = float(w[j:] @ w[:-j])
        sq += 2 * j ** q * sj
        sz += 2 * sj
    pw = 1 / (2 * q + 1)
    gam = cy * ((sq / sz) ** 2) ** pw if sz != 0 else 0.0
    if intb:
        hb = int(gam * T ** pw)
        return hb - 1 if hb < T else T - 2
    hb = gam * T ** pw
    return hb - 1 if hb <= T else T - 1


def _kw(kernel, j, m):
    z = j / (m + 1)
    if kernel == 'bartlett':
        return 1 - z if z <= 1 else 0.0
    if kernel == 'parzen':
        return (1 - 6 * z ** 2 + 6 * z ** 3) if z <= .5 else (2 * (1 - z) ** 3 if z <= 1 else 0.0)
    th = 6 * math.pi * z / 5
    return 3 * (math.sin(th) / th - math.cos(th)) / th ** 2


@command('xtcointtest')
def cmd_xtcointtest(ip, cmd, rest, by):
    sub, _, tail = rest.strip().partition(' ')
    sub = sub.lower()
    if sub in ('pedroni', 'westerlund'):
        raise StataError(f'xtcointtest {sub} is not available in Stata Web (Stata computes it in a compiled library); '
                         'use xtcointtest kao', 199)
    if sub != 'kao':
        raise StataError(f'{sub} is not a valid test', 198)
    p = parse_command(tail)
    d = ip.data
    if not d.ts or not d.ts.get('panel') or not d.ts.get('time'):
        raise StataError('panel and time variables must be set; use xtset panelvar timevar', 459)
    ivar = d.ts['panel']
    from .xtmg import _expand, _vals
    vl = _expand(ip, split_top(p.body))
    if len(vl) < 2:
        raise StataError('at least one regressor must be specified', 198)
    dep, ind = vl[0], vl[1:]
    if dep in ind:
        raise StataError('may not specify dependent variable as one of the regressors', 198)
    lagsel, maxl = None, 1
    lg = p.opts.get('lags', 1)
    if lg:
        w = lg.split()
        if w[0].lower() in ('aic', 'bic', 'hqic'):
            lagsel, maxl = w[0].lower(), int(w[1]) if len(w) > 1 else 1
        else:
            maxl = int(w[0])
    ks = p.opts.get('kernel', 3)
    kernel, hac_lags, nwest = 'bartlett', None, True
    if ks:
        w = ks.split()
        k0 = w[0].lower()
        kernel = 'bartlett' if 'bartlett'.startswith(k0) and len(k0) >= 3 else (
            'parzen' if 'parzen'.startswith(k0) and len(k0) >= 3 else 'quadraticspectral')
        if len(w) > 1 and w[1].lower() != 'nwest':
            hac_lags, nwest = float(w[1]), False
    y = _vals(ip, dep)
    X = {v: _vals(ip, v) for v in ind}
    touse = ip.mark(p, d) & ~np.isnan(y)
    for v in ind:
        touse &= ~np.isnan(X[v])
    pid = d.cols[ivar].astype(float)
    tv = d.cols[d.ts['time']].astype(float)
    delta = d.ts.get('delta', 1) or 1
    for g in np.unique(pid[touse]):
        t = np.sort(tv[touse & (pid == g)])
        if t.size > 1 and np.any(np.round(np.diff(t) / delta) != 1):
            raise StataError('Kao test does not allow gaps in data', 498)
    cnts = [int(np.sum(touse & (pid == g))) for g in np.unique(pid[touse])]
    unbal = len(set(cnts)) > 1
    if p.opts.has('demean'):
        for nm in [dep] + ind:
            v = y if nm == dep else X[nm]
            out = np.full(d.N, np.nan)
            for t in np.unique(tv[touse]):
                s = touse & (tv == t)
                out[s] = v[s] - v[s].mean()
            if nm == dep:
                y = out
            else:
                X[nm] = out
    L = lambda v: ip._lag(d, v, 1)
    # residuals of AR(1) (no constant) for y and each x
    def ar1res(v):
        lv = L(np.where(touse, v, np.nan))
        ok = touse & ~np.isnan(lv)
        r = float(v[ok] @ lv[ok]) / float(lv[ok] @ lv[ok])
        e = np.full(d.N, np.nan)
        e[ok] = v[ok] - r * lv[ok]
        return e
    ey = ar1res(y)
    ex = [ar1res(X[v]) for v in ind]
    mk = touse & ~np.isnan(ey)
    for e in ex:
        mk &= ~np.isnan(e)
    ids = np.unique(pid[mk])
    N = ids.size
    # long-run covariance (averaged over panels)
    k1 = 1 + len(ind)
    Sig = np.zeros((k1, k1))
    Om = np.zeros((k1, k1))
    lagsused = []
    for g in ids:
        s = np.flatnonzero(mk & (pid == g))
        s = s[np.argsort(tv[s])]
        W = np.column_stack([ey[s]] + [e[s] for e in ex])
        T = s.size
        S0 = W.T @ W / T
        m = _nw_lags(W.sum(1), kernel) if nwest else hac_lags
        lagsused.append(m)
        O = S0.copy()
        top = int(m) if kernel != 'quadraticspectral' else T - 1
        for j in range(1, top + 1):
            if j >= T:
                break
            Gj = W[j:].T @ W[:-j] / T
            O += _kw(kernel, j, m) * (Gj + Gj.T)
        Sig += S0 / N
        Om += O / N
    def cond(M):
        return float(M[0, 0] - M[0, 1:] @ np.linalg.solve(M[1:, 1:], M[1:, 0]))
    s2v, s20v = cond(Sig), cond(Om)
    haclag = float(np.mean(lagsused))
    # within (fe) residuals of y on x
    gidx = np.unique(pid[touse], return_inverse=True)[1]
    G = gidx.max() + 1
    cnt = np.bincount(gidx)
    yy = y[touse]
    XX = np.column_stack([X[v][touse] for v in ind])
    dm = lambda v: v - (np.bincount(gidx, v) / cnt)[gidx]
    Xw = np.column_stack([dm(XX[:, j]) for j in range(XX.shape[1])])
    b = np.linalg.lstsq(Xw, dm(yy), rcond=None)[0]
    resd = np.full(d.N, np.nan)
    resd[touse] = dm(yy) - Xw @ b
    lr = L(resd)
    ok = touse & ~np.isnan(lr)
    rho = float(resd[ok] @ lr[ok]) / float(lr[ok] @ lr[ok])
    Ntot = int(ok.sum())
    Tn = Ntot / N
    ev = resd[ok] - rho * lr[ok]
    se2 = float(ev @ ev) / Ntot
    trho = (rho - 1) * math.sqrt(float(lr[ok] @ lr[ok])) / math.sqrt(se2)
    dfr = (math.sqrt(N) * Tn * (rho - 1) + 3 * math.sqrt(N) * s2v / s20v) / math.sqrt(3 + 36 * s2v ** 2 /
                                                                                  (5 * s20v ** 2))
    dft = (trho + math.sqrt(6 * N * s2v) / (2 * math.sqrt(s20v))) / math.sqrt(s20v / (2 * s2v) +
                                                                            3 * s2v / (10 * s20v))
    dres = resd - lr
    lagd = {j: ip._lag(d, dres, j) for j in range(1, maxl + 1)}

    def adf(pl, base):
        okk = base & ~np.isnan(lr)
        for j in range(1, pl + 1):
            okk &= ~np.isnan(lagd[j])
        Z = np.column_stack([lr[okk]] + [lagd[j][okk] for j in range(1, pl + 1)])
        bb = np.linalg.lstsq(Z, resd[okk], rcond=None)[0]
        e = resd[okk] - Z @ bb
        return bb, e, Z, okk
    pl = maxl
    if lagsel:
        _, _, _, samp = adf(maxl, touse)
        best = np.inf
        for i in range(maxl, 0, -1):
            bb, e, Z, okk = adf(i, samp)
            n_, kk = Z.shape
            llv = -0.5 * n_ * (1 + math.log(2 * math.pi) + math.log(float(e @ e) / n_))
            ic = {'aic': -2 * llv + 2 * kk, 'bic': -2 * llv + math.log(n_) * kk,
                  'hqic': -2 * llv + 2 * math.log(math.log(n_)) * kk}[lagsel]
            if ic <= best:
                best, pl = ic, i
    bb, e, Z, okk = adf(pl, touse)
    rhoa = bb[0]
    if Z.shape[1] > 1:
        D = Z[:, 1:]
        lres = Z[:, 0] - D @ np.linalg.lstsq(D, Z[:, 0], rcond=None)[0]
    else:
        lres = Z[:, 0]
    sv2 = float(e @ e) / e.size
    tadf = (rhoa - 1) * math.sqrt(float(lres @ lres)) / math.sqrt(sv2)
    adfs = (tadf + math.sqrt(6 * N * s2v) / (2 * math.sqrt(s20v))) / math.sqrt(s20v / (2 * s2v) +
                                                                             3 * s2v / (10 * s20v))
    dfr_u = (math.sqrt(N) * Tn * (rho - 1) + 3 * math.sqrt(N)) / math.sqrt(51 / 5)
    dft_u = math.sqrt(5 / 4) * trho + math.sqrt(15 * N / 8)
    stats_ = [dfr, dft, adfs, dfr_u, dft_u]
    pvals = [1 - sps.norm.cdf(abs(s)) for s in stats_]
    ip.txt('\nKao test for cointegration\n' + '-' * 26 + '\n')
    ip.txt(f'{"Ho: No cointegration":<44}{"Number of panels":<23}= ')
    ip.res(fmt_num(N, '%7.0g') + '\n')
    ip.txt(f'{"Ha: All panels are cointegrated":<44}{"Avg. number of periods" if unbal else "Number of periods":<23}= ')
    ip.res(fmt_num(Tn, '%7.0g') + '\n\n')
    ip.txt('Cointegrating vector: ')
    ip.res('Same\n')
    kn = {'bartlett': 'Bartlett', 'parzen': 'Parzen', 'quadraticspectral': 'Quad. Spectral'}[kernel]
    ip.txt('Panel means: ' + ' ' * 9)
    ip.res(f'{"Included":<22}')
    ip.txt(f'{"Kernel:":<18}')
    ip.res(kn + '\n')
    ip.txt('Time trend: ' + ' ' * 10)
    ip.res(f'{"Not included":<22}')
    ip.txt(f'{"Lags:":<17}')
    ip.res(f'{haclag:5.2f}')
    if nwest:
        ip.txt(' (')
        ip.res('Newey-West')
        ip.txt(')')
    ip.txt('\nAR parameter: ' + ' ' * 8)
    ip.res(f'{"Same":<22}')
    ip.txt(f'{"Augmented lags:":<18}')
    ip.res(f'{pl} ')
    if lagsel:
        ip.txt('(')
        ip.res(lagsel.upper())
        ip.txt(')')
    ip.txt('\n')
    if p.opts.has('demean'):
        ip.txt('\nCross-sectional means removed\n')
    ip.txt('-' * 78 + '\n' + ' ' * 44 + 'Statistic' + ' ' * 9 + 'p-value\n' + '-' * 78 + '\n')
    for lab, s, pv in zip(('Modified Dickey-Fuller t', 'Dickey-Fuller t', 'Augmented Dickey-Fuller t',
                           'Unadjusted modified Dickey-Fuller t', 'Unadjusted Dickey-Fuller t'), stats_, pvals):
        ip.txt(f' {lab:<43}')
        ip.res(f'{s:8.4f}' + ' ' * 10 + f'{pv:6.4f}\n')
    ip.txt('-' * 78 + '\n')
    from .expr import Matrix
    cn = ['DF rho*', 'DF t*', 'ADF', 'DF rho', 'DF t']
    ip.set_r(scalars={'adf_lags': pl, 'hac_lagm': haclag, 'N_g': N, 'N_t': Tn, 'N': Ntot},
             macros={'test': 'kao', 'deterministics': 'constant', 'hac_kernel': kernel,
                     'adf_method': lagsel or '', 'demean': 'demean' if p.opts.has('demean') else '',
                     'hac_method': 'nwest' if nwest else ''},
             matrices={'stats': Matrix(np.array([stats_]), ['r1'], cn),
                       'p': Matrix(np.array([pvals]), ['r1'], cn)})
