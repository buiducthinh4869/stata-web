"""vecrank, vec (Johansen ML, no constraints), irf after vec, vecstable (ported from Stata's vecrank.ado,
vec.ado, _vecu.ado, _vecauxdisp.ado, _vecgetcv.ado, _vecmka.ado, vecirf_create.ado)."""
import math

import numpy as np
from scipy import stats as sps

from .design import coef_table, Col
from .errors import StataError
from .fmt import fmt_num, g_fmt, abbrev
from .interp import command
from .syntax import parse_command
from .tsuni import need_ts, series
from .tsops import split_name, canonical
from .stab import Tab
from . import tsvar as TV

# Osterwald-Lenum (1992) critical values, rows K-r = 1..11, columns none rconstant constant rtrend trend
_CV = {
    ('max', 95): [[3.84, 9.24, 3.76, 12.52, 3.74], [11.44, 15.67, 14.07, 18.96, 16.87],
                  [17.89, 22.00, 20.97, 25.54, 23.78], [23.80, 28.14, 27.07, 31.46, 30.33],
                  [30.04, 34.40, 33.46, 37.52, 36.41], [36.36, 40.30, 39.37, 43.97, 42.48],
                  [41.51, 46.45, 45.28, 49.42, 48.45], [47.99, 52.00, 51.42, 55.50, 54.25],
                  [53.69, 57.42, 57.12, 61.29, 60.29], [59.06, 63.57, 62.81, 66.23, 66.10],
                  [65.30, 69.74, 68.83, 72.72, 71.68]],
    ('max', 99): [[6.51, 12.97, 6.65, 16.26, 6.40], [15.69, 20.20, 18.63, 23.65, 21.47],
                  [22.99, 26.81, 25.52, 30.34, 28.83], [28.82, 33.24, 32.24, 36.65, 35.68],
                  [35.17, 39.79, 38.77, 42.36, 41.58], [41.00, 46.82, 45.10, 49.51, 48.17],
                  [47.15, 51.91, 51.57, 54.71, 54.48], [53.90, 57.95, 57.69, 62.46, 60.81],
                  [59.78, 63.71, 62.80, 67.88, 66.91], [65.21, 69.94, 69.09, 73.73, 72.96],
                  [72.36, 76.63, 75.95, 79.23, 78.51]],
    ('trace', 95): [[3.84, 9.42, 3.76, 12.25, 3.74], [12.53, 19.96, 15.41, 25.32, 18.17],
                    [24.31, 34.91, 29.68, 42.44, 34.55], [39.89, 53.12, 47.21, 62.99, 54.64],
                    [59.46, 76.07, 68.52, 87.31, 77.74], [82.49, 102.14, 94.15, 114.90, 104.94],
                    [109.99, 131.70, 124.24, 146.76, 136.61], [141.20, 165.58, 156.00, 182.82, 170.80],
                    [175.77, 202.92, 192.89, 222.21, 208.97], [212.67, 244.15, 233.13, 263.42, 250.84],
                    [255.27, 291.40, 277.71, 310.81, 295.99]],
    ('trace', 99): [[6.51, 12.97, 6.65, 16.26, 6.40], [16.31, 24.60, 20.04, 30.45, 23.46],
                    [29.75, 41.07, 35.65, 48.45, 40.49], [45.58, 60.16, 54.46, 70.05, 61.21],
                    [66.52, 84.45, 76.07, 96.58, 85.78], [90.45, 111.01, 103.18, 124.75, 114.36],
                    [119.80, 143.09, 133.57, 158.49, 146.99], [152.32, 177.20, 168.36, 196.08, 182.51],
                    [187.31, 215.74, 204.95, 234.41, 222.46], [226.40, 257.68, 247.18, 279.07, 263.94],
                    [269.81, 307.64, 293.44, 327.45, 312.58]],
}
_TRENDS = ['none', 'rconstant', 'constant', 'rtrend', 'trend']


def _cv(test, level, Kr, trend):
    if Kr < 1 or Kr > 11:
        return np.nan
    return _CV[(test, level)][Kr - 1][_TRENDS.index(trend)]


def _msqrt_inv(S):
    w, V = np.linalg.eigh(S)
    d = np.where(w <= 1e-15, 0.0, 1 / np.sqrt(np.maximum(w, 1e-300)))
    return V @ np.diag(d) @ V.T


class VecData:
    def __init__(self, ip, p, toks, lags, trend, base_mask=None):
        d = ip.data
        need_ts(d)
        keys = [x for x in (d.ts.get('panel'), d.ts.get('time')) if x]
        if d.sortedby[:len(keys)] != keys:
            d.sort(keys)
        if trend not in _TRENDS:
            raise StataError(f'trend({trend}) not valid', 198)
        self.toks = toks
        K = len(toks)
        self.K = K
        self.p = lags
        pm1 = lags - 1
        Ys = [series(ip, v)[0] for v in toks]
        mask = ip.mark(p, d) if base_mask is None else base_mask.copy()
        tv = d.cols[d.ts['time']].astype(float)
        mask &= ~np.isnan(tv)
        for y in Ys:
            for l in range(0, lags + 1):
                mask &= ~np.isnan(ip._lag(d, y, l) if l else y)
        if d.ts.get('panel'):
            pp = d.cols[d.ts['panel']].astype(float)
            if len(np.unique(pp[mask])) > 1:
                raise StataError('sample may not include multiple panels', 459)
        idx = np.flatnonzero(mask)
        T = idx.size
        delta = d.ts.get('delta', 1) or 1
        t = tv[idx]
        if T > 1 and np.any(np.round(np.diff(t) / delta) > 1):
            raise StataError('gaps not allowed', 498)
        self.idx, self.mask, self.T = idx, mask, T
        self.tmin, self.tmax = float(t.min()), float(t.max())
        self.tmins, self.tmaxs = TV._tsfmt_range(d, self.tmin, self.tmax)
        trendv = (t - self.tmin) / delta + 1
        self.trendv = trendv
        dY = [y - ip._lag(d, y, 1) for y in Ys]
        self.Z0 = np.column_stack([v[idx] for v in dY])
        self.Ylev = np.column_stack([y[idx] for y in Ys])
        Z1 = [ip._lag(d, y, 1)[idx] for y in Ys]
        z1n = []
        if trend == 'rconstant':
            Z1.append(np.ones(T))
            z1n = ['_cons']
        elif trend == 'rtrend':
            Z1.append(trendv)
            z1n = ['_trend']
        self.Z1 = np.column_stack(Z1)
        Zl = [ip._lag(d, y, 1)[idx] for y in Ys]
        if trend == 'rconstant':
            Zl.append(np.ones(T))
        elif trend == 'rtrend':
            Zl.append(trendv - 1)
        self.Zlag = np.column_stack(Zl)
        self.z1n = z1n
        lagd, lagn, lagi = [], [], []
        for v, dy in zip(toks, dY):
            ch, base = split_name(v)
            for l in range(1, pm1 + 1):
                lagd.append(ip._lag(d, dy, l)[idx])
                c = f'L{l}D' + (ch or '')
                lagn.append(canonical(c, base))
                lagi.append((base, c))
        dt, dtn = [], []
        if trend in ('constant', 'rtrend'):
            dt, dtn = [np.ones(T)], ['_cons']
        elif trend == 'trend':
            dt, dtn = [trendv, np.ones(T)], ['_trend', '_cons']
        self.lagd, self.lagn, self.lagi = lagd, lagn, lagi
        self.dt, self.dtn = dt, dtn
        Z2 = lagd + dt
        self.Z2 = np.column_stack(Z2) if Z2 else np.zeros((T, 0))
        self.nz1 = self.Z1.shape[1]
        self.nz2 = K * pm1 + len(dt)
        self.trend = trend
        if self.Z2.shape[1]:
            G = np.linalg.lstsq(self.Z2, np.column_stack([self.Z0, self.Z1]), rcond=None)[0]
            R = np.column_stack([self.Z0, self.Z1]) - self.Z2 @ G
        else:
            R = np.column_stack([self.Z0, self.Z1])
        R0, R1 = R[:, :K], R[:, K:]
        self.S00 = R0.T @ R0 / T
        self.S01 = R0.T @ R1 / T
        self.S11 = R1.T @ R1 / T
        s11i = _msqrt_inv(self.S11)
        M = s11i @ self.S01.T @ np.linalg.inv(self.S00) @ self.S01 @ s11i
        lam, U = np.linalg.eigh((M + M.T) / 2)
        o = np.argsort(-lam)
        self.lam = lam[o]
        self.beta = s11i @ U[:, o]

    def ll_rank(self, r):
        T, K = self.T, self.K
        c = -.5 * T * K * math.log(2 * math.pi) - .5 * T * math.log(np.linalg.det(self.S00)) - .5 * T * K
        return c - .5 * T * float(np.sum(np.log1p(-self.lam[:r])))

    def parms(self, r):
        return self.K * self.nz2 + (self.K + self.nz1 - r) * r


def _vec_common_opts(ip, p):
    toks = TV._toks(ip, p.body)
    if len(toks) < 2:
        raise StataError('too few variables specified', 102)
    lg = p.opts.get('lags', 2)
    lags = int(lg) if lg not in (None, '') else 2
    if lags < 1:
        raise StataError('lags() must be a positive integer', 198)
    tr = (p.opts.get('trend', 1) or 'constant').strip().lower()
    if p.opts.get('sindicators', 2) or p.opts.has('seasonal'):
        raise StataError('sindicators() is not available in Stata Web', 198)
    return toks, lags, tr


@command('vecrank')
def cmd_vecrank(ip, cmd, rest, by):
    p = parse_command(rest)
    if not p.body.strip():
        e = ip.est
        if not e or e.get('cmd') != 'vecrank':
            raise StataError('vecrank results not found', 301)
        return e['replay'](ip, p)
    toks, lags, trend = _vec_common_opts(ip, p)
    V = VecData(ip, p, toks, lags, trend)
    K, T = V.K, V.T
    lam = V.lam[:K]
    ll = [V.ll_rank(r) for r in range(K + 1)]
    parms = [V.parms(r) for r in range(K + 1)]
    mx = [-T * math.log1p(-lam[i]) for i in range(K)]
    tr = [sum(mx[i:]) for i in range(K)]
    sel = {}
    for lev in (95, 99):
        r_ = None
        for r in range(K):
            if tr[r] <= _cv('trace', lev, K - r, trend):
                r_ = r
                break
        sel[lev] = r_ if r_ is not None else (K if tr and False else np.nan)
    aic = [(-2 * l_ + 2 * k_) / T for l_, k_ in zip(ll, parms)]
    bic = [(-2 * l_ + math.log(T) * k_) / T for l_, k_ in zip(ll, parms)]
    hq = [(-2 * l_ + 2 * math.log(math.log(T)) * k_) / T for l_, k_ in zip(ll, parms)]
    sel['sbic'] = int(np.argmin(bic))
    sel['hqic'] = int(np.argmin(hq))
    res = {'K': K, 'T': T, 'lam': lam, 'll': ll, 'parms': parms, 'max': mx, 'trace': tr, 'sel': sel,
           'aic': aic, 'sbic': bic, 'hqic': hq, 'trend': trend, 'lags': lags, 'tmins': V.tmins, 'tmaxs': V.tmaxs}

    def replay(ip_, pp):
        _vecrank_disp(ip_, res, pp)
    from .expr import Matrix
    est = {'cmd': 'vecrank', 'replay': replay, 'names': [], 'scalars': {
        'N': T, 'k_eq': K, 'n_lags': lags, 'tmin': V.tmin, 'tmax': V.tmax,
        'k_ce95': sel[95] if sel[95] == sel[95] else np.nan, 'k_ce99': sel[99] if sel[99] == sel[99] else np.nan,
        'k_cesbic': sel['sbic'], 'k_cehqic': sel['hqic']},
        'macros': {'cmd': 'vecrank', 'trend': trend},
        'matrices': {'ll': Matrix([ll]), 'k_rank': Matrix([parms]), 'lambda': Matrix([list(lam)]),
                     'trace': Matrix([tr]), 'max': Matrix([mx]), 'aic': Matrix([aic]), 'sbic': Matrix([bic]),
                     'hqic': Matrix([hq])}}
    ip.set_estimation(est)
    replay(ip, p)


def _vecrank_disp(ip, R, p):
    l99, la = p.opts.has('level99', 7), p.opts.has('levela', 6)
    disp95 = not l99 or la
    disp99 = l99 or la
    both = disp95 and disp99
    tests = []
    if not p.opts.has('notrace', 5):
        tests.append('trace')
    if p.opts.has('max', 3):
        tests.append('max')
    if p.opts.has('ic'):
        tests.append('ic')
    K, T = R['K'], R['T']
    head = True
    for test in tests:
        if head:
            ip.txt('\n' + 'Johansen tests for cointegration'.center(79).rstrip() + '\n')
            ip.txt(f"{'Trend: ' + R['trend']:<56}Number of obs = ")
            ip.res(f'{T:7.0f}\n')
            s = f"Sample: {R['tmins']} thru {R['tmaxs']}"
            ip.txt(f'{s:<65}Lags = ')
            ip.res(f"{R['lags']:7.0f}\n")
            ip.txt('-' * 79 + '\n')
            head = False
        if test != 'ic':
            vals = R['trace'] if test == 'trace' else R['max']
            tn = 'trace' if test == 'trace' else ' max'
            def hl(*parts):
                L_ = _Line()
                for c_, t_ in parts:
                    L_.col(c_, t_)
                ip.txt(L_.s + '\n')
            if both:
                hl((1, 'maximum'), (46, tn), (56, '5% critical'), (69, '1% critical'))
                hl((1, '  rank'), (11, 'parms'), (23, 'LL'), (32, 'eigenvalue'), (44, 'statistic'), (59, 'value'),
                   (72, 'value'))
            else:
                hl((58, f'{5 if disp95 else 1}%'))
                hl((1, 'maximum'), (46, tn), (55, 'critical'))
                hl((1, '  rank'), (11, 'parms'), (23, 'LL'), (32, 'eigenvalue'), (44, 'statistic'), (57, 'value'))
            for i in range(K + 1):
                line = _Line()
                line.col(3, f'{i:3.0f}')
                line.col(12, f'{R["parms"][i]:<4.0f}')
                line.col(19, fmt_num(R['ll'][i], '%10.9g'))
                if i > 0 or not both:
                    ev = R['lam'][i - 1] if i > 0 else np.nan
                    line.col(33, fmt_num(ev, '%8.5f'))
                if i < K:
                    star = ''
                    if test == 'trace':
                        if both:
                            star = ('*1' if R['sel'][99] == i else '') + ('*5' if R['sel'][95] == i else '')
                        else:
                            star = '*' if R['sel'][95 if disp95 else 99] == i else ''
                    line.col(44, f'{vals[i]:9.4f}' + star)
                    if both:
                        line.col(57, f'{_cv(test, 95, K - i, R["trend"]):7.2f}')
                        line.col(70, f'{_cv(test, 99, K - i, R["trend"]):7.2f}')
                    else:
                        line.col(55, f'{_cv(test, 95 if disp95 else 99, K - i, R["trend"]):7.2f}')
                ip.res(line.s + '\n')
            ip.txt('-' * 79 + '\n')
        else:
            ip.txt('maximum\n')
            L_ = _Line()
            for c_, t_ in ((1, '  rank'), (11, 'parms'), (23, 'LL'), (32, 'eigenvalue'), (47, 'SBIC'), (58, 'HQIC'),
                           (69, 'AIC')):
                L_.col(c_, t_)
            ip.txt(L_.s + '\n')
            for i in range(K + 1):
                line = _Line()
                line.col(3, f'{i:3.0f}')
                line.col(12, f'{R["parms"][i]:<4.0f}')
                line.col(19, fmt_num(R['ll'][i], '%10.9g'))
                if i > 0:
                    line.col(33, fmt_num(R['lam'][i - 1], '%8.5f'))
                line.col(44, fmt_num(R['sbic'][i], '%9.8g') + ('*' if R['sel']['sbic'] == i else ''))
                line.col(55, fmt_num(R['hqic'][i], '%9.8g') + ('*' if R['sel']['hqic'] == i else ''))
                line.col(66, fmt_num(R['aic'][i], '%9.8g'))
                ip.res(line.s + '\n')
            ip.txt('-' * 79 + '\n')


class _Line:
    def __init__(self):
        self.s = ''

    def col(self, c, txt):
        if len(self.s) < c - 1:
            self.s += ' ' * (c - 1 - len(self.s))
        self.s += txt


# ---------------------------------------------------------------- vec
def _ols_qsur(Y, X, df):
    """_qsur with dfmodel(df): per-equation OLS on common X; Sigma = E'E/(T-df); V = Sigma (x) (X'X)^-1."""
    T, K = Y.shape
    XtXi = np.linalg.inv(X.T @ X)
    B = XtXi @ X.T @ Y
    E = Y - X @ B
    Sig = E.T @ E / (T - df)
    V = np.kron(Sig, XtXi)
    return B, E, Sig, V, XtXi


@command('vec')
def cmd_vec(ip, cmd, rest, by):
    p = parse_command(rest)
    if not p.body.strip():
        e = ip.est
        if not e or e.get('cmd') != 'vec':
            raise StataError('vec results not found', 301)
        return e['replay'](ip, p)
    if p.opts.get('aconstraints', 2) or p.opts.get('bconstraints', 2):
        raise StataError('aconstraints()/bconstraints() are not available in Stata Web', 198)
    toks, lags, trend = _vec_common_opts(ip, p)
    rk = p.opts.get('rank', 1)
    r = int(rk) if rk not in (None, '') else 1
    V = VecData(ip, p, toks, lags, trend)
    K, T, nz1 = V.K, V.T, V.nz1
    if r < 1 or r >= K:
        raise StataError(f'rank() must be less than {K}', 198)
    parms1 = K * V.nz2 + (K + nz1 - r) * r
    df = int(parms1 / K)
    if T <= df:
        raise StataError('there are too many parameters in the model', 498)
    b0 = V.beta[:, :r]
    if np.min(np.abs(np.diag(np.linalg.inv(b0.T @ b0)))) <= 1e-15:
        raise StataError('the parameters of the cointegrating equations are not identified by the sample', 498)
    beta2 = b0 @ np.linalg.inv(b0[:r, :r])
    alpha2 = V.S01 @ beta2 @ np.linalg.inv(beta2.T @ V.S11 @ beta2)
    omega = V.S00 - alpha2 @ beta2.T @ V.S01.T
    detsig = float(np.linalg.det(omega))
    llv = -.5 * T * K * math.log(2 * math.pi) - .5 * T * math.log(abs(detsig)) - .5 * T * K
    aic = (-2 * llv + 2 * parms1) / T
    bic = (-2 * llv + math.log(T) * parms1) / T
    hq = (-2 * llv + 2 * math.log(math.log(T)) * parms1) / T
    ce = V.Zlag @ beta2                     # L._ce (beta' y_{t-1} [+ restricted term at t-1])
    X = np.column_stack([ce] + V.lagd + V.dt)
    xn = [f'L._ce{j + 1}' for j in range(r)] + V.lagn + V.dtn
    xi = [(f'_ce{j + 1}', 'L') for j in range(r)] + V.lagi + [(None, '')] * len(V.dtn)
    B, E, Sig, Vb, XtXi = _ols_qsur(V.Z0, X, df)
    p0 = p1 = None
    beta_full = beta2
    if trend in ('constant', 'rtrend', 'trend'):
        ci = xn.index('_cons')
        mu0 = B[ci, :]
        p0 = np.linalg.solve(alpha2.T @ alpha2, alpha2.T @ mu0)
        rows = [beta2]
        if trend == 'trend':
            ti = xn.index('_trend')
            p1 = np.linalg.solve(alpha2.T @ alpha2, alpha2.T @ B[ti, :])
            rows.append(p1[None, :])
        rows.append(p0[None, :])
        beta3 = np.vstack(rows)
        # _ce now includes the constant (and trend) of the cointegrating relation
        Zc = [V.Zlag]
        if trend == 'trend':
            Zc.append((V.trendv - 1)[:, None])
        Zc.append(np.ones((T, 1)))
        ce = np.hstack(Zc) @ beta3
        X = np.column_stack([ce] + V.lagd + V.dt)
        B, E, Sig, Vb, XtXi = _ols_qsur(V.Z0, X, df)
        beta_full = beta3
    m = X.shape[1]
    eqn = [('D_' + v.replace('.', '_')) for v in toks]
    eqstats = []
    for i in range(K):
        y = V.Z0[:, i]
        e_ = E[:, i]
        ssr = float(e_ @ e_)
        nc = [j for j in range(m) if xn[j] != '_cons']
        b = B[nc, i]
        Vi = Sig[i, i] * XtXi[np.ix_(nc, nc)]
        chi2 = float(b @ np.linalg.solve(Vi, b))
        cons = '_cons' in xn
        tss = float(np.sum((y - y.mean()) ** 2)) if cons else float(y @ y)
        eqstats.append({'k': m, 'rmse': math.sqrt(ssr / (T - m)), 'r2': 1 - ssr / tss, 'chi2': chi2,
                        'df_m': len(nc), 'p': float(sps.chi2.sf(chi2, len(nc)))})
    # variance of beta (Johansen normalisation)
    omegai = np.linalg.inv(omega)
    rho = alpha2.T @ omegai @ alpha2
    H = np.vstack([np.zeros((r, nz1 - r)), np.eye(nz1 - r)])
    base = H.T @ V.S11 @ H
    Ith = (1 / (T - df)) * np.linalg.inv(np.kron(rho, base))
    H2 = np.kron(np.eye(r), H)
    Ith = H2 @ Ith @ H2.T
    nb = nz1
    if trend in ('constant', 'rtrend'):
        blk = np.kron(np.eye(r), np.vstack([np.eye(nz1), np.zeros((1, nz1))]))
        Ith = blk @ Ith @ blk.T
        nb = nz1 + 1
    elif trend == 'trend':
        blk = np.kron(np.eye(r), np.vstack([np.eye(K), np.zeros((2, K))]))
        Ith = blk @ Ith @ blk.T
        nb = K + 2
    bvec = beta_full.reshape(-1, order='F')
    brow = list(toks) + V.z1n + (['_trend'] if trend == 'trend' else []) + \
        (['_cons'] if trend in ('constant', 'rtrend', 'trend') else [])
    sigbb = np.linalg.inv(beta2.T @ V.S11 @ beta2)
    alpha_V = (1 / (T - df)) * np.kron(omega, sigbb)
    pi = alpha2 @ beta2.T
    full_names = [f'{q}:{nm}' for q in eqn for nm in xn]
    est = {'cmd': 'vec', 'names': full_names, 'b': B.T.reshape(-1), 'V': Vb, 'colinfo': None, 'sample': V.mask,
           'depvar': ' '.join(toks), 'nomargins': True, 'endog': toks, 'eqnames': eqn, 'rank': r,
           'n_lags': lags, 'trend': trend, 'alpha': alpha2, 'beta': beta2, 'beta_full': beta_full,
           'omega': omega, 'Bmat': B, 'xn': xn, 'xi': xi, 'T': T, 'df_m': df, 'pi': pi, 'K': K, 'Z0': V.Z0,
           'E': E, 'idx': V.idx, 'tmin': V.tmin, 'tmax': V.tmax}
    sc = {'N': T, 'k_ce': r, 'n_lags': lags, 'k_eq': K, 'k_dv': K, 'k_rank': parms1, 'df_m': df, 'll': llv,
          'aic': aic, 'sbic': bic, 'hqic': hq, 'detsig_ml': detsig, 'tmin': V.tmin, 'tmax': V.tmax,
          'beta_iden': 1, 'beta_icnt': r * r, 'df_lr': 0, 'chi2_res': 0.0, 'N_gaps': 0}
    for i, rr in enumerate(eqstats, start=1):
        sc.update({f'k_{i}': rr['k'], f'rmse_{i}': rr['rmse'], f'r2_{i}': rr['r2'], f'df_m{i}': rr['df_m'],
                   f'chi2_{i}': rr['chi2']})
    from .expr import Matrix
    bn = [f'_ce{j + 1}:{v}' for j in range(r) for v in brow]
    est['scalars'] = sc
    est['matrices'] = {'beta': Matrix([bvec], ['beta'], bn), 'V_beta': Matrix(Ith, bn, bn),
                       'alpha': Matrix([alpha2.reshape(-1)], ['alpha'],
                                       [f'{q}:L._ce{j + 1}' for q in eqn for j in range(r)]),
                       'V_alpha': Matrix(alpha_V), 'omega': Matrix(omega, eqn, eqn),
                       'pi': Matrix([pi.reshape(-1)], ['pi'])}
    est['macros'] = {'cmd': 'vec', 'title': 'Vector error-correction model', 'trend': trend,
                     'endog': ' '.join(toks), 'eqnames': ' '.join(eqn),
                     'cenames': ' '.join(f'_ce{j + 1}' for j in range(r))}

    def replay(ip_, pp):
        ip_.txt('\nVector error-correction model\n\n')
        s = f'Sample: {V.tmins} thru {V.tmaxs}'
        ip_.txt(f'{s:<48}{"Number of obs":<18}= ')
        ip_.res(f'{T:>10,}\n')
        ip_.txt(f'{"":<48}{"AIC":<18}=  ')
        ip_.res(fmt_num(aic, '%9.8g') + '\n')
        ip_.txt('Log likelihood = ')
        ip_.res(fmt_num(llv, '%9.8g'))
        ip_.txt(' ' * (48 - 17 - 9) + f'{"HQIC":<18}=  ')
        ip_.res(fmt_num(hq, '%9.8g') + '\n')
        ip_.txt(f'{"Det(Sigma_ml)":<15}= ')
        ip_.res(fmt_num(detsig, '%9.8g'))
        ip_.txt(' ' * (48 - 17 - 9) + f'{"SBIC":<18}=  ')
        ip_.res(fmt_num(bic, '%9.8g') + '\n')
        if not pp.opts.has('noetable', 5):
            ip_.txt('\n')
            t = Tab(ip_, '17 8 11 10 11 7', lmargin=0)
            t.set_numfmt(None, '%6.0f', '%8.7g', '%7.4f', '%9.8g', '%7.4f')
            t.set_pad(None, None, 2, 1, None, None)
            t.set_strfmt('%-16s', None, None, None, None, None)
            t.set_titlefmt('%-16s', None, None, None, None, None)
            t.titles('Equation', 'Parms', 'RMSE ', 'R-sq  ', 'chi2   ', ' P>chi2')
            t.sep('top')
            for i, rr in enumerate(eqstats):
                t.row(abbrev(eqn[i], 16), rr['k'], rr['rmse'], rr['r2'], rr['chi2'], rr['p'])
            t.sep('bot')
            ip_.txt('\n')
            for i, q in enumerate(eqn):
                ix = list(range(i * m, (i + 1) * m))
                info = []
                for nm, (bse, ch) in zip(xn, xi):
                    c = Col(name=nm, values=None, term=0, group='', level='',
                            kind='cons' if nm == '_cons' else 'cont', vars=[])
                    c.tsvar, c.tschain = bse, ch
                    info.append(c)
                sub = {'names': xn, 'b': B[:, i], 'V': Vb[np.ix_(ix, ix)], 'colinfo': info, 'df_r': None}
                coef_table(ip_, sub, 'z', depvar=' ', header=(i == 0), footer=(i == K - 1), eq_title=q)
        if not pp.opts.has('nobtable', 5):
            _beta_disp(ip_, bvec, Ith, r, brow, nb)
        if pp.opts.has('alpha', 2):
            _alpha_disp(ip_, alpha2, alpha_V, eqn, r)
    est['replay'] = replay
    est['predict'] = None
    est['predict_multi'] = lambda ip_, pp, typ, tk: _vec_predict(ip_, est, pp, typ, tk)
    ip.set_estimation(est)
    replay(ip, p)


def _aux_table(ip, title, eqs, rows_by_eq, b, Vm, depname):
    """Cointegrating-equations style chi2 table and coefficient table (zero variances shown as '.')."""
    se = np.sqrt(np.maximum(np.diag(Vm), 0))
    zc = sps.norm.ppf(.975)
    ip.txt(f'\n{title}\n\n')
    t = Tab(ip, '17 8 11 7', lmargin=0)
    t.set_numfmt(None, '%6.0f', '%9.8g', '%7.4f')
    t.set_strfmt('%-16s', None, None, None)
    t.set_titlefmt('%-16s', None, None, None)
    t.titles('Equation', 'Parms', 'chi2   ', ' P>chi2')
    t.sep('top')
    k = 0
    for q, rows in zip(eqs, rows_by_eq):
        ix = [k + j for j in range(len(rows)) if se[k + j] > 0 and rows[j] != '_cons']
        bb = b[ix]
        chi2 = float(bb @ np.linalg.solve(Vm[np.ix_(ix, ix)], bb)) if ix else np.nan
        t.row(q, float(len(ix)), chi2, float(sps.chi2.sf(chi2, len(ix))) if ix else np.nan)
        k += len(rows)
    t.sep('bot')
    return se, zc


def _coef_rows(ip, eqs, rows_by_eq, b, se, zc, depname):
    ip.txt('-' * 78 + '\n')
    ip.txt(f'{depname:>12} | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]\n')
    k = 0
    for q, rows in zip(eqs, rows_by_eq):
        ip.txt('-' * 13 + '+' + '-' * 64 + '\n')
        ip.res(f'{q:<12}')
        ip.txt(' |\n')
        for nm in rows:
            if nm.startswith('L.'):
                ip.txt(f'{abbrev(nm[2:], 12):>12} |\n')
                nm = 'L1.'
            ip.txt(f'{abbrev(nm, 12):>12} |')
            if se[k] > 0:
                z = b[k] / se[k]
                ip.res(f'{g_fmt(b[k], 9):>11}{g_fmt(se[k], 9):>11}{z:9.2f}{2 * sps.norm.sf(abs(z)):8.3f}'
                       f'{g_fmt(b[k] - zc * se[k], 9):>13}{g_fmt(b[k] + zc * se[k], 9):>12}\n')
            else:
                ip.res(f'{g_fmt(b[k], 9):>11}{".":>11}{".":>9}{".":>8}{".":>13}{".":>12}\n')
            k += 1
    ip.txt('-' * 78 + '\n')


def _beta_disp(ip, bvec, Ith, r, brow, nb):
    eqs = [f'_ce{j + 1}' for j in range(r)]
    rows = [brow] * r
    se, zc = _aux_table(ip, 'Cointegrating equations', eqs, rows, bvec, Ith, 'beta')
    ip.txt('\nIdentification:  beta is exactly identified\n\n')
    ip.txt(' ' * 17 + f'Johansen normalization restriction{"s" if r > 1 else ""} imposed\n')
    _coef_rows(ip, eqs, rows, bvec, se, zc, 'beta')


def _alpha_disp(ip, alpha2, aV, eqn, r):
    b = alpha2.reshape(-1)
    rows = [[f'L._ce{j + 1}' for j in range(r)]] * len(eqn)
    se, zc = _aux_table(ip, 'Adjustment parameters', eqn, rows, b, aV, 'alpha')
    _coef_rows(ip, eqn, rows, b, se, zc, 'alpha')


def _vec_predict(ip, est, p, typ, toks):
    from .fmt import comma_int
    d = ip.data
    if not toks:
        raise StataError('varlist required', 100)
    out = np.full(d.n, np.nan)
    if p.opts.has('ce'):
        eq = (p.opts.get('equation', 2) or '#1').strip()
        j = int(eq[1:]) - 1 if eq.startswith('#') else [f'_ce{k + 1}' for k in range(est['rank'])].index(eq)
        Ys = np.column_stack([series(ip, v)[0] for v in est['endog']])
        bf = est['beta_full'][:, j]
        K = est['K']
        val = Ys @ bf[:K]
        extra = bf[K:]
        if est['trend'] == 'rconstant':
            val = val + extra[0]
        elif est['trend'] == 'rtrend':
            tv = d.cols[d.ts['time']].astype(float)
            delta = d.ts.get('delta', 1) or 1
            val = val + extra[0] * ((tv - est['tmin']) / delta + 1) + (extra[1] if extra.size > 1 else 0)
        elif est['trend'] == 'trend':
            tv = d.cols[d.ts['time']].astype(float)
            delta = d.ts.get('delta', 1) or 1
            val = val + extra[0] * ((tv - est['tmin']) / delta + 1) + extra[1]
        elif est['trend'] == 'constant':
            val = val + extra[0]
        out = val
        lab = f'Predicted cointegrated equation'
    else:
        eq = (p.opts.get('equation', 2) or '#1').strip()
        i = int(eq[1:]) - 1 if eq.startswith('#') else est['eqnames'].index(eq)
        fit = est['Z0'][:, i] - est['E'][:, i]
        if p.opts.has('residuals', 1):
            out[est['idx']] = est['E'][:, i]
            lab = 'Residuals'
        else:
            out[est['idx']] = fit
            lab = 'Linear prediction'
        lab += f', {est["eqnames"][i]}'
    out = np.where(ip.mark(p), out, np.nan)
    d.add_var(toks[0], out, typ or 'float')
    d.varlab[toks[0]] = lab
    nm = int(np.isnan(out).sum())
    if nm:
        ip.txt(f'({comma_int(nm)} missing value{"s" if nm != 1 else ""} generated)\n')


def vec_levels_A(est):
    """Levels-VAR matrices A_1..A_p implied by the VECM (_vecmka.ado)."""
    K, r = est['K'], est['rank']
    p = est['n_lags']
    pm1 = p - 1
    B = est['Bmat']
    xn = est['xn']
    Pi = est['alpha'] @ est['beta'][:K].T
    G = []
    for l in range(1, pm1 + 1):
        Gl = np.zeros((K, K))
        for j, v in enumerate(est['endog']):
            ch, base = split_name(v)
            nm = canonical(f'L{l}D' + (ch or ''), base)
            Gl[:, j] = B[xn.index(nm), :]
        G.append(Gl)
    A = []
    if pm1 > 0:
        A.append(Pi + G[0] + np.eye(K))
        for i in range(2, pm1 + 1):
            A.append(G[i - 1] - G[i - 2])
        A.append(-G[pm1 - 1])
    else:
        A.append(Pi + np.eye(K))
    return A


def vec_irf_create(ip, p, name):
    from . import tsirf as TI
    e = ip.est
    stp = p.opts.get('step', 2)
    step = int(stp) if stp not in (None, '') else 8
    sv = p.opts.get('set', 3)
    if sv:
        TI._set_file(ip, sv)
    fobj, f = TI._active(ip)
    if name in fobj['res']:
        if p.opts.has('replace'):
            fobj['irfnames'].remove(name)
            del fobj['res'][name]
        else:
            raise StataError(f'irfname {name} is already in {f}', 498)
    T, dfm = e['T'], e['df_m']
    om = e['omega'] * (1.0 if p.opts.has('dfk') else T / (T - dfm))
    A = vec_levels_A(e)
    K = e['K']
    st = TI.irf_stats(A, om, step, list(range(K)))
    fobj['res'][name] = {'order': list(e['endog']), 'step': step, 'stats': st, 'std': {}, 'model': 'vec',
                         'endog': list(e['endog'])}
    fobj['irfnames'].append(name)
    ip.txt(f'(file {f} updated)\n')


@command('vecstable')
def cmd_vecstable(ip, cmd, rest, by):
    e = ip.est
    if not e or e.get('cmd') != 'vec':
        raise StataError('vecstable only works after vec', 301)
    p = parse_command(rest)
    K = e['K']
    A = vec_levels_A(e)
    C = TV.companion(A, K)
    ev = np.linalg.eigvals(C)
    ev = sorted(ev, key=lambda z: (-round(abs(z), 12), -z.imag))
    re_, im_, mod = [z.real for z in ev], [z.imag for z in ev], [abs(z) for z in ev]
    ip.txt('\n   Eigenvalue stability condition\n')
    t1 = Tab(ip, '|26|13|')
    t2 = Tab(ip, '|12 14|13|')
    t2.set_numfmt('%10.7g', None, '%8.7g')
    t2.set_pad(1, None, 2)
    t1.sep('top')
    t1.titles('Eigenvalue       ', 'Modulus  ')
    t1.sep('mid')
    for rr, c, md in zip(re_, im_, mod):
        if abs(c) > 1e-13:
            t2.row(rr, ('-' if c < 0 else '+') + fmt_num(abs(c), '%10.7g') + '{it:i}', md)
        else:
            t2.row(rr, '', md)
    t1.sep('bot')
    nu = K - e['rank']
    ip.txt(f'   The VECM specification imposes {nu} unit modul{"us" if nu == 1 else "i"}.\n')
    from .expr import Matrix
    ip.set_r(matrices={'Re': Matrix([re_]), 'Im': Matrix([im_]), 'Modulus': Matrix([mod])})
    if p.opts.has('graph', 2):
        TV._stable_graph(ip, re_, im_, p)
