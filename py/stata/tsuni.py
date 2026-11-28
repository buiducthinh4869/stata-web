"""Univariate time-series tests, ported from the official/SSC ado code:
dfuller, pperron, kpss (Baum), zandrews (Baum), corrgram, ac, pac, wntestq, tsfilter hp."""
import math

import numpy as np
from scipy import stats as sps

from .errors import StataError
from .fmt import g_fmt, comma_int, abbrev
from .interp import command, alias
from .syntax import parse_command, split_top
from .tsops import apply as ts_apply

alias('dfuller', 'dfu', 'dful', 'dfull', 'dfulle')
alias('pperron', 'pperr', 'pperro')
alias('corrgram', 'corrgr', 'corrgra')
alias('wntestq', 'wntestq')


# ================================================================ helpers
def ts_unit(d):
    """Stata tsset unit from the time variable's display format."""
    t = (d.ts or {}).get('time')
    f = d.formats.get(t, '') if t else ''
    f = f.strip().lstrip('%').lstrip('-')
    return {'td': 'daily', 'tw': 'weekly', 'tm': 'monthly', 'tq': 'quarterly', 'th': 'halfyearly',
            'ty': 'yearly'}.get(f[:2], 'generic')


def need_ts(d):
    if not d.ts or not d.ts.get('time'):
        raise StataError('time variable not set, use tsset varname ...', 111)


def ts_sample(ip, p, var, sort=True):
    """touse for a single-series command (onepanel), sorted by panel/time."""
    d = ip.data
    need_ts(d)
    if sort:
        keys = [x for x in (d.ts.get('panel'), d.ts.get('time')) if x]
        if d.sortedby[:len(keys)] != keys:
            d.sort(keys)
    v = d.resolve(var) if '.' not in var else None
    mask = ip.mark(p, d)
    t = d.cols[d.ts['time']].astype(float)
    mask &= ~np.isnan(t)
    pv = d.ts.get('panel')
    if pv:
        pp = d.cols[pv].astype(float)
        mask &= ~np.isnan(pp)
        if len(np.unique(pp[mask])) > 1:
            raise StataError('sample may not include multiple panels', 459)
    return mask, v


def series(ip, var):
    """Values of a (possibly operated) variable over the whole dataset."""
    from .tsops import has_tsop, expand_token, split_name
    d = ip.data
    if has_tsop(var):
        ex = expand_token(var)
        ch, v = split_name(ex[0])
        v = d.resolve(v)
        return (ts_apply(ip, d, ch, v) if ch else d.cols[v].astype(float)), (f'{ch}.{v}' if ch else v)
    v = d.resolve(var)
    if d.isstr(v):
        raise StataError(f'{v}: string variables not allowed', 109)
    return d.cols[v].astype(float), v


def lagged(ip, x, k):
    """L^k of an arbitrary array respecting tsset (panel-aware)."""
    return ip._lag(ip.data, x, k)


def n_gaps(ip, mask):
    d = ip.data
    t = d.cols[d.ts['time']].astype(float)[mask]
    if t.size < 2:
        return 0
    delta = d.ts.get('delta', 1) or 1
    return int(np.sum(np.round(np.diff(np.sort(t)) / delta) > 1))


def ols(y, X):
    """Plain OLS on complete rows. Returns dict with b, se, V, df_r, rss, N, rmse, resid, df_m."""
    n, k = X.shape
    XtX = X.T @ X
    try:
        XtXi = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtXi = np.linalg.pinv(XtX)
    b = XtXi @ (X.T @ y)
    e = y - X @ b
    rss = float(e @ e)
    df_r = n - k
    s2 = rss / df_r if df_r > 0 else np.nan
    V = XtXi * s2
    return {'b': b, 'se': np.sqrt(np.maximum(np.diag(V), 0)), 'V': V, 'df_r': df_r, 'rss': rss, 'N': n,
            'rmse': math.sqrt(s2) if s2 == s2 else np.nan, 'resid': e, 'k': k}


def regress_ts(y, cols, mask, cons=True):
    """OLS of y on cols (full-length arrays) over mask & complete cases."""
    M = mask & ~np.isnan(y)
    for c in cols:
        M &= ~np.isnan(c)
    X = np.column_stack([c[M] for c in cols] + ([np.ones(int(M.sum()))] if cons else [])) if (cols or cons) \
        else np.zeros((int(M.sum()), 0))
    r = ols(y[M], X)
    r['sample'] = M
    r['df_m'] = X.shape[1] - (1 if cons else 0)
    return r


def _interp_cv(table, N):
    """Fuller-table interpolation exactly as in dfuller.ado / pperron.ado (columns 25,50,100,250,500,inf)."""
    out = []
    for row in table:
        if N <= 25:
            v = row[0]
        elif N <= 50:
            v = row[0] + (N - 25) / 25 * (row[1] - row[0])
        elif N <= 100:
            v = row[1] + (N - 50) / 50 * (row[2] - row[1])
        elif N <= 250:
            v = row[2] + (N - 100) / 150 * (row[3] - row[2])
        elif N <= 500:
            v = row[3] + (N - 250) / 250 * (row[4] - row[3])
        else:
            v = row[5]
        out.append(v)
    return out


_ZT = {1: [[-2.66, -2.62, -2.60, -2.58, -2.58, -2.58], [-1.95, -1.95, -1.95, -1.95, -1.95, -1.95],
           [-1.60, -1.61, -1.61, -1.62, -1.62, -1.62]],
       2: [[-3.75, -3.58, -3.51, -3.46, -3.44, -3.43], [-3.00, -2.93, -2.89, -2.88, -2.87, -2.86],
           [-2.63, -2.60, -2.58, -2.57, -2.57, -2.57]],
       4: [[-4.38, -4.15, -4.04, -3.99, -3.98, -3.96], [-3.60, -3.50, -3.45, -3.43, -3.42, -3.41],
           [-3.24, -3.18, -3.15, -3.13, -3.13, -3.12]]}
_ZP = {1: [[-11.9, -12.9, -13.3, -13.6, -13.7, -13.8], [-7.3, -7.7, -7.9, -8.0, -8.0, -8.1],
           [-5.3, -5.5, -5.6, -5.7, -5.7, -5.7]],
       2: [[-17.2, -18.9, -19.8, -20.3, -20.5, -20.7], [-12.5, -13.3, -13.7, -14.0, -14.0, -14.1],
           [-10.2, -10.7, -11.0, -11.2, -11.2, -11.3]],
       4: [[-22.5, -25.7, -27.4, -28.4, -28.9, -29.5], [-17.9, -19.8, -20.7, -21.3, -21.5, -21.8],
           [-15.6, -16.8, -17.5, -18.0, -18.1, -18.3]]}


def mackinnon_p(tau, kind):
    """MacKinnon (1994) approximate p-value, N=1, as in dfuller.ado (kind 'c' or 'ct')."""
    if kind == 'c':
        if tau > -1.61:
            mn, mx, g = -9999, 2.74, (1.7339, 0.93202, -0.12745, -0.010368)
        else:
            mn, mx, g = -18.83, None, (2.1659, 1.4412, 0.038269, 0.0)
    else:
        if tau > -2.89:
            mn, mx, g = -9999, 0.70, (2.5261, 0.61654, -0.37956, -0.060285)
        else:
            mn, mx, g = -16.18, None, (3.2512, 1.6047, 0.049588, 0.0)
    h = g[0] + g[1] * tau + g[2] * tau ** 2 + g[3] * tau ** 3
    if tau < mn:
        return 0.0
    if mx is not None and tau > mx:
        return 1.0
    return float(sps.norm.cdf(h))


_H0 = {1: 'H0: Random walk without drift, a = 0, d = 0', 2: 'H0: Random walk without drift, d = 0',
       3: 'H0: Random walk with drift, d = 0', 4: 'H0: Random walk with or without drift'}


def _cv_table(ip, rows, drift=False):
    ip.txt('\n')
    if drift:
        ip.txt(f'{"":<35}Student\'s t\n')
    else:
        ip.txt(f'{"":<39}Dickey–Fuller\n')
    ip.txt(f'{"":<19}Test      -------- critical value ---------\n')
    ip.txt(f'{"":<14}statistic           1%           5%          10%\n')
    ip.txt('-' * 62 + '\n')
    for lab, st, c1, c5, c10 in rows:
        ip.txt(f' {lab:<7}')
        ip.res(f'{st:>15.3f}{c1:>13.3f}{c5:>13.3f}{c10:>13.3f}\n')
    ip.txt('-' * 62 + '\n')


def _case(p):
    nocons = p.opts.has('noconstant', 4)
    trend = p.opts.has('trend', 2)
    drift = p.opts.has('drift', 2)
    if drift and nocons:
        raise StataError('cannot specify drift if constant is excluded', 198)
    if drift and trend:
        raise StataError('cannot specify drift if time trend is included', 198)
    if nocons and trend:
        raise StataError('cannot choose trend if constant is excluded', 198)
    if nocons:
        return 1
    if drift:
        return 3
    if trend:
        return 4
    return 2


def _reg_table(ip, r, depvar, names, info_vars):
    """Coefficient table (no header) for the auxiliary regression (regress option)."""
    from .design import coef_table, Col
    info = []
    for nm, (tv, ch) in zip(names, info_vars):
        c = Col(name=nm, values=None, term=0, group='', level='', kind='cons' if nm == '_cons' else 'cont', vars=[])
        c.tsvar, c.tschain = tv, ch
        info.append(c)
    est = {'names': names, 'b': r['b'], 'V': r['V'], 'colinfo': info, 'df_r': r['df_r']}
    coef_table(ip, est, 't', depvar=depvar)


# ================================================================ dfuller
@command('dfuller')
def cmd_dfuller(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    toks = split_top(p.body)
    if len(toks) != 1:
        raise StataError('too many variables specified' if len(toks) > 1 else 'varlist required', 103 if toks else 100)
    case = _case(p)
    mask, _ = ts_sample(ip, p, toks[0])
    y, yname = series(ip, toks[0])
    mask &= ~np.isnan(y)
    lags = p.opts.get('lags', 1)
    lags = int(float(lags)) if lags not in (None, '') else 0
    if lags < 0:
        lags = 0
    dy = y - lagged(ip, y, 1)
    cols = [lagged(ip, y, 1)]
    names = [f'L.{yname}']
    tsinfo = [(yname, 'L')]
    for j in range(1, lags + 1):
        cols.append(lagged(ip, dy, j))
        names.append(f'L{"" if j == 1 else j}D.{yname}')
        tsinfo.append((yname, f'L{"" if j == 1 else j}D'))
    tv = d.cols[d.ts['time']].astype(float)
    if case == 4:
        tt = tv - np.nanmin(tv[mask])
        cols.append(tt)
        names.append('_trend')
        tsinfo.append((None, ''))
    r = regress_ts(dy, cols, mask, cons=(case != 1))
    if case != 1:
        names.append('_cons')
        tsinfo.append((None, ''))
    T = r['N']
    Zt = r['b'][0] / r['se'][0]
    ztp = None
    if case == 2:
        ztp = mackinnon_p(Zt, 'c')
    elif case == 4:
        ztp = mackinnon_p(Zt, 'ct')
    if case == 3:
        df = r['df_r']
        ztp = float(sps.t.cdf(Zt, df))
        cv = [float(sps.t.ppf(q, df)) for q in (0.01, 0.05, 0.10)]
    else:
        cv = _interp_cv(_ZT[1 if case == 1 else (2 if case == 2 else 4)], T)
    aug = 'Augmented ' if lags > 0 else ''
    ip.txt(f'\n{aug}Dickey–Fuller test for unit root\n\n')
    ip.txt(f'{"Variable: " + yname:<48}Number of obs  = ')
    ip.res(f'{comma_int(T):>6}\n')
    ip.txt(f'{"":<48}Number of lags = ')
    ip.res(f'{lags:>6}\n\n')
    ip.txt(_H0[case] + '\n')
    _cv_table(ip, [('Z(t)', Zt, *cv)], drift=(case == 3))
    if case == 3:
        ip.txt('p-value for Z(t) = ')
        ip.res(f'{ztp:.4f}')
        ip.txt('.\n')
    elif ztp is not None:
        ip.txt('MacKinnon approximate p-value for Z(t) = ')
        ip.res(f'{ztp:.4f}')
        ip.txt('.\n')
    if p.opts.has('regress', 3):
        ip.txt('\n')
        _reg_table(ip, r, f'D.{yname}', names, tsinfo)
    sc = {'Zt': Zt, 'N': T, 'lags': lags, 'cv1': cv[0], 'cv5': cv[1], 'cv10': cv[2]}
    if ztp is not None:
        sc['p'] = ztp
    ip.set_r(scalars=sc)


# ================================================================ pperron
@command('pperron')
def cmd_pperron(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    toks = split_top(p.body)
    if len(toks) != 1:
        raise StataError('too many variables specified' if len(toks) > 1 else 'varlist required', 103 if toks else 100)
    nocons = p.opts.has('noconstant', 4)
    trend = p.opts.has('trend', 2)
    if nocons and trend:
        raise StataError('cannot choose trend if constant is excluded', 198)
    case = 1 if nocons else (4 if trend else 2)
    mask, _ = ts_sample(ip, p, toks[0])
    y, yname = series(ip, toks[0])
    mask &= ~np.isnan(y)
    cols = [lagged(ip, y, 1)]
    names = [f'L.{yname}']
    tsinfo = [(yname, 'L')]
    tv = d.cols[d.ts['time']].astype(float)
    if trend:
        cols.append(tv - np.nanmin(tv[mask]))
        names.append('_trend')
        tsinfo.append((None, ''))
    r = regress_ts(y, cols, mask, cons=not nocons)
    if not nocons:
        names.append('_cons')
        tsinfo.append((None, ''))
    T = r['N']
    s = r['rmse']
    rho, sig = r['b'][0], r['se'][0]
    lags = p.opts.get('lags', 1)
    lags = int(float(lags)) if lags not in (None, '') else -1
    if lags < 0:
        lags = int(4 * (T / 100) ** (2 / 9))
    u = np.full(d.n, np.nan)
    u[r['sample']] = r['resid']
    gam0 = np.nansum(u * u) / T
    lam = gam0
    for i in range(1, lags + 1):
        gi = np.nansum(u * lagged(ip, u, i)) / T
        lam += 2 * (1 - i / (lags + 1)) * gi
    Zp0 = T * (rho - 1)
    Zt0 = (rho - 1) / sig
    Zp = Zp0 - ((T * sig / s) ** 2) * (lam - gam0) / 2
    Zt = Zt0 * math.sqrt(gam0 / lam) - 0.5 * (lam - gam0) * (T * sig / s) / math.sqrt(lam)
    ztp = mackinnon_p(Zt, 'c') if case == 2 else (mackinnon_p(Zt, 'ct') if case == 4 else None)
    cvp = _interp_cv(_ZP[case], T)
    cvt = _interp_cv(_ZT[case], T)
    ip.txt(f'\n{"Phillips–Perron test for unit root":<48}Number of obs   = ')
    ip.res(f'{comma_int(T):>6}\n')
    ip.txt(f'{"Variable: " + yname:<48}Newey–West lags = ')
    ip.res(f'{lags:>6}\n\n')
    ip.txt(_H0[case] + '\n')
    _cv_table(ip, [('Z(rho)', Zp, *cvp), ('Z(t)', Zt, *cvt)])
    if ztp is not None:
        ip.txt('MacKinnon approximate p-value for Z(t) = ')
        ip.res(f'{ztp:.4f}')
        ip.txt('.\n')
    if p.opts.has('regress', 3):
        ip.txt('\n')
        _reg_table(ip, r, yname, names, tsinfo)
    sc = {'Zt': Zt, 'Zrho': Zp, 'lags': lags, 'N': T}
    if ztp is not None:
        sc['pval'] = ztp
        sc['p'] = ztp
    ip.set_r(scalars=sc)


# ================================================================ kpss (C F Baum, SSC)
def _g3(x):
    """Stata %8.3g display (3 significant digits, no leading zero)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '.'
    s = f'{x:.3g}'
    if 'e' in s:
        m, e = s.split('e')
        s = f'{m}e{e[0]}{int(e[1:]):02d}'
    if s.startswith('0.'):
        s = s[1:]
    elif s.startswith('-0.'):
        s = '-' + s[2:]
    return s


@command('kpss')
def cmd_kpss(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    toks = split_top(p.body)
    if len(toks) != 1:
        raise StataError('varlist required', 100)
    mask, _ = ts_sample(ip, p, toks[0])
    y, yname = series(ip, toks[0])
    mask &= ~np.isnan(y)
    if n_gaps(ip, mask):
        ip.err('sample may not contain gaps\n')
        return
    schwert = 12 if ts_unit(d) in ('monthly', 'generic') else 4
    trend = not p.opts.has('notrend', 5)
    if trend:
        stat, cv = 'trend stationary', (0.216, 0.176, 0.146, 0.119)
    else:
        stat, cv = 'level stationary', (0.739, 0.574, 0.463, 0.347)
    nobs = int(mask.sum())
    ml = p.opts.get('maxlag', 1)
    if ml not in (None, ''):
        maxlag = int(float(ml))
        kmax = f'Maxlag = {maxlag}'
    else:
        maxlag = int(schwert * (nobs / 100) ** 0.25)
        kmax = f'Maxlag = {maxlag} chosen by Schwert criterion'
    qs = p.opts.has('qs')
    kernel = 'Quadratic Spectral kernel' if qs else 'Bartlett kernel'
    nt = 0
    if p.opts.has('auto'):
        nt = int(nobs ** (2 / 25)) if qs else int(nobs ** (2 / 9))
        maxlag = nt
    trd = np.arange(1, d.n + 1, dtype=float)
    r = regress_ts(y, [trd] if trend else [], mask, cons=True)
    e = np.full(d.n, np.nan)
    e[r['sample']] = r['resid']
    s20 = r['rss'] / r['N']
    nobs = r['N']
    psum = np.cumsum(np.nan_to_num(e))
    sumst2 = float(np.mean((psum ** 2)[r['sample']])) / nobs
    kpss0 = sumst2 / s20
    ACv = {}

    def acov(l):
        if l not in ACv:
            ACv[l] = float(np.nansum(e * lagged(ip, e, l)))
        return ACv[l]
    if nt > 0:
        s0 = s1 = es2 = 0.0
        # mat accum dty L(1/nt).dty, noc : cross products over rows where all lags exist
        Le = [lagged(ip, e, l) for l in range(1, nt + 1)]
        ok = ~np.isnan(e)
        for z in Le:
            ok &= ~np.isnan(z)
        for l in range(1, nt + 1):
            a = float(np.sum(e[ok] * Le[l - 1][ok]))
            s0 += 2.0 * a
            s1 += 2.0 * l * a
            es2 += 2.0 * l * l * a
        # mat accum uses raw cross products; long-run quantities as in kpss.ado
        s0 = s0 + s20
        if not qs:
            gamma = 1.1447 * ((s1 / s0) ** 2) ** (1 / 3)
            maxlag = min(nobs, int(gamma * nobs ** (1 / 3)))
        else:
            gamma = 1.3221 * ((es2 / s0) ** 2) ** (1 / 5)
            maxlag = min(nobs, int(gamma * nobs ** (1 / 5)))
        kmax = f'Automatic bandwidth selection (maxlag) = {maxlag}'
    stats_ = {}
    for l in range(1, maxlag + 1):
        s2 = 0.0
        for s in range(1, l + 1):
            if not qs:
                w = 1 - s / (l + 1)
            else:
                x = 6 * math.pi * (s / l) / 5
                w = (25 / (12 * math.pi ** 2 * (s / l) ** 2)) * (math.sin(x) / x - math.cos(x))
            s2 += w * acov(s)
        s2 = 2.0 * s2 / nobs + s20
        stats_[l] = sumst2 / s2
    ip.txt(' \n')
    ip.txt(f'KPSS test for {yname}\n \n{kmax}\nAutocovariances weighted by {kernel}\n \n')
    ip.txt(f'Critical values for H0: {yname} is {stat}\n \n')
    ip.txt(f'10%: {cv[3]}  5% : {cv[2]}  2.5%: {cv[1]}  1% : {cv[0]}\n \n')
    ip.txt('Lag order    Test statistic\n')
    if (nt > 0 and maxlag == 0) or nt == 0:
        ip.res(f'    0       {_g3(kpss0):>8}\n')
    for l in range(1, maxlag + 1):
        if (nt > 0 and maxlag == l) or nt == 0:
            ip.res(f'   {l:>2}       {_g3(stats_[l]):>8}\n')
    sc = {'N': nobs, 'kpss0': kpss0, 'p01': cv[0], 'p025': cv[1], 'p05': cv[2], 'p10': cv[3]}
    if maxlag > 0:
        sc['maxlag'] = maxlag
        for l, v in stats_.items():
            sc[f'kpss{l}'] = v
    ip.set_r(scalars=sc)


# ================================================================ zandrews (C F Baum, SSC)
@command('zandrews')
def cmd_zandrews(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    toks = split_top(p.body)
    if len(toks) != 1:
        raise StataError('varlist required', 100)
    brk = (p.opts.get('break', 1) or 'intercept').strip()
    crits = {'intercept': (-5.34, -4.80, -4.58, 'intercept'), 'trend': (-4.93, -4.42, -4.11, 'trend'),
             'both': (-5.57, -5.08, -4.82, 'both intercept and trend')}
    if brk not in crits:
        raise StataError('Error: break must be intercept, trend, or both.', 198)
    c01, c05, c10, obr = crits[brk]
    lagtype = (p.opts.get('lagmethod', 1) or 'TTest').strip()
    if lagtype not in ('input', 'AIC', 'BIC', 'TTest'):
        raise StataError('Error: lagmethod must be input, AIC, BIC, or TTest.', 198)
    lvl = p.opts.get('level', 3)
    siglevel = float(lvl) if lvl not in (None, '') else 0.10
    if not (0 < siglevel < 0.25):
        raise StataError('Error: Level must be a real number between 0 and 0.25.', 198)
    tr = p.opts.get('trim', 2)
    trim = float(tr) if tr not in (None, '') else 0.15
    if not (0 <= trim < 0.25):
        raise StataError('Error: Trim must be a positive real number greater than 0 and less than 0.25', 198)
    mask, _ = ts_sample(ip, p, toks[0])
    y, var = series(ip, toks[0])
    ok = mask & ~np.isnan(y)
    if ok.any() and np.nanmax(y[ok]) == np.nanmin(y[ok]):
        raise StataError(f'Error: variable {var} has no variance.', 198)
    touse = ok
    if n_gaps(ip, touse):
        ip.err('Sample may not contain gaps\n')
        return
    count = np.cumsum(touse)
    nobs = int(count[-1]) if d.n else 0
    ntrim = int(trim * nobs + 0.49)
    maxl = int(nobs ** 0.25)
    ml = p.opts.get('maxlags', 1)
    if lagtype == 'input':
        if ml in (None, ''):
            raise StataError('Error: If lagmethod=input you must provide a value for maxlags.', 198)
        maxl = int(float(ml))
        bestlag = maxl
    lagout = touse & (count > maxl + 1)
    dy = np.where(touse, y - lagged(ip, y, 1), np.nan)
    trend = np.where(touse, np.arange(1, d.n + 1, dtype=float), np.nan)
    Ly = lagged(ip, y, 1)
    Ldy = {j: lagged(ip, dy, j) for j in range(1, max(maxl, 1) + 1)}
    if lagtype != 'input':
        icmult = 2.0 / nobs if lagtype == 'AIC' else math.log(nobs) / nobs
        r0 = regress_ts(dy, [trend, Ly], lagout)
        icmin = math.log(r0['rss'] / r0['N']) + icmult * (r0['df_m'] + 1)
        bestlag = 0
        for lag in range(maxl, 0, -1):
            r1 = regress_ts(dy, [trend, Ly] + [Ldy[j] for j in range(1, lag + 1)], lagout)
            if lagtype == 'TTest':
                tee = abs(r1['b'][1 + lag] / r1['se'][1 + lag])
                tsig = float(sps.t.sf(tee, r1['df_m']))
                if tsig <= siglevel:
                    bestlag = lag
                    break
            else:
                ic = math.log(r1['rss'] / r1['N']) + icmult * (r1['df_m'] + 1)
                if ic < icmin:
                    icmin = ic
                    bestlag = lag
    if bestlag > max(maxl, 1):
        for j in range(max(maxl, 1) + 1, bestlag + 1):
            Ldy[j] = lagged(ip, dy, j)
    lagout = touse & (count > bestlag)
    rows = np.arange(1, d.n + 1)
    mint = int(rows[touse].min()) - 1
    samp = lagout & ~((count <= ntrim + bestlag) | (count > nobs - ntrim))
    tstart = int(count[samp].min()) + mint
    tend = int(count[samp].max()) + mint
    minbreak, minent = 1.0e10, None
    breakstats = np.full(d.n, np.nan)
    base = [Ly, trend] + [Ldy[j] for j in range(1, bestlag + 1)]
    enn = 0
    for time in range(tstart, tend + 1):
        cpos = count + mint
        bint = np.where(touse, (cpos >= time).astype(float), np.nan)
        btr = np.where(touse, np.where(cpos + 1 - time > 0, cpos + 1 - time, 0.0), np.nan)
        cols = list(base)
        if brk != 'intercept':
            cols.append(btr)
        if brk != 'trend':
            cols.append(bint)
        r = regress_ts(dy, cols, lagout)
        tee = r['b'][0] / r['se'][0]
        enn = r['N']
        breakstats[time - 1] = tee
        if tee < minbreak:
            minbreak, minent = tee, time
    tvar = d.ts['time']
    from .fmt import fmt_any
    tfmt = d.formats.get(tvar, '%9.0g')
    minb = fmt_any(float(d.cols[tvar][minent - 1]), tfmt).strip()
    gen = p.opts.get('generate', 1)
    if gen:
        d.add_var(gen.strip(), breakstats, 'float')
    if p.opts.has('graph'):
        _za_graph(ip, breakstats, var, tvar, tstart, tend, minb, touse)
    ip.txt('\nZivot-Andrews unit root test for ')
    ip.res(f' {var}\n')
    ip.txt('\nAllowing for break in ')
    ip.res(f'{obr}\n')
    ip.txt('\nLag selection via ')
    ip.res(lagtype)
    ip.txt(f': lags of D.{var} included = {bestlag}\n')
    ip.txt(f'\nMinimum t-statistic {minbreak:6.3f} at {minb}  (obs {minent})\n')
    ip.txt(f'\nCritical values: 1%: {c01:5.2f} 5%: {c05:5.2f} 10%: {c10:5.2f}\n')
    ip.set_r(scalars={'tmin': minbreak, 'crit01': c01, 'crit05': c05, 'crit10': c10, 'tminobs': minent},
             macros={'breaktype': brk, 'cmd': 'zandrews', 'var': var, 'nobs': str(enn), 'trim': g_fmt(trim).strip(),
                     'lagtype': lagtype, 'bestlag': str(bestlag)})


def _za_graph(ip, stats, var, tvar, tstart, tend, minb, touse):
    from .graphs import Figure, finish_graph, stata_color
    from .fmt import fmt_any
    d = ip.data
    t = d.cols[tvar].astype(float)
    fig = Figure(ip)
    ok = ~np.isnan(stats)
    fig.ax.plot(t[ok], stats[ok], color=stata_color('navy'), lw=1.4)
    tf = d.formats.get(tvar, '%9.0g')
    minp = fmt_any(float(t[tstart - 1]), tf).strip()
    maxp = fmt_any(float(t[tend - 1]), tf).strip()
    _tick_dates(fig.ax, t[ok], tf)
    fig.set_default_titles(title=f'Zivot-Andrews test for {var}, {minp}-{maxp}', ytitle='Breakpoint t-statistics',
                           xtitle=tvar)
    fig.titles['subtitle'] = f'Min breakpoint at {minb}'
    finish_graph(ip, fig, None)


def _tick_dates(ax, t, tf):
    from .dates import is_tfmt, fmt_t
    if is_tfmt(tf) and t.size:
        ticks = ax.get_xticks()
        ticks = [x for x in ticks if t.min() <= x <= t.max()]
        ax.set_xticks(ticks)
        ax.set_xticklabels([fmt_t(x, tf).strip() for x in ticks])


# ================================================================ ac / pac / corrgram / wntestq
def _acf(ip, x, mask, lags):
    """AC as in ac.ado: sum(xm*L.xm)/R0 over the sample (gap-aware)."""
    xs = x[mask]
    n = xs.size
    xm = np.where(mask, x - xs.mean(), np.nan)
    R0 = float(np.sum((xs - xs.mean()) ** 2))
    ac = np.full(lags, np.nan)
    pse = np.full(lags, np.nan)
    for i in range(1, lags + 1):
        ac[i - 1] = float(np.nansum(xm * lagged(ip, xm, i))) / R0
    pse[0] = 1 / math.sqrt(n)
    for i in range(2, lags + 1):
        pse[i - 1] = math.sqrt((1 + 2 * np.sum(ac[:i - 1] ** 2)) / n)
    return ac, pse, n, xm


def _pacf(ip, x, mask, lags, yw=False):
    xs = x[mask]
    n = xs.size
    xm = np.where(mask, x - xs.mean(), np.nan)
    pac = np.full(lags, np.nan)
    if yw:
        ac, _, _, _ = _acf(ip, x, mask, lags)
        P = np.zeros((lags, lags))
        P[0, 0] = ac[0]
        for s in range(2, lags + 1):
            sumsmj = sum(P[s - 2, j - 1] * ac[s - j - 1] for j in range(1, s))
            sumj = sum(P[s - 2, j - 1] * ac[j - 1] for j in range(1, s))
            P[s - 1, s - 1] = (ac[s - 1] - sumsmj) / (1 - sumj)
            for j in range(1, s):
                P[s - 1, j - 1] = P[s - 2, j - 1] - P[s - 1, s - 1] * P[s - 2, s - j - 1]
        return np.diag(P).copy(), n
    Lx = {}
    for i in range(1, lags + 1):
        Lx[i] = lagged(ip, xm, i)
        r = regress_ts(xm, [Lx[j] for j in range(1, i + 1)], mask & ~np.isnan(xm))
        if r['N'] > r['k'] and r['se'][i - 1] != 0 and np.isfinite(r['se'][i - 1]):
            pac[i - 1] = r['b'][i - 1]
    return pac, n


def _default_lags(n):
    return max(1, min(int(n / 2) - 2, 40))


def _note_gaps(ip, mask, x):
    g = n_gaps(ip, mask & ~np.isnan(x))
    if g > 1:
        ip.txt(f'(note: time series has {g} gaps)\n')
    elif g == 1:
        ip.txt('(note: time series has 1 gap)\n')


def _mkstring(c):
    if c is None or not np.isfinite(c):
        return ''
    if c >= 0:
        vb, ve = 9, int(8 * c) + 9
    else:
        ve, vb = 9, int(8 * c) + 9
    out = []
    for k in range(1, 18):
        ch = ' '
        if vb <= k <= ve:
            ch = '-'
        if k == 9:
            ch = '|' if (vb == 9 or ve == 9) else ch
        out.append(ch)
    return ''.join(out)


def _col(s, col):
    """Pad string s so that the next text starts at 1-based column col."""
    return s + ' ' * max(1 if s and not s.endswith(' ') else 0, col - 1 - len(s))


@command('corrgram')
def cmd_corrgram(ip, cmd, rest, by):
    p = parse_command(rest)
    toks = split_top(p.body)
    mask, _ = ts_sample(ip, p, toks[0])
    x, xname = series(ip, toks[0])
    mask &= ~np.isnan(x)
    n = int(mask.sum())
    lg = p.opts.get('lags', 1)
    if lg in (None, ''):
        lags = _default_lags(n)
    else:
        lags = int(float(lg))
        if lags >= n:
            raise StataError(f'lags() too large; must be less than {n}', 498)
        if lags <= 0:
            raise StataError('lags() must be greater than zero', 498)
    _note_gaps(ip, mask, x)
    ac, _, _, _ = _acf(ip, x, mask, lags)
    plags = min(lags, int(n / 2) - 2)
    pac = np.full(lags, np.nan)
    if plags > 0:
        pp, _ = _pacf(ip, x, mask, plags, yw=p.opts.has('yw'))
        pac[:plags] = pp
    q = n * (n + 2) * np.cumsum(ac ** 2 / (n - np.arange(1, lags + 1)))
    noplot = p.opts.has('noplot', 4)
    if not noplot:
        ip.txt('\n' + ' ' * 42 + '-1       0       1 -1       0       1\n')
        ip.txt(' LAG       AC       PAC      Q     Prob>Q  [Autocorrelation]  [Partial Autocor]\n')
        ip.txt('-' * 79 + '\n')
    else:
        ip.txt('\n LAG       AC       PAC      Q     Prob>Q\n')
        ip.txt('-' * 41 + '\n')
    for i in range(1, lags + 1):
        a, pa, qq = ac[i - 1], pac[i - 1], q[i - 1]
        pr = float(sps.chi2.sf(qq, i)) if np.isfinite(qq) else np.nan
        lag_s = g_fmt(i, 6).strip()
        ip.txt(f'{lag_s:<8}')
        line = (f'{_f74(a):>7}' + '  ' + f'{_f74(pa):>7}' + '  ' + f'{g_fmt(qq, 7).strip() if np.isfinite(qq) else ".":>7}'
                + '  ' + (f'{pr:6.4f}' if np.isfinite(pr) else f'{".":>6}'))
        if not noplot:
            line += '  ' + f'{_mkstring(a):<17}' + '  ' + _mkstring(pa)
        ip.res(line.rstrip() + '\n')
    AC = ac[~np.isnan(ac)]
    sc = {'lags': lags}
    for i in range(1, min(lags, 10) + 1):
        sc[f'ac{i}'] = ac[i - 1]
        sc[f'pac{i}'] = pac[i - 1]
        sc[f'q{i}'] = q[i - 1]
    from .expr import Matrix

    def first_run(v):
        k = 0
        while k < len(v) and np.isfinite(v[k]):
            k += 1
        return v[:k]
    mats = {}
    for nm, v in (('AC', ac), ('PAC', pac), ('Q', q)):
        fr = first_run(v)
        if fr.size:
            mats[nm] = Matrix(fr.reshape(1, -1), ['r1'], [f'lag{j + 1}' for j in range(fr.size)])
    ip.set_r(scalars=sc, matrices=mats)


def _f74(v):
    return f'{v:7.4f}' if v is not None and np.isfinite(v) else '.'


@command('wntestq')
def cmd_wntestq(ip, cmd, rest, by):
    p = parse_command(rest)
    toks = split_top(p.body)
    mask, _ = ts_sample(ip, p, toks[0])
    x, xname = series(ip, toks[0])
    mask &= ~np.isnan(x)
    n = int(mask.sum())
    lg = p.opts.get('lags', 1)
    lags = _default_lags(n) if lg in (None, '') else int(float(lg))
    if lags >= n:
        raise StataError(f'lags() too large; must be less than {n}', 498)
    if lags <= 0:
        raise StataError('lags() must be greater than zero', 498)
    _note_gaps(ip, mask, x)
    ac, _, _, _ = _acf(ip, x, mask, lags)
    ok = ~np.isnan(ac)
    df = int(ok.sum())
    stat = float(np.sum((ac[ok] ** 2) / (n - np.arange(1, lags + 1)[ok]))) * n * (n + 2)
    pv = float(sps.chi2.sf(stat, df))
    ip.txt('\nPortmanteau test for white noise\n')
    ip.txt('-' * 39 + '\n')
    ip.txt(' Portmanteau (Q) statistic = ')
    ip.res(f'{stat:10.4f}\n')
    ip.txt(' Prob > chi2(')
    ip.res(str(df))
    ip.txt(')')
    ip.txt(' ' * max(1, 27 - (14 + len(str(df)))) + '= ')
    ip.res(f'{pv:10.4f}\n')
    ip.set_r(scalars={'stat': stat, 'p': pv, 'df': df})


def _ac_graph(ip, p, x, xname, mask, partial):
    from .graphs import Figure, finish_graph, stata_color, apply_global_options
    n = int(mask.sum())
    lg = p.opts.get('lags', 1)
    if partial:
        lags = min(int(n / 2) - 2, 40) if lg in (None, '') else int(float(lg))
        if lags > int(n / 2) - 2:
            raise StataError(f'lags() too large; must be less than {int(n / 2) - 1}', 498)
    else:
        lags = _default_lags(n) if lg in (None, '') else int(float(lg))
        if lags >= n:
            raise StataError(f'lags() too large; must be less than {n}', 498)
    if lags <= 0:
        raise StataError('lags() must be greater than zero', 498)
    lv = p.opts.get('level', 1)
    level = float(lv) if lv not in (None, '') else 95.0
    z = sps.norm.ppf((100 + level) / 200)
    _note_gaps(ip, mask, x)
    if partial:
        vals, _ = _pacf(ip, x, mask, lags, yw=p.opts.has('yw'))
        band = np.full(lags, z / math.sqrt(n))
        ytitle = f'Partial autocorrelations of {xname}'
        note = f'{g_fmt(level).strip()}% Confidence bands [se = 1/sqrt(n)]'
    else:
        vals, pse, _, _ = _acf(ip, x, mask, lags)
        band = z * pse
        ytitle = f'Autocorrelations of {xname}'
        note = f"Bartlett's formula for MA(q) {g_fmt(level).strip()}% confidence bands"
    gen = p.opts.get('generate', 3)
    if gen:
        d = ip.data
        col = np.full(d.n, np.nan)
        col[:lags] = vals
        d.add_var(gen.strip(), col, 'double')
        d.varlab[gen.strip()] = ytitle
    if p.opts.has('nograph', 4):
        return
    lagsx = np.arange(1, lags + 1)
    fig = Figure(ip)
    ax = fig.ax
    ax.fill_between(lagsx, -band, band, color=(0.82, 0.84, 0.86), lw=0, step=None)
    ax.vlines(lagsx, 0, vals, color=stata_color('navy'), lw=1.2)
    ax.scatter(lagsx, vals, s=14, color=stata_color('navy'), zorder=3)
    ax.axhline(0, color=(0.6, 0.6, 0.6), lw=0.6)
    fig.set_default_titles(xtitle='Lag', ytitle=ytitle)
    fig.titles['note'] = note
    apply_global_options(fig, p.opts)
    finish_graph(ip, fig, p.opts)


@command('ac')
def cmd_ac(ip, cmd, rest, by):
    p = parse_command(rest)
    toks = split_top(p.body)
    mask, _ = ts_sample(ip, p, toks[0])
    x, xname = series(ip, toks[0])
    mask &= ~np.isnan(x)
    _ac_graph(ip, p, x, xname, mask, partial=False)


@command('pac')
def cmd_pac(ip, cmd, rest, by):
    p = parse_command(rest)
    toks = split_top(p.body)
    mask, _ = ts_sample(ip, p, toks[0])
    x, xname = series(ip, toks[0])
    mask &= ~np.isnan(x)
    _ac_graph(ip, p, x, xname, mask, partial=True)


# ================================================================ tsfilter hp
@command('tsfilter')
def cmd_tsfilter(ip, cmd, rest, by):
    rest = rest.strip()
    parts = rest.split(None, 1)
    if not parts:
        raise StataError('filter type required: hp', 198)
    flt = parts[0].lower()
    if flt in ('bk', 'cf', 'bw'):
        raise StataError(f'tsfilter {flt} is not yet available in Stata Web; use tsfilter hp', 199)
    if flt != 'hp':
        raise StataError('filter must be bk, bw, cf, or hp', 198)
    p = parse_command(parts[1] if len(parts) > 1 else '')
    body = p.body
    if '=' not in body:
        raise StataError('invalid syntax: tsfilter hp newvar = varname', 198)
    lhs, rhs = body.split('=', 1)
    lt = lhs.split()
    if lt and lt[0] in ('float', 'double'):
        lt = lt[1:]
    d = ip.data
    need_ts(d)
    rvars = d.expand_varlist(rhs.split())
    if len(lt) == 1 and lt[0].endswith('*'):
        newvars = [f'{lt[0][:-1]}{i + 1}' for i in range(len(rvars))]
    else:
        newvars = lt
    if len(newvars) != len(rvars):
        raise StataError('number of new variables must equal number of variables to filter', 198)
    tr = p.opts.get('trend', 2)
    trends = None
    if tr:
        tt = tr.split()
        trends = [f'{tt[0][:-1]}{i + 1}' for i in range(len(rvars))] if len(tt) == 1 and tt[0].endswith('*') else tt
        if len(trends) != len(rvars):
            raise StataError('number of trend variables must equal number of variables to filter', 198)
    sm = p.opts.get('smooth', 2)
    unit = ts_unit(d)
    if sm not in (None, ''):
        lam = float(sm)
    else:
        lam = {'yearly': 6.25, 'monthly': 129600.0, 'weekly': 1600.0 * 12 ** 4, 'daily': 1600.0 * (365.0 / 4) ** 4,
               'halfyearly': 100.0}.get(unit, 1600.0)
    if lam <= 0:
        raise StataError('smooth() must be greater than 0', 198)
    for nv in newvars + (trends or []):
        if d.has(nv):
            raise StataError(f'variable {nv} already defined', 110)
    keys = [x for x in (d.ts.get('panel'), d.ts.get('time')) if x]
    d.sort(keys)
    base = ip.mark(p, d)
    pv = d.cols[d.ts['panel']].astype(float) if d.ts.get('panel') else np.zeros(d.n)
    for vi, v in enumerate(rvars):
        x = d.cols[v].astype(float)
        cyc = np.full(d.n, np.nan)
        trd = np.full(d.n, np.nan)
        m = base & ~np.isnan(x) & ~np.isnan(pv)
        for g in np.unique(pv[m]):
            idx = np.flatnonzero(m & (pv == g))
            if idx.size <= 2:
                raise StataError('insufficient observations for the specified sample', 498)
            if n_gaps(ip, np.isin(np.arange(d.n), idx)):
                raise StataError('sample may not contain gaps', 498)
            tau = hp_trend(x[idx], lam)
            trd[idx] = tau
            cyc[idx] = x[idx] - tau
        d.add_var(newvars[vi], cyc, 'double')
        d.varlab[newvars[vi]] = f'{v} cyclical component from hp filter'
        if trends:
            d.add_var(trends[vi], trd, 'double')
            d.varlab[trends[vi]] = f'{v} trend component from hp filter'
    ip.set_r(scalars={'smooth': lam}, macros={'unit': unit, 'method': 'Hodrick-Prescott'})


def hp_trend(y, lam):
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    n = y.size
    e = np.ones(n)
    D = sparse.diags([e[:-2], -2 * e[:-2], e[:-2]], [0, 1, 2], shape=(n - 2, n))
    A = sparse.eye(n) + lam * (D.T @ D)
    return spsolve(A.tocsc(), y)
