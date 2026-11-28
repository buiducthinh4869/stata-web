"""Time-series regression: newey, prais, and estat dwatson/durbinalt/bgodfrey/archlm/imtest after regress.
Ported from the official ado files (newey.ado, prais.ado, _crcar1.ado, dwstat.ado, bgodfrey.ado, durbina.ado,
archlm.ado, imtest.ado)."""
import math

import numpy as np
from scipy import stats as sps

from .design import coef_table, right_lines
from .errors import StataError
from .estim import (prepare_linear, parse_vce, ols_core, wald_test, _anova_header, _predict_linear, _store_rhs,
                    _ols_ll, ESTAT_EXT)
from .fmt import g_fmt, comma_int, fmt_num
from .interp import command, alias
from .syntax import parse_command, split_top, parse_numlist
from .tsuni import need_ts, n_gaps

alias('prais', 'prai')


def _ts_lag(ip, x, k):
    return ip._lag(ip.data, x, k)


def _full(mask, vals):
    out = np.full(mask.size, np.nan)
    out[mask] = vals
    return out


def _check_onepanel(ip, mask):
    d = ip.data
    pv = d.ts.get('panel')
    if pv:
        pp = d.cols[pv].astype(float)
        if len(np.unique(pp[mask])) > 1:
            raise StataError('sample may not include multiple panels', 459)


# ================================================================ newey
@command('newey')
def cmd_newey(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    toks = split_top(p.body)
    if not toks:
        if ip.est and ip.est.get('cmd') == 'newey':
            return ip.est['replay'](ip)
        raise StataError('varlist required', 100)
    lag = p.opts.get('lag', 3)
    if lag in (None, ''):
        raise StataError('option lag() required', 198)
    lag = int(float(lag))
    if lag < 0:
        raise StataError(f'lag({lag}) invalid', 198)
    need_ts(d)
    nocons = p.opts.has('noconstant', 4)
    if nocons and len(toks) == 1:
        raise StataError('may not specify noconstant without regressors', 198)
    yv, y, cols, mask, w, wtype = prepare_linear(ip, p, d, toks[1:], toks[0])
    if w is not None and wtype != 'aweight':
        raise StataError(f'{wtype}s not allowed', 101)
    tv = d.cols[d.ts['time']].astype(float)
    mask &= ~np.isnan(tv)
    # regular spacing check (Checkt)
    tt = np.sort(tv[mask])
    delta = d.ts.get('delta', 1) or 1
    if tt.size > 1:
        dd = np.diff(tt) / delta
        if dd.min() != dd.max() and lag > 0 and not p.opts.has('force'):
            raise StataError(f'{d.ts["time"]} is not regularly spaced', 198)
        if dd.min() == dd.max() and dd.mean() > 1:
            raise StataError(f'{d.ts["time"]} is regularly spaced, but does not have intervals of 1', 198)
    f = ols_core(ip, d, y, cols, mask, w, wtype, 'ols', None, nocons)
    f.depvar = yv
    X = f.Xfull[:, f.active]
    n, k = X.shape
    wn = f.wn
    e = f.e
    # full-length arrays to apply time-series lags
    Xf = [_full(mask, X[:, j]) for j in range(k)]
    ef = _full(mask, e)
    wf = _full(mask, wn if wtype == 'aweight' else np.ones(n))
    ow = _full(mask, wn) if wtype == 'aweight' else np.ones(d.n)
    Om = None
    for j in range(lag + 1):
        Le = _ts_lag(ip, ef, j)
        Lw = _ts_lag(ip, wf, j)
        LX = np.column_stack([_ts_lag(ip, c, j) for c in Xf])
        Xc = np.column_stack(Xf)
        vt = ef * Le * Lw * ow
        okr = mask & ~np.isnan(vt) & ~np.isnan(LX).any(axis=1)
        tt_ = (LX[okr] * vt[okr][:, None]).T @ Xc[okr]
        tt_ = (tt_ + tt_.T) * (1 - j / (1 + lag))
        Om = tt_ * 0.5 if j == 0 else Om + tt_
    XtX = X.T @ (X * wn[:, None])
    XtXi = np.linalg.pinv(XtX)
    Vact = XtXi @ Om @ XtXi * (f.N / f.df_r)
    Vact = (Vact + Vact.T) / 2
    from .estim import full_bV
    b, V = full_bV(len(f.names), f.active, f.b[f.active], Vact)
    f.V = V
    slopes = [j for j in f.active if f.info[j].kind != 'cons']
    F, df_m, pF = wald_test(b, V, slopes, f.df_r) if slopes else (np.nan, 0, np.nan)
    est = {'names': f.names, 'b': b, 'V': V, 'colinfo': f.info, 'df_r': f.df_r, 'depvar': yv, 'sample': mask,
           'cmd': 'newey', 'stat': 't', 'vce': 'newey', 'fit': f,
           'scalars': {'N': f.N, 'df_m': df_m, 'df_r': f.df_r, 'F': F, 'lag': lag, 'rank': k},
           'macros': {'cmd': 'newey', 'depvar': yv, 'vcetype': 'Newey–West',
                      'title': 'Regression with Newey–West standard errors'}}

    def replay(ip_):
        ip_.txt('\n')
        right_lines(ip_, ['Regression with Newey–West standard errors', f'Maximum lag = {lag}'],
                    [('Number of obs', comma_int(f.N)), (f'F({df_m}, {f.df_r})', f'{F:.2f}' if F == F else '.'),
                     ('Prob > F', f'{pF:.4f}' if pF == pF else '.')])
        ip_.txt('\n')
        coef_table(ip_, est, 't', robust='Newey–West')
    est['replay'] = replay
    est['predict'] = lambda ip_, kind, m_: _predict_linear(ip_, est, kind)
    _store_rhs(est, toks[1:], cols, d)
    ip.set_estimation(est)
    replay(ip)


# ================================================================ prais
def _rho(ip, res, kind, k, n):
    """_crcar1: estimate rho from residuals (full-length array, nan outside sample)."""
    L = _ts_lag(ip, res, 1)
    if kind in ('regress', ''):
        v = res * L
        ok = ~np.isnan(v)
        return float(np.sum(v[ok]) / np.sum(L[ok] ** 2))
    if kind == 'freg':
        F = _ts_lag(ip, res, -1)
        v = res * F
        ok = ~np.isnan(v)
        return float(np.sum(v[ok]) / np.sum(F[ok] ** 2))
    ok_e = ~np.isnan(res)
    sse = float(np.sum(res[ok_e] ** 2))
    sse_n = int(ok_e.sum())
    if kind in ('dw', 'nagar'):
        dv = (res - L) ** 2
        dwv = float(np.nansum(dv)) / sse
        if kind == 'dw':
            return 1 - dwv / 2
        return (sse_n ** 2 * (1 - dwv / 2) + k ** 2) / (sse_n ** 2 - k ** 2)
    v = res * L
    okv = ~np.isnan(v)
    cov = float(np.sum(v[okv]))
    if kind == 'tscorr':
        return cov / sse
    if kind == 'theil':
        return ((sse_n - k) * cov) / (int(okv.sum()) * sse)
    raise StataError(f'Invalid option for rho(rho_method):  {kind}', 198)


def _dw(ip, res):
    L = _ts_lag(ip, res, 1)
    num = float(np.nansum((res - L) ** 2))
    den = float(np.nansum(res ** 2))
    return num / den


@command('prais')
def cmd_prais(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    toks = split_top(p.body)
    if not toks:
        if ip.est and ip.est.get('cmd') == 'prais':
            return ip.est['replay'](ip)
        raise StataError('varlist required', 100)
    need_ts(d)
    corc = p.opts.has('corc')
    nocons = p.opts.has('noconstant', 4)
    twostep = p.opts.has('twostep', 3)
    ssesearch = p.opts.has('ssesearch', 3)
    if twostep and ssesearch:
        raise StataError('options twostep and ssesearch are mutually exclusive', 198)
    if ssesearch:
        raise StataError('ssesearch is not available in Stata Web; use the default iterated method', 199)
    it_max = p.opts.get('iterate', 2)
    it_max = int(float(it_max)) if it_max not in (None, '') else 100
    if twostep:
        it_max = 1
    tol = p.opts.get('tolerance', 3)
    tol = float(tol) if tol not in (None, '') else 1e-6
    rhotype = (p.opts.get('rhotype', 3) or 'regress').strip().lower()
    if rhotype not in ('regress', 'freg', 'tscorr', 'dw', 'theil', 'nagar'):
        raise StataError(f'Invalid option for rho(rho_method):  {rhotype}', 198)
    nolog = p.opts.has('nolog') or ip.quiet
    vce, clv = parse_vce(p.opts)
    yv, y, cols, mask, w, wtype = prepare_linear(ip, p, d, toks[1:], toks[0])
    if w is not None:
        raise StataError('weights not allowed', 101)
    tv = d.cols[d.ts['time']].astype(float)
    mask &= ~np.isnan(tv)
    if d.ts.get('panel'):
        mask &= ~np.isnan(d.cols[d.ts['panel']].astype(float))
    gaps = n_gaps(ip, mask) if not d.ts.get('panel') else 0
    if gaps > 0:
        ip.txt('(note: computations for rho restarted at each gap)\n')
    # initial OLS
    f0 = ols_core(ip, d, y, cols, mask, None, None, 'ols', None, nocons)
    act = f0.active
    Xs = f0.Xfull[:, act]
    k = Xs.shape[1]
    yf = y.copy()
    Xf = [_full(mask, Xs[:, j]) for j in range(k)]
    b = f0.b[act]
    rho, lastrho = 0.0, 2.0
    it = 0
    dw0 = None
    log = []
    touse_f = mask.astype(float)
    Ltouse = _ts_lag(ip, touse_f, 1)
    Lyf = _ts_lag(ip, np.where(mask, yf, np.nan), 1)
    LXf = [_ts_lag(ip, c, 1) for c in Xf]
    first = mask & ~(Ltouse == 1)
    inner = mask & (Ltouse == 1)
    fit = None

    def transform(rh):
        yt = np.full(d.n, np.nan)
        yt[inner] = yf[inner] - rh * Lyf[inner]
        Xt = []
        for c, Lc in zip(Xf, LXf):
            z = np.full(d.n, np.nan)
            z[inner] = c[inner] - rh * Lc[inner]
            Xt.append(z)
        if not corc:
            s = math.sqrt(1 - rh ** 2) if abs(rh) < 1 else np.nan
            yt[first] = yf[first] * s
            for c, z in zip(Xf, Xt):
                z[first] = c[first] * s
        return yt, Xt

    while it < it_max and abs(rho - lastrho) > tol:
        lastrho = rho
        res = np.full(d.n, np.nan)
        res[mask] = yf[mask] - np.column_stack([c[mask] for c in Xf]) @ b
        rho = _rho(ip, res, rhotype, k, int(mask.sum()))
        if it == 0:
            dw0 = _dw(ip, res)
            log.append(0.0)
        yt, Xt = transform(rho)
        m2 = mask & ~np.isnan(yt)
        Xpre = np.column_stack([z[m2] for z in Xt])
        fit = ols_core(ip, d, y, cols, m2, None, None, vce if vce in ('ols', 'robust') else 'ols', None,
                       nocons, Xpre=_pad_omitted(Xpre, f0, act), ypre=yt[m2])
        b = fit.b[act]
        it += 1
        log.append(rho)
    if fit is None:
        raise StataError('no observations', 2000)
    # transformed Durbin-Watson
    yt, Xt = transform(rho)
    rt = yt - np.column_stack([np.nan_to_num(z) for z in Xt]) @ b
    rt[np.isnan(yt)] = np.nan
    dw = _dw(ip, rt)
    fit.depvar = yv
    f = fit
    N, df_r = f.N, f.df_r
    r2 = f.mss / f.tss if f.tss > 0 else np.nan
    r2a = 1 - (1 - r2) * (N - (0 if nocons else 1)) / df_r if df_r > 0 else np.nan
    rmse = math.sqrt(f.s2)
    slopes = [j for j in f.active if f.info[j].kind != 'cons']
    if f.vce == 'ols':
        df_m = f.df_m
        F = (f.mss / df_m) / (f.rss / df_r) if df_m > 0 else np.nan
        pF = sps.f.sf(F, df_m, df_r) if F == F else np.nan
    else:
        F, df_m, pF = wald_test(f.b, f.V, slopes, df_r)
    method = 'twostep' if twostep else 'iterated'
    conv = not (it == it_max and not twostep and abs(rho - lastrho) > tol)
    title = 'Cochrane–Orcutt AR(1) regression' if corc else 'Prais–Winsten AR(1) regression'
    est = {'names': f.names, 'b': f.b, 'V': f.V, 'colinfo': f.info, 'df_r': df_r, 'depvar': yv, 'sample': mask,
           'cmd': 'prais', 'stat': 't', 'vce': f.vce, 'fit': f,
           'scalars': {'N': N, 'df_m': df_m, 'df_r': df_r, 'F': F, 'r2': r2, 'r2_a': r2a, 'rmse': rmse,
                       'mss': f.mss, 'rss': f.rss, 'rho': rho, 'dw': dw, 'dw_0': dw0, 'ic': it, 'max_ic': it_max,
                       'tol': tol, 'N_gaps': gaps, 'll': _ols_ll(f), 'rank': f.k},
           'macros': {'cmd': 'prais', 'depvar': yv, 'title': 'Prais-Winsten AR(1) regression',
                      'tranmeth': 'corc' if corc else 'prais', 'method': method, 'rhotype': rhotype,
                      'vce': f.vce, 'vcetype': 'Semirobust' if f.vce == 'robust' else ''}}
    nodw = p.opts.has('nodw', 3)

    def replay(ip_, show_log=False):
        if show_log and not nolog:
            ip_.txt('\n')
            for i, r in enumerate(log):
                ip_.txt(f'Iteration {i}:  rho = ')
                ip_.res(f'{r:6.4f}\n')
        ip_.txt(f'\n{title} -- {method} estimates\n\n')
        if f.vce == 'ols':
            _anova_header(ip_, f, F, pF, r2, r2a, rmse)
        else:
            right_lines(ip_, [''], [('Number of obs', comma_int(N)), (f'F({df_m}, {df_r})', f'{F:.2f}'),
                                    ('Prob > F', f'{pF:.4f}'), ('R-squared', f'{r2:.4f}'),
                                    ('Root MSE', g_fmt(rmse, 9))])
            ip_.txt('\n')
        coef_table(ip_, est, 't', robust='Semirobust' if f.vce == 'robust' else None, footer=False)
        ip_.txt(f'{"rho":>12} |')
        ip_.res(f' {g_fmt(rho, 9):>10}\n')
        ip_.txt('-' * 13 + '+' + '-' * 64 + '\n')
        if not nodw:
            ip_.txt('Durbin–Watson statistic (original)    = ')
            ip_.res(f'{dw0:8.6f}\n')
            ip_.txt('Durbin–Watson statistic (transformed) = ')
            ip_.res(f'{dw:8.6f}\n')
    est['replay'] = lambda ip_: replay(ip_, False)
    est['predict'] = lambda ip_, kind, m_: _predict_linear(ip_, est, kind)
    _store_rhs(est, toks[1:], cols, d)
    ip.set_estimation(est)
    replay(ip, True)
    if not conv:
        raise StataError('convergence not achieved', 430)


def _pad_omitted(Xpre, f0, act):
    """Rebuild a matrix with all design columns (omitted/base ones zero) in the order of f0.info."""
    full = np.zeros((Xpre.shape[0], len(f0.info)))
    for j, a in enumerate(act):
        full[:, a] = Xpre[:, j]
    return full


# ================================================================ estat after regress
def _after_regress(ip, allow=('regress',)):
    est = ip.est
    if est is None or est.get('cmd') not in allow:
        raise StataError('This command only works after regress.', 301)
    d = ip.data
    need_ts(d)
    f = est['fit']
    mask = f.mask.copy()
    keys = [x for x in (d.ts.get('panel'), d.ts.get('time')) if x]
    tv = d.cols[d.ts['time']].astype(float)
    mask &= ~np.isnan(tv)
    _check_onepanel(ip, mask)
    res = np.full(d.n, np.nan)
    res[f.mask] = f.e
    res[~mask] = np.nan
    X = f.Xfull[:, f.active]
    return est, f, mask, res, X


def _ols_lm(y, X, cons):
    """OLS with collinear columns dropped; returns N, R2, rss, df_m, df_r, b, V, cols kept."""
    from .design import mark_collinear
    keep = list(range(X.shape[1]))
    if X.shape[1]:
        om = mark_collinear(X, None)
        keep = [j for j, o in enumerate(om) if not o]
    Xk = X[:, keep]
    if cons:
        Xk = np.column_stack([Xk, np.ones(len(y))])
    b, *_ = np.linalg.lstsq(Xk, y, rcond=None)
    e = y - Xk @ b
    rss = float(e @ e)
    tss = float(np.sum((y - y.mean()) ** 2)) if cons else float(y @ y)
    n, kk = Xk.shape
    df_r = n - kk
    XtXi = np.linalg.pinv(Xk.T @ Xk)
    V = XtXi * rss / df_r
    return {'N': n, 'r2': 1 - rss / tss, 'rss': rss, 'df_m': kk - (1 if cons else 0), 'df_r': df_r, 'b': b,
            'V': V, 'keep': keep, 'e': e, 'X': Xk}


def _lags_opt(p):
    lg = p.opts.get('lags', 1)
    return [int(v) for v in sorted(parse_numlist(lg))] if lg not in (None, '') else [1]


def _lm_table_head(ip, title, testname):
    ip.txt(f'\n{title}\n')
    ip.txt('-' * 13 + '+' + '-' * 61 + '\n')
    ip.txt(f'{"":4}lags(p){"":2}|{"":10}{testname:<19}df{"":17}Prob > {testname}\n')
    ip.txt('-' * 13 + '+' + '-' * 61 + '\n')


def _lm_row(ip, lag, stat, df, pv, small=False, df_r=None):
    ip.txt(f'{"":5}{lag:>3}{"":5}|')
    if small:
        ip.res(f'{"":3}{stat:>10.3f}{"":11}(' + f'{df:>3.0f},{df_r:>5.0f} )' + f'{"":11}{pv:>8.4f}\n')
    else:
        ip.res(f'{"":5}{stat:>10.3f}{"":13}{df:>3.0f}{"":17}{pv:>8.4f}\n')


def estat_dwatson(ip, p):
    est, f, mask, res, X = _after_regress(ip)
    N = int(mask.sum())
    k = f.N - f.df_r
    dw = _dw(ip, res)
    ip.txt('\nDurbin–Watson d-statistic(')
    ip.res(f'{k:>3}')
    ip.txt(',')
    ip.res(f'{N:>6}')
    ip.txt(') = ')
    ip.res(f'{g_fmt(dw, 9).strip()}\n')
    ip.set_r(scalars={'dw': dw, 'N': N, 'k': k, 'N_gaps': n_gaps(ip, mask)})


def _resid_lags(ip, res, mask, p_, miss0):
    Ls = []
    for j in range(1, p_ + 1):
        L = _ts_lag(ip, res, j)
        if miss0:
            L = np.where(mask & np.isnan(L), 0.0, L)
        Ls.append(L)
    return Ls


def estat_bgodfrey(ip, p):
    est, f, mask, res, X = _after_regress(ip)
    if est.get('vce') in ('robust', 'cluster'):
        raise StataError('This command does not work after regress, robust', 198)
    lags = _lags_opt(p)
    small = p.opts.has('small', 1)
    miss0 = not p.opts.has('nomiss0', 5)
    N = int(mask.sum())
    for l in lags:
        if l >= N - 1:
            raise StataError(f'lags({l}) is too large for the number of observations in the sample', 198)
    cons = not f.nocons
    Xf = [_full(f.mask, X[:, j]) for j in range(X.shape[1]) if f.info[f.active[j]].kind != 'cons']
    tn = 'F' if small else 'chi2'
    _lm_table_head(ip, 'Breusch–Godfrey LM test for autocorrelation', tn)
    out = {'stat': [], 'df': [], 'p': [], 'df_r': []}
    for l in lags:
        Ls = _resid_lags(ip, res, mask, l, miss0)
        cols = Ls + Xf
        ok = mask & ~np.isnan(res)
        for c in cols:
            ok &= ~np.isnan(c)
        r = _ols_lm(res[ok], np.column_stack([c[ok] for c in cols]), cons)
        if small:
            F = r['N'] * r['r2'] / l
            pv = float(sps.f.sf(F, l, r['df_r']))
            _lm_row(ip, l, F, l, pv, True, r['df_r'])
            out['stat'].append(F)
            out['df_r'].append(r['df_r'])
        else:
            chi2 = r['N'] * r['r2']
            pv = float(sps.chi2.sf(chi2, l))
            _lm_row(ip, l, chi2, l, pv)
            out['stat'].append(chi2)
        out['df'].append(l)
        out['p'].append(pv)
    ip.txt('-' * 13 + '+' + '-' * 61 + '\n')
    ip.txt(f'{"":24}H0: no serial correlation\n')
    _store_lm(ip, out, lags, tn, mask, N, f)


def _store_lm(ip, out, lags, tn, mask, N, f, statname=None):
    from .expr import Matrix
    cn = [str(l) for l in lags]
    mats = {(statname or tn): Matrix(np.array([out['stat']]), ['r1'], cn),
            'df': Matrix(np.array([out['df']], dtype=float), ['r1'], cn),
            'p': Matrix(np.array([out['p']]), ['r1'], cn)}
    if out.get('df_r'):
        mats['df_r'] = Matrix(np.array([out['df_r']], dtype=float), ['r1'], cn)
    ip.set_r(scalars={'N': N, 'k': f.N - f.df_r, 'N_gaps': n_gaps(ip, mask)},
             macros={'lags': ' '.join(cn)}, matrices=mats)


def estat_durbinalt(ip, p):
    est = ip.est
    if est is None or est.get('cmd') not in ('regress', 'newey'):
        raise StataError('This command only works after regress and newey', 301)
    if est.get('cmd') == 'newey' and not p.opts.has('force'):
        raise StataError('You must specify force for this command to work after newey', 198)
    if est.get('vce') == 'cluster':
        raise StataError('This command does not work after specifying the cluster() option', 198)
    if est.get('vce') == 'robust' and not p.opts.has('force'):
        raise StataError('you must specify force after regress, robust', 198)
    est, f, mask, res, X = _after_regress(ip, allow=('regress', 'newey'))
    robust = p.opts.has('robust', 1)
    small = p.opts.has('small', 1)
    if robust and small:
        raise StataError('-small- cannot be specified with -robust-.', 198)
    if robust:
        small = True
    lags = _lags_opt(p)
    miss0 = not p.opts.has('nomiss0', 5)
    N = int(mask.sum())
    for l in lags:
        if l >= N - 2:
            raise StataError(f'lags({l}) is too large for the number of observations in the sample', 198)
    cons = not f.nocons
    Xf = [_full(f.mask, X[:, j]) for j in range(X.shape[1]) if f.info[f.active[j]].kind != 'cons']
    tn = 'F' if small else 'chi2'
    _lm_table_head(ip, "Durbin's alternative test for autocorrelation", tn)
    out = {'stat': [], 'df': [], 'p': [], 'df_r': []}
    for l in lags:
        Ls = _resid_lags(ip, res, mask, l, miss0)
        cols = Ls + Xf
        ok = mask & ~np.isnan(res)
        for c in cols:
            ok &= ~np.isnan(c)
        r = _ols_lm(res[ok], np.column_stack([c[ok] for c in cols]), cons)
        idx = [r['keep'].index(j) for j in range(l) if j in r['keep']]
        Vt = r['V']
        if robust:
            Xk, e = r['X'], r['e']
            XtXi = np.linalg.pinv(Xk.T @ Xk)
            S = Xk * e[:, None]
            n_, kk = Xk.shape
            Vt = XtXi @ (S.T @ S) @ XtXi * n_ / (n_ - kk)
        bb = r['b'][idx]
        Vi = Vt[np.ix_(idx, idx)]
        W = float(bb @ np.linalg.pinv(Vi) @ bb)
        df = len(idx)
        F = W / df
        if small:
            pv = float(sps.f.sf(F, df, r['df_r']))
            _lm_row(ip, l, F, df, pv, True, r['df_r'])
            out['stat'].append(F)
            out['df_r'].append(r['df_r'])
        else:
            chi2 = F * df
            pv = float(sps.chi2.sf(chi2, df))
            _lm_row(ip, l, chi2, df, pv)
            out['stat'].append(chi2)
        out['df'].append(df)
        out['p'].append(pv)
    ip.txt('-' * 13 + '+' + '-' * 61 + '\n')
    ip.txt(f'{"":24}H0: no serial correlation\n')
    _store_lm(ip, out, lags, tn, mask, N, f)


def estat_archlm(ip, p):
    est = ip.est
    if est is not None and est.get('vce') == 'cluster':
        raise StataError('estat archlm does not work after regress, vce(cluster clustvar)', 198)
    if est is not None and est.get('vce') == 'robust' and not p.opts.has('force'):
        raise StataError('you must specify force after regress, vce(robust)', 198)
    est, f, mask, res, X = _after_regress(ip)
    lags = _lags_opt(p)
    N = int(mask.sum())
    for l in lags:
        if l >= N - 1:
            raise StataError(f'lags({l}) is too large for the number of observations in the sample', 198)
    r2v = res * res
    ip.txt('\nLM test for autoregressive conditional heteroskedasticity (ARCH)\n')
    ip.txt('-' * 13 + '+' + '-' * 61 + '\n')
    ip.txt(f'{"":4}lags(p){"":2}|{"":10}chi2{"":15}df{"":17}Prob > chi2\n')
    ip.txt('-' * 13 + '+' + '-' * 61 + '\n')
    out = {'stat': [], 'df': [], 'p': []}
    for l in lags:
        cols = [_ts_lag(ip, r2v, j) for j in range(1, l + 1)]
        ok = ~np.isnan(r2v)
        for c in cols:
            ok &= ~np.isnan(c)
        r = _ols_lm(r2v[ok], np.column_stack([c[ok] for c in cols]), True)
        a = r['N'] * r['r2']
        pv = float(sps.chi2.sf(a, l))
        _lm_row(ip, l, a, l, pv)
        out['stat'].append(a)
        out['df'].append(l)
        out['p'].append(pv)
    ip.txt('-' * 13 + '+' + '-' * 61 + '\n')
    ip.txt(f'{"":9}H0: no ARCH effects{"":5}vs.  H1: ARCH(p) disturbance\n')
    _store_lm(ip, out, lags, 'arch', mask, N, f, statname='arch')


def estat_imtest(ip, p):
    est = ip.est
    if est is None or est.get('cmd') != 'regress':
        raise StataError('imtest is only possible after regress', 301)
    f = est['fit']
    if f.wtype:
        raise StataError('imtest does not support weights', 101)
    if f.vce == 'cluster':
        raise StataError('imtest does not support cluster()', 498)
    rhs_idx = [j for j in f.active if f.info[j].kind != 'cons']
    if not rhs_idx:
        ip.txt('(imtest not allowed without covariates)\n')
        return
    Xr = f.Xfull[:, rhs_idx]
    n = Xr.shape[0]
    res = f.e
    s2 = f.rss / f.N
    C = Xr - Xr.mean(axis=0)
    cross = []
    for i in range(Xr.shape[1]):
        for j in range(i + 1):
            cross.append(C[:, i] * C[:, j])
    X2 = np.column_stack([Xr] + cross)
    out = {}
    if p.opts.has('white', 1):
        r = _ols_lm(res ** 2, X2, True)
        out['chi2'] = r['N'] * r['r2']
        out['df'] = r['df_m']
        out['p'] = float(sps.chi2.sf(out['chi2'], out['df']))

    def piece(yv, Xm):
        uss = float(np.sum(yv ** 2))
        if Xm is None:
            rss = float(np.sum((yv - yv.mean()) ** 2))
            return n * (1 - rss / uss), 0
        r = _ols_lm(yv, Xm, True)
        return r['N'] * (1 - r['rss'] / uss), r['df_m']
    ch, dh = piece(res ** 2 - s2, X2)
    cs, ds = piece(res ** 3 - 3 * s2 * res, Xr)
    ck, _ = piece(res ** 4 - 6 * s2 * res ** 2 + 3 * s2 * s2, None)
    dk = 1
    ct, dt = ch + cs + ck, dh + ds + dk
    if 'chi2' in out:
        ip.txt("\nWhite's test for Ho: homoskedasticity\n         against Ha: unrestricted heteroskedasticity\n\n")
        ip.txt(f'{"":9}chi2(')
        ip.res(str(out['df']))
        ip.txt(')')
        ip.txt(' ' * max(1, 22 - (14 + len(str(out['df'])) + 1)) + '= ')
        ip.res(f'{out["chi2"]:9.2f}\n')
        ip.txt(f'{"":9}Prob > chi2  = ')
        ip.res(f'{out["p"]:9.4f}\n')
    ip.txt("\nCameron & Trivedi's decomposition of IM-test\n\n")
    ip.txt('-' * 21 + '+' + '-' * 29 + '\n')
    ip.txt(f'{"Source":>20} |       chi2     df      p\n')
    ip.txt('-' * 21 + '+' + '-' * 29 + '\n')
    for lab, c, df_ in (('Heteroskedasticity', ch, dh), ('Skewness', cs, ds), ('Kurtosis', ck, dk)):
        ip.txt(f'{lab:>20} |')
        ip.res(f'{c:11.2f}{df_:7.0f}{sps.chi2.sf(c, df_):10.4f}\n')
    ip.txt('-' * 21 + '+' + '-' * 29 + '\n')
    ip.txt(f'{"Total":>20} |')
    ip.res(f'{ct:11.2f}{dt:7.0f}{sps.chi2.sf(ct, dt):10.4f}\n')
    ip.txt('-' * 21 + '+' + '-' * 29 + '\n')
    sc = {'chi2_h': ch, 'df_h': dh, 'chi2_s': cs, 'df_s': ds, 'chi2_k': ck, 'df_k': dk, 'chi2_t': ct, 'df_t': dt}
    sc.update(out)
    ip.set_r(scalars=sc)


ESTAT_EXT.extend([
    ('dwa', 'dwatson', estat_dwatson),
    ('dur', 'durbinalt', estat_durbinalt),
    ('bgo', 'bgodfrey', estat_bgodfrey),
    ('archlm', 'archlm', estat_archlm),
    ('imt', 'imtest', estat_imtest),
])
