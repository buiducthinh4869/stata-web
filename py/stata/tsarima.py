"""arima: regression with (seasonal) ARMA disturbances by exact Kalman-filter ML (as Stata's arima_dr),
conditional ML (condition), diffuse initialisation; OPG default variance; predict (xb, y, residuals,
yresiduals, mse, dynamic()); tsappend."""
import math

import numpy as np
from scipy import stats as sps

from .design import coef_table, right_lines, Col
from .errors import StataError
from .fmt import g_fmt, comma_int, fmt_any
from .interp import command, alias
from .syntax import parse_command, split_top, parse_numlist
from .tsuni import need_ts, series, ts_unit
from .tsops import apply as ts_apply, expand_token
from . import tsml

alias('arima', 'arim')


# ------------------------------------------------------------------ polynomials
def ar_poly(phi, lags, sphi, slags, s):
    """Coefficients c_k (k=1..K) of 1 - sum c_k L^k = (1 - sum phi L^lags)(1 - sum sphi L^(s*slags))."""
    a = np.zeros(1 + (max(lags) if lags else 0))
    a[0] = 1
    for c, l in zip(phi, lags):
        a[l] -= c
    b = np.zeros(1 + (s * max(slags) if slags else 0))
    b[0] = 1
    for c, l in zip(sphi, slags):
        b[s * l] -= c
    prod = np.convolve(a, b)
    return -prod[1:]


def ma_poly(th, lags, sth, slags, s):
    a = np.zeros(1 + (max(lags) if lags else 0))
    a[0] = 1
    for c, l in zip(th, lags):
        a[l] += c
    b = np.zeros(1 + (s * max(slags) if slags else 0))
    b[0] = 1
    for c, l in zip(sth, slags):
        b[s * l] += c
    return np.convolve(a, b)[1:]


def _ssm(c, m):
    """Hamilton representation (as Stata's arima): state = (mu_t, ..., mu_{t-r+1}), Z = (1, theta)."""
    r = max(len(c), len(m) + 1, 1)
    T = np.zeros((r, r))
    T[0, :len(c)] = c
    for i in range(1, r):
        T[i, i - 1] = 1.0
    Z = np.zeros(r)
    Z[0] = 1.0
    Z[1:1 + len(m)] = m
    return T, Z


def _uncond_P(T):
    r = T.shape[0]
    Q = np.zeros((r, r))
    Q[0, 0] = 1.0
    A = np.eye(r * r) - np.kron(T, T)
    vecP = np.linalg.solve(A, Q.reshape(-1))
    return vecP.reshape(r, r)


def kalman(mu, c, m, sigma2, diffuse=False, want=False):
    """Exact Gaussian likelihood contributions of mu (nan = missing) for ARMA(c, m) errors."""
    T, Z = _ssm(c, m)
    r = T.shape[0]
    n = mu.size
    llv = np.zeros(n)
    pred = np.full(n, np.nan)
    Fv = np.full(n, np.nan)
    if diffuse:
        P = np.eye(r) * 1e9
    else:
        # as Stata: the Lyapunov solution is used even outside the stationary region
        try:
            P = _uncond_P(T) * sigma2
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(P)):
            return None
    a = np.zeros(r)
    for t in range(n):
        if t > 0:
            a = T @ a
            P = T @ P @ T.T
            P[0, 0] += sigma2
        pa = float(Z @ a)
        PZ = P @ Z
        F = float(Z @ PZ)
        pred[t] = pa
        Fv[t] = F
        x = mu[t]
        if not np.isnan(x):
            if not F > 0:
                return None
            v = x - pa
            K = PZ / F
            a = a + K * v
            P = P - np.outer(K, PZ)
            llv[t] = -0.5 * (math.log(2 * math.pi) + math.log(F) + v * v / F)
    if want:
        return llv, pred, Fv
    return llv


def css(mu, c, m, sigma2):
    """Conditional (presample zeros) likelihood contributions."""
    n = mu.size
    eps = np.zeros(n)
    mm = np.nan_to_num(mu)
    K, J = len(c), len(m)
    for t in range(n):
        s = mm[t]
        for k in range(1, K + 1):
            if t - k >= 0:
                s -= c[k - 1] * mm[t - k]
        for j in range(1, J + 1):
            if t - j >= 0:
                s -= m[j - 1] * eps[t - j]
        eps[t] = s
    llv = np.where(np.isnan(mu), 0.0, -0.5 * (math.log(2 * math.pi) + math.log(sigma2) + eps ** 2 / sigma2))
    return llv, eps


def _lagg(x, k):
    out = np.full(x.size, np.nan)
    if k < x.size:
        out[k:] = x[:x.size - k]
    return out


def _ols_r(y, X, cons):
    ok = ~np.isnan(y)
    for c in X:
        ok &= ~np.isnan(c)
    M = np.column_stack(X + ([np.ones(y.size)] if cons else []))[ok]
    b = np.linalg.lstsq(M, y[ok], rcond=None)[0]
    e = np.full(y.size, np.nan)
    e[ok] = y[ok] - M @ b
    return b, e, int(ok.sum()), M.shape[1]


def _start_values(yg, Xd, obs, kx, ar, ma, nsar, nsma, no_xb):
    G = yg.size
    if not no_xb and kx:
        bb = np.linalg.lstsq(Xd[obs], yg[obs], rcond=None)[0]
        u = np.where(obs, yg - Xd @ bb, np.nan)
        rss = float(np.nansum(u ** 2))
        sigma = math.sqrt(rss / obs.sum())
    else:
        bb = np.zeros(kx)
        u = np.where(obs, yg, np.nan)
        sigma = float(np.nanstd(yg, ddof=1))
    arma = []
    if ar or ma:
        lags_all = sorted(ar + ma)
        minlag = lags_all[0]
        maxlag = (max(ar) if ar else 0) + (max(ma) if ma else 0)
        k = 30
        usemax = usemin = None
        while k > 0:
            usemin = min(minlag, 1) if k > 10 else minlag
            usemax = max(maxlag, usemin + k)
            ok = ~np.isnan(u)
            for l in range(usemin, usemax + 1):
                ok &= ~np.isnan(_lagg(u, l))
            if ok.sum() > 2 * k:
                break
            k -= 5
        et = None
        while usemax is not None and usemax > usemin:
            try:
                _, et, _, _ = _ols_r(u, [_lagg(u, l) for l in range(usemin, usemax + 1)], True)
                break
            except np.linalg.LinAlgError:
                usemax -= 1
        cols = [_lagg(u, l) for l in ar] + ([_lagg(et, l) for l in ma] if et is not None else [])
        try:
            T, e2, n2, k2 = _ols_r(u, cols, False)
            nar = len(ar)
            bs = float(np.sum(T[:nar]))
            if abs(bs) > .95:
                T[:nar] = (.95 / abs(bs)) * T[:nar]
            for i in range(nar, T.size):
                if abs(T[i]) > .95:
                    T[i] = .95 * T[i] / abs(T[i])
            arma = list(T)
            sigma = math.sqrt(float(np.nansum(e2 ** 2)) / n2)
        except (np.linalg.LinAlgError, ValueError):
            arma = [0.0] * (len(ar) + len(ma))
    ar0 = arma[:len(ar)]
    ma0 = arma[len(ar):]
    return np.concatenate([bb, ar0, [0.0] * nsar, ma0, [0.0] * nsma, [sigma]])


# ------------------------------------------------------------------ tsappend
@command('tsappend')
def cmd_tsappend(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    need_ts(d)
    add = p.opts.get('add', 1)
    last = p.opts.get('last', 1)
    tvar = d.ts['time']
    delta = d.ts.get('delta', 1) or 1
    t = d.cols[tvar].astype(float)
    if add in (None, '') and last in (None, ''):
        raise StataError('must specify either add() or last()', 198)
    panel = d.ts.get('panel')
    pv = d.cols[panel].astype(float) if panel else np.zeros(d.n)
    newrows = []
    for g in np.unique(pv[~np.isnan(pv)]):
        tmax = np.nanmax(t[pv == g])
        if add not in (None, ''):
            k = int(float(add))
        else:
            from . import expr as E
            lt = float(E.eval_expr(last, E.Ctx(ip, d)))
            k = int(round((lt - tmax) / delta))
        for i in range(1, k + 1):
            newrows.append((g, tmax + i * delta))
    if not newrows:
        return
    from .cmds_ext import _append_rows
    n0 = d.n
    _append_rows(d, len(newrows))
    for i, (g, tt) in enumerate(newrows):
        d.cols[tvar][n0 + i] = tt
        if panel:
            d.cols[panel][n0 + i] = g
    d.sort([x for x in (panel, tvar) if x])


# ------------------------------------------------------------------ arima
def _numlist(s):
    return [int(v) for v in parse_numlist(s)] if s not in (None, '') else []


@command('arima')
def cmd_arima(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    toks = split_top(p.body)
    if not toks:
        if ip.est and ip.est.get('cmd') == 'arima':
            return ip.est['replay'](ip)
        raise StataError('varlist required', 100)
    need_ts(d)
    keys = [x for x in (d.ts.get('panel'), d.ts.get('time')) if x]
    d.sort(keys)
    for o in ('mar1', 'mma1', 'mar2', 'mma2'):
        if p.opts.get(o, 4):
            raise StataError(f'{o}() is not available in Stata Web; use sarima()', 199)
    ar = _numlist(p.opts.get('ar', 2))
    ma = _numlist(p.opts.get('ma', 2))
    dd = 0
    if p.opts.get('arima', 5):
        a3 = _numlist(p.opts.get('arima', 5))
        if len(a3) != 3:
            raise StataError('arima() requires three integers', 198)
        if ar or ma:
            raise StataError('arima() may not be combined with ar() or ma()', 198)
        ar = list(range(1, a3[0] + 1))
        dd = a3[1]
        ma = list(range(1, a3[2] + 1))
    sar, sma, sD, s = [], [], 0, 0
    if p.opts.get('sarima', 6):
        s4 = _numlist(p.opts.get('sarima', 6))
        if len(s4) != 4:
            raise StataError('sarima() requires four integers', 198)
        sar = list(range(1, s4[0] + 1))
        sD = s4[1]
        sma = list(range(1, s4[2] + 1))
        s = s4[3]
        if s < 2:
            raise StataError('seasonal period in sarima() must be at least 2', 198)
    nocons = p.opts.has('noconstant', 4)
    condition = p.opts.has('condition', 4)
    diffuse = p.opts.has('diffuse', 2)
    vk = (p.opts.get('vce', 3) or '').strip().lower()
    if p.opts.has('robust', 1):
        vk = 'robust'
    vk = {'': 'opg', 'opg': 'opg', 'oim': 'oim', 'robust': 'robust', 'r': 'robust'}.get(vk)
    if vk is None:
        raise StataError('vce() must be opg, oim, or robust', 198)
    chain = 'D' * dd + ''.join(f'S{s}' for _ in range(sD))
    from .tsops import opname, _parse_chain
    chain = opname(_parse_chain(chain)) if chain else ''

    def transformed(tok):
        x, nm = series(ip, tok)
        if not chain:
            return x, nm
        from .tsops import split_name
        ch0, base = split_name(nm)
        full = opname(_parse_chain(ch0 + chain))
        return ts_apply(ip, d, full, base), f'{full}.{base}'
    y, yname = transformed(toks[0])
    Xs, xnames = [], []
    for tok in toks[1:]:
        from .tsops import expand_token
        for t2 in expand_token(tok):
            x, nm = transformed(t2)
            Xs.append(x)
            xnames.append(nm)
    mask = ip.mark(p, d) & ~np.isnan(y)
    for x in Xs:
        mask &= ~np.isnan(x)
    tv = d.cols[d.ts['time']].astype(float)
    mask &= ~np.isnan(tv)
    if d.ts.get('panel'):
        pp = d.cols[d.ts['panel']].astype(float)
        if len(np.unique(pp[mask])) > 1:
            raise StataError('sample may not include multiple panels', 459)
    if mask.sum() == 0:
        raise StataError('no observations', 2000)
    delta = d.ts.get('delta', 1) or 1
    tmin, tmax = tv[mask].min(), tv[mask].max()
    grid_t = np.arange(tmin, tmax + delta / 2, delta)
    pos = np.round((tv - tmin) / delta).astype(int)
    G = grid_t.size
    yg = np.full(G, np.nan)
    yg[pos[mask]] = y[mask]
    Xg = np.full((G, len(Xs)), np.nan)
    for j, x in enumerate(Xs):
        Xg[pos[mask], j] = x[mask]
    obs = ~np.isnan(yg)
    N = int(obs.sum())
    gaps = int(np.sum(~obs))
    if condition and gaps:
        raise StataError('condition may not be used with gaps in the data', 498)
    kx = len(Xs) + (0 if nocons else 1)
    Xd = np.column_stack([np.nan_to_num(Xg)] + ([np.ones(G)] if not nocons else [])) if kx else np.zeros((G, 0))
    nar, nma, nsar, nsma = len(ar), len(ma), len(sar), len(sma)

    def unpack(th):
        i = 0
        beta = th[i:i + kx]
        i += kx
        phi = th[i:i + nar]
        i += nar
        sphi = th[i:i + nsar]
        i += nsar
        tht = th[i:i + nma]
        i += nma
        stht = th[i:i + nsma]
        i += nsma
        return beta, phi, sphi, tht, stht, th[i]

    def llvec(th):
        beta, phi, sphi, tht, stht, sig = unpack(th)
        if sig == 0 or not np.isfinite(sig):
            return np.full(G, -np.inf)
        mu = np.where(obs, yg - Xd @ beta, np.nan)
        c = ar_poly(phi, ar, sphi, sar, s)
        m = ma_poly(tht, ma, stht, sma, s)
        if condition:
            return css(mu, c, m, sig * sig)[0]
        r = kalman(mu, c, m, sig * sig, diffuse=diffuse)
        if r is None:
            return np.full(G, -np.inf)
        return r
    # starting values exactly as arima.ado: OLS, Monfort long-AR whitening, Hannan-Rissanen, FixARMA
    th0 = _start_values(yg, Xd, obs, kx, ar, ma, nsar, nsma, nocons and not Xs)
    if not np.isfinite(np.sum(llvec(th0))):
        th0[kx:kx + nar + nsar] = 0.0
    frm = p.opts.get('from', 4)
    if frm not in (None, ''):
        vals = frm.replace(',', ' ').split()
        vals = [v for v in vals if v.lower() != 'copy']
        try:
            th0 = np.array([float(v) for v in vals])
        except ValueError:
            M = ip.get_matrix(vals[0])
            th0 = np.asarray(M.a, float).ravel()
        if th0.size != kx + nar + nsar + nma + nsma + 1:
            raise StataError('from(): wrong number of values', 503)
        # user order is Stata's (xb, ARMA ar, ARMA ma, ARMAs ar, ARMAs ma, sigma) -> internal order
        th0 = np.concatenate([th0[:kx], th0[kx:kx + nar], th0[kx + nar + nma:kx + nar + nma + nsar],
                              th0[kx + nar:kx + nar + nma], th0[kx + nar + nma + nsar:-1], th0[-1:]])
    itmax = p.opts.get('iterate', 4)
    itmax = int(float(itmax)) if itmax not in (None, '') else 300
    if itmax == 0:
        th = th0.copy()
        ll = float(np.sum(llvec(th)))
        it_log = [(0, ll)]
    else:
        th, ll, it_log = tsml.maximize(llvec, th0, maxit=itmax)
    V = tsml.vce(llvec, th, vk)
    se = np.sqrt(np.maximum(np.diag(V), 0))
    # names / colinfo
    eqdep = yname.split('.')[-1]
    names, info = [], []

    def add(nm, kind, tsvar=None, chain_=''):
        c = Col(name=nm, values=None, term=0, group='', level='', kind=kind, vars=[])
        c.tsvar, c.tschain = tsvar, chain_
        names.append(nm)
        info.append(c)
    for nm in xnames:
        from .tsops import split_name
        ch, base = split_name(nm)
        add(nm, 'cont', base, ch)
    if not nocons:
        add('_cons', 'cons')
    for l in ar:
        add(f'L{"" if l == 1 else l}.ar', 'cont', 'ar', f'L{"" if l == 1 else l}')
    for l in sar:
        add(f'L{"" if l == 1 else l}.ar', 'cont', 'ar', f'L{"" if l == 1 else l}')
    for l in ma:
        add(f'L{"" if l == 1 else l}.ma', 'cont', 'ma', f'L{"" if l == 1 else l}')
    for l in sma:
        add(f'L{"" if l == 1 else l}.ma', 'cont', 'ma', f'L{"" if l == 1 else l}')
    sig = th[-1]
    # Wald test: regressors (excluding constant) and ARMA terms
    widx = [j for j in range(len(xnames))] + list(range(kx, th.size - 1))
    if widx:
        bw = th[widx]
        Vw = V[np.ix_(widx, widx)]
        chi2 = float(bw @ tsml.safe_inv(Vw) @ bw)
        dfm = len(widx)
        pchi = float(sps.chi2.sf(chi2, dfm))
    else:
        chi2, dfm, pchi = np.nan, 0, np.nan
    tf = d.formats.get(d.ts['time'], '%9.0g')
    smin = fmt_any(float(tmin), tf).strip()
    smax = fmt_any(float(tmax), tf).strip()
    k_all = th.size
    full_names = ([f'{eqdep}:{n}' for n in names[:kx]] + [f'ARMA:{n}' for n in names[kx:kx + nar]]
                  + [f'ARMA{s}:{n}' for n in names[kx + nar:kx + nar + nsar]]
                  + [f'ARMA:{n}' for n in names[kx + nar + nsar:kx + nar + nsar + nma]]
                  + [f'ARMA{s}:{n}' for n in names[kx + nar + nsar + nma:k_all - 1]] + ['sigma:_cons'])
    # reorder to Stata's equation order: xb, ARMA (ar, ma), ARMAs (ar, ma), sigma
    order = (list(range(kx)) + list(range(kx, kx + nar)) + list(range(kx + nar + nsar, kx + nar + nsar + nma))
             + list(range(kx + nar, kx + nar + nsar)) + list(range(kx + nar + nsar + nma, k_all - 1)) + [k_all - 1])
    b_st = th[order]
    V_st = V[np.ix_(order, order)]
    nm_st = [full_names[i] for i in order]
    est = {'names': nm_st, 'b': b_st, 'V': V_st, 'colinfo': None, 'df_r': None, 'depvar': yname,
           'sample': mask, 'cmd': 'arima', 'stat': 'z', 'vce': vk, 'nomargins': True,
           'scalars': {'N': N, 'll': ll, 'chi2': chi2, 'df_m': dfm, 'p': pchi, 'sigma': sig, 'N_gaps': gaps,
                       'k': k_all, 'ar_max': max(ar) if ar else 0, 'ma_max': max(ma) if ma else 0,
                       'tmin': tmin, 'tmax': tmax},
           'macros': {'cmd': 'arima', 'depvar': yname, 'title': 'ARIMA regression', 'vce': vk,
                      'vcetype': {'opg': 'OPG', 'oim': 'OIM', 'robust': 'Semirobust'}[vk],
                      'tmins': smin, 'tmaxs': smax, 'ar': ' '.join(map(str, ar)), 'ma': ' '.join(map(str, ma)),
                      'diffuse': 'diffuse' if diffuse else ''}}
    lab = {'opg': 'OPG', 'oim': 'OIM', 'robust': 'Semirobust'}[vk]
    quiet_log = p.opts.has('nolog') or ip.quiet

    def subtab(idx, eqt, first, last):
        sub = {'names': [names[i] for i in idx], 'b': th[idx], 'V': V[np.ix_(idx, idx)],
               'colinfo': [info[i] for i in idx], 'df_r': None}
        coef_table(ip_ref[0], sub, 'z', robust=lab, depvar=yname, header=first, footer=False, eq_title=eqt)

    ip_ref = [ip]

    def replay(ip_, show_log=False):
        ip_ref[0] = ip_
        if show_log and not quiet_log:
            tsml.print_log(ip_, it_log)
        ip_.txt('ARIMA regression\n\n')
        right_lines(ip_, [f'Sample: {smin} thru {smax}' + (', but with gaps' if gaps else ''), '',
                          f'Log likelihood = {g_fmt(ll, 9).strip()}'],
                    [('Number of obs', comma_int(N)),
                     (f'Wald chi2({dfm if dfm else "."})', f'{chi2:.2f}' if chi2 == chi2 else '.'),
                     ('Prob > chi2', f'{pchi:.4f}' if pchi == pchi else '.')])
        ip_.txt('\n')
        first = True
        if kx:
            subtab(list(range(kx)), eqdep, first, False)
            first = False
        if nar + nma:
            idx = list(range(kx, kx + nar)) + list(range(kx + nar + nsar, kx + nar + nsar + nma))
            subtab(idx, 'ARMA', first, False)
            first = False
        if nsar + nsma:
            idx = list(range(kx + nar, kx + nar + nsar)) + list(range(kx + nar + nsar + nma, k_all - 1))
            subtab(idx, f'ARMA{s}', first, False)
            first = False
        if first:
            coef_table(ip_, {'names': [], 'b': np.zeros(0), 'V': np.zeros((0, 0)), 'colinfo': [], 'df_r': None},
                       'z', robust=lab, depvar=yname, footer=False)
        ip_.txt('-' * 13 + '+' + '-' * 64 + '\n')
        ss = se[-1]
        z = sig / ss if ss > 0 else np.nan
        crit = sps.norm.ppf(0.975)
        lo = max(0.0, sig - crit * ss)
        hi = sig + crit * ss
        ip_.txt(f'{"/sigma":>12} |')
        ip_.res(f'{g_fmt(sig, 9):>11}{g_fmt(ss, 9):>11}{z:9.2f}{2 * sps.norm.sf(abs(z)):8.3f}'
                f'{g_fmt(lo, 9):>13}{g_fmt(hi, 9):>12}\n')
        ip_.txt('-' * 78 + '\n')
        ip_.txt('Note: The test of the variance against zero is one sided, and the two-sided\n')
        ip_.txt('      confidence interval is truncated at zero.\n')

    # ---------------- predict
    def predict_multi(ip_, pp, typ, tk):
        kinds = pp.opts.names()
        kind = 'xb'
        for kname in ('xb', 'y', 'residuals', 'yresiduals', 'mse', 'stdp'):
            for kk in kinds:
                if kname.startswith(kk) and kk != 'dynamic' and len(kk) >= 1:
                    kind = kname
        if pp.opts.has('residuals', 1):
            kind = 'residuals'
        if pp.opts.has('yresiduals', 2):
            kind = 'yresiduals'
        if pp.opts.has('y', 1) and not pp.opts.has('yresiduals', 2):
            kind = 'y'
        if pp.opts.has('mse', 3):
            kind = 'mse'
        dd_ = ip_.data
        dyn = pp.opts.get('dynamic', 3)
        tv_ = dd_.cols[dd_.ts['time']].astype(float)
        hi_t = np.nanmax(tv_)
        gt = np.arange(tmin, hi_t + delta / 2, delta)
        Gp = gt.size
        pos2 = np.round((tv_ - tmin) / delta).astype(int)
        inr = (pos2 >= 0) & (pos2 < Gp) & ~np.isnan(tv_)
        y_full, _ = transformed(toks[0])
        lev, _ = series(ip_, toks[0])
        ygp = np.full(Gp, np.nan)
        ygp[pos2[inr]] = y_full[inr]
        levg = np.full(Gp, np.nan)
        levg[pos2[inr]] = lev[inr]
        Xgp = np.full((Gp, len(Xs)), np.nan)
        for j, tok in enumerate([t2 for tk_ in toks[1:] for t2 in expand_token(tk_)]):
            x, _ = transformed(tok)
            Xgp[pos2[inr], j] = x[inr]
        Xdp = np.column_stack([Xgp] + ([np.ones(Gp)] if not nocons else [])) if kx else np.zeros((Gp, 0))
        beta, phi, sphi, tht, stht, sg = unpack(th)
        xb0 = Xdp @ beta if kx else np.zeros(Gp)
        mu = ygp - xb0
        t0 = None
        if dyn not in (None, ''):
            from . import expr as E
            t0 = float(E.eval_expr(dyn, E.Ctx(ip_, dd_)))
            mu = np.where(gt >= t0, np.nan, mu)
        c = ar_poly(phi, ar, sphi, sar, s)
        m = ma_poly(tht, ma, stht, sma, s)
        if condition:
            _, eps = css(np.where(np.isnan(mu), 0, mu), c, m, sg * sg)
            pr = np.where(np.isnan(ygp), np.nan, (ygp - xb0) - eps)
            Fv = np.full(Gp, sg * sg)
        else:
            out = kalman(mu, c, m, sg * sg, diffuse=diffuse, want=True)
            _, pr, Fv = out
        xbp = xb0 + pr
        if kind == 'xb':
            vals = xbp
        elif kind == 'residuals':
            vals = ygp - xbp
        elif kind == 'mse':
            vals = Fv
        else:
            # undifference: y_t = y*_t + sum c_k y_{t-k}, c from (1-L)^d (1-L^s)^D
            dp = np.array([1.0])
            for _ in range(dd):
                dp = np.convolve(dp, [1.0, -1.0])
            for _ in range(sD):
                sp = np.zeros(s + 1)
                sp[0], sp[s] = 1.0, -1.0
                dp = np.convolve(dp, sp)
            ck = -dp[1:]
            yhat = np.full(Gp, np.nan)
            for t in range(Gp):
                acc = xbp[t]
                okk = True
                for k_, cv in enumerate(ck, start=1):
                    if cv == 0:
                        continue
                    if t - k_ < 0:
                        okk = False
                        break
                    src = levg[t - k_]
                    if t0 is not None and gt[t - k_] >= t0:
                        src = yhat[t - k_]
                    if np.isnan(src):
                        okk = False
                        break
                    acc += cv * src
                yhat[t] = acc if okk else np.nan
            vals = yhat if kind == 'y' else levg - yhat
        outv = np.full(dd_.n, np.nan)
        outv[inr] = vals[pos2[inr]]
        mk = ip_.mark(pp)
        outv[~mk] = np.nan
        name = tk[0]
        dd_.add_var(name, outv, typ or 'float')
        dd_.varlab[name] = {'xb': f'xb prediction, one-step', 'y': f'y prediction, one-step',
                            'residuals': 'residual, one-step', 'yresiduals': 'y residual, one-step',
                            'mse': 'MSE of xb prediction'}.get(kind, '')
        if t0 is not None:
            dd_.varlab[name] = dd_.varlab[name].replace('one-step', f'dyn({fmt_any(t0, tf).strip()})')
        nm = int(np.isnan(outv).sum())
        if nm:
            ip_.txt(f'({comma_int(nm)} missing value{"s" if nm != 1 else ""} generated)\n')
    est['replay'] = lambda ip_: replay(ip_, False)
    est['predict'] = None
    est['predict_multi'] = predict_multi
    ip.set_estimation(est)
    replay(ip, True)
