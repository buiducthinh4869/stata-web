"""Nonlinear models: poisson, nbreg, ppmlhdfe, tobit, ologit/oprobit, mlogit."""
import math

import numpy as np
from scipy import stats as sps
from scipy import special as spc

from . import expr as E
from .design import coef_table, right_lines, Col
from .errors import StataError
from .estim import (prepare_linear, assemble, detect_omitted, full_bV, wald_test, parse_vce, cluster_ids,
                    _predict_linear, _store_rhs, sandwich, _omit_name)
from .design import mark_collinear
from .fmt import g_fmt, comma_int, abbrev
from .interp import command, alias
from .syntax import parse_command, split_top

alias('poisson', 'pois', 'poiss', 'poisso')
alias('ologit', 'olog', 'ologi')
alias('oprobit', 'oprob', 'oprobi')
alias('mlogit', 'mlog', 'mlogi')


# ================================================================ helpers
def _sexp(v):
    return math.exp(v) if v < 709.0 else float('nan')


def _num_hess(grad, b, h=1e-5):
    k = len(b)
    H = np.zeros((k, k))
    for j in range(k):
        e = np.zeros(k)
        step = h * max(1.0, abs(b[j]))
        e[j] = step
        H[:, j] = (grad(b + e) - grad(b - e)) / (2 * step)
    return (H + H.T) / 2


def _newton(fun, b0, maxit=300):
    """Maximise; fun(b) -> (ll, g, H). Returns b, ll, H, list of ll per iteration."""
    b = np.array(b0, float)
    ll, g, H = fun(b)
    lls = [ll]
    for _ in range(maxit):
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            raise StataError('convergence not achieved', 430)
        s = 1.0
        while True:
            bn = b - s * step
            lln, gn, Hn = fun(bn)
            if np.isfinite(lln) and lln >= ll - 1e-12 * (1 + abs(ll)):
                break
            s /= 2
            if s < 1e-12:
                bn, lln, gn, Hn = b, ll, g, H
                break
        crit = abs(float(g @ step))  # scaled gradient g'H^-1 g
        dll = abs(lln - ll)
        b, ll, g, H = bn, lln, gn, Hn
        lls.append(ll)
        if (dll < 1e-11 * (1 + abs(ll)) and crit < 1e-5) or s < 1e-12:
            break
    return b, ll, H, lls


def _print_log(ip, lls, title=None, pseudo=False):
    if title:
        ip.txt(f'{title}\n\n')
    word = 'Log pseudolikelihood' if pseudo else 'Log likelihood'
    for i, l in enumerate(lls):
        ip.txt(f'Iteration {i}:  {word} = ')
        ip.res(f'{g_fmt(l, 10)}\n')
    ip.txt('\n')


def _vce_adjust(ip, p, d, mask, Vinv, S, n):
    """Robust / cluster sandwich for ML estimators. Returns V, vce, clv, G."""
    vce, clv = parse_vce(p.opts)
    if vce == 'cluster':
        gid, cname = cluster_ids(d, clv, mask)
        G = int(gid.max()) + 1
        return sandwich(None, S, Vinv, gid) * G / (G - 1), vce, cname, G
    if vce == 'robust':
        return sandwich(None, S, Vinv) * n / (n - 1), vce, None, None
    return Vinv, 'ols', None, None


def _setup(ip, p, depvar_tok, rhs, cons=True):
    d = ip.data
    yv, y, cols, mask, w, wtype = prepare_linear(ip, p, d, rhs, depvar_tok)
    if w is not None and wtype not in ('fweight', 'pweight', 'iweight'):
        raise StataError(f'{wtype}s not allowed', 101)
    vce, clv = parse_vce(p.opts)
    if clv:
        cv = d.resolve(clv)
        mask &= (d.cols[cv] != '') if d.isstr(cv) else ~np.isnan(d.cols[cv])
    Xall, names, info = assemble(cols, mask, cons=cons)
    active = detect_omitted(Xall, info, cons=cons)
    return d, yv, y[mask], cols, mask, Xall, names, info, active


def _offset(ip, p, d, mask):
    ex = p.opts.get('exposure', 3)
    of = p.opts.get('offset', 3)
    if ex:
        v = d.resolve(ex)
        x = d.cols[v].astype(float)
        if np.any(x[mask & ~np.isnan(x)] <= 0):
            raise StataError(f'{v} has nonpositive values', 459)
        return np.log(x), f'ln({v})', 'exposure'
    if of:
        v = d.resolve(of)
        return d.cols[v].astype(float), v, 'offset'
    return None, None, None


def _header_stat(vce, k, lr, b, V, slopes, df_extra=None):
    if vce != 'ols':
        stat, _, pv = wald_test(b, V, slopes)
        return f'Wald chi2({k})', stat, pv
    return f'LR chi2({k})', lr, sps.chi2.sf(lr, k) if k else np.nan


def _robust_label(vce, clv, G):
    if vce == 'cluster':
        return f'(Std. err. adjusted for {comma_int(G)} clusters in {clv})', 'Robust'
    if vce == 'robust':
        return None, 'Robust'
    return None, None


def _eform_head(name, robust):
    return f'{name:>11}  {"std. err." if robust else "Std. err."}'


def _finish_predict(ip, p, typ, toks, vals, label):
    d = ip.data
    name = toks[0]
    mask = ip.mark(p)
    vals = np.where(mask, vals, np.nan)
    d.add_var(name, vals, typ or 'float')
    d.varlab[name] = label
    nm = int(np.isnan(d.cols[name]).sum())
    if nm:
        ip.txt(f'({comma_int(nm)} missing value{"s" if nm != 1 else ""} generated)\n')


# ================================================================ poisson
def _poisson_fit(X, y, off, b0=None):
    offv = off if off is not None else 0.0

    def fun(b):
        xb = X @ b + offv
        mu = np.exp(xb)
        ll = float(np.sum(-mu + y * xb - spc.gammaln(y + 1)))
        g = X.T @ (y - mu)
        H = -(X.T @ (X * mu[:, None]))
        return ll, g, H
    if b0 is None:
        b0 = np.zeros(X.shape[1])
    return _newton(fun, b0)


@command('poisson')
def cmd_poisson(ip, cmd, rest, by):
    p = parse_command(rest)
    toks = split_top(p.body)
    cons = not p.opts.has('noconstant', 4)
    d, yv, yy, cols, mask, Xall, names, info, active = _setup(ip, p, toks[0], toks[1:], cons)
    off_all, offname, offkind = _offset(ip, p, d, mask)
    if off_all is not None and np.isnan(off_all[mask]).any():
        raise StataError('offset/exposure variable has missing values in the estimation sample', 416)
    if np.any(yy < 0):
        raise StataError(f'{yv} must be greater than or equal to zero', 459)
    off = off_all[mask] if off_all is not None else None
    X = Xall[:, active]
    n = len(yy)
    ybar = yy.mean()
    # constant-only model
    if cons and off is None:
        ll0 = float(np.sum(-ybar + yy * math.log(ybar) - spc.gammaln(yy + 1))) if ybar > 0 else 0.0
    elif cons:
        _, ll0, _, _ = _poisson_fit(np.ones((n, 1)), yy, off)
    else:
        ll0 = np.nan
    b0 = np.zeros(X.shape[1])
    ci = [k for k, j in enumerate(active) if info[j].kind == 'cons']
    if ci and ybar > 0:
        b0[ci[0]] = math.log(ybar) - (off.mean() if off is not None else 0)
    bb, ll, H, lls = _poisson_fit(X, yy, off, b0)
    Vinv = np.linalg.inv(-H)
    mu = np.exp(X @ bb + (off if off is not None else 0))
    S = X * (yy - mu)[:, None]
    Vact, vce, clv, G = _vce_adjust(ip, p, d, mask, Vinv, S, n)
    b, V = full_bV(len(names), active, bb, Vact)
    names = [c.name for c in info]
    slopes = [j for j in active if info[j].kind != 'cons']
    k = len(slopes)
    lr = 2 * (ll - ll0)
    statlab, stat, pv = _header_stat(vce, k, lr, b, V, slopes)
    r2p = 1 - ll / ll0 if ll0 and not np.isnan(ll0) else np.nan
    est = {'nomargins': True, 'names': names, 'b': b, 'V': V, 'colinfo': info, 'df_r': None, 'depvar': yv, 'sample': mask,
           'cmd': 'poisson', 'stat': 'z', 'vce': vce,
           'scalars': {'N': n, 'll': ll, 'll_0': ll0, 'chi2': stat, 'r2_p': r2p, 'df_m': k, 'p': pv,
                       'k': len(active)},
           'macros': {'cmd': 'poisson', 'depvar': yv, 'title': 'Poisson regression', 'vce': vce,
                      'offset': offname or '', 'chi2type': 'Wald' if vce != 'ols' else 'LR'}}
    if G:
        est['scalars']['N_clust'] = G
        est['macros']['clustvar'] = clv
    est['xb_offset_var'] = (off_all if off_all is not None else None)
    irr = p.opts.has('irr')
    pseudo = vce != 'ols'
    note, rob = _robust_label(vce, clv, G)

    def replay(ip_, show_iter=False):
        if show_iter and not (p.opts.has('nolog') or ip_.quiet):
            _print_log(ip_, lls, pseudo=pseudo)
        else:
            ip_.txt('\n')
        word = 'Log pseudolikelihood' if pseudo else 'Log likelihood'
        right_lines(ip_, ['Poisson regression', '', '', f'{word} = {g_fmt(ll, 10)}'],
                    [('Number of obs', comma_int(n)), (statlab, f'{stat:.2f}'),
                     ('Prob > chi2', f'{pv:.4f}'), ('Pseudo R2', f'{r2p:.4f}')])
        ip_.txt('\n')
        if note:
            ip_.txt(f'{note:>78}\n')
        coef_table(ip_, est, 'z', robust=rob, eform=irr, footer=not offname,
                   colhead=_eform_head('IRR', rob) if irr else None)
        if offname:
            ip_.txt(f'{abbrev(offname, 12):>12} |')
            ip_.res(f'{"1":>11}')
            ip_.txt(f'  ({offkind})\n')
            ip_.txt('-' * 78 + '\n')
        if irr and cons:
            ip_.txt('Note: _cons estimates baseline incidence rate.\n')

    def predict(ip_, kind, m_):
        xb = _predict_linear(ip_, est, 'xb')
        if est['xb_offset_var'] is not None and kind != 'xb':
            xb = xb + est['xb_offset_var']
        if kind in (None, '', 'n'):
            return np.exp(xb)
        if kind == 'ir':
            return np.exp(_predict_linear(ip_, est, 'xb'))
        if kind == 'xb':
            return xb
        raise StataError(f'predict option {kind} not supported', 198)
    est['replay'] = lambda ip_: replay(ip_, False)
    est['predict'] = predict
    _store_rhs(est, toks[1:], cols, d)
    ip.set_estimation(est)
    replay(ip, True)


# ================================================================ nbreg (NB2, mean dispersion)
def _nb_fun(X, y, off):
    offv = off if off is not None else 0.0
    k = X.shape[1]
    yint = bool(np.all(y == np.round(y))) and (y.max() if y.size else 0) <= 20000
    ymax = int(y.max()) if yint and y.size else 0
    ks = [(j, y > j) for j in range(ymax)]
    lgy1 = spc.gammaln(y + 1)

    def parts(theta):
        b, la = theta[:k], theta[k]
        la = float(np.clip(la, -30, 30))
        a = math.exp(la)
        xb = X @ b + offv
        mu = np.exp(xb)
        l1 = np.log1p(a * mu)
        gx = (y - mu) / (1 + a * mu)
        if yint:
            # gammaln(y+m)-gammaln(m) = -y*log(a) + sum_{j<y} log1p(a*j): no cancellation as alpha -> 0
            slog = np.zeros_like(y)
            sinv = np.zeros_like(y)
            for j, msk in ks:
                slog[msk] += math.log1p(a * j)
                sinv[msk] += 1.0 / (1.0 + a * j)
            ll = np.sum(slog - lgy1 + y * xb - (1 / a + y) * l1)
            # d ll / d lnalpha = [log1p(a mu)/a - sum 1/(1+aj) + (y-mu)/(1+a mu)]
            if a * max(float(mu.max()), float(y.max()) if y.size else 0.0, 1.0) < 1e-5:
                gl = a * ((y - mu) ** 2 - y) / 2
            else:
                gl = l1 / a - sinv + (y - mu) / (1 + a * mu)
        else:
            m = 1 / a
            ll = np.sum(spc.gammaln(y + m) - lgy1 - spc.gammaln(m) - m * l1 + y * (np.log(a * mu) - l1))
            gl = a * (m * m * (l1 - spc.digamma(y + m) + spc.digamma(m)) + (y - mu) / (a * (1 + a * mu)))
        return float(ll), gx, gl

    def grad(theta):
        _, gx, gl = parts(theta)
        return np.concatenate([X.T @ gx, [gl.sum()]])

    def fun(theta):
        ll, gx, gl = parts(theta)
        g = np.concatenate([X.T @ gx, [gl.sum()]])
        return ll, g, _num_hess(grad, theta)

    def scores(theta):
        _, gx, gl = parts(theta)
        return np.column_stack([X * gx[:, None], gl])
    return fun, scores


@command('nbreg')
def cmd_nbreg(ip, cmd, rest, by):
    p = parse_command(rest)
    toks = split_top(p.body)
    disp = (p.opts.get('dispersion', 1) or 'mean').strip()
    if not disp.startswith('m'):
        raise StataError('only dispersion(mean) is supported', 198)
    cons = not p.opts.has('noconstant', 4)
    d, yv, yy, cols, mask, Xall, names, info, active = _setup(ip, p, toks[0], toks[1:], cons)
    off_all, offname, offkind = _offset(ip, p, d, mask)
    if off_all is not None:
        if not np.all(~np.isnan(off_all[mask])):
            raise StataError('offset/exposure variable has missing values in the estimation sample', 416)
    off = off_all[mask] if off_all is not None else None
    if np.any(yy < 0):
        raise StataError(f'{yv} must be greater than or equal to zero', 459)
    X = Xall[:, active]
    n, k = X.shape
    quiet_log = p.opts.has('nolog') or ip.quiet
    pseudo = parse_vce(p.opts)[0] != 'ols'
    # 1) Poisson
    ybar = yy.mean()
    b0 = np.zeros(k)
    ci = [j for j, a in enumerate(active) if info[a].kind == 'cons']
    if ci and ybar > 0:
        b0[ci[0]] = math.log(ybar) - (off.mean() if off is not None else 0)
    bp, llp, _, lls_p = _poisson_fit(X, yy, off, b0)
    # 2) constant-only NB
    if cons:
        fun0, _ = _nb_fun(np.ones((n, 1)), yy, off)
        th0 = np.array([math.log(ybar) - (off.mean() if off is not None else 0), 0.0])
        th0, ll0, _, lls_0 = _newton(fun0, th0)
    else:
        ll0, lls_0, th0 = np.nan, [], np.array([0.0, 0.0])
    # 3) full model
    fun, scores = _nb_fun(X, yy, off)
    th = np.concatenate([bp, [th0[-1]]])
    th, ll, H, lls = _newton(fun, th)
    Vinv = np.linalg.inv(-H)
    S = scores(th)
    Vall, vce, clv, G = _vce_adjust(ip, p, d, mask, Vinv, S, n)
    K = len(names)
    b = np.zeros(K + 1)
    V = np.zeros((K + 1, K + 1))
    idx = list(active) + [K]
    b[idx] = th
    V[np.ix_(idx, idx)] = Vall
    info = list(info) + [Col(name='/lnalpha', values=None, term=-1, group='', level='', kind='aux', vars=[])]
    names = [c.name for c in info]
    slopes = [j for j in active if info[j].kind != 'cons']
    kk = len(slopes)
    lr = 2 * (ll - ll0)
    statlab, stat, pv = _header_stat(vce, kk, lr, b, V, slopes)
    r2p = 1 - ll / ll0 if cons else np.nan
    lna, se_lna = th[-1], math.sqrt(max(Vall[-1, -1], 0))
    alpha = math.exp(lna)
    z = sps.norm.ppf(0.975)
    chib = max(2 * (ll - llp), 0.0)
    pchib = 0.5 * sps.chi2.sf(chib, 1) if chib > 0 else 1.0
    est = {'nomargins': True, 'names': names, 'b': b, 'V': V, 'colinfo': info, 'df_r': None, 'depvar': yv, 'sample': mask,
           'cmd': 'nbreg', 'stat': 'z', 'vce': vce,
           'scalars': {'N': n, 'll': ll, 'll_0': ll0, 'll_c': llp, 'chi2': stat, 'chi2_c': chib, 'r2_p': r2p,
                       'df_m': kk, 'p': pv, 'alpha': alpha, 'k': len(idx)},
           'macros': {'cmd': 'nbreg', 'depvar': yv, 'title': 'Negative binomial regression', 'vce': vce,
                      'dispers': 'mean', 'offset': offname or ''}}
    if G:
        est['scalars']['N_clust'] = G
        est['macros']['clustvar'] = clv
    est['xb_offset_var'] = off_all
    irr = p.opts.has('irr')
    note, rob = _robust_label(vce, clv, G)

    def replay(ip_, show_iter=False):
        if show_iter and not quiet_log:
            _print_log(ip_, lls_p, 'Fitting Poisson model:', pseudo)
            if cons:
                _print_log(ip_, lls_0, 'Fitting constant-only model:', pseudo)
            _print_log(ip_, lls, 'Fitting full model:', pseudo)
        else:
            ip_.txt('\n')
        word = 'Log pseudolikelihood' if pseudo else 'Log likelihood'
        right_lines(ip_, ['Negative binomial regression', '', 'Dispersion: mean', f'{word} = {g_fmt(ll, 10)}'],
                    [('Number of obs', comma_int(n)), (statlab, f'{stat:.2f}'),
                     ('Prob > chi2', f'{pv:.4f}'), ('Pseudo R2', f'{r2p:.4f}')])
        ip_.txt('\n')
        if note:
            ip_.txt(f'{note:>78}\n')
        aux = [[('/lnalpha', lna, se_lna, False, None, None)],
               [('alpha', alpha, alpha * se_lna, False, _sexp(lna - z * se_lna), _sexp(lna + z * se_lna))]]
        coef_table(ip_, est, 'z', robust=rob, eform=irr, aux=aux,
                   colhead=_eform_head('IRR', rob) if irr else None)
        if irr and cons:
            ip_.txt('Note: Estimates are transformed only in the first equation to incidence-rate ratios.\n')
            ip_.txt('Note: _cons estimates baseline incidence rate.\n')
        if vce == 'ols':
            ip_.txt(f'LR test of alpha=0: chibar2(01) = ')
            ip_.res(f'{chib:.2f}')
            ip_.txt(f'{"":<16}Prob >= chibar2 = ')
            ip_.res(f'{pchib:.3f}\n')

    def predict(ip_, kind, m_):
        xb = _predict_linear(ip_, est, 'xb')
        if est['xb_offset_var'] is not None and kind != 'xb':
            xb = xb + est['xb_offset_var']
        if kind in (None, '', 'n'):
            return np.exp(xb)
        if kind == 'ir':
            return np.exp(_predict_linear(ip_, est, 'xb'))
        if kind == 'xb':
            return xb
        raise StataError(f'predict option {kind} not supported', 198)
    est['replay'] = lambda ip_: replay(ip_, False)
    est['predict'] = predict
    _store_rhs(est, toks[1:], cols, d)
    ip.set_estimation(est)
    replay(ip, True)


# ================================================================ tobit
@command('tobit')
def cmd_tobit(ip, cmd, rest, by):
    p = parse_command(rest)
    toks = split_top(p.body)
    cons = not p.opts.has('noconstant', 4)
    d, yv, yy, cols, mask, Xall, names, info, active = _setup(ip, p, toks[0], toks[1:], cons)
    X = Xall[:, active]
    n, k = X.shape
    llo = p.opts.get('ll')
    ulo = p.opts.get('ul')
    if llo is None and ulo is None:
        raise StataError('must specify censoring point', 198)
    lo = (float(yy.min()) if llo == '' else float(E.eval_expr(llo, E.Ctx(ip, d)))) if llo is not None else -np.inf
    hi = (float(yy.max()) if ulo == '' else float(E.eval_expr(ulo, E.Ctx(ip, d)))) if ulo is not None else np.inf
    left = yy <= lo
    right = yy >= hi
    unc = ~left & ~right
    if not unc.any():
        raise StataError('no uncensored observations', 2000)

    def make(Xm):
        km = Xm.shape[1]

        def parts(theta):
            b, ls = theta[:km], float(np.clip(theta[km], -30, 30))
            s = math.exp(ls)
            xb = Xm @ b
            ll = 0.0
            gx = np.zeros(n)
            gs = np.zeros(n)
            if unc.any():
                r = (yy[unc] - xb[unc]) / s
                ll += np.sum(sps.norm.logpdf(r) - ls)
                gx[unc] = r / s
                gs[unc] = r * r - 1
            if left.any():
                a = (lo - xb[left]) / s
                lc = sps.norm.logcdf(a)
                ll += np.sum(lc)
                lam = np.exp(sps.norm.logpdf(a) - lc)
                gx[left] = -lam / s
                gs[left] = -lam * a
            if right.any():
                a = (xb[right] - hi) / s
                lc = sps.norm.logcdf(a)
                ll += np.sum(lc)
                lam = np.exp(sps.norm.logpdf(a) - lc)
                gx[right] = lam / s
                gs[right] = -lam * a
            return float(ll), gx, gs

        def grad(theta):
            _, gx, gs = parts(theta)
            return np.concatenate([Xm.T @ gx, [gs.sum()]])

        def fun(theta):
            ll, gx, gs = parts(theta)
            return ll, np.concatenate([Xm.T @ gx, [gs.sum()]]), _num_hess(grad, theta)

        def scores(theta):
            _, gx, gs = parts(theta)
            return np.column_stack([Xm * gx[:, None], gs])
        return fun, scores
    # OLS start
    bo = np.linalg.lstsq(X, yy, rcond=None)[0]
    so = max(np.std(yy - X @ bo), 1e-6)
    fun, scores = make(X)
    th, ll, H, lls = _newton(fun, np.concatenate([bo, [math.log(so)]]))
    if cons:
        fun0, _ = make(np.ones((n, 1)))
        _, ll0, _, _ = _newton(fun0, np.array([yy.mean(), math.log(max(yy.std(), 1e-6))]))
    else:
        ll0 = np.nan
    Vinv = np.linalg.inv(-H)
    Vall, vce, clv, G = _vce_adjust(ip, p, d, mask, Vinv, scores(th), n)
    K = len(names)
    b = np.zeros(K + 1)
    V = np.zeros((K + 1, K + 1))
    idx = list(active) + [K]
    b[idx] = th
    V[np.ix_(idx, idx)] = Vall
    info = list(info) + [Col(name='/lnsigma', values=None, term=-1, group='', level='', kind='aux', vars=[])]
    names = [c.name for c in info]
    slopes = [j for j in active if info[j].kind != 'cons']
    kk = len(slopes)
    lr = 2 * (ll - ll0)
    statlab, stat, pv = _header_stat(vce, kk, lr, b, V, slopes)
    r2p = 1 - ll / ll0 if cons else np.nan
    ls, se_ls = th[-1], math.sqrt(max(Vall[-1, -1], 0))
    sig2 = math.exp(2 * ls)
    zc = sps.norm.ppf(0.975)
    nunc, nl, nr = int(unc.sum()), int(left.sum()), int(right.sum())
    est = {'nomargins': True, 'names': names, 'b': b, 'V': V, 'colinfo': info, 'df_r': None, 'depvar': yv, 'sample': mask,
           'cmd': 'tobit', 'stat': 'z', 'vce': vce,
           'scalars': {'N': n, 'N_unc': nunc, 'N_lc': nl, 'N_rc': nr, 'll': ll, 'll_0': ll0, 'chi2': stat,
                       'r2_p': r2p, 'df_m': kk, 'p': pv, 'sigma': math.sqrt(sig2),
                       'llopt': lo if np.isfinite(lo) else E.MISS, 'ulopt': hi if np.isfinite(hi) else E.MISS},
           'macros': {'cmd': 'tobit', 'depvar': yv, 'title': 'Tobit regression', 'vce': vce}}
    if G:
        est['scalars']['N_clust'] = G
        est['macros']['clustvar'] = clv
    note, rob = _robust_label(vce, clv, G)
    pseudo = vce != 'ols'
    lo_s = g_fmt(lo) if np.isfinite(lo) else '-inf'
    hi_s = g_fmt(hi) if np.isfinite(hi) else '+inf'

    def replay(ip_, show_iter=False):
        if show_iter and not (p.opts.has('nolog') or ip_.quiet):
            _print_log(ip_, lls, 'Fitting full model:', pseudo)
        else:
            ip_.txt('\n')
        word = 'Log pseudolikelihood' if pseudo else 'Log likelihood'
        def row(left, key, val):
            ip_.txt(f'{left:<48}{key:<17} = ' if key == 'Number of obs' else f'{left:<48}{key:>17} = ')
            ip_.res(f'{val:>10}\n')
        row('Tobit regression', 'Number of obs', comma_int(n))
        row('', 'Uncensored', comma_int(nunc))
        row('Limits: Lower = ' + lo_s, 'Left-censored', comma_int(nl))
        row('        Upper = ' + hi_s, 'Right-censored', comma_int(nr))
        ip_.txt('\n')
        ip_.txt(f'{"":<48}{statlab:<17} = ')
        ip_.res(f'{stat:>10.2f}\n')
        ip_.txt(f'{"":<48}{"Prob > chi2":<17} = ')
        ip_.res(f'{pv:>10.4f}\n')
        ip_.txt(f'{word + " = " + g_fmt(ll, 10):<48}{"Pseudo R2":<17} = ')
        ip_.res(f'{r2p:>10.4f}\n\n')
        if note:
            ip_.txt(f'{note:>78}\n')
        aux = [[(f'var(e.{yv})', sig2, 2 * sig2 * se_ls, False,
                 _sexp(2 * (ls - zc * se_ls)), _sexp(2 * (ls + zc * se_ls)))]]
        coef_table(ip_, est, 'z', robust=rob, aux=aux)

    def predict(ip_, kind, m_):
        xb = _predict_linear(ip_, est, 'xb')
        if kind in (None, '', 'xb'):
            return xb
        s = math.sqrt(sig2)
        if kind in ('ystar', 'e'):
            raise StataError(f'predict option {kind}() requires arguments; use xb', 198)
        raise StataError(f'predict option {kind} not supported', 198)
    est['replay'] = lambda ip_: replay(ip_, False)
    est['predict'] = predict
    _store_rhs(est, toks[1:], cols, d)
    ip.set_estimation(est)
    replay(ip, True)


# ================================================================ ologit / oprobit
def _cat_labels(d, v, cats):
    lab = d.vallab.get(v)
    tab = d.labels.get(lab, {}) if lab else {}
    return [tab.get(float(c), g_fmt(c)) for c in cats]


@command('ologit', 'oprobit')
def cmd_ologit(ip, cmd, rest, by):
    p = parse_command(rest)
    toks = split_top(p.body)
    d, yv, yy, cols, mask, Xall, names, info, active = _setup(ip, p, toks[0], toks[1:], cons=True)
    # drop the constant: identified through cutpoints
    act = [j for j in active if info[j].kind != 'cons']
    info = [c for c in info if c.kind != 'cons']
    names = [c.name for c in info]
    X = Xall[:, act]
    n, k = X.shape
    cats = np.unique(yy)
    J = len(cats)
    if J < 2:
        raise StataError('outcome does not vary', 2000)
    if J > 50:
        raise StataError('too many outcomes', 149)
    yi = np.searchsorted(cats, yy)
    probit = cmd == 'oprobit'
    F = (lambda z: sps.norm.cdf(z)) if probit else (lambda z: spc.expit(z))
    f = (lambda z: sps.norm.pdf(z)) if probit else (lambda z: spc.expit(z) * (1 - spc.expit(z)))

    def parts(theta):
        b, c = theta[:k], theta[k:]
        xb = X @ b
        cu = np.concatenate([[-np.inf], c, [np.inf]])
        a = cu[yi + 1] - xb
        bl = cu[yi] - xb
        Pa = np.where(np.isfinite(a), F(np.where(np.isfinite(a), a, 0)), np.where(a > 0, 1.0, 0.0))
        Pb = np.where(np.isfinite(bl), F(np.where(np.isfinite(bl), bl, 0)), np.where(bl > 0, 1.0, 0.0))
        P = np.maximum(Pa - Pb, 1e-300)
        fa = np.where(np.isfinite(a), f(np.where(np.isfinite(a), a, 0)), 0.0)
        fb = np.where(np.isfinite(bl), f(np.where(np.isfinite(bl), bl, 0)), 0.0)
        ll = float(np.sum(np.log(P)))
        gx = -(fa - fb) / P
        S = np.zeros((n, k + J - 1))
        S[:, :k] = X * gx[:, None]
        rows = np.arange(n)
        up = yi < J - 1
        S[rows[up], k + yi[up]] += fa[up] / P[up]
        lw = yi > 0
        S[rows[lw], k + yi[lw] - 1] -= fb[lw] / P[lw]
        return ll, S

    def grad(theta):
        return parts(theta)[1].sum(axis=0)

    def fun(theta):
        ll, S = parts(theta)
        if np.any(np.diff(theta[k:]) <= 0):
            return -np.inf, S.sum(axis=0), -np.eye(len(theta))
        return ll, S.sum(axis=0), _num_hess(grad, theta)
    cum = np.cumsum(np.bincount(yi, minlength=J))[:-1] / n
    finv = (lambda q: sps.norm.ppf(q)) if probit else (lambda q: spc.logit(q))
    th0 = np.concatenate([np.zeros(k), finv(cum)])
    cnt = np.bincount(yi, minlength=J)
    ll0 = float(np.sum(cnt * np.log(cnt / n)))
    th, ll, H, lls = _newton(fun, th0)
    lls[0] = ll0
    Vinv = np.linalg.inv(-H)
    _, S = parts(th)
    Vall, vce, clv, G = _vce_adjust(ip, p, d, mask, Vinv, S, n)
    K = len(names)
    cutn = [f'/cut{j + 1}' for j in range(J - 1)]
    b = np.zeros(K + J - 1)
    V = np.zeros((K + J - 1, K + J - 1))
    actn = [info.index(c) for c in info if not c.base and not c.omitted]
    idx = actn + list(range(K, K + J - 1))
    b[idx] = th
    V[np.ix_(idx, idx)] = Vall
    info = info + [Col(name=c, values=None, term=-1, group='', level='', kind='aux', vars=[]) for c in cutn]
    names = [c.name for c in info]
    slopes = actn
    lr = 2 * (ll - ll0)
    statlab, stat, pv = _header_stat(vce, k, lr, b, V, slopes)
    r2p = 1 - ll / ll0
    title = 'Ordered probit regression' if probit else 'Ordered logistic regression'
    est = {'names': names, 'b': b, 'V': V, 'colinfo': info, 'df_r': None, 'depvar': yv, 'sample': mask,
           'cmd': cmd, 'stat': 'z', 'vce': vce,
           'scalars': {'N': n, 'll': ll, 'll_0': ll0, 'chi2': stat, 'r2_p': r2p, 'df_m': k, 'p': pv,
                       'k_cat': J, 'k_aux': J - 1},
           'macros': {'cmd': cmd, 'depvar': yv, 'title': title, 'vce': vce},
           'matrices': {'cat': E.Matrix(cats.reshape(1, -1).astype(float), ['r1'],
                                        [f'c{j + 1}' for j in range(J)])}}
    if G:
        est['scalars']['N_clust'] = G
        est['macros']['clustvar'] = clv
    note, rob = _robust_label(vce, clv, G)
    pseudo = vce != 'ols'
    orr = p.opts.has('or') and not probit
    cuts = th[k:]
    se_c = np.sqrt(np.maximum(np.diag(Vall)[k:], 0))

    def replay(ip_, show_iter=False):
        if show_iter and not (p.opts.has('nolog') or ip_.quiet):
            _print_log(ip_, lls, pseudo=pseudo)
        else:
            ip_.txt('\n')
        word = 'Log pseudolikelihood' if pseudo else 'Log likelihood'
        right_lines(ip_, [title, '', '', f'{word} = {g_fmt(ll, 10)}'],
                    [('Number of obs', comma_int(n)), (statlab, f'{stat:.2f}'),
                     ('Prob > chi2', f'{pv:.4f}'), ('Pseudo R2', f'{r2p:.4f}')])
        ip_.txt('\n')
        if note:
            ip_.txt(f'{note:>78}\n')
        aux = [[(cutn[j], cuts[j], se_c[j], False, None, None) for j in range(J - 1)]]
        coef_table(ip_, est, 'z', robust=rob, aux=aux, eform=orr,
                   colhead=_eform_head('Odds ratio', rob) if orr else None)
        if orr:
            ip_.txt('Note: Estimates are transformed only in the first equation to odds ratios.\n')

    def probs(ip_):
        xb = _predict_linear(ip_, {**est, 'names': [nm for nm in names if not nm.startswith('/')]}, 'xb')
        cu = np.concatenate([[-np.inf], cuts, [np.inf]])
        out = []
        for j in range(J):
            a = cu[j + 1] - xb
            bl = cu[j] - xb
            Pa = np.ones_like(xb) if j == J - 1 else F(a)
            Pb = np.zeros_like(xb) if j == 0 else F(bl)
            out.append(Pa - Pb)
        return xb, out

    def predict_multi(ip_, pp, typ, tk):
        tk = _stub(tk, J)
        kinds = pp.opts.names()
        kind = 'pr' if not kinds or kinds == ['outcome'] else kinds[0]
        xb, P = probs(ip_)
        if kind == 'xb':
            return _finish_predict(ip_, pp, typ, tk[:1], xb, 'Linear prediction (cutpoints excluded)')
        if kind != 'pr':
            raise StataError(f'predict option {kind} not supported', 198)
        oc = pp.opts.get('outcome')
        labs = _cat_labels(d, yv, cats)
        if oc:
            j = _outcome_index(oc, cats, labs)
            return _finish_predict(ip_, pp, typ, tk[:1], P[j], f'Pr({yv}=={g_fmt(cats[j])})')
        if len(tk) == 1:
            raise StataError(f'too few variables specified; {J} new variables required (or use outcome())', 102)
        if len(tk) != J:
            raise StataError(f'{J} new variables required', 102 if len(tk) < J else 103)
        for j in range(J):
            _finish_predict(ip_, pp, typ, [tk[j]], P[j], f'Pr({yv}=={g_fmt(cats[j])})')
    est['replay'] = lambda ip_: replay(ip_, False)
    est['predict'] = None
    est['predict_multi'] = predict_multi
    est['nomargins'] = True
    _store_rhs(est, toks[1:], cols, d)
    ip.set_estimation(est)
    replay(ip, True)


def _stub(tk, J):
    """predict stub*, pr -> stub1 ... stubJ"""
    if len(tk) == 1 and tk[0].endswith('*'):
        return [f'{tk[0][:-1]}{j + 1}' for j in range(J)]
    return tk


def _outcome_index(oc, cats, labs):
    oc = oc.strip()
    if oc.startswith('#'):
        j = int(oc[1:]) - 1
        if 0 <= j < len(cats):
            return j
    for j, c in enumerate(cats):
        if g_fmt(c) == oc or labs[j] == oc:
            return j
    try:
        v = float(oc)
        for j, c in enumerate(cats):
            if c == v:
                return j
    except ValueError:
        pass
    raise StataError(f'outcome {oc} not found', 303)


# ================================================================ mlogit
@command('mlogit')
def cmd_mlogit(ip, cmd, rest, by):
    p = parse_command(rest)
    toks = split_top(p.body)
    cons = not p.opts.has('noconstant', 4)
    d, yv, yy, cols, mask, Xall, names0, info0, active = _setup(ip, p, toks[0], toks[1:], cons)
    X = Xall[:, active]
    n, k = X.shape
    cats = np.unique(yy)
    J = len(cats)
    if J < 2:
        raise StataError('outcome does not vary', 2000)
    if J > 50:
        raise StataError('too many outcomes', 149)
    yi = np.searchsorted(cats, yy)
    cnt = np.bincount(yi, minlength=J)
    labs = _cat_labels(d, yv, cats)
    bo = p.opts.get('baseoutcome', 1)
    base = _outcome_index(bo, cats, labs) if bo else int(np.argmax(cnt))
    others = [j for j in range(J) if j != base]
    Y = np.zeros((n, J))
    Y[np.arange(n), yi] = 1

    def fun(theta):
        B = np.zeros((k, J))
        B[:, others] = theta.reshape(J - 1, k).T
        U = X @ B
        U -= U.max(axis=1, keepdims=True)
        P = np.exp(U)
        P /= P.sum(axis=1, keepdims=True)
        ll = float(np.sum(np.log(np.maximum(P[np.arange(n), yi], 1e-300))))
        R = (Y - P)[:, others]
        g = (X.T @ R).T.reshape(-1)
        m = J - 1
        H = np.zeros((m * k, m * k))
        Po = P[:, others]
        for a in range(m):
            for c in range(a, m):
                wgt = Po[:, a] * ((1.0 if a == c else 0.0) - Po[:, c])
                blk = -(X.T @ (X * wgt[:, None]))
                H[a * k:(a + 1) * k, c * k:(c + 1) * k] = blk
                H[c * k:(c + 1) * k, a * k:(a + 1) * k] = blk.T
        return ll, g, H
    ll0 = float(np.sum(cnt[cnt > 0] * np.log(cnt[cnt > 0] / n)))
    th, ll, H, lls = _newton(fun, np.zeros((J - 1) * k))
    if cons:
        lls[0] = ll0
    Vinv = np.linalg.inv(-H)
    B = np.zeros((k, J))
    B[:, others] = th.reshape(J - 1, k).T
    U = X @ B
    U -= U.max(axis=1, keepdims=True)
    P = np.exp(U)
    P /= P.sum(axis=1, keepdims=True)
    R = (Y - P)[:, others]
    S = np.column_stack([X * R[:, a][:, None] for a in range(J - 1)])
    Vall, vce, clv, G = _vce_adjust(ip, p, d, mask, Vinv, S, n)
    K0 = len(names0)
    eqn = [abbrev(lb.replace(' ', '_'), 32) for lb in labs]
    names, info, b, idx = [], [], [], []
    for a, j in enumerate(others):
        for c in info0:
            nc = Col(name=f'{eqn[j]}:{c.name}', values=c.values, term=c.term, group=c.group, level=c.level,
                     kind=c.kind, vars=c.vars)
            nc.base, nc.omitted, nc.compact = c.base, c.omitted, c.compact
            info.append(nc)
            names.append(nc.name)
    bfull = np.zeros(len(names))
    Vfull = np.zeros((len(names), len(names)))
    pos = []
    for a in range(J - 1):
        pos += [a * K0 + j for j in active]
    bfull[pos] = th
    Vfull[np.ix_(pos, pos)] = Vall
    slopes = [a * K0 + j for a in range(J - 1) for j in active if info0[j].kind != 'cons']
    kk = len(slopes)
    lr = 2 * (ll - ll0)
    statlab, stat, pv = _header_stat(vce, kk, lr, bfull, Vfull, slopes)
    r2p = 1 - ll / ll0 if cons else np.nan
    est = {'names': names, 'b': bfull, 'V': Vfull, 'colinfo': info, 'df_r': None, 'depvar': yv, 'sample': mask,
           'cmd': 'mlogit', 'stat': 'z', 'vce': vce,
           'scalars': {'N': n, 'll': ll, 'll_0': ll0, 'chi2': stat, 'r2_p': r2p, 'df_m': kk, 'p': pv,
                       'k_out': J, 'ibaseout': base + 1, 'baseout': float(cats[base])},
           'macros': {'cmd': 'mlogit', 'depvar': yv, 'title': 'Multinomial logistic regression', 'vce': vce,
                      'eqnames': ' '.join(eqn)},
           'matrices': {'out': E.Matrix(cats.reshape(1, -1).astype(float), ['r1'],
                                        [f'o{j + 1}' for j in range(J)])}}
    if G:
        est['scalars']['N_clust'] = G
        est['macros']['clustvar'] = clv
    note, rob = _robust_label(vce, clv, G)
    pseudo = vce != 'ols'
    rrr = p.opts.has('rrr')

    def replay(ip_, show_iter=False):
        if show_iter and not (p.opts.has('nolog') or ip_.quiet):
            _print_log(ip_, lls, pseudo=pseudo)
        else:
            ip_.txt('\n')
        word = 'Log pseudolikelihood' if pseudo else 'Log likelihood'
        right_lines(ip_, ['Multinomial logistic regression', '', '', f'{word} = {g_fmt(ll, 10)}'],
                    [('Number of obs', comma_int(n)), (statlab, f'{stat:.2f}'),
                     ('Prob > chi2', f'{pv:.4f}'), ('Pseudo R2', f'{r2p:.4f}')])
        ip_.txt('\n')
        if note:
            ip_.txt(f'{note:>78}\n')
        first = True
        for j in range(J):
            last = j == J - 1
            if j == base:
                if first:
                    _mlogit_head(ip_, yv, rob, rrr)
                    first = False
                ip_.txt('-' * 13 + '+' + '-' * 64 + '\n')
                ip_.res(f'{abbrev(eqn[j], 12):<12}')
                ip_.txt(' |  (base outcome)\n')
                if last:
                    ip_.txt('-' * 78 + '\n')
                continue
            a = others.index(j)
            sl = list(range(a * K0, (a + 1) * K0))
            sub = {'names': [info0[i].name for i in range(K0)], 'b': bfull[sl], 'V': Vfull[np.ix_(sl, sl)],
                   'colinfo': [info[i] for i in sl], 'df_r': None}
            for i, c in zip(range(K0), sub['colinfo']):
                pass
            sub['colinfo'] = [_rename(info[s], info0[i].name) for i, s in zip(range(K0), sl)]
            if first:
                _mlogit_head(ip_, yv, rob, rrr)
                first = False
            coef_table(ip_, sub, 'z', robust=rob, header=False, footer=last, eq_title=abbrev(eqn[j], 12),
                       eform=rrr)
        if rrr and cons:
            ip_.txt('Note: _cons estimates baseline relative risk for each outcome.\n')

    def predict_multi(ip_, pp, typ, tk):
        tk = _stub(tk, J)
        kinds = pp.opts.names()
        kind = 'pr' if not kinds or kinds == ['outcome'] else kinds[0]
        dd = ip_.data
        Xn_cols = []
        xbs = []
        for a, j in enumerate(others):
            sub = {**est, 'names': [info0[i].name for i in range(K0)], 'b': bfull[a * K0:(a + 1) * K0]}
            xbs.append(_predict_linear(ip_, sub, 'xb'))
        XB = np.zeros((dd.n, J))
        for a, j in enumerate(others):
            XB[:, j] = xbs[a]
        miss = np.isnan(XB).any(axis=1)
        XB = np.nan_to_num(XB)
        XB -= XB.max(axis=1, keepdims=True)
        PP = np.exp(XB)
        PP /= PP.sum(axis=1, keepdims=True)
        PP[miss] = np.nan
        oc = pp.opts.get('outcome')
        if kind == 'xb':
            j = _outcome_index(oc, cats, labs) if oc else others[0]
            return _finish_predict(ip_, pp, typ, tk[:1], xbs[others.index(j)] if j != base else np.zeros(dd.n),
                                   'Linear prediction')
        if kind != 'pr':
            raise StataError(f'predict option {kind} not supported', 198)
        if oc:
            j = _outcome_index(oc, cats, labs)
            return _finish_predict(ip_, pp, typ, tk[:1], PP[:, j], f'Pr({yv}=={labs[j]})')
        if len(tk) != J:
            raise StataError(f'{J} new variables required (or use outcome())', 102 if len(tk) < J else 103)
        for j in range(J):
            _finish_predict(ip_, pp, typ, [tk[j]], PP[:, j], f'Pr({yv}=={labs[j]})')
    est['replay'] = lambda ip_: replay(ip_, False)
    est['predict'] = None
    est['predict_multi'] = predict_multi
    est['nomargins'] = True
    _store_rhs(est, toks[1:], cols, d)
    ip.set_estimation(est)
    replay(ip, True)


def _rename(c, name):
    nc = Col(name=name, values=c.values, term=c.term, group=c.group, level=c.level, kind=c.kind, vars=c.vars)
    nc.base, nc.omitted, nc.compact = c.base, c.omitted, c.compact
    return nc


def _mlogit_head(ip, yv, rob, rrr):
    ip.txt('-' * 78 + '\n')
    if rob:
        ip.txt(f'{"":>12} |{rob:>21}\n')
    lab = _eform_head('RRR', rob) if rrr else ('Coefficient  std. err.' if rob else 'Coefficient  Std. err.')
    ip.txt(f'{abbrev(yv, 12):>12} | {lab}      z    P>|z|     [95% conf. interval]\n')


# ================================================================ ppmlhdfe (Correia, Guimarães, Zylkin)
@command('ppmlhdfe')
def cmd_ppmlhdfe(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    toks = split_top(p.body)
    absorb = p.opts.get('absorb', 1)
    fe_vars = d.expand_varlist([t.replace('i.', '') for t in absorb.split()]) if absorb else []
    yv, y, cols, mask, w, wtype = prepare_linear(ip, p, d, toks[1:], toks[0])
    if w is not None:
        raise StataError('weights not supported in this version of ppmlhdfe', 101)
    for v in fe_vars:
        mask &= (d.cols[v] != '') if d.isstr(v) else ~np.isnan(d.cols[v])
    vce, clv = parse_vce(p.opts)
    if vce == 'ols':
        vce = 'robust'
    if clv:
        cv = d.resolve(clv)
        mask &= (d.cols[cv] != '') if d.isstr(cv) else ~np.isnan(d.cols[cv])
    off_all, offname, _ = _offset(ip, p, d, mask)
    if off_all is not None:
        mask &= ~np.isnan(off_all)
    if np.any(y[mask] < 0):
        raise StataError(f'{yv} must be greater than or equal to zero', 459)
    n_start = int(mask.sum())
    # iteratively drop singletons and observations separated by a fixed effect (all-zero groups)
    while fe_vars:
        drop = np.zeros(d.n, dtype=bool)
        idx = np.flatnonzero(mask)
        for v in fe_vars:
            vals = d.cols[v][idx]
            _, inv, cnt = np.unique(vals.astype(str) if d.isstr(v) else vals, return_inverse=True,
                                    return_counts=True)
            ysum = np.bincount(inv, weights=y[idx])
            bad = (cnt[inv] == 1) | (ysum[inv] <= 0)
            drop[idx[bad]] = True
        if not drop.any():
            break
        mask &= ~drop
    ndrop = n_start - int(mask.sum())
    n = int(mask.sum())
    yy = y[mask]
    cons = True
    Xall, names, info = assemble(cols, mask, cons=True)
    fe_ids = []
    for v in fe_vars:
        vals = d.cols[v][mask]
        _, inv = np.unique(vals.astype(str) if d.isstr(v) else vals, return_inverse=True)
        fe_ids.append(inv)
    off = off_all[mask] if off_all is not None else np.zeros(n)
    k_all = Xall.shape[1]
    Xs = Xall[:, :k_all - 1] if fe_ids else Xall

    subit = [0]

    def demean(Z, wt):
        if not fe_ids:
            return Z.copy()
        Z = Z.copy()
        for _ in range(10000):
            subit[0] += 1
            old = Z.copy()
            for g in fe_ids:
                sw = np.bincount(g, weights=wt)
                m = np.column_stack([np.bincount(g, weights=wt * Z[:, j]) for j in range(Z.shape[1])]) / sw[:, None]
                Z = Z - m[g]
            if len(fe_ids) == 1 or np.max(np.abs(Z - old)) < 1e-11 * (1 + np.max(np.abs(Z))):
                break
        return Z
    # collinearity among regressors after partialling FE (unit weights)
    Xt0 = demean(Xs, np.ones(n))
    cand = [j for j in range(Xs.shape[1]) if not info[j].base]
    om = mark_collinear(Xt0[:, cand], None) if cand else []
    for j, o in zip(cand, om):
        if o:
            info[j].omitted = True
            info[j].name = _omit_name(info[j].name) if info[j].kind == 'factor' else 'o.' + info[j].name
    active = [j for j in range(Xs.shape[1]) if not info[j].base and not info[j].omitted]
    X = Xs[:, active]
    mu = (yy + yy.mean()) / 2
    eta = np.log(mu)
    dev_old = np.inf
    it_log = []
    bb = np.zeros(len(active))
    for it in range(1, 1001):
        z = eta - off + (yy - mu) / mu
        M = demean(np.column_stack([z, X]), mu)
        zt, Xt = M[:, 0], M[:, 1:]
        if Xt.shape[1]:
            WX = Xt * mu[:, None]
            bb = np.linalg.solve(Xt.T @ WX, WX.T @ zt)
            resid = zt - Xt @ bb
        else:
            resid = zt
        eta = z - resid + off
        mu = np.exp(eta)
        with np.errstate(divide='ignore', invalid='ignore'):
            dev = 2 * float(np.sum(np.where(yy > 0, yy * np.log(yy / mu), 0) - (yy - mu)))
        eps = abs(dev - dev_old) / max(min(dev, dev_old), 0.1)
        it_log.append((dev, eps))
        dev_old = dev
        if eps < 1e-8:
            break
    else:
        raise StataError('convergence not achieved', 430)
    # constant: recover from fitted values
    if fe_ids:
        info = info[:k_all - 1]
        names = names[:k_all - 1]
    cons_b = None
    if not fe_ids:
        cons_b = None
    Xt = demean(X, mu)
    bread = np.linalg.inv(Xt.T @ (Xt * mu[:, None])) if Xt.shape[1] else np.zeros((0, 0))
    S = Xt * (yy - mu)[:, None]
    G = None
    if vce == 'cluster':
        gid, cname = cluster_ids(d, clv, mask)
        G = int(gid.max()) + 1
        Vact = sandwich(None, S, bread, gid) * G / (G - 1)
        clv = cname
    else:
        Vact = sandwich(None, S, bread) * n / (n - 1)
    if not fe_ids:
        # constant is the last active column; already estimated within X
        pass
    b, V = full_bV(len(names), active, bb, Vact)
    names = [c.name for c in info]
    slopes = [j for j in active if info[j].kind != 'cons']
    kk = len(slopes)
    stat, _, pv = wald_test(b, V, slopes)
    ll = float(np.sum(-mu + yy * np.log(np.where(mu > 0, mu, 1)) - spc.gammaln(yy + 1)))
    ybar = yy.mean()
    ll0 = float(np.sum(-ybar + yy * math.log(ybar) - spc.gammaln(yy + 1)))
    r2p = 1 - ll / ll0
    dev = it_log[-1][0]
    est = {'names': names, 'b': b, 'V': V, 'colinfo': info, 'df_r': None, 'depvar': yv, 'sample': mask,
           'cmd': 'ppmlhdfe', 'stat': 'z', 'vce': vce,
           'scalars': {'N': n, 'll': ll, 'll_0': ll0, 'chi2': stat, 'r2_p': r2p, 'df_m': kk, 'p': pv,
                       'deviance': dev, 'N_full': n_start, 'num_singletons': ndrop},
           'macros': {'cmd': 'ppmlhdfe', 'depvar': yv, 'title': 'HDFE PPML regression', 'vce': vce,
                      'absvars': ' '.join(fe_vars) or '_cons'}}
    if G:
        est['scalars']['N_clust'] = G
        est['macros']['clustvar'] = clv
    note, rob = _robust_label(vce, clv, G)
    irr = p.opts.has('irr') or p.opts.has('eform')

    def replay(ip_, show_iter=False):
        if show_iter and not ip_.quiet:
            if fe_vars and ndrop:
                ip_.txt(f'(dropped {comma_int(ndrop)} observations that are either singletons or separated by a fixed effect)\n')
            if not p.opts.has('nolog'):
                for i, (dv_, ep) in enumerate(it_log, 1):
                    ip_.txt(f'Iteration {i}:{"":<3}deviance = ')
                    ip_.res(f'{dv_:.4e}')
                    ip_.txt('  eps = ')
                    ip_.res(f'{ep:.4e}\n')
                msg = f'Converged in {len(it_log)} iterations'
                if fe_vars:
                    msg += f' and {subit[0]} HDFE sub-iterations'
                ip_.res(msg + ' (tol = 1.0e-08)\n')
        nabs = len(fe_vars)
        left = ['HDFE PPML regression',
                f'Absorbing {nabs} HDFE group{"s" if nabs != 1 else ""}' if nabs else '',
                'Statistics robust to heteroskedasticity',
                f'Deviance             = {g_fmt(dev, 10)}',
                f'Log pseudolikelihood = {g_fmt(ll, 10)}']
        right = [('No. of obs', comma_int(n)), None,
                 (f'Wald chi2({kk})', f'{stat:.2f}'), ('Prob > chi2', f'{pv:.4f}'), ('Pseudo R2', f'{r2p:.4f}')]
        if G:
            right[1] = (f'No. of clusters', comma_int(G))
        if not nabs:
            del left[1]
            if right[1] is None:
                del right[1]
            else:
                left.insert(1, '')
        ip_.txt('\n')
        right_lines(ip_, left, right)
        ip_.txt('\n')
        if note:
            ip_.txt(f'{note:>78}\n')
        coef_table(ip_, est, 'z', robust=rob, eform=irr,
                   colhead=_eform_head('exp(b)', rob) if irr else None)
        if fe_vars:
            ip_.txt('\nAbsorbed degrees of freedom:\n')
            ip_.txt('-----------------------------------------------------+\n')
            ip_.txt(' Absorbed FE | Categories  - Redundant  = Num. Coefs |\n')
            ip_.txt('-------------+---------------------------------------|\n')
            first = True
            marks = []
            gid_c = cluster_ids(d, clv, mask)[0] if G else None
            for v, g in zip(fe_vars, fe_ids):
                L = int(g.max()) + 1
                nested = gid_c is not None and len(np.unique(np.column_stack([g, gid_c]), axis=0)) == L
                red = L if nested else (0 if first else 1)
                mk = '*' if nested else ('?' if not first else ' ')
                first = False
                marks.append(mk)
                ip_.txt(f'{abbrev(v):>12} |')
                ip_.res(f'{L:>11}{red:>13}{L - red:>13}')
                ip_.txt(f' {mk}|\n')
            ip_.txt('-----------------------------------------------------+\n')
            if '?' in marks:
                ip_.txt('? = number of redundant parameters may be higher\n')
            if '*' in marks:
                ip_.txt('* = FE nested within cluster; treated as redundant for DoF computation\n')

    def predict(ip_, kind, m_):
        if kind == 'xb' or kind in (None, '', 'mu'):
            raise StataError('predict after ppmlhdfe: use the d() option of ppmlhdfe (not available in Stata Web)', 198)
        raise StataError(f'predict option {kind} not supported', 198)
    est['replay'] = lambda ip_: replay(ip_, False)
    est['predict'] = predict
    est['nomargins'] = True
    _store_rhs(est, toks[1:], cols, d)
    ip.set_estimation(est)
    replay(ip, True)
