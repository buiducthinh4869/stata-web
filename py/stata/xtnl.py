"""Nonlinear panel models: xtlogit (fe/re), xtprobit (re), xtpoisson (fe/re gamma/normal), xtnbreg (fe/re).
Likelihoods follow the Stata manuals (Methods and formulas); headers follow _crcphdr.ado and
_lrtest_note_xt.ado."""
import math

import numpy as np
from scipy import special as spc
from scipy import stats as sps

from .design import coef_table
from .errors import StataError
from .estim import prepare_linear, assemble, detect_omitted, full_bV, wald_test, _predict_linear, _store_rhs
from .fmt import g_fmt, comma_int, fmt_num
from .interp import command, alias
from .models import _newton, _num_hess, _print_log, _eform_head
from .syntax import parse_command, split_top

alias('xtlogit', 'xtlog', 'xtlogi')
alias('xtprobit', 'xtprob', 'xtprobi')
alias('xtpoisson', 'xtpois')


# ---------------------------------------------------------------- setup
def _panel_setup(ip, p, toks, fe):
    d = ip.data
    if not d.ts or not d.ts.get('panel'):
        raise StataError('panel variable not set; use xtset varname ...', 459)
    ivar = d.ts['panel']
    if len(toks) < 1:
        raise StataError('varlist required', 100)
    cons = not fe and not p.opts.has('noconstant', 4)
    yv, y, cols, mask, w, wtype = prepare_linear(ip, p, d, toks[1:], toks[0])
    if w is not None:
        raise StataError('weights not allowed in Stata Web for this command', 101)
    pid = d.cols[ivar].astype(float)
    mask &= ~np.isnan(pid)
    Xall, names, info = assemble(cols, mask, cons=cons)
    active = detect_omitted(Xall, info, cons=cons)
    yy = y[mask]
    g = pid[mask]
    _, gid = np.unique(g, return_inverse=True)
    p._xtcols = cols
    return d, ivar, yv, yy, gid, cols, mask, Xall, names, info, active, cons


def _within_drop(ip, X, gid, info, active):
    """FE models: drop regressors without within-panel variation (Stata note)."""
    keep = []
    for k, j in enumerate(active):
        x = X[:, j]
        m = np.bincount(gid, x) / np.bincount(gid)
        if np.max(np.abs(x - m[gid])) < 1e-12 * (1 + np.max(np.abs(x))):
            ip.txt(f'note: {info[j].name} omitted because of no within-group variance.\n')
            info[j].omitted = True
        else:
            keep.append(j)
    return keep


def _gstats(gid):
    c = np.bincount(gid)
    c = c[c > 0]
    return int(c.size), int(c.min()), float(c.mean()), int(c.max())


def _header(ip, title, ivar, n, ng, gmin, gavg, gmax, distrib, chi2type, df, chi2, ll, pseudo=False,
            intmethod=None, nq=None):
    ip.txt('\n')
    ip.txt(f'{title:<48}{"Number of obs":<18}= ')
    ip.res(fmt_num(n, '%10.0fc') + '\n')
    ip.txt(f'Group variable: ')
    ip.res(f'{ivar[:14]:<32}')
    ip.txt(f'{"Number of groups":<18}= ')
    ip.res(fmt_num(ng, '%10.0fc') + '\n\n')
    if distrib:
        ip.txt('Random effects u_i ~ ')
        ip.res(f'{distrib:<27}')
        ip.txt('Obs per group:\n')
    else:
        ip.txt(' ' * 48 + 'Obs per group:\n')
    for k, v, f in (('min', gmin, '%10.0fc'), ('avg', gavg, '%10.1fc'), ('max', gmax, '%10.0fc')):
        ip.txt(' ' * 62 + f'{k} = ')
        ip.res(fmt_num(v, f) + '\n')
    if intmethod:
        ip.txt('\nIntegration method: ')
        ip.res(f'{intmethod:<28}')
        ip.txt('Integration pts.  =  ')
        ip.res(fmt_num(nq, '%9.0g') + '\n')
    lab = f'{chi2type} chi2('
    ip.txt('\n' + ' ' * 48 + lab)
    ip.res(f'{df}')
    ip.txt(')' + ' ' * max(1, 66 - 48 - len(lab) - len(str(df)) - 1) + '=  ')
    ip.res(f'{abs(chi2):9.2f}\n')
    crt = 'Log pseudolikelihood' if pseudo else 'Log likelihood'
    left = f'{crt}  = '
    ip.txt(left)
    ip.res(fmt_num(ll, '%10.0g'))
    ip.txt(' ' * max(1, 48 - len(left) - 10) + 'Prob > chi2' + ' ' * 7 + '=')
    pv = sps.chi2.sf(abs(chi2), df) if df else np.nan
    ip.res(' ' * 5 + (f'{pv:6.4f}' if pv == pv else '     .') + '\n\n')
    return pv


def _lrnote(ip, msg, chi2c):
    pv = 1.0 if chi2c == 0 else 0.5 * sps.chi2.sf(chi2c, 1)
    s = f'{chi2c:8.2f}' if (0.005 < chi2c < 1e4) or chi2c == 0 else f'{chi2c:8.2e}'
    left = f'{msg}: chibar2(01) = {s.strip()}'
    ip.txt(f'{msg}: chibar2(01) = ')
    ip.res(s.strip())
    ip.txt(' ' * max(1, 55 - len(left)) + 'Prob >= chibar2 = ')
    ip.res(f'{pv:5.3f}\n')
    return pv


def _robust(p):
    v = (p.opts.get('vce', 1) or '').strip().lower()
    if v == '' or v.startswith('oim'):
        return False
    if v.startswith('r') or v.startswith('cl'):
        return True
    raise StataError(f'vce({v}) not allowed', 198)


def _cluster_V(Vinv, S, gid):
    G = int(gid.max()) + 1
    Sg = np.zeros((G, S.shape[1]))
    np.add.at(Sg, gid, S)
    return Vinv @ (Sg.T @ Sg) @ Vinv * G / (G - 1), G


def _pooled(kind, X, y):
    """Pooled logit / probit / poisson by Newton; returns b, ll."""
    def fun(b):
        xb = X @ b
        if kind == 'logit':
            pr = spc.expit(xb)
            ll = float(np.sum(y * xb - np.logaddexp(0, xb)))
            return ll, X.T @ (y - pr), -(X.T @ (X * (pr * (1 - pr))[:, None]))
        if kind == 'probit':
            q = 2 * y - 1
            cdf = sps.norm.logcdf(q * xb)
            lam = q * np.exp(sps.norm.logpdf(q * xb) - cdf)
            ll = float(cdf.sum())
            wgt = lam * (lam + xb)
            return ll, X.T @ lam, -(X.T @ (X * wgt[:, None]))
        mu = np.exp(xb)
        return float(np.sum(-mu + y * xb - spc.gammaln(y + 1))), X.T @ (y - mu), -(X.T @ (X * mu[:, None]))
    b, ll, H, _ = _newton(fun, np.zeros(X.shape[1]))
    return b, ll


# ---------------------------------------------------------------- conditional logit (FE)
def _clogit_panel_ll(xb, y):
    """log P(y | sum y) for one panel and gradient weights (inclusion probabilities)."""
    T = xb.size
    k = int(round(y.sum()))
    m = xb.max()
    w = np.exp(xb - m)

    def esp(wv, kk):
        e = np.zeros(kk + 1)
        e[0] = 1.0
        for wt in wv:
            e[1:] = e[1:] + wt * e[:-1]
        return e[kk]
    Ek = esp(w, k)
    ll = float(xb @ y) - k * m - math.log(Ek)
    pin = np.empty(T)
    for t in range(T):
        pin[t] = w[t] * esp(np.delete(w, t), k - 1) / Ek
    return ll, pin


def _clogit(X, y, gid):
    groups = [np.flatnonzero(gid == g) for g in range(gid.max() + 1)]

    def ll_grad(b):
        xb = X @ b
        ll = 0.0
        g = np.zeros(X.shape[1])
        S = np.zeros_like(X)
        for idx in groups:
            l, pin = _clogit_panel_ll(xb[idx], y[idx])
            ll += l
            s = (y[idx] - pin) @ X[idx]
            g += s
            S[idx[0]] = s
        return ll, g, S

    def fun(b):
        ll, g, _ = ll_grad(b)
        H = _num_hess(lambda bb: ll_grad(bb)[1], b)
        return ll, g, H
    b, ll, H, lls = _newton(fun, np.zeros(X.shape[1]))
    return b, ll, H, lls, ll_grad(b)[2]


# ---------------------------------------------------------------- RE logit/probit by adaptive Gauss-Hermite
def _gh(nq):
    z, w = np.polynomial.hermite_e.hermegauss(nq)
    return z, w / math.sqrt(2 * math.pi)


def _logf(kind, y, eta):
    if kind == 'logit':
        return y * eta - np.logaddexp(0, eta)
    q = 2 * y - 1
    return sps.norm.logcdf(q * eta)


def _dlogf(kind, y, eta):
    if kind == 'logit':
        return y - spc.expit(eta)
    q = 2 * y - 1
    return q * np.exp(sps.norm.logpdf(q * eta) - sps.norm.logcdf(q * eta))


class _Pad:
    def __init__(self, X, y, gid):
        G = gid.max() + 1
        cnt = np.bincount(gid)
        Tm = cnt.max()
        pos = np.zeros(gid.size, int)
        seen = np.zeros(G, int)
        for i, g in enumerate(gid):
            pos[i] = seen[g]
            seen[g] += 1
        self.G, self.Tm, self.pos, self.gid = G, Tm, pos, gid
        self.M = np.zeros((G, Tm), bool)
        self.M[gid, pos] = True
        self.Y = np.zeros((G, Tm))
        self.Y[gid, pos] = y
        self.X = np.zeros((G, Tm, X.shape[1]))
        self.X[gid, pos] = X

    def pad(self, v):
        out = np.zeros((self.G, self.Tm))
        out[self.gid, self.pos] = v
        return out


def _re_binary(kind, X, y, gid, nq=12, b0=None, th0=0.0):
    P = _Pad(X, y, gid)
    z, wq = _gh(nq)
    k = X.shape[1]

    def adapt(b, th):
        sig = math.exp(th / 2)
        xb = P.pad(X @ b)
        mu = np.zeros(P.G)

        def obj(m):
            return (_logf(kind, P.Y, xb + sig * m[:, None]) * P.M).sum(1) - 0.5 * m ** 2

        def derivs(m):
            eta = xb + sig * m[:, None]
            lam = _dlogf(kind, P.Y, eta)
            d1 = (lam * P.M).sum(1) * sig - m
            if kind == 'logit':
                pr = spc.expit(eta)
                d2 = -(pr * (1 - pr) * P.M).sum(1) * sig ** 2 - 1
            else:
                d2 = -(lam * (lam + eta) * P.M).sum(1) * sig ** 2 - 1
            return d1, d2
        f0 = obj(mu)
        for _ in range(100):
            d1, d2 = derivs(mu)
            step = -d1 / d2
            t = np.ones(P.G)
            for _h in range(30):
                cand = mu + t * step
                fc = obj(cand)
                okk = fc >= f0 - 1e-12 * (1 + np.abs(f0))
                if okk.all():
                    break
                t = np.where(okk, t, t / 2)
            mu, f0 = cand, fc
            if np.max(np.abs(t * step)) < 1e-10:
                break
        d1, d2 = derivs(mu)
        return mu, 1 / np.sqrt(-d2)

    def ll_grad(par, mu, s):
        b, th = par[:k], par[k]
        sig = math.exp(th / 2)
        xb = P.pad(X @ b)
        u = mu[:, None] + s[:, None] * z[None, :]                      # G x Q
        eta = xb[:, :, None] + sig * u[:, None, :]                     # G x T x Q
        lf = (_logf(kind, P.Y[:, :, None], eta) * P.M[:, :, None]).sum(1)  # G x Q
        lw = np.log(wq)[None, :] + np.log(s)[:, None] - 0.5 * u ** 2 + 0.5 * z[None, :] ** 2 + lf
        mx = lw.max(1, keepdims=True)
        L = mx[:, 0] + np.log(np.exp(lw - mx).sum(1))
        post = np.exp(lw - L[:, None])                                 # G x Q
        h = _dlogf(kind, P.Y[:, :, None], eta) * P.M[:, :, None]       # G x T x Q
        gb = np.einsum('gq,gtq,gtk->gk', post, h, P.X)
        gth = np.einsum('gq,gtq,gq->g', post, h, u) * sig / 2
        S = np.column_stack([gb, gth])
        return float(L.sum()), S.sum(0), S

    par = np.r_[b0 if b0 is not None else np.zeros(k), th0]
    lls = []
    for it in range(200):
        mu, s = adapt(par[:k], par[k])
        ll, g, _ = ll_grad(par, mu, s)
        lls.append(ll)
        H = _num_hess(lambda q: ll_grad(q, mu, s)[1], par)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        lam = 1.0
        while True:
            pn = par - lam * step
            ln = ll_grad(pn, *adapt(pn[:k], pn[k]))[0]
            if np.isfinite(ln) and ln >= ll - 1e-10 * (1 + abs(ll)):
                break
            lam /= 2
            if lam < 1e-10:
                pn = par
                break
        par = pn
        if (abs(float(g @ step)) < 1e-7 and it > 0) or lam < 1e-10 or \
                (it > 0 and abs(ln - ll) < 1e-10 * (1 + abs(ll))):
            break
    mu, s = adapt(par[:k], par[k])
    ll, g, S = ll_grad(par, mu, s)
    H = _num_hess(lambda q: ll_grad(q, mu, s)[1], par)
    lls.append(ll)
    return par, ll, H, lls, S


# ---------------------------------------------------------------- Poisson / NB panel likelihoods
def _sums(v, gid, G):
    return np.bincount(gid, v, minlength=G)


def _pois_fe(X, y, gid):
    G = gid.max() + 1
    sy = _sums(y, gid, G)
    const = float(np.sum(spc.gammaln(sy + 1)) - np.sum(spc.gammaln(y + 1)))

    def fun(b):
        xb = X @ b
        m = np.full(G, -np.inf)
        np.maximum.at(m, gid, xb)
        e = np.exp(xb - m[gid])
        se = _sums(e, gid, G)
        lse = m + np.log(se)
        ll = const + float(np.sum(y * (xb - lse[gid])))
        pr = e / se[gid]
        S = X * (y - sy[gid] * pr)[:, None]
        g = S.sum(0)
        # Hessian: -sum_i sy_i [sum_t p x x' - (sum p x)(sum p x)']
        Xp = X * pr[:, None]
        mx = np.zeros((G, X.shape[1]))
        np.add.at(mx, gid, Xp)
        H = -(X.T @ (Xp * sy[gid][:, None])) + (mx.T @ (mx * sy[:, None]))
        return ll, g, H, S
    b, ll, H, lls = _newton(lambda bb: fun(bb)[:3], np.zeros(X.shape[1]))
    return b, ll, H, lls, fun(b)[3]


def _gsum(M, gid, G):
    out = np.zeros((G, M.shape[1]))
    np.add.at(out, gid, M)
    return out


def _pois_re_gamma(par, X, y, gid, G, k):
    """Per-panel log likelihood and scores; par = (b, lnalpha)."""
    b, lna = par[:k], par[k]
    a = math.exp(-lna)
    xb = X @ b
    mu = np.exp(xb)
    smu, sy = _sums(mu, gid, G), _sums(y, gid, G)
    li = (spc.gammaln(a + sy) - spc.gammaln(a) - _sums(spc.gammaln(y + 1), gid, G)
          + a * math.log(a) - (a + sy) * np.log(a + smu) + _sums(y * xb, gid, G))
    rat = (a + sy) / (a + smu)
    Sb = _gsum(X * y[:, None], gid, G) - rat[:, None] * _gsum(X * mu[:, None], gid, G)
    da = spc.digamma(a + sy) - spc.digamma(a) + math.log(a) + 1 - np.log(a + smu) - rat
    return li, np.column_stack([Sb, -a * da])


def _nb_fe(par, X, y, gid, G, k=None):
    lam = np.exp(X @ par)
    sl, sy = _sums(lam, gid, G), _sums(y, gid, G)
    li = (spc.gammaln(sl) + spc.gammaln(sy + 1) - spc.gammaln(sl + sy)
          + _sums(spc.gammaln(lam + y) - spc.gammaln(lam) - spc.gammaln(y + 1), gid, G))
    w = lam * ((spc.digamma(sl) - spc.digamma(sl + sy))[gid] + spc.digamma(lam + y) - spc.digamma(lam))
    return li, _gsum(X * w[:, None], gid, G)


def _nb_re(par, X, y, gid, G, k):
    b, lr, ls = par[:k], par[k], par[k + 1]
    r, s = math.exp(lr), math.exp(ls)
    lam = np.exp(X @ b)
    sl, sy = _sums(lam, gid, G), _sums(y, gid, G)
    tot = r + s + sl + sy
    li = (spc.gammaln(r + s) + spc.gammaln(r + sl) + spc.gammaln(s + sy)
          - spc.gammaln(r) - spc.gammaln(s) - spc.gammaln(tot)
          + _sums(spc.gammaln(lam + y) - spc.gammaln(lam) - spc.gammaln(y + 1), gid, G))
    w = lam * ((spc.digamma(r + sl) - spc.digamma(tot))[gid] + spc.digamma(lam + y) - spc.digamma(lam))
    dr = spc.digamma(r + s) + spc.digamma(r + sl) - spc.digamma(r) - spc.digamma(tot)
    ds = spc.digamma(r + s) + spc.digamma(s + sy) - spc.digamma(s) - spc.digamma(tot)
    return li, np.column_stack([_gsum(X * w[:, None], gid, G), r * dr, s * ds])


def _ml_scores(fs, x0, maxit=300):
    """Newton using analytic per-panel scores fs(x) -> (li, S); Hessian by differencing the gradient."""
    def fun(x):
        li, S = fs(x)
        ll = float(li.sum())
        if not np.isfinite(ll):
            return -np.inf, np.zeros_like(x), -np.eye(x.size)
        H = _num_hess(lambda q: fs(q)[1].sum(0), x, h=1e-6)
        return ll, S.sum(0), H
    b, ll, H, lls = _newton(fun, x0, maxit)
    return b, ll, H, lls, fs(b)[1]


# ---------------------------------------------------------------- common output
def _finish(ip, p, cmd, title, toks, d, ivar, yv, mask, gid, names, info, active, bb, Vb, ll, lls, chi2type,
            distrib=None, aux=None, lrmsg=None, chi2c=None, llc=None, extra_sc=None, robust=False, G=None,
            eform_lab=None, intmethod=None, nq=None, cons=True, model='', cols=None):
    b, V = full_bV(len(names), active, bb, Vb)
    slopes = [j for j in active if info[j].kind != 'cons']
    ng, gmin, gavg, gmax = _gstats(gid)
    n = int(gid.size)
    if chi2type == 'Wald' or robust:
        chi2, _, _ = wald_test(b, V, slopes)
        chi2type = 'Wald'
    else:
        chi2 = 2 * (ll - extra_sc.pop('ll_0'))
    df = len(slopes)
    pv = sps.chi2.sf(chi2, df) if df else np.nan
    sc = {'N': n, 'N_g': ng, 'g_min': gmin, 'g_avg': gavg, 'g_max': gmax, 'll': ll, 'chi2': chi2, 'df_m': df,
          'p': pv, 'k': len(active)}
    if chi2c is not None:
        sc['chi2_c'] = chi2c
        sc['ll_c'] = llc
    sc.update(extra_sc or {})
    est = {'nomargins': True, 'names': names, 'b': b, 'V': V, 'colinfo': info, 'df_r': None, 'depvar': yv,
           'sample': mask, 'cmd': cmd, 'stat': 'z',
           'scalars': sc,
           'macros': {'cmd': cmd, 'depvar': yv, 'ivar': ivar, 'title': title, 'model': model,
                      'chi2type': chi2type, 'vce': 'cluster' if robust else 'oim',
                      'distrib': distrib or '', 'intmethod': intmethod or ''}}
    if robust:
        sc['N_clust'] = G
        est['macros']['clustvar'] = ivar
    eform = bool(eform_lab) and any(p.opts.has(o) for o in ('or', 'irr'))
    quiet = p.opts.has('nolog')

    def replay(ip_, show=False):
        if show and not quiet and lls:
            _print_log(ip_, lls, 'Fitting full model:' if distrib else None, robust)
        _header(ip_, title, ivar, n, ng, gmin, gavg, gmax, distrib, chi2type, df, chi2, ll, robust,
                intmethod, nq)
        rob = None
        if robust:
            ip_.txt(f'{f"(Std. err. adjusted for {comma_int(G)} clusters in {ivar})":>78}\n')
            rob = 'Robust'
        coef_table(ip_, est, 'z', robust=rob, aux=aux, eform=eform,
                   colhead=_eform_head(eform_lab, rob) if eform else None)
        if lrmsg and not robust:
            _lrnote(ip_, lrmsg, chi2c)

    def predict(ip_, kind, m_):
        xb = _predict_linear(ip_, est, 'xb')
        if kind in ('xb',):
            return xb
        if kind in (None, '', 'pu0', 'pr', 'pc1', 'nu0', 'n', 'ir'):
            if cmd == 'xtlogit':
                return spc.expit(xb)
            if cmd == 'xtprobit':
                return sps.norm.cdf(xb)
            return np.exp(xb)
        if kind == 'stdp':
            return _predict_linear(ip_, est, 'stdp')
        raise StataError(f'predict option {kind} not supported', 198)
    est['replay'] = lambda ip_: replay(ip_, False)
    est['predict'] = predict
    _store_rhs(est, toks[1:], p._xtcols, d)
    ip.set_estimation(est)
    replay(ip, True)


def _model(p, allowed):
    m = [o for o in allowed if p.opts.has(o)]
    if len(m) > 1:
        raise StataError(f'choose only one of {", ".join(allowed)}', 198)
    return m[0] if m else 're'


def _aux_re(th, se_th, link):
    zc = sps.norm.ppf(.975)
    sig = math.exp(th / 2)
    c = math.pi ** 2 / 3 if link == 'logit' else 1.0
    lo, hi = th - zc * se_th, th + zc * se_th
    rho = sig ** 2 / (sig ** 2 + c)
    drho = rho * (1 - rho)          # d rho / d th
    return [[('/lnsig2u', th, se_th, False, None, None)],
            [('sigma_u', sig, sig * se_th / 2, False, math.exp(lo / 2), math.exp(hi / 2)),
             ('rho', rho, drho * se_th, False, math.exp(lo) / (math.exp(lo) + c), math.exp(hi) / (math.exp(hi) + c))]]


# ---------------------------------------------------------------- commands
@command('xtlogit', 'xtprobit')
def cmd_xtbinary(ip, cmd, rest, by):
    p = parse_command(rest)
    toks = split_top(p.body)
    kind = 'logit' if cmd == 'xtlogit' else 'probit'
    model = _model(p, ('fe', 're', 'pa') if kind == 'logit' else ('re', 'pa'))
    if model == 'pa':
        raise StataError('pa option is not available in Stata Web; use xtgee', 198)
    fe = model == 'fe'
    d, ivar, yv, yy, gid, cols, mask, Xall, names, info, active, cons = _panel_setup(ip, p, toks, fe)
    yy = (yy != 0).astype(float)
    robust = _robust(p)
    if fe:
        G0 = gid.max() + 1
        sy, cnt = np.bincount(gid, yy), np.bincount(gid)
        keepg = (sy > 0) & (sy < cnt)
        drop = ~keepg[gid]
        if drop.any():
            ng_d, no_d = int((~keepg).sum()), int(drop.sum())
            ip.txt(f'note: {comma_int(ng_d)} group{"s" if ng_d != 1 else ""} ({comma_int(no_d)} obs) dropped '
                   f'because of all positive or\n      all negative outcomes.\n')
            sub = np.flatnonzero(mask)
            mask = mask.copy()
            mask[sub[drop]] = False
            Xall, yy = Xall[~drop], yy[~drop]
            _, gid = np.unique(gid[~drop], return_inverse=True)
        if gid.size == 0:
            raise StataError('outcome does not vary in any group', 2000)
        active = _within_drop(ip, Xall, gid, info, active)
        X = Xall[:, active]
        bb, ll, H, lls, S = _clogit(X, yy, gid)
        Vinv = np.linalg.inv(-H)
        G = None
        if robust:
            Vinv, G = _cluster_V(Vinv, S, gid)
        ll0 = sum(_clogit_panel_ll(np.zeros(int((gid == g).sum())), yy[gid == g])[0] for g in range(gid.max() + 1))
        return _finish(ip, p, cmd, 'Conditional fixed-effects logistic regression', toks, d, ivar, yv, mask, gid,
                       names, info, active, bb, Vinv, ll, lls, 'LR', extra_sc={'ll_0': ll0}, robust=robust, G=G,
                       eform_lab='Odds ratio', cons=False, model='fe')
    X = Xall[:, active]
    nq = int(p.opts.get('intpoints', 4) or 12)
    b0, llp = _pooled(kind, X, yy)
    par, ll, H, lls, S = _re_binary(kind, X, yy, gid, nq, b0, 0.0)
    Vinv = np.linalg.inv(-H)
    G = None
    if robust:
        Sg = S
        G = Sg.shape[0]
        Vinv = Vinv @ (Sg.T @ Sg) @ Vinv * G / (G - 1)
    k = len(active)
    bb, Vb = par[:k], Vinv[:k, :k]
    th, se_th = par[k], math.sqrt(max(Vinv[k, k], 0))
    chi2c = max(2 * (ll - llp), 0.0)
    sig = math.exp(th / 2)
    title = 'Random-effects logistic regression' if kind == 'logit' else 'Random-effects probit regression'
    c = math.pi ** 2 / 3 if kind == 'logit' else 1.0
    return _finish(ip, p, cmd, title, toks, d, ivar, yv, mask, gid, names, info, active, bb, Vb, ll, lls,
                   'Wald', distrib='Gaussian', aux=_aux_re(th, se_th, kind), lrmsg='LR test of rho=0',
                   chi2c=chi2c, llc=llp, extra_sc={'sigma_u': sig, 'rho': sig ** 2 / (sig ** 2 + c),
                                                   'n_quad': nq},
                   robust=robust, G=G, eform_lab='Odds ratio' if kind == 'logit' else None,
                   intmethod='mvaghermite', nq=nq, cons=cons, model='re')


@command('xtpoisson')
def cmd_xtpoisson(ip, cmd, rest, by):
    p = parse_command(rest)
    toks = split_top(p.body)
    model = _model(p, ('fe', 're', 'pa'))
    if model == 'pa':
        raise StataError('pa option is not available in Stata Web; use xtgee', 198)
    fe = model == 'fe'
    d, ivar, yv, yy, gid, cols, mask, Xall, names, info, active, cons = _panel_setup(ip, p, toks, fe)
    if np.any(yy < 0):
        raise StataError(f'{yv} must be greater than or equal to zero', 459)
    robust = _robust(p)
    if fe:
        sy = np.bincount(gid, yy)
        drop = ~(sy > 0)[gid]
        if drop.any():
            ng_d, no_d = int((sy == 0).sum()), int(drop.sum())
            ip.txt(f'note: {comma_int(ng_d)} group{"s" if ng_d != 1 else ""} ({comma_int(no_d)} obs) dropped '
                   f'because of all zero outcomes\n')
            sub = np.flatnonzero(mask)
            mask = mask.copy()
            mask[sub[drop]] = False
            Xall, yy = Xall[~drop], yy[~drop]
            _, gid = np.unique(gid[~drop], return_inverse=True)
        active = _within_drop(ip, Xall, gid, info, active)
        X = Xall[:, active]
        bb, ll, H, lls, S = _pois_fe(X, yy, gid)
        Vinv = np.linalg.inv(-H)
        G = None
        if robust:
            Vinv, G = _cluster_V(Vinv, S, gid)
        return _finish(ip, p, cmd, 'Conditional fixed-effects Poisson regression', toks, d, ivar, yv, mask, gid,
                       names, info, active, bb, Vinv, ll, lls, 'Wald', robust=robust, G=G, eform_lab='IRR',
                       cons=False, model='fe')
    if p.opts.has('normal', 3):
        raise StataError('xtpoisson, re normal is not available in Stata Web; the default gamma model is', 198)
    X = Xall[:, active]
    k = X.shape[1]
    G = gid.max() + 1
    b0, llp = _pooled('poisson', X, yy)
    par, ll, H, lls, S = _ml_scores(lambda x: _pois_re_gamma(x, X, yy, gid, G, k), np.r_[b0, 0.0])
    Vinv = np.linalg.inv(-H)
    Gc = None
    if robust:
        Gc = G
        Vinv = Vinv @ (S.T @ S) @ Vinv * G / (G - 1)
    lna, se = par[k], math.sqrt(max(Vinv[k, k], 0))
    zc = sps.norm.ppf(.975)
    alpha = math.exp(lna)
    aux = [[('/lnalpha', lna, se, False, None, None)],
           [('alpha', alpha, alpha * se, False, math.exp(lna - zc * se), math.exp(lna + zc * se))]]
    chi2c = max(2 * (ll - llp), 0.0)
    return _finish(ip, p, cmd, 'Random-effects Poisson regression', toks, d, ivar, yv, mask, gid, names, info,
                   active, par[:k], Vinv[:k, :k], ll, lls, 'Wald', distrib='Gamma', aux=aux,
                   lrmsg='LR test of alpha=0', chi2c=chi2c, llc=llp, extra_sc={'alpha': alpha},
                   robust=robust, G=Gc, eform_lab='IRR', cons=cons, model='re')


@command('xtnbreg')
def cmd_xtnbreg(ip, cmd, rest, by):
    p = parse_command(rest)
    toks = split_top(p.body)
    model = _model(p, ('fe', 're', 'pa'))
    if model == 'pa':
        raise StataError('pa option is not available in Stata Web; use xtgee', 198)
    fe = model == 'fe'
    # conditional FE NB identifies an intercept (Hausman, Hall & Griliches 1984)
    d, ivar, yv, yy, gid, cols, mask, Xall, names, info, active, cons = _panel_setup(ip, p, toks, False)
    if np.any(yy < 0):
        raise StataError(f'{yv} must be greater than or equal to zero', 459)
    if _robust(p):
        raise StataError('vce(robust) not allowed with xtnbreg', 198)
    sy = np.bincount(gid, yy)
    if fe:
        cnt = np.bincount(gid)
        drop = ~((sy > 0) & (cnt > 1))[gid]
        if drop.any():
            ng_d, no_d = int(((sy == 0) | (cnt == 1)).sum()), int(drop.sum())
            ip.txt(f'note: {comma_int(ng_d)} group{"s" if ng_d != 1 else ""} ({comma_int(no_d)} obs) dropped '
                   f'because of only one obs per group or all zero outcomes\n')
            sub = np.flatnonzero(mask)
            mask = mask.copy()
            mask[sub[drop]] = False
            Xall, yy = Xall[~drop], yy[~drop]
            _, gid = np.unique(gid[~drop], return_inverse=True)
    X = Xall[:, active]
    k = X.shape[1]
    G = gid.max() + 1
    b0, llp = _pooled('poisson', X, yy)
    if fe:
        par, ll, H, lls, _ = _ml_scores(lambda x: _nb_fe(x, X, yy, gid, G), b0)
        Vinv = np.linalg.inv(-H)
        return _finish(ip, p, cmd, 'Conditional FE negative binomial regression', toks, d, ivar, yv, mask, gid,
                       names, info, active, par, Vinv, ll, lls, 'Wald', eform_lab='IRR', cons=cons, model='fe')
    # pooled NB for the LR test vs. pooled
    from .models import _nb_fun
    try:
        fnb, _ = _nb_fun(X, yy, None)
        bnb, llnb, _, _ = _newton(fnb, np.r_[b0, 0.0])
    except Exception:
        llnb = np.nan
    par, ll, H, lls, _ = _ml_scores(lambda x: _nb_re(x, X, yy, gid, G, k), np.r_[b0, 1.0, 1.0])
    Vinv = np.linalg.inv(-H)
    zc = sps.norm.ppf(.975)
    rows = []
    for j, nm in ((k, 'r'), (k + 1, 's')):
        v, se = par[j], math.sqrt(max(Vinv[j, j], 0))
        rows.append((f'/ln_{nm}', v, se, False, None, None))
    aux = [rows, [(nm, math.exp(par[j]), math.exp(par[j]) * math.sqrt(max(Vinv[j, j], 0)), False,
                   math.exp(par[j] - zc * math.sqrt(max(Vinv[j, j], 0))),
                   math.exp(par[j] + zc * math.sqrt(max(Vinv[j, j], 0)))) for j, nm in ((k, 'r'), (k + 1, 's'))]]
    chi2c = max(2 * (ll - llnb), 0.0) if llnb == llnb else 0.0
    return _finish(ip, p, cmd, 'Random-effects negative binomial regression', toks, d, ivar, yv, mask, gid, names,
                   info, active, par[:k], Vinv[:k, :k], ll, lls, 'Wald', distrib='Beta', aux=aux,
                   lrmsg='LR test vs. pooled', chi2c=chi2c, llc=llnb,
                   extra_sc={'r': math.exp(par[k]), 's': math.exp(par[k + 1])}, eform_lab='IRR', cons=cons,
                   model='re')
