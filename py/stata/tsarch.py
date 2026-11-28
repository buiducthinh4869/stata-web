"""arch: ARCH/GARCH/TARCH/EGARCH (EARCH) family, ARCH-in-mean, ARMA disturbances, Gaussian or t errors.
Likelihood follows arch_dr.ado: presample sigma^2 = mean(e^2) at the current beta (arch0(xb)), every presample
lag in the ARCH equation set to that value, EARCH presample z = 0, EGARCH presample ln(sigma^2_0)."""
import math

import numpy as np
from scipy import special as spc
from scipy import stats as sps

from .design import coef_table, right_lines, Col
from .errors import StataError
from .fmt import g_fmt, comma_int, fmt_any
from .interp import command
from .syntax import parse_command, split_top, parse_numlist
from .tsuni import need_ts, series
from .tsops import expand_token
from . import tsml

_NORMADJ = math.sqrt(2 / math.pi)


def _nl(s):
    return [int(v) for v in parse_numlist(s)] if s not in (None, '') else []


def _lag(x, k):
    out = np.zeros_like(x)
    if k < x.size:
        out[k:] = x[:x.size - k]
    return out


@command('arch')
def cmd_arch(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    toks = split_top(p.body)
    if not toks:
        if ip.est and ip.est.get('cmd') == 'arch':
            return ip.est['replay'](ip)
        raise StataError('varlist required', 100)
    need_ts(d)
    d.sort([x for x in (d.ts.get('panel'), d.ts.get('time')) if x])
    for o in ('saarch', 'parch', 'tparch', 'aparch', 'nparch', 'nparchk', 'pgarch', 'aarch', 'narch', 'narchk',
              'abarch', 'atarch', 'sdgarch', 'het', 'archmexp'):
        if p.opts.get(o, len(o)):
            raise StataError(f'{o}() is not available in Stata Web', 199)
    A = _nl(p.opts.get('arch', 4))
    Gg = _nl(p.opts.get('garch', 1))
    Tt = _nl(p.opts.get('tarch', 2))
    Ea = _nl(p.opts.get('earch', 2))
    Eg = _nl(p.opts.get('egarch', 2))
    ar = _nl(p.opts.get('ar', 2))
    ma = _nl(p.opts.get('ma', 2))
    archm = p.opts.has('archm', 5)
    archml = _nl(p.opts.get('archmlags', 6))
    mlags = sorted(set(([0] if archm else []) + archml)) if (archm or archml) else []
    if (Ea or Eg) and (A or Gg or Tt):
        raise StataError('earch()/egarch() may not be combined with arch(), garch() or tarch() in Stata Web', 198)
    nocons = p.opts.has('noconstant', 4)
    dist = (p.opts.get('distribution', 4) or 'gaussian').strip().lower().split()
    dname = dist[0] if dist else 'gaussian'
    if dname.startswith('gau') or dname.startswith('nor'):
        dname = 'gaussian'
    elif dname == 't':
        pass
    else:
        raise StataError('distribution() must be gaussian or t', 198)
    tdf_fixed = float(dist[1]) if dname == 't' and len(dist) > 1 else None
    vk = (p.opts.get('vce', 3) or '').strip().lower()
    if p.opts.has('robust', 1):
        vk = 'robust'
    vk = {'': 'opg', 'opg': 'opg', 'oim': 'oim', 'robust': 'robust', 'r': 'robust'}.get(vk)
    if vk is None:
        raise StataError('vce() must be opg, oim, or robust', 198)
    y, yname = series(ip, toks[0])
    Xs, xnames = [], []
    for tok in toks[1:]:
        for t2 in expand_token(tok):
            x, nm = series(ip, t2)
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
    delta = d.ts.get('delta', 1) or 1
    tt = np.sort(tv[mask])
    gaps = int(np.sum(np.round(np.diff(tt) / delta) > 1)) if tt.size > 1 else 0
    if gaps:
        ip.txt('(note: conditioning reset at each gap)\n\n')
    idx = np.flatnonzero(mask)
    yy = y[idx]
    n = yy.size
    # segment starts (conditioning reset at gaps)
    seg = np.zeros(n, dtype=int)
    if n > 1:
        brk = np.round(np.diff(tv[idx]) / delta) > 1
        seg[1:] = np.cumsum(brk)
    kx = len(Xs) + (0 if nocons else 1)
    X = np.column_stack([x[idx] for x in Xs] + ([np.ones(n)] if not nocons else [])) if kx else np.zeros((n, 0))
    archany = bool(A or Gg or Tt or Ea or Eg)
    egarch_mode = bool(Ea or Eg)
    # parameter layout: xb | archm | ar | ma | ARCH: arch, tarch, garch, earch, earch_a, egarch, _cons | dist
    nam, nar_, nma_ = len(mlags), len(ar), len(ma)
    nA, nT, nG, nEa, nEg = len(A), len(Tt), len(Gg), len(Ea), len(Eg)
    has_vcons = True
    ndf = 1 if (dname == 't' and tdf_fixed is None) else 0
    K = kx + nam + nar_ + nma_ + nA + nT + nG + 2 * nEa + nEg + 1 + ndf

    def split(th):
        i = 0
        out = {}
        for key, k in (('b', kx), ('m', nam), ('ar', nar_), ('ma', nma_), ('A', nA), ('T', nT), ('G', nG),
                       ('Ea', nEa), ('Eaa', nEa), ('Eg', nEg), ('c', 1), ('df', ndf)):
            out[key] = th[i:i + k]
            i += k
        return out

    def recurse(th, want=False):
        P = split(th)
        xb = X @ P['b'] if kx else np.zeros(n)
        c0 = P['c'][0]
        if dname == 't':
            nu = (math.exp(min(P['df'][0], 50.0)) + 2) if tdf_fixed is None else tdf_fixed
            if not nu > 2.0000001:
                return None
        # first pass: ARMA errors without arch-in-mean to obtain e^2 for priming
        u = yy - xb
        e = _arma_filter(u, P['ar'], ar, P['ma'], ma, seg)
        if archm:
            sig2_0 = prime[0]
        else:
            sig2_0 = float(np.mean(e * e))
        if sig2_0 <= 0:
            return None
        lsig2_0 = math.log(sig2_0)
        s2 = np.zeros(n)
        ee = np.zeros(n)
        uu = np.zeros(n)
        z = np.zeros(n)
        ll = np.zeros(n)
        for t in range(n):
            st = seg[t]

            def lagv(arr, k, pre):
                j = t - k
                return arr[j] if (j >= 0 and seg[j] == st) else pre
            if egarch_mode:
                v = c0
                for co, l in zip(P['A'], A):
                    v += co * lagv(ee * ee, l, sig2_0)
                for co, l in zip(P['T'], Tt):
                    v += co * lagv(np.where(ee > 0, ee * ee, 0.0), l, sig2_0)
                for co, l in zip(P['G'], Gg):
                    v += co * lagv(s2, l, sig2_0)
                lv = v
                for co, l in zip(P['Ea'], Ea):
                    lv += co * lagv(z, l, 0.0)
                for co, l in zip(P['Eaa'], Ea):
                    j = t - l
                    lv += co * ((abs(z[j]) - _NORMADJ) if (j >= 0 and seg[j] == st) else 0.0)
                for co, l in zip(P['Eg'], Eg):
                    j = t - l
                    lv += co * (math.log(s2[j]) if (j >= 0 and seg[j] == st) else lsig2_0)
                if lv > 700:
                    return None
                s2t = math.exp(lv)
            else:
                v = c0
                for co, l in zip(P['A'], A):
                    j = t - l
                    v += co * (ee[j] * ee[j] if (j >= 0 and seg[j] == st) else sig2_0)
                for co, l in zip(P['T'], Tt):
                    j = t - l
                    v += co * ((ee[j] * ee[j] if ee[j] > 0 else 0.0) if (j >= 0 and seg[j] == st) else sig2_0)
                for co, l in zip(P['G'], Gg):
                    j = t - l
                    v += co * (s2[j] if (j >= 0 and seg[j] == st) else sig2_0)
                s2t = v
            if not (s2t > 0) or not np.isfinite(s2t):
                return None
            s2[t] = s2t
            ut = yy[t] - xb[t]
            for co, l in zip(P['m'], mlags):
                ut -= co * (s2[t - l] if (t - l >= 0 and seg[t - l] == st) else sig2_0)
            uu[t] = ut
            et = ut
            for co, l in zip(P['ar'], ar):
                et -= co * (uu[t - l] if (t - l >= 0 and seg[t - l] == st) else 0.0)
            for co, l in zip(P['ma'], ma):
                et -= co * (ee[t - l] if (t - l >= 0 and seg[t - l] == st) else 0.0)
            ee[t] = et
            z[t] = et / math.sqrt(s2t)
            if dname == 'gaussian':
                ll[t] = -0.5 * (math.log(2 * math.pi) + math.log(s2t) + et * et / s2t)
            else:
                ll[t] = (spc.gammaln((nu + 1) / 2) - 0.5 * math.log(math.pi) - spc.gammaln(nu / 2)
                         - 0.5 * math.log(nu - 2) - 0.5 * math.log(s2t)
                         - (nu + 1) / 2 * math.log1p(et * et / (s2t * (nu - 2))))
        if want:
            return ll, s2, ee, xb
        return ll

    prime = [None]

    def llvec(th):
        r = recurse(th)
        return np.full(n, -np.inf) if r is None else r
    # ---------------- starting values (arch.ado)
    if kx:
        b0 = np.linalg.lstsq(X, yy, rcond=None)[0]
        u0 = yy - X @ b0
    else:
        b0 = np.zeros(0)
        u0 = yy.copy()
    from .tsarima import _start_values
    armab = []
    if ar or ma:
        sv = _start_values(np.r_[u0], np.zeros((n, 0)), np.ones(n, bool), 0, ar, ma, 0, 0, True)
        armab = list(sv[:nar_ + nma_])
    e0 = _arma_filter(u0, np.array(armab[:nar_]), ar, np.array(armab[nar_:]), ma, seg) if (ar or ma) else u0
    e2 = e0 * e0
    prime[0] = float(np.mean(e2))
    vstart = []
    if A or Gg or Tt:
        vstart = _arch_start(e2, A, Tt, Gg, egarch_mode)
    if egarch_mode:
        le2 = np.log(np.maximum(e2, 1e-300))
        vA = vstart[:nA + nT + nG] if vstart else []
        vc = float(np.mean(le2)) if not (A or Gg or Tt) else vstart[-1]
        vstart = list(vA) + [0.0] * (2 * nEa) + [0.0] * nEg + [vc]
    elif not (A or Gg or Tt):
        vstart = [float(np.mean(e2))]
    dfst = [math.log(max(10.0 - 2, 1e-3))] if ndf else []
    th0 = np.concatenate([b0, [0.0] * nam, armab, vstart, dfst]).astype(float)
    if th0.size != K:
        raise StataError('internal: parameter count mismatch', 9999)
    if not np.isfinite(np.sum(llvec(th0))):
        th0[kx + nam + nar_ + nma_:K - 1 - ndf] = 0.0
        th0[K - 1 - ndf] = prime[0] if not egarch_mode else math.log(prime[0])
    itmax = p.opts.get('iterate', 4)
    itmax = int(float(itmax)) if itmax not in (None, '') else 300
    frm = p.opts.get('from', 4)
    if frm not in (None, ''):
        th0 = np.array([float(v) for v in frm.replace(',', ' ').split() if v.lower() != 'copy'])
    if itmax == 0:
        th, ll, it_log = th0, float(np.sum(llvec(th0))), [(0, float(np.sum(llvec(th0))))]
    else:
        th, ll, it_log = tsml.maximize(llvec, th0, maxit=itmax)
        if egarch_mode and nEg and frm in (None, ''):
            # EGARCH likelihoods are often bimodal in the persistence parameter: also start from a
            # persistent solution and keep the better optimum
            th1 = th0.copy()
            j0 = kx + nam + nar_ + nma_ + nA + nT + nG + 2 * nEa
            th1[j0:j0 + nEg] = 0.9 / nEg
            th1[j0 + nEg] = th0[j0 + nEg] * 0.1
            try:
                th_b, ll_b, log_b = tsml.maximize(llvec, th1, maxit=itmax)
                if ll_b > ll + 1e-6:
                    th, ll, it_log = th_b, ll_b, log_b
            except StataError:
                pass
    V = tsml.vce(llvec, th, vk)
    # names
    eqdep = yname.split('.')[-1]
    names, info, eqs = [], [], []

    def add(eq, nm, kind, tsvar=None, ch=''):
        c = Col(name=nm, values=None, term=0, group='', level='', kind=kind, vars=[])
        c.tsvar, c.tschain = tsvar, ch
        names.append(nm)
        info.append(c)
        eqs.append(eq)

    def L(l):
        return f'L{"" if l == 1 else l}'
    from .tsops import split_name
    for nm in xnames:
        ch, base = split_name(nm)
        add(eqdep, nm, 'cont', base, ch)
    if not nocons:
        add(eqdep, '_cons', 'cons')
    for l in mlags:
        add('ARCHM', f'{L(l)}.sigma2' if l else 'sigma2', 'cont', 'sigma2', L(l) if l else '')
    for l in ar:
        add('ARMA', f'{L(l)}.ar', 'cont', 'ar', L(l))
    for l in ma:
        add('ARMA', f'{L(l)}.ma', 'cont', 'ma', L(l))
    for l in A:
        add('ARCH', f'{L(l)}.arch', 'cont', 'arch', L(l))
    for l in Tt:
        add('ARCH', f'{L(l)}.tarch', 'cont', 'tarch', L(l))
    for l in Gg:
        add('ARCH', f'{L(l)}.garch', 'cont', 'garch', L(l))
    for l in Ea:
        add('ARCH', f'{L(l)}.earch', 'cont', 'earch', L(l))
    for l in Ea:
        add('ARCH', f'{L(l)}.earch_a', 'cont', 'earch_a', L(l))
    for l in Eg:
        add('ARCH', f'{L(l)}.egarch', 'cont', 'egarch', L(l))
    add('ARCH', '_cons', 'cons')
    k_main = len(names)
    se = np.sqrt(np.maximum(np.diag(V), 0))
    # Wald test: mean-equation regressors (excluding constant), ARCHM and ARMA terms
    widx = list(range(len(xnames))) + list(range(kx, kx + nam + nar_ + nma_))
    if widx:
        bw = th[widx]
        Vw = V[np.ix_(widx, widx)]
        chi2 = float(bw @ tsml.safe_inv(Vw) @ bw)
        dfm = len(widx)
        pchi = float(sps.chi2.sf(chi2, dfm))
    else:
        chi2, dfm, pchi = np.nan, 0, np.nan
    tf = d.formats.get(d.ts['time'], '%9.0g')
    smin, smax = fmt_any(float(tt[0]), tf).strip(), fmt_any(float(tt[-1]), tf).strip()
    title = 'ARCH family regression' if archany else 'Time-series regression'
    if ar and ma:
        title += ' -- ARMA disturbances'
    elif ar:
        title += ' -- AR disturbances'
    elif ma:
        title += ' -- MA disturbances'
    full_names = [f'{q}:{nm}' for q, nm in zip(eqs, names)]
    if ndf:
        full_names.append('lndfm2:_cons')
    lab = {'opg': 'OPG', 'oim': 'OIM', 'robust': 'Semirobust'}[vk]
    _, s2fit, efit, xbfit = recurse(th, want=True)
    est = {'names': full_names, 'b': th, 'V': V, 'colinfo': None, 'df_r': None, 'depvar': yname, 'sample': mask,
           'cmd': 'arch', 'stat': 'z', 'vce': vk, 'nomargins': True,
           'scalars': {'N': n, 'll': ll, 'chi2': chi2, 'df_m': dfm, 'p': pchi, 'N_gaps': gaps, 'k': K,
                       'tmin': float(tt[0]), 'tmax': float(tt[-1])},
           'macros': {'cmd': 'arch', 'depvar': yname, 'title': title, 'vce': vk, 'vcetype': lab, 'tmins': smin,
                      'tmaxs': smax, 'dist': dname, 'arch': ' '.join(map(str, A)), 'garch': ' '.join(map(str, Gg))}}
    if ndf:
        est['scalars']['tdf'] = math.exp(th[-1]) + 2
    quiet_log = p.opts.has('nolog') or ip.quiet

    def replay(ip_, show_log=False):
        if show_log and not quiet_log:
            tsml.print_log(ip_, it_log)
        ip_.txt(title + '\n\n')
        dline = 'Distribution: Gaussian' if dname == 'gaussian' else (
            f'Distribution: t({g_fmt(tdf_fixed, 5).strip()})' if tdf_fixed else 'Distribution: t')
        right_lines(ip_, [f'Sample: {smin} thru {smax}' + (', but with gaps' if gaps > 1 else (
            ', but with a gap' if gaps == 1 else '')), dline, f'Log likelihood = {g_fmt(ll, 9).strip()}'],
            [('Number of obs', comma_int(n)), (f'Wald chi2({dfm if dfm else "."})', f'{chi2:.2f}' if chi2 == chi2 else '.'),
             ('Prob > chi2', f'{pchi:.4f}' if pchi == pchi else '.')])
        ip_.txt('\n')
        first = True
        for eqn in [eqdep, 'ARCHM', 'ARMA', 'ARCH']:
            ix = [i for i in range(k_main) if eqs[i] == eqn]
            if not ix:
                continue
            sub = {'names': [names[i] for i in ix], 'b': th[ix], 'V': V[np.ix_(ix, ix)],
                   'colinfo': [info[i] for i in ix], 'df_r': None}
            coef_table(ip_, sub, 'z', robust=lab, depvar=yname, header=first, footer=False, eq_title=eqn)
            first = False
        if ndf:
            ip_.txt('-' * 13 + '+' + '-' * 64 + '\n')
            b_, s_ = th[-1], se[-1]
            zc = sps.norm.ppf(0.975)
            ip_.txt(f'{"/lndfm2":>12} |')
            ip_.res(f'{g_fmt(b_, 9):>11}{g_fmt(s_, 9):>11}{b_ / s_:9.2f}{2 * sps.norm.sf(abs(b_ / s_)):8.3f}'
                    f'{g_fmt(b_ - zc * s_, 9):>13}{g_fmt(b_ + zc * s_, 9):>12}\n')
            ip_.txt('-' * 13 + '+' + '-' * 64 + '\n')
            df_ = math.exp(b_) + 2
            ip_.txt(f'{"df":>12} |')
            ip_.res(f'{g_fmt(df_, 9):>11}{g_fmt(math.exp(b_) * s_, 9):>11}{"":17}'
                    f'{g_fmt(math.exp(b_ - zc * s_) + 2, 9):>13}{g_fmt(math.exp(b_ + zc * s_) + 2, 9):>12}\n')
        ip_.txt('-' * 78 + '\n')

    def predict(ip_, kind, m_):
        out = np.full(ip_.data.n, np.nan)
        if kind in ('variance', 'var', 'v'):
            out[idx] = s2fit
        elif kind in ('residuals', 'resid', 'r'):
            out[idx] = efit
        elif kind in (None, '', 'xb'):
            out[idx] = xbfit
        else:
            raise StataError(f'predict option {kind} not supported after arch', 198)
        return out
    est['replay'] = lambda ip_: replay(ip_, False)
    est['predict'] = predict
    ip.set_estimation(est)
    replay(ip, True)


def _arma_filter(u, arc, ar, mac, ma, seg):
    if not ar and not ma:
        return u.copy()
    n = u.size
    e = np.zeros(n)
    for t in range(n):
        v = u[t]
        for co, l in zip(arc, ar):
            j = t - l
            v -= co * (u[j] if (j >= 0 and seg[j] == seg[t]) else 0.0)
        for co, l in zip(mac, ma):
            j = t - l
            v -= co * (e[j] if (j >= 0 and seg[j] == seg[t]) else 0.0)
        e[t] = v
    return e


def _arch_start(e2, A, Tt, G, egarch_mode):
    """Starting values for the ARCH equation (arch.ado): Monfort whitening of e^2, regression of e^2 on
    its lags and the whitened series, ARMA->GARCH mapping, FixARCH."""
    from .tsarima import _lagg, _ols_r
    lags = sorted(A + Tt + G)
    minlag = lags[0]
    maxlag = ((max(sorted(A + Tt)) if (A or Tt) else 0) + (max(G) if G else 0)) if (A or Tt) else max(G)
    n = e2.size
    k = 30
    usemin = usemax = None
    while k > 0:
        usemin = min(minlag, 1) if k > 10 else minlag
        usemax = max(maxlag, usemin + k)
        ok = np.ones(n, bool)
        ok[:usemax] = False
        if ok.sum() > 2 * k:
            break
        k -= 5
    try:
        _, sig, _, _ = _ols_r(e2, [_lagg(e2, l) for l in range(usemin, usemax + 1)], True)
    except (np.linalg.LinAlgError, ValueError):
        sig = None
    etar = np.where(e2 > 0, e2, 0.0)
    cols = [_lagg(e2, l) for l in A]
    if Tt:
        # tarch term uses (e>0)*e^2; e sign is not available here, use half of e^2 as a proxy
        cols += [_lagg(etar * 0.5, l) for l in Tt]
    if G and sig is not None:
        cols += [_lagg(sig, l) for l in G]
    try:
        T, _, _, _ = _ols_r(e2, cols, True)
        T = list(T)
    except (np.linalg.LinAlgError, ValueError):
        T = [0.0] * (len(A) + len(Tt) + len(G)) + [float(np.mean(e2))]
    if len(T) != len(A) + len(Tt) + len(G) + 1:
        T = [0.0] * (len(A) + len(Tt) + len(G)) + [float(np.mean(e2))]
    nar = len(A) + len(Tt)
    for i, l in enumerate(A):
        if l in G:
            gj = nar + G.index(l)
            T[i] = T[i] + T[gj]
            T[gj] = -T[gj]
    if not Tt:
        cols_ = len(T) - 1
        bsum = 0.0
        for i in range(cols_):
            if T[i] < .01:
                T[i] = .01
            bsum += T[i]
        if T[-1] <= 0:
            T[-1] = .01
        if bsum > .95:
            for i in range(cols_):
                T[i] = (.95 / bsum) * T[i]
    return T
