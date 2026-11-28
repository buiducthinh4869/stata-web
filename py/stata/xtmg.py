"""xtmg (Eberhardt: MG / CCEMG / AMG) and xtpmg (Blackburne & Frank: PMG / MG / DFE), ported from the SSC ados."""
import math

import numpy as np
from scipy import stats as sps

from .design import coef_table, Col
from .errors import StataError
from .fmt import g_fmt, comma_int, fmt_num
from .interp import command
from .syntax import parse_command, split_top


def _expand(ip, toks):
    from .tsops import expand_token
    d = ip.data
    out = []
    for t in toks:
        out += expand_token(t) if '.' in t or '(' in t else d.expand_varlist([t])
    return out


def _vals(ip, v):
    from .tsuni import series
    if '.' in v:
        return series(ip, v)[0]
    d = ip.data
    return d.cols[d.resolve(v)].astype(float)


def _col(nm):
    from .tsops import split_name
    if '.' in nm:
        ch, base = split_name(nm)
        c = Col(name=nm, values=None, term=0, group='', level='', kind='cont', vars=[base])
        c.tsvar, c.tschain = base, ch
        return c
    return Col(name=nm, values=None, term=0, group='', level='', kind='cons' if nm == '_cons' else 'cont',
               vars=[nm])


def _ols(y, X):
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ (X.T @ y)
    e = y - X @ b
    n, k = X.shape
    rss = float(e @ e)
    V = XtXi * rss / (n - k) if n > k else np.full((k, k), np.nan)
    ll = -0.5 * n * (1 + math.log(2 * math.pi) + math.log(rss / n))
    return b, V, e, rss, ll


def _mge(B):
    n = B.shape[0]
    m = B.mean(0)
    D = B - m
    return m, D.T @ D / (n * (n - 1))


def _need_panel(ip):
    d = ip.data
    if not d.ts or not d.ts.get('panel') or not d.ts.get('time'):
        raise StataError('panel and time variables must be set; use xtset panelvar timevar', 459)
    return d.ts['panel'], d.ts['time']


def _xtheader(ip, title_lines, ivar, n, ng, gmin, gavg, gmax, chi2, df):
    ip.txt('\n')
    for i, (txt, emph) in enumerate(title_lines):
        if emph is None:
            ip.txt(txt + '\n')
        else:
            ip.txt(txt)
            ip.res(emph[0])
            ip.txt(emph[1] + '\n')
    ip.txt('\n' + f'{"Mean Group type estimation":<48}{"Number of obs":<18}= ')
    ip.res(fmt_num(n, '%10.0fc') + '\n')
    ip.txt('Group variable: ')
    ip.res(f'{ivar[:14]:<32}')
    ip.txt(f'{"Number of groups":<18}= ')
    ip.res(fmt_num(ng, '%10.0fc') + '\n\n')
    ip.txt(' ' * 48 + 'Obs per group:\n')
    for k, v, f in (('min', gmin, '%10.0fc'), ('avg', gavg, '%10.1fc'), ('max', gmax, '%10.0fc')):
        ip.txt(' ' * 62 + f'{k} = ')
        ip.res(fmt_num(v, f) + '\n')
    lab = f'Wald chi2({df})'
    ip.txt('\n' + ' ' * 48 + f'{lab:<18}=  ')
    ip.res(f'{chi2:9.2f}\n')
    ip.txt(' ' * 48 + f'{"Prob > chi2":<18}=')
    pv = sps.chi2.sf(chi2, df) if df else np.nan
    ip.res(' ' * 5 + (f'{pv:6.4f}' if pv == pv else '     .') + '\n\n')


@command('xtmg')
def cmd_xtmg(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    ivar, tvar = _need_panel(ip)
    vl = _expand(ip, split_top(p.body))
    dep, ind = vl[0], vl[1:]
    trend, nocons = p.opts.has('trend'), p.opts.has('noconstant', 6)
    aug, impose, cce = p.opts.has('augment', 3), p.opts.has('impose', 3), p.opts.has('cce')
    full, robust = p.opts.has('full'), p.opts.has('robust', 3)
    resname = p.opts.get('res', 3)
    if robust:
        raise StataError('option robust (rreg-based means) is not available in Stata Web', 198)
    if aug and cce:
        raise StataError('You can only select one of options -cce- or -augment-.', 198)
    if impose and not aug:
        raise StataError('You cannot select option -impose- without -augment-.', 198)
    if (cce or aug) and nocons:
        raise StataError('The CCEMG/AMG routines require a group-specific intercept.', 198)
    y = _vals(ip, dep)
    X = {v: _vals(ip, v) for v in ind}
    touse = ip.mark(p, d) & ~np.isnan(y) & ~np.isnan(d.cols[ivar].astype(float))
    for v in ind:
        touse &= ~np.isnan(X[v])
    rhs = len(ind) + (0 if nocons else 1) + (1 if trend else 0) + (1 if (aug and not impose) else 0)
    pid = d.cols[ivar].astype(float)
    tv = d.cols[tvar].astype(float)
    cnt = {g: int(np.sum(touse & (pid == g))) for g in np.unique(pid[touse])}
    nobso = int(touse.sum())
    small = np.array([cnt.get(g, 0) <= rhs for g in pid])
    touse &= ~small
    nobs = int(touse.sum())
    if nobs < nobso:
        ip.txt(' Note: ')
        ip.res(f'{nobso - nobs}')
        ip.txt(' obs. dropped (panels too small)\n')
    ids = np.unique(pid[touse])
    ng = ids.size
    if ng < 2:
        raise StataError('insufficient number of groups', 2001)
    tmin_all = np.nanmin(tv)
    trendv = tv - tmin_all
    extra = []                 # names of extra regressors (in order) and their values
    if aug:
        # first stage: pooled FD regression with differenced year dummies (no constant)
        yrs = np.unique(tv[~np.isnan(tv)])
        Dy = y - ip._lag(d, y, 1)
        DX = [X[v] - ip._lag(d, X[v], 1) for v in ind]
        dums = [(tv == t).astype(float) for t in yrs]
        Ddums = [u - ip._lag(d, u, 1) for u in dums]
        ok = touse & ~np.isnan(Dy)
        for a in DX + Ddums:
            ok &= ~np.isnan(a)
        Z = np.column_stack([a[ok] for a in DX] + [u[ok] for u in Ddums[1:]])
        bz = np.linalg.lstsq(Z, Dy[ok], rcond=None)[0]
        cdp = np.r_[0.0, bz[len(ind):]]
        tmap = {t: cdp[i] for i, t in enumerate(yrs)}
        aug_c = np.array([tmap.get(t, 0.0) for t in tv])
        if impose:
            y = y - aug_c
        else:
            extra.append(('c_d_p', aug_c))
    if trend:
        extra.append(('trend', trendv))
    if cce:
        for v, val in [(dep, y)] + [(v, X[v]) for v in ind]:
            avg = np.full(d.N, np.nan)
            for t in np.unique(tv[touse]):
                s = touse & (tv == t)
                avg[s] = val[s].mean()
            extra.append(('_' + v.replace('.', '_'), avg))
    names = list(ind)
    if aug and not impose:
        names.append('c_d_p')
    if trend:
        names.append('trend')
    if cce:
        names += [nm for nm, _ in extra if nm.startswith('_')]
    if not nocons:
        names.append('_cons')
    ext = dict(extra)
    B, Vs, waldt = [], [], 0
    resid = np.full(d.N, np.nan)
    gmin, gmax = 10 ** 9, 0
    for g in ids:
        s = np.flatnonzero(touse & (pid == g))
        cols = [X[v][s] for v in ind]
        if aug and not impose:
            cols.append(ext['c_d_p'][s])
        if trend:
            cols.append(trendv[s])
        if cce:
            cols += [ext[nm][s] for nm in names if nm.startswith('_') and nm != '_cons']
        if not nocons:
            cols.append(np.ones(s.size))
        Xi = np.column_stack(cols)
        b, V, e, rss, _ = _ols(y[s], Xi)
        B.append(b)
        Vs.append(np.diag(V))
        resid[s] = e
        gmin, gmax = min(gmin, s.size), max(gmax, s.size)
        if trend:
            j = names.index('trend')
            if V[j, j] > 0:
                tstat = b[j] / math.sqrt(V[j, j])
                if 2 * sps.t.sf(abs(tstat), s.size - Xi.shape[1]) < .05:
                    waldt += 1
    B = np.array(B)
    bm, Vm = _mge(B)
    sig2 = float(np.nanmean(resid[touse] ** 2))
    gavg = nobs / ng
    slopes = [names.index(v) for v in ind]
    if trend:
        slopes.append(names.index('trend'))
    try:
        Vss = Vm[np.ix_(slopes, slopes)]
        chi2 = float(bm[slopes] @ np.linalg.solve(Vss, bm[slopes]))
        dfm = len(slopes)
    except np.linalg.LinAlgError:
        chi2, dfm = np.nan, 0
    if resname:
        if d.has(resname):
            raise StataError(f'Variable {resname} to hold residuals already exists.', 110)
        d.add_var(resname, np.where(touse, resid, np.nan), 'float')
    info = [_col(nm) for nm in names]
    title2 = 'AMG' if aug else ('CCEMG' if cce else 'MG')
    est = {'nomargins': True, 'names': names, 'b': bm, 'V': Vm, 'colinfo': info, 'df_r': None,
           'depvar': ('adjusted ' + dep) if (aug and impose) else dep, 'sample': touse, 'cmd': 'xtmg', 'stat': 'z',
           'scalars': {'N': nobs, 'N_g': ng, 'g_min': gmin, 'g_avg': gavg, 'g_max': gmax, 'chi2': chi2,
                       'df_m': dfm, 'sigma': sig2},
           'macros': {'cmd': 'xtmg', 'depvar': dep, 'ivar': ivar, 'tvar': tvar, 'title': 'Mean Group type estimation',
                      'title2': title2, 'chi2type': 'Wald'},
           'matrices_extra': {'betas': B}}
    if trend:
        est['scalars']['trend_sig'] = waldt / ng

    def replay(ip_):
        tl = []
        if cce:
            tl.append(('Pesaran (2006) Common Correlated Effects Mean Group estimator', None))
        elif aug:
            tl.append(('Augmented Mean Group estimator (Bond & Eberhardt, 2009; Eberhardt & Teal, 2010)', None))
            tl.append(('', None))
            if impose:
                tl.append(('Common dynamic process ', ('imposed', ' with unit coefficient ')))
                tl.append(('    Dependent variable', (f' adjusted {dep}', '')))
            else:
                tl.append(('Common dynamic process ', ('included', ' as additional regressor')))
        else:
            tl.append(('Pesaran & Smith (1995) Mean Group estimator', None))
        tl.append(('', None))
        tl.append(('All coefficients present represent averages across groups (', (ivar, ')')))
        if robust:
            tl.append(('Coefficient averages computed as', (' outlier-robust', ' means (using rreg)')))
        else:
            tl.append(('Coefficient averages computed as ', ('unweighted', ' means')))
        _xtheader(ip_, tl, ivar, nobs, ng, gmin, gavg, gmax, chi2, dfm)
        coef_table(ip_, est, 'z', depvar=est['depvar'])
        ip_.txt('Root Mean Squared Error (sigma): ')
        ip_.res(f'{math.sqrt(sig2):5.4f}\n')
        if robust:
            ip_.txt("(RMSE uses residuals from group-specific regressions: unaffected by 'robust').\n")
        if aug and not impose:
            ip_.txt('Variable ')
            ip_.res('c_d_p')
            ip_.txt(' refers to the common dynamic process.\n')
        if cce:
            ip_.txt('Cross-section averaged ')
            ip_.res('regressors')
            ip_.txt(' are marked by the suffix: \n')
            ip_.res('   ' + ', '.join('_' + v.replace('.', '_') for v in [dep] + ind))
            ip_.txt(' respectively.\n')
        if resname:
            ip_.txt('Residual series based on country regressions stored in variable: ')
            ip_.res(resname + '\n')
        if trend:
            ip_.txt('Variable ')
            ip_.res('trend')
            ip_.txt(' refers to a group-specific linear trend.\n')
            ip_.txt('Share of group-specific trends significant at 5% level: ')
            ip_.res(f'{waldt / ng:5.3f}')
            ip_.txt(f' (= {waldt} trends)\n')
        if full:
            ip_.txt('\n\n' + ' ' * 24 + 'Group-specific coefficients\n')
            for gi, g in enumerate(ids):
                sub = {'names': names, 'b': B[gi], 'V': np.diag(Vs[gi]), 'colinfo': [_col(nm) for nm in names],
                       'df_r': None}
                coef_table(ip_, sub, 'z', depvar=f'Group {gi + 1}')
    est['replay'] = replay
    est['predict'] = None
    ip.set_estimation(est)
    replay(ip)


# ---------------------------------------------------------------- xtpmg
def _multi_eq_table(ip, eqs, b, V, depvar=''):
    """eqs: list of (eqname, [names]) in order; b, V cover the concatenation."""
    pos = 0
    K = len(eqs)
    for i, (eq, nms) in enumerate(eqs):
        ix = list(range(pos, pos + len(nms)))
        pos += len(nms)
        sub = {'names': nms, 'b': b[ix], 'V': V[np.ix_(ix, ix)], 'colinfo': [_col(n) for n in nms], 'df_r': None}
        coef_table(ip, sub, 'z', depvar=depvar or ' ', header=(i == 0), footer=(i == K - 1), eq_title=eq)


@command('xtpmg')
def cmd_xtpmg(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    ivar, tvar = _need_panel(ip)
    mods = [m for m in ('mg', 'dfe', 'pmg') if p.opts.has(m)]
    if len(mods) > 1:
        raise StataError('choose only one of pmg, mg or dfe', 198)
    model = mods[0] if mods else 'pmg'
    vl = _expand(ip, split_top(p.body))
    SRy, SRx = vl[0], vl[1:]
    lr = _expand(ip, split_top(p.opts.get('lr', 2) or ''))
    if not lr:
        raise StataError('option lr() required', 198)
    LRy, LRx = lr[0], lr[1:]
    ecname = p.opts.get('ec', 2) or '__ec'
    if p.opts.has('replace') and d.has(ecname):
        d.drop_vars([ecname])
    if model != 'dfe' and d.has(ecname):
        raise StataError(f'Variable {ecname} already exists. Either drop the variable or specify another name as '
                         'EC option.', 110)
    nocons = p.opts.has('noconstant', 6)
    full = p.opts.has('full')
    allv = [SRy] + SRx + [LRy] + LRx
    V_ = {v: _vals(ip, v) for v in allv}
    touse = ip.mark(p, d) & ~np.isnan(d.cols[ivar].astype(float))
    for v in allv:
        touse &= ~np.isnan(V_[v])
    pid = d.cols[ivar].astype(float)
    ids = np.unique(pid[touse])
    ng = ids.size
    N = int(touse.sum())
    rows = [np.flatnonzero(touse & (pid == g)) for g in ids]
    gmin, gmax = min(r.size for r in rows), max(r.size for r in rows)
    gavg = N / ng
    yS = V_[SRy]
    XS = np.column_stack([V_[v] for v in SRx]) if SRx else np.zeros((d.N, 0))
    yL = V_[LRy]
    XL = np.column_stack([V_[v] for v in LRx])
    kl = len(LRx)

    def panel_X(s, ec):
        cols = [ec[s]] + [XS[s, j] for j in range(XS.shape[1])] + ([] if nocons else [np.ones(s.size)])
        return np.column_stack(cols)
    srnames = [ecname] + SRx + ([] if nocons else ['_cons'])
    if model == 'pmg':
        th0 = np.linalg.lstsq(XL[touse], yL[touse], rcond=None)[0]

        def cll(th):
            ec = yL - XL @ th
            tot = 0.0
            for s in rows:
                tot += _ols(yS[s], panel_X(s, ec))[4]
            return tot
        from scipy import optimize
        r = optimize.minimize(lambda t: -cll(t), th0, method='BFGS', options={'gtol': 1e-7})
        th = r.x
        for _ in range(50):          # Newton polish with numerical derivatives
            h = 1e-5 * np.maximum(1, np.abs(th))
            g = np.zeros(kl)
            H = np.zeros((kl, kl))
            f0 = cll(th)
            for j in range(kl):
                ej = np.zeros(kl)
                ej[j] = h[j]
                fp, fm = cll(th + ej), cll(th - ej)
                g[j] = (fp - fm) / (2 * h[j])
                H[j, j] = (fp - 2 * f0 + fm) / h[j] ** 2
                for m in range(j):
                    em = np.zeros(kl)
                    em[m] = h[m]
                    H[j, m] = H[m, j] = (cll(th + ej + em) - cll(th + ej - em) - cll(th - ej + em)
                                         + cll(th - ej - em)) / (4 * h[j] * h[m])
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(step)) or cll(th - step) < f0 - 1e-9:
                break
            th = th - step
            if np.max(np.abs(step)) < 1e-10:
                break
        ec = yL - XL @ th
        Bs, sig, ll = [], [], 0.0
        ks = len(srnames)
        Gxx = np.zeros((kl, kl))
        Grow = []
        Gblk = []
        resid = np.full(d.N, np.nan)
        for s in rows:
            Xi = panel_X(s, ec)
            b, V, e, rss, l = _ols(yS[s], Xi)
            ll += l
            s2 = rss / s.size
            Bs.append(b)
            sig.append(s2)
            resid[s] = e
            phi = b[0]
            Gxx += (XL[s].T @ XL[s]) * (phi ** 2 / s2)
            Grow.append(-(phi / s2) * (XL[s].T @ Xi))
            Gblk.append(Xi.T @ Xi / s2)
        Bs = np.array(Bs)
        dim = kl + ng * ks
        Gm = np.zeros((dim, dim))
        Gm[:kl, :kl] = Gxx
        for i in range(ng):
            a = kl + i * ks
            Gm[:kl, a:a + ks] = Grow[i]
            Gm[a:a + ks, :kl] = Grow[i].T
            Gm[a:a + ks, a:a + ks] = Gblk[i]
        Vfull = np.linalg.pinv(Gm)
        thV = Vfull[:kl, :kl]
        mb, mV = _mge(Bs)
        b_out = np.r_[th, mb]
        V_out = np.zeros((kl + ks, kl + ks))
        V_out[:kl, :kl] = thV
        V_out[kl:, kl:] = mV
        sigma = float(np.nanmean(resid[touse] ** 2))
        d.add_var(ecname, np.where(touse, ec, np.nan), 'double')
        title = 'Pooled Mean Group Regression'
        saved = 'pmg' if not full else 'PMG'
        if full:
            b_out = np.r_[th, Bs.ravel()]
            V_out = Vfull
    elif model == 'mg':
        Bs, ll = [], 0.0
        resid = np.full(d.N, np.nan)
        Vblocks = []
        for s in rows:
            cols = [yL[s]] + [XL[s, j] for j in range(kl)] + [XS[s, j] for j in range(XS.shape[1])] + \
                ([] if nocons else [np.ones(s.size)])
            Xi = np.column_stack(cols)
            b, V, e, rss, l = _ols(yS[s], Xi)
            ll += l
            resid[s] = e
            phi = b[0]
            g = np.r_[-b[1:1 + kl] / phi, phi, b[1 + kl:]]
            J = np.zeros((g.size, b.size))
            for j in range(kl):
                J[j, 0] = b[1 + j] / phi ** 2
                J[j, 1 + j] = -1 / phi
            J[kl, 0] = 1
            for j in range(b.size - 1 - kl):
                J[kl + 1 + j, 1 + kl + j] = 1
            Bs.append(g)
            Vblocks.append(J @ V @ J.T)
        Bs = np.array(Bs)
        mb, mV = _mge(Bs)
        b_out, V_out = mb, mV
        th = mb[:kl]
        ec = yL - XL @ th
        d.add_var(ecname, np.where(touse, ec, np.nan), 'double')
        sigma = float(np.nanmean(resid[touse] ** 2))
        title = 'Mean Group Estimation: '
        saved = 'mg' if not full else 'MG'
        if full:
            b_out = Bs.ravel()
            V_out = np.zeros((b_out.size, b_out.size))
            m = Bs.shape[1]
            for i, Vb in enumerate(Vblocks):
                V_out[i * m:(i + 1) * m, i * m:(i + 1) * m] = Vb
    else:
        # dynamic fixed effects: xtreg fe of SRy on LRy LRx SRx, then nlcom
        _, gid = np.unique(pid[touse], return_inverse=True)
        G = ng
        W = np.column_stack([yL[touse], XL[touse], XS[touse]])
        yy = yS[touse]
        Ti = np.bincount(gid)

        def dm(v):
            return v - (np.bincount(gid, v) / Ti)[gid] + v.mean()
        Wt = np.column_stack([dm(W[:, j]) for j in range(W.shape[1])] + [np.ones(yy.size)])
        yt = dm(yy)
        XtXi = np.linalg.pinv(Wt.T @ Wt)
        b = XtXi @ Wt.T @ yt
        e = yt - Wt @ b
        dfr = yy.size - Wt.shape[1] - G + 1
        Vb = XtXi * float(e @ e) / dfr
        cl = p.opts.get('cluster', 2)
        if cl:
            cv = d.cols[d.resolve(cl)][touse]
            _, cg = np.unique(cv, return_inverse=True)
            Gc = int(cg.max()) + 1
            S = np.zeros((Gc, Wt.shape[1]))
            np.add.at(S, cg, Wt * e[:, None])
            Vb = XtXi @ (S.T @ S) @ XtXi * (Gc / (Gc - 1)) * ((yy.size - 1) / (yy.size - Wt.shape[1]))
        if nocons:
            pass
        phi = b[0]
        kk = b.size
        g = np.r_[-b[1:1 + kl] / phi, phi, b[1 + kl:]]
        J = np.zeros((g.size, kk))
        for j in range(kl):
            J[j, 0] = b[1 + j] / phi ** 2
            J[j, 1 + j] = -1 / phi
        J[kl, 0] = 1
        for j in range(kk - 1 - kl):
            J[kl + 1 + j, 1 + kl + j] = 1
        b_out, V_out = g, J @ Vb @ J.T
        sigma = math.sqrt(float(e @ e) / dfr)
        ll = np.nan
        title = 'Dynamic Fixed Effects Regression: '
        saved = 'DFE'
        full = False
    lrnames = list(LRx)
    if full and model in ('pmg', 'mg'):
        eqs = [(ecname, lrnames)] if model == 'pmg' else []
        for g in ids:
            lab = f'{ivar}_{int(g) if float(g).is_integer() else g}'
            eqs.append((lab, (lrnames + srnames) if model == 'mg' else srnames))
        if model == 'mg':
            eqs = [(lab, [f'{ecname}:{n}' if i < kl else n for i, n in enumerate(nms)])
                   for lab, nms in eqs]
            eqs = [(lab, [n.split(':')[-1] for n in nms]) for lab, nms in eqs]
    else:
        eqs = [(ecname, lrnames), ('SR', srnames)]
    allnames = [n for _, nms in eqs for n in nms]
    fullnames = [f'{eq}:{n}' for eq, nms in eqs for n in nms]
    est = {'nomargins': True, 'names': fullnames, 'b': b_out, 'V': V_out,
           'colinfo': [_col(n) for n in allnames], 'df_r': None, 'depvar': SRy, 'sample': touse, 'cmd': 'xtpmg',
           'stat': 'z',
           'scalars': {'N': N, 'n_g': ng, 'g_min': gmin, 'g_avg': gavg, 'g_max': gmax, 'sigma': sigma},
           'macros': {'cmd': 'xtpmg', 'model': saved, 'depvar': SRy, 'ivar': ivar, 'tvar': tvar}}
    if ll == ll:
        est['scalars']['ll'] = ll

    def replay(ip_):
        if model == 'pmg':
            ip_.txt('\nPooled Mean Group Regression\n(Estimate results saved as ')
            ip_.res(saved)
            ip_.txt(')\n\n')
            ip_.txt('Panel Variable (i): ')
            ip_.res(f'{ivar[:12]:<28}')
            ip_.txt(f'{"Number of obs":<19}= ')
            ip_.res(fmt_num(N, '%9.0f') + '\n')
            ip_.txt('Time Variable (t): ')
            ip_.res(f'{tvar[:12]:<29}')
            ip_.txt(f'{"Number of groups ":<19}= ')
            ip_.res(fmt_num(ng, '%9.0g') + '\n')
            ip_.txt(' ' * 48 + f'{"Obs per group: min":<19}= ')
            ip_.res(fmt_num(gmin, '%9.0g') + '\n')
            ip_.txt(' ' * 63 + f'{"avg":<4}= ')
            ip_.res(fmt_num(gavg, '%9.1f') + '\n')
            ip_.txt(' ' * 63 + f'{"max":<4}= ')
            ip_.res(fmt_num(gmax, '%9.0g') + '\n\n')
            ip_.txt(' ' * 48 + f'{"Log Likelihood":<19}= ')
            ip_.res(fmt_num(ll, '%9.0g') + '\n')
        elif model == 'mg':
            ip_.txt('\n' + '-' * 78 + '\nMean Group Estimation: ')
            ip_.res('Error Correction Form\n')
            ip_.txt('(Estimate results saved as ')
            ip_.res(saved)
            ip_.txt(')\n' + '-' * 78 + '\n')
        else:
            if p.opts.get('cluster', 2):
                ip_.txt('\n' + '-' * 78 + '\n')
                ip_.res(f'Standard errors adjusted with cluster({p.opts.get("cluster", 2)}) option.\n')
            ip_.txt('-' * 78 + '\nDynamic Fixed Effects Regression: ')
            ip_.res('Estimated Error Correction Form\n')
            ip_.txt('(Estimate results saved as ')
            ip_.res('DFE')
            ip_.txt(')\n' + '-' * 78 + '\n')
        _multi_eq_table(ip_, eqs, b_out, V_out, depvar=' ')
    est['replay'] = replay
    est['predict'] = None
    ip.set_estimation(est)
    replay(ip)
