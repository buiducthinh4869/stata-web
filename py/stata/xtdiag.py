"""Panel diagnostics and corrections: xttest0 (official), xttest3, xtserial, xtcsd, xtscc (SSC; ported from
xttest0.ado, xttest3.ado (Baum), xtserial.ado (Drukker), xtcsd.ado (De Hoyos & Sarafidis), xtscc.ado
(Hoechle))."""
import math

import numpy as np
from scipy import stats as sps

from .design import coef_table, Col
from .errors import StataError
from .estim import prepare_linear, depvar_values, _gmeans, _group_index
from .fmt import fmt_num, g_fmt, comma_int
from .interp import command
from .syntax import parse_command, split_top


def _need_xtreg(ip, models, what):
    e = ip.est
    if not e or e.get('cmd') != 'xtreg':
        raise StataError('last estimates not found' if not e else f'{what}: last estimates not xtreg', 301)
    if e.get('model') not in models:
        raise StataError(f"last estimates not xtreg, {models[0]}", 301)
    return e


def _xt_e(ip, e):
    """idiosyncratic residual e_it (predict, e) on e(sample)."""
    d = ip.data
    return e['predict'](ip, 'e', None)


# ---------------------------------------------------------------- xttest0
@command('xttest0')
def cmd_xttest0(ip, cmd, rest, by):
    e = _need_xtreg(ip, ('re',), 'xttest0')
    d = ip.data
    ivar = e['macros']['ivar']
    mask = e['sample']
    yv = e['depvar']
    y = depvar_values(ip, d, yv)[1]
    X = e['Xpool'] if 'Xpool' in e else None
    yy = y[mask]
    b = np.linalg.lstsq(X, yy, rcond=None)[0]
    u = yy - X @ b
    gid = _group_index(d, ivar, mask)
    Ti = np.bincount(gid).astype(float)
    M11 = float(np.sum(Ti ** 2))
    s = np.bincount(gid, weights=u)
    A1 = 1 - float(np.sum(s ** 2)) / float(u @ u)
    N = yy.size
    su = e['scalars']['sigma_u']
    se = e['scalars']['sigma_e']
    if su <= 0:
        LM, pv = 0.0, 1.0
    else:
        LM = (N ** 2 / 2) * (A1 ** 2 / (M11 - N))
        pv = sps.chi2.sf(LM, 1) / 2
    vy = float(np.var(yy, ddof=1))
    ip.txt('\nBreusch and Pagan Lagrangian multiplier test for random effects\n\n')
    ip.txt(f'        {yv}[{ivar},t] = Xb + u[{ivar}] + e[{ivar},t]\n\n')
    ip.txt('        Estimated results:\n')
    ip.txt(f'{"":25}|{"":7}Var{"":5}sd = sqrt(Var)\n')
    ip.txt(f'{"":16}---------+-----------------------------\n')
    for lab, v in ((yv[:9], vy), ('e', se ** 2), ('u', su ** 2)):
        ip.txt(f'{lab:>24} |  ')
        ip.res(f'{fmt_num(v, "%9.0g")}{"":6}{fmt_num(math.sqrt(v), "%9.0g")}\n')
    ip.txt(f'\n        Test:   Var(u) = 0\n{"":29}chibar2(01) = ')
    ip.res(f'{LM:8.2f}\n')
    ip.txt(f'{"":26}Prob > chibar2 = ')
    ip.res(f'{pv:7.4f}\n')
    ip.set_r(scalars={'lm': LM, 'df': 1, 'p': pv})


# ---------------------------------------------------------------- xttest3
@command('xttest3')
def cmd_xttest3(ip, cmd, rest, by):
    e = ip.est
    if not e or e.get('cmd') not in ('xtreg', 'xtgls'):
        raise StataError('last estimates not found', 301)
    if e['cmd'] == 'xtreg' and e.get('model') != 'fe':
        raise StataError('last estimates not xtreg, fe', 301)
    d = ip.data
    mask = e['sample']
    ivar = e['macros'].get('ivar') or d.ts['panel']
    if e['cmd'] == 'xtreg':
        r = _xt_e(ip, e)[mask]
    else:
        y = depvar_values(ip, d, e['depvar'])[1]
        r = (y - e['predict'](ip, 'xb', None))[mask]
    gid = _group_index(d, ivar, mask)
    G = int(gid.max()) + 1
    Ti = np.bincount(gid).astype(float)
    e2 = r * r
    si = np.bincount(gid, weights=e2) / Ti
    esig = (e2 - si[gid]) ** 2
    v = np.bincount(gid, weights=esig) / Ti / (Ti - 1)
    N = r.size
    var_all = float(np.var(r, ddof=1)) * (N - 1) / N
    W = float(np.sum((si - var_all) ** 2 / v))
    pv = float(sps.chi2.sf(W, G))
    ip.txt('\nModified Wald test for groupwise heteroskedasticity\n')
    ip.txt('in fixed effect regression model\n\n' if e['cmd'] == 'xtreg' else
           'in cross-sectional time-series FGLS regression model\n\n')
    ip.txt('H0: sigma(i)^2 = sigma^2 for all i\n\n')
    ip.txt(f'chi2 ({G})  =   ')
    ip.res(f'{W:8.2f}\n')
    ip.txt('Prob>chi2 =    ')
    ip.res(f'{pv:8.4f}\n\n')
    ip.set_r(scalars={'wald': W, 'df': G, 'p': pv})


# ---------------------------------------------------------------- xtserial
@command('xtserial')
def cmd_xtserial(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    if not d.ts or not d.ts.get('panel'):
        raise StataError('panel variable not specified', 498)
    if not d.ts.get('time'):
        raise StataError('time variable not specified', 498)
    toks = split_top(p.body)
    if not toks:
        raise StataError('varlist required', 100)
    ivar = d.ts['panel']
    vlist = []
    for t in toks:
        vlist += d.expand_varlist([t])
    base = ip.mark(p, d)
    for v in vlist:
        base &= ~np.isnan(d.cols[v].astype(float))
    show = p.opts.has('output', 1)
    from .estim import ols_core, make_ols_est, cluster_ids
    q0 = ip.quiet
    if not show:
        ip.quiet += 1
    try:
        dv = [f'D.{v}' for v in vlist]
        yname, y, cols, m1, w, wt = prepare_linear(ip, p, d, dv[1:], dv[0], extra_mask=base)
        f1 = ols_core(ip, d, y, cols, m1, None, None, 'cluster', ivar, nocons=True)
        f1.depvar = yname
        est1 = make_ols_est(ip, f1, 'regress', p)
        if show:
            est1['replay'](ip)
    finally:
        ip.quiet = q0
    res = np.full(d.n, np.nan)
    res[f1.mask] = f1.e
    lres = ip._lag(d, res, 1)
    m2 = f1.mask & ~np.isnan(lres) & base
    y2, x2 = res[m2], lres[m2]
    bb = float(x2 @ y2 / (x2 @ x2))
    u = y2 - bb * x2
    cg, _ = cluster_ids(d, ivar, m2)
    G = int(cg.max()) + 1
    n = y2.size
    sc = np.bincount(cg, weights=x2 * u)
    Vb = float(np.sum(sc ** 2)) / (x2 @ x2) ** 2 * (G / (G - 1)) * ((n - 1) / (n - 1))
    F = (bb + .5) ** 2 / Vb
    df_r = G - 1
    pv = float(sps.f.sf(F, 1, df_r))
    ip.txt('\nWooldridge test for autocorrelation in panel data\nH0: no first order autocorrelation\n')
    ip.txt(f'    F({1:3d},{df_r:8d}) = ')
    ip.res(f'{F:10.3f}\n')
    ip.txt(f'{"":11}Prob > F = ')
    ip.res(f'{pv:11.4f}\n')
    ip.set_r(scalars={'corr': bb, 'df_r': df_r, 'F': F, 'df': 1, 'p': pv})


# ---------------------------------------------------------------- xtcsd
@command('xtcsd')
def cmd_xtcsd(ip, cmd, rest, by):
    p = parse_command(rest)
    e = ip.est
    if not e or e.get('cmd') != 'xtreg':
        raise StataError('last estimates not xtreg', 301)
    if e.get('model') not in ('fe', 're'):
        raise StataError('last estimates not xtreg, fe or re', 301)
    pes, fre, fri = p.opts.has('pesaran', 3), p.opts.has('frees', 3), p.opts.has('friedman', 3)
    if not (pes or fre or fri):
        raise StataError('Method option not specified', 198)
    if pes + fre + fri > 1:
        raise StataError('Too many method options specified', 198)
    d = ip.data
    mask = e['sample']
    ivar, tvar = d.ts['panel'], d.ts['time']
    r = _xt_e(ip, e)
    ids = np.unique(d.cols[ivar][mask])
    times = np.unique(d.cols[tvar][mask])
    G = ids.size
    W = np.full((times.size, G), np.nan)
    ii = np.searchsorted(ids, d.cols[ivar][mask])
    tt = np.searchsorted(times, d.cols[tvar][mask])
    W[tt, ii] = r[mask]
    tb = int(np.bincount(ii).min())
    if fre or fri:
        W = W[~np.isnan(W).any(axis=1)]
        minT = W.shape[0]
    E = np.ones((G, G))
    COR = np.ones((G, G))
    ABS = np.ones((G, G))
    Fm = np.ones((G, G))
    for i in range(G):
        for j in range(G):
            ok = ~np.isnan(W[:, i]) & ~np.isnan(W[:, j])
            a, b_ = W[ok, i], W[ok, j]
            if pes:
                if ok.sum() <= 2:
                    raise StataError('The panel is highly unbalanced.\nNot enough common observations across panel '
                                     "to perform Pesaran's test.", 2001)
                rho = float(np.corrcoef(a, b_)[0, 1])
                E[i, j] = rho * math.sqrt(ok.sum())
                ABS[i, j] = abs(rho)
                COR[i, j] = rho
            else:
                rs = float(sps.spearmanr(a, b_)[0])
                E[i, j] = rs ** 2
                Fm[i, j] = rs
                rho = float(np.corrcoef(a, b_)[0, 1])
                ABS[i, j] = abs(rho)
                COR[i, j] = rho
    off = lambda M: (M.sum() - np.trace(M)) / 2  # noqa: E731
    ip.txt(' \n')
    if p.opts.has('show'):
        ip.txt('Correlation matrix of residuals:\n')
        from .expr import Matrix
        nm = [f'__e{k + 1}' for k in range(G)]
        ip.dispatch_matrix_list = None
        for i in range(G):
            ip.txt(f'{nm[i]:>8}' + ''.join(fmt_num(COR[i, j], '%9.4f').rjust(10) for j in range(i + 1)) + '\n')
    out = {}
    if pes:
        cd = math.sqrt(2 / (G * (G - 1))) * off(E)
        pv = 2 * (1 - sps.norm.cdf(abs(cd)))
        ip.txt(' \n')
        ip.txt("Pesaran's test of cross sectional independence = ")
        ip.res(f'{cd:9.3f}')
        ip.txt(', Pr = ')
        ip.res(f'{pv:6.4f}\n')
        out['pesaran'] = cd
    elif fre:
        rave2 = (2 / (G * (G - 1))) * off(E)
        q = G * (rave2 - 1 / (tb - 1))
        if minT < 31:
            if minT < 4:
                raise StataError("the distribution of Frees' Q statistic requires T>3", 198)
            a1, a2, a3 = _FREES[minT]
            ip.txt(' \n')
            ip.txt("  Frees' test of cross sectional independence = ")
            ip.res(f'{q:9.3f}\n')
            ip.txt('|--------------------------------------------------------|\n')
            ip.txt("  Critical values from Frees' Q distribution\n")
            for al, v in (('0.10', a1), ('0.05', a2), ('0.01', a3)):
                ip.txt(f'                        alpha = {al} :')
                ip.res(f'{v:9.4f}\n')
        else:
            T_ = minT
            s1 = (32 / 25) * ((T_ + 2) ** 2 / ((T_ - 1) ** 3 * (T_ + 1) ** 2))
            s2 = (4 / 25) * (((5 * T_ + 6) ** 2 * (T_ - 3)) / ((T_ * (T_ - 1) ** 2) * (T_ + 1) ** 2))
            seq = math.sqrt(s1 + s2)
            ip.txt(' \n')
            ip.txt("Frees' test of cross sectional independence = ")
            ip.res(f'{q:9.3f}')
            ip.txt(', Pr = ')
            ip.res(f'{2 * (1 - sps.norm.cdf(q / seq)):6.4f}\n \n')
            ip.res('Warning: A normal distribution had been used to approximate Frees\' Q distribution\n')
        out['frees'] = q
    else:
        rave = (2 / (G * (G - 1))) * off(Fm)
        fr = (tb - 1) * ((G - 1) * rave + 1)
        ip.txt(' \n')
        ip.txt("Friedman's test of cross sectional independence = ")
        ip.res(f'{fr:9.3f}')
        ip.txt(', Pr = ')
        ip.res(f'{sps.chi2.sf(fr, G - 1):6.4f}\n')
        out['fried'] = fr
    if p.opts.has('abs'):
        absvm = off(ABS) * 2 / (G * (G - 1))
        ip.txt(' \nAverage absolute value of the off-diagonal elements = ')
        ip.res(f'{absvm:9.3f}\n')
    ip.set_r(scalars=out)


_FREES = {4: (0.582210758, 0.839114984, 1.421135642), 5: (0.489238114, 0.685977181, 1.10456324),
          6: (0.412658998, 0.567636868, 0.902706907), 7: (0.358288118, 0.492251034, 0.767775969),
          8: (0.316868873, 0.432452369, 0.660545679), 9: (0.282784751, 0.382564287, 0.581080225),
          10: (0.255853719, 0.342858814, 0.519780382), 11: (0.233292559, 0.310295348, 0.464867739),
          12: (0.213585942, 0.283763514, 0.425186549), 13: (0.198350566, 0.262031641, 0.390109604),
          14: (0.184133643, 0.243101807, 0.360299337), 15: (0.171938209, 0.226202271, 0.335078763),
          16: (0.161163782, 0.211642005, 0.312465021), 17: (0.152050026, 0.199631597, 0.292842932),
          18: (0.143773435, 0.188844992, 0.276263341), 19: (0.135985199, 0.178176611, 0.260105614),
          20: (0.129385699, 0.169520436, 0.246825304), 21: (0.1230993, 0.161076942, 0.23378533),
          22: (0.117399185, 0.153662139, 0.222533364), 23: (0.112413921, 0.147049964, 0.212859171),
          24: (0.107792641, 0.140834061, 0.203408604), 25: (0.103534605, 0.134976492, 0.19467908),
          26: (0.099552324, 0.129701827, 0.186993823), 27: (0.095832193, 0.124819992, 0.179400199),
          28: (0.092370839, 0.120406029, 0.172627827), 29: (0.089158218, 0.11597072, 0.16598793),
          30: (0.086149177, 0.111918056, 0.159835289)}


# ---------------------------------------------------------------- xtscc
@command('xtscc')
def cmd_xtscc(ip, cmd, rest, by):
    p = parse_command(rest)
    toks = split_top(p.body)
    if not toks:
        if ip.est and ip.est.get('cmd') == 'xtscc':
            return ip.est['replay'](ip)
        raise StataError('varlist required', 100)
    d = ip.data
    if not d.ts or not d.ts.get('panel') or not d.ts.get('time'):
        raise StataError('panel and time variables must be set; use xtset panelvar timevar', 459)
    ivar, tvar = d.ts['panel'], d.ts['time']
    fe, re_ = p.opts.has('fe'), p.opts.has('re')
    nocons = p.opts.has('noconstant', 5)
    if (fe or re_) and nocons:
        raise StataError(f"option noconstant not allowed with option {'fe' if fe else 're'}", 101)
    lag = p.opts.get('lag', 3)
    lag = abs(int(lag)) if lag not in (None, '') else None
    if p.weight:
        raise StataError('weights are not available with xtscc in Stata Web', 101)
    yname, y, cols, mask, w, wt = prepare_linear(ip, p, d, toks[1:], toks[0])
    mask &= ~np.isnan(d.cols[ivar].astype(float)) & ~np.isnan(d.cols[tvar].astype(float))
    tv = d.cols[tvar].astype(float)
    ut = np.unique(tv[mask])
    delta = d.ts.get('delta', 1) or 1
    if ut.size < int(round((ut.max() - ut.min()) / delta)) + 1:
        raise StataError(f'{tvar} is not regularly spaced: there are contemporaneous gap(s) across all subjects in '
                         f'{ivar}', 101)
    from .estim import assemble, detect_omitted
    Xall, names, info = assemble(cols, mask, cons=not nocons)
    active = detect_omitted(Xall, info, cons=not nocons)
    X = Xall[:, active]
    names = [info[j].name for j in active]
    info = [info[j] for j in active]
    yy = y[mask]
    gid = _group_index(d, ivar, mask)
    G = int(gid.max()) + 1
    Ti = np.bincount(gid).astype(float)
    n = yy.size
    kx = X.shape[1] - (0 if nocons else 1)
    extra = {}
    if not (fe or re_):
        Xr = X
        yr = yy
        b0 = np.linalg.lstsq(Xr, yr, rcond=None)[0]
        e0 = yr - Xr @ b0
        tss = float(np.sum((yr - yr.mean()) ** 2)) if not nocons else float(yr @ yr)
        extra['r2'] = 1 - float(e0 @ e0) / tss
        extra['rmse'] = math.sqrt(float(e0 @ e0) / (n - Xr.shape[1]))
        rank = Xr.shape[1]
        method = 'Pooled OLS'
    else:
        Xs = X[:, :-1]
        ym, Xm = _gmeans(yy, gid), _gmeans(Xs, gid)
        yw = yy - ym[gid] + yy.mean()
        Xw = Xs - Xm[gid] + Xs.mean(axis=0)
        Z = np.column_stack([Xw, np.ones(n)])
        bw = np.linalg.lstsq(Z, yw, rcond=None)[0]
        ew = yw - Z @ bw
        sse = float(ew @ ew)
        r2w = 1 - sse / float(np.sum((yw - yw.mean()) ** 2))
        dfm = kx
        sig_e = math.sqrt(sse / (n - dfm - G))
        rank = Z.shape[1]
        if fe:
            Xr, yr = Z, yw
            extra['r2_w'] = r2w
            method = 'Fixed-effects regression'
        else:
            Tbar = 1 / float(np.mean(1 / Ti))
            Zb = np.column_stack([Xm, np.ones(G)])
            bb = np.linalg.lstsq(Zb, ym, rcond=None)[0]
            eb = ym - Zb @ bb
            rmse_b = math.sqrt(float(eb @ eb) / (G - Zb.shape[1]))
            sig_u = max(math.sqrt(max(rmse_b ** 2 - sig_e ** 2 / Tbar, 0)), 0)
            ratio = sig_u / sig_e
            th = 1 - 1 / np.sqrt(Ti * ratio ** 2 + 1)
            Xr = np.column_stack([Xs - th[gid][:, None] * Xm[gid], 1 - th[gid]])
            yr = yy - th[gid] * ym[gid]
            method = 'Random-effects GLS regression'
            extra.update({'sigma_e': sig_e, 'sigma_u': sig_u, 'rho': sig_u ** 2 / (sig_u ** 2 + sig_e ** 2)})
    # Driscoll-Kraay
    tvals = tv[mask]
    order = np.lexsort((d.cols[ivar][mask].astype(float), tvals))
    Xo, yo, to = Xr[order], yr[order], tvals[order]
    XtXi = np.linalg.inv(Xo.T @ Xo)
    b = XtXi @ Xo.T @ yo
    u = yo - Xo @ b
    tu, tinv = np.unique(to, return_inverse=True)
    T = tu.size
    if lag is None or lag == 9999:
        lag = int(math.floor(4 * (T / 100) ** (2 / 9)))
    h = np.zeros((T, Xo.shape[1]))
    np.add.at(h, tinv, Xo * u[:, None])
    S = h.T @ h
    for j in range(1, lag + 1):
        Om = h[j:].T @ h[:-j]
        S += (1 - j / (lag + 1)) * (Om + Om.T)
    vcv = XtXi @ S @ XtXi
    if not p.opts.has('ase'):
        V = (T / (T - 1)) * ((n - 1) / (n - rank)) * vcv
    else:
        V = vcv
    df_r = T - 1
    sl = [j for j, c in enumerate(info) if c.kind != 'cons']
    if sl:
        bs = b[sl]
        Fv = float(bs @ np.linalg.solve(V[np.ix_(sl, sl)], bs)) / len(sl)
    else:
        Fv = np.nan
    dfm = len(sl)
    if re_:
        xb = X @ b
        r2o = float(np.corrcoef(xb, yy)[0, 1] ** 2)
        extra['r2_o'] = r2o
    sc = {'N': n, 'N_g': G, 'df_m': dfm, 'df_r': df_r, 'lag': lag}
    if re_:
        sc['chi2'] = Fv * dfm
    else:
        sc['F'] = Fv
    sc.update(extra)
    est = {'cmd': 'xtscc', 'names': names, 'b': b, 'V': V, 'colinfo': info, 'df_r': df_r, 'depvar': yname,
           'sample': mask, 'stat': 't', 'vce': 'dkraay', 'scalars': sc,
           'macros': {'cmd': 'xtscc', 'title': 'Regression with Driscoll-Kraay standard errors',
                      'vcetype': 'Drisc/Kraay', 'depvar': yname, 'groupvar': ivar, 'method': method,
                      'predict': 'xtscc_p'}}

    def replay(ip_):
        ip_.txt('\nRegression with Driscoll-Kraay standard errors')
        ip_.txt(' ' * (49 - 46) + 'Number of obs     =')
        ip_.res(f'{n:10.0f}\n')
        s = 'Method: '
        ip_.txt(s)
        ip_.res(method)
        ip_.txt(' ' * max(49 - len(s) - len(method), 1) + 'Number of groups  =')
        ip_.res(f'{G:10.0f}\n')
        s = 'Group variable (i): '
        ip_.txt(s)
        ip_.res(ivar[:16])
        ip_.txt(' ' * max(49 - len(s) - len(ivar[:16]), 1))
        if not re_:
            ip_.txt('F(')
            ip_.res(f'{dfm:3.0f}')
            ip_.txt(',')
            ip_.res(f'{df_r:6.0f}')
            ip_.txt(')' + ' ' * 5 + '=')
            ip_.res(f'{Fv:10.2f}\n')
        else:
            lab = f'Wald chi2({dfm})'
            ip_.txt(f'{lab:<18}=')
            ip_.res(f'{Fv * dfm:10.2f}\n')
        s = 'maximum lag: '
        ip_.txt(s)
        ip_.res(str(lag))
        ip_.txt(' ' * max(49 - len(s) - len(str(lag)), 1))
        if not re_:
            ip_.txt('Prob > F          =    ')
            ip_.res(f'{sps.f.sf(Fv, dfm, df_r):6.4f}\n')
        else:
            ip_.txt('Prob > chi2       =    ')
            ip_.res(f'{sps.chi2.sf(Fv * dfm, dfm):6.4f}\n')
        if re_:
            ip_.txt('corr(u_i, Xb) = ')
            ip_.res('0 ')
            ip_.txt('(assumed)' + ' ' * (49 - 27) + 'overall R-squared =    ')
            ip_.res(f"{extra['r2_o']:5.4f}\n")
        elif fe:
            ip_.txt(' ' * 49 + 'within R-squared  =    ')
            ip_.res(f"{extra['r2_w']:5.4f}\n")
        else:
            ip_.txt(' ' * 49 + 'R-squared         =    ')
            ip_.res(f"{extra['r2']:5.4f}\n")
            ip_.txt(' ' * 49 + 'Root MSE          =  ')
            ip_.res(f"{extra['rmse']:8.4f}\n")
        ip_.txt('\n')
        rows = None
        if re_:
            rows = [('sigma_u', extra['sigma_u'], ''), ('sigma_e', extra['sigma_e'], ''), ('rho', extra['rho'], '')]
        coef_table(ip_, est, 't', robust='Drisc/Kraay', extra_rows=rows)
        ip_.txt('\n')
    est['replay'] = replay

    def predict(ip_, kind, m_):
        Xf = np.column_stack([c.values for c in info if c.kind != 'cons'] + ([np.ones(d.n)] if not nocons else []))
        xb = Xf @ b
        if kind in (None, '', 'xb'):
            return xb
        if kind in ('residuals', 'resid', 'r'):
            return depvar_values(ip_, d, yname)[1] - xb
        raise StataError(f'predict option {kind} not allowed after xtscc', 198)
    est['predict'] = predict
    ip.set_estimation(est)
    replay(ip)
