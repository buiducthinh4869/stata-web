"""xtgls (FGLS: panels iid/heteroskedastic/correlated, corr independent/ar1/psar1, igls) and xtpcse (OLS or
Prais-Winsten with panel-corrected standard errors).  Ported from Stata's xtgls.ado, xtpcse.ado and
_crcar1.ado."""
import math

import numpy as np
from scipy import stats as sps

from .design import coef_table
from .errors import StataError
from .estim import prepare_linear, assemble, detect_omitted, depvar_values
from .fmt import fmt_num, comma_int
from .interp import command
from .syntax import parse_command, split_top


def _crcar1(e, lag_e, kind, k=None):
    """rho from residuals; lag_e are the (aligned) lagged residuals (nan where unavailable)."""
    ok = ~np.isnan(lag_e) & ~np.isnan(e)
    if kind in ('', 'regress', 'reg', 'regr', 'regre', 'regres'):
        den = float(np.sum(lag_e[ok] ** 2))
        return (float(np.sum(e[ok] * lag_e[ok])) / den if den > 0 else np.nan), int(ok.sum())
    if kind in ('freg', 'fre'):
        # forward: e_t * e_{t+1} / sum e_{t+1}^2 ; same pairs, roles reversed
        den = float(np.sum(e[ok] ** 2))
        return (float(np.sum(e[ok] * lag_e[ok])) / den if den > 0 else np.nan), int(ok.sum())
    sse = float(np.sum(e[~np.isnan(e)] ** 2))
    if kind in ('dw',):
        dwv = float(np.sum((e[ok] - lag_e[ok]) ** 2)) / sse
        return 1 - dwv / 2, int(ok.sum())
    if kind in ('tscorr', 'tsc', 'tsco', 'tscor'):
        return float(np.sum(e[ok] * lag_e[ok])) / sse, int(ok.sum())
    if kind in ('nagar', 'nag', 'theil', 'th'):
        raise StataError(f'invalid method for rhotype(), rhotype({kind})', 198)
    raise StataError(f'rhotype({kind}) : argument is invalid', 198)


def _setup(ip, p):
    d = ip.data
    if not d.ts or not d.ts.get('panel'):
        raise StataError('panel variable not set; use xtset varname ...', 459)
    toks = split_top(p.body)
    if not toks:
        raise StataError('varlist required', 100)
    nocons = p.opts.has('noconstant', 3)
    yname, y, cols, mask, w, wt = prepare_linear(ip, p, d, toks[1:], toks[0])
    if w is not None:
        raise StataError('weights are not available with this command in Stata Web', 101)
    ivar = d.ts['panel']
    tvar = d.ts.get('time')
    mask &= ~np.isnan(d.cols[ivar].astype(float))
    if tvar:
        mask &= ~np.isnan(d.cols[tvar].astype(float))
    Xall, names, info = assemble(cols, mask, cons=not nocons)
    active = detect_omitted(Xall, info, cons=not nocons)
    X = Xall[:, active]
    info = [info[j] for j in active]
    names = [c.name for c in info]
    return d, toks, yname, y, X, names, info, mask, ivar, tvar, nocons


def _order(d, ivar, tvar, mask):
    pv = d.cols[ivar][mask].astype(float)
    tv = d.cols[tvar][mask].astype(float) if tvar else np.arange(mask.sum(), dtype=float)
    o = np.lexsort((tv, pv))
    return o, pv[o], tv[o]


# ================================================================ xtgls
@command('xtgls')
def cmd_xtgls(ip, cmd, rest, by):
    p = parse_command(rest)
    if not p.body.strip():
        if ip.est and ip.est.get('cmd') == 'xtgls':
            return ip.est['replay'](ip)
        raise StataError('last estimates not found', 301)
    if p.opts.has('pcse'):
        raise StataError('option pcse is obsolete; use xtpcse', 198)
    pan = (p.opts.get('panels', 1) or 'iid').strip().lower()
    if 'iid'.startswith(pan) or pan in ('0',):
        var = 0
    elif 'heteroskedastic'.startswith(pan) or 'heteroscedastic'.startswith(pan) or pan == '1':
        var = 1
    elif 'correlated'.startswith(pan) or pan == '2':
        var = 2
    else:
        raise StataError(f'panel({pan}) invalid', 198)
    cr = (p.opts.get('corr', 1) or 'independent').strip().lower()
    if 'independent'.startswith(cr) or cr == '0':
        corr = 0
    elif cr in ('ar1', 'ar', '1'):
        corr = 1
    elif cr in ('psar1', 'psar', 'psa', 'ps', '2'):
        corr = 2
    else:
        raise StataError(f'corr({cr}) invalid', 198)
    rhotype = (p.opts.get('rhotype', 3) or '').strip().lower()
    if rhotype and corr == 0:
        raise StataError('rhotype() not allowed with default corr(independent)', 198)
    igls = p.opts.has('igls')
    it = p.opts.get('iterate', 4)
    iterate = int(it) if it not in (None, '') else 16000
    tl = p.opts.get('tolerance', 3)
    tol = float(tl) if tl not in (None, '') else 1e-7
    d, toks, yname, y, X0, names, info, mask, ivar, tvar, nocons = _setup(ip, p)
    if (corr > 0 or var == 2) and not tvar:
        raise StataError('time variable not set', 459)
    o, pv, tv = _order(d, ivar, tvar, mask)
    yy = y[mask][o]
    X = X0[o]
    idx = np.flatnonzero(mask)[o]
    if corr > 0:
        # drop singleton panels
        up, cnt = np.unique(pv, return_counts=True)
        single = np.isin(pv, up[cnt == 1])
        if single.any():
            ip.txt(f'(note: {int(single.sum())} observations dropped because only 1 obs in group)\n')
            keep = ~single
            yy, X, pv, tv, idx = yy[keep], X[keep], pv[keep], tv[keep], idx[keep]
    ug, gid = np.unique(pv, return_inverse=True)
    M = ug.size
    utimes, tid = np.unique(tv, return_inverse=True)
    NT = utimes.size
    n = yy.size
    Ti = np.bincount(gid, minlength=M)
    if var == 2:
        if np.unique(Ti).size != 1:
            raise StataError('panels must be balanced', 459)
        if np.any(np.bincount(tid) != M):
            raise StataError('time periods must be the same across panels', 459)
    if corr > 0:
        # time positions: lag is previous group-time (egen group of tvar), as in xtgls.ado
        first = np.r_[True, gid[1:] != gid[:-1]]
        prev_ok = np.r_[False, (gid[1:] == gid[:-1]) & (tid[1:] == tid[:-1] + 1)]
    if M == 1 and var == 2:
        var = 1
    if NT < 2 and corr > 0:
        corr = 0
    # ---- correlation step
    rho_txt = ''
    rhos = []
    Xt, yt = X.copy(), yy.copy()
    if corr > 0:
        b0 = np.linalg.lstsq(X, yy, rcond=None)[0]
        e0 = yy - X @ b0
        lag = np.where(prev_ok, np.r_[np.nan, e0[:-1]], np.nan)
        r_i = []
        for g in range(M):
            s = gid == g
            r, _ = _crcar1(e0[s], lag[s], rhotype)
            r_i.append(r)
        if corr == 1:
            ok = [g for g in range(M) if r_i[g] == r_i[g]]
            rho = sum(r_i[g] * Ti[g] for g in ok) / sum(Ti[g] for g in ok)
            rvec = np.full(M, rho)
            rhos = [rho]
            corr_txt = 'common AR(1) coefficient for all panels'
        else:
            for g in range(M):
                if r_i[g] != r_i[g]:
                    raise StataError(f'rho_i could not be computed for panel {ug[g]:g}', 459)
            rvec = np.array(r_i)
            rhos = list(rvec)
            corr_txt = 'panel-specific AR(1)'
        rr = rvec[gid]
        # Prais-Winsten using the previous observation within the panel ([_n-1])
        Zy, ZX = yy.copy(), X.copy()
        nf = ~first
        Zy[nf] = yy[nf] - rr[nf] * yy[np.flatnonzero(nf) - 1]
        ZX[nf] = X[nf] - rr[nf][:, None] * X[np.flatnonzero(nf) - 1]
        Zy[first] = np.sqrt(1 - rr[first] ** 2) * yy[first]
        ZX[first] = np.sqrt(1 - rr[first] ** 2)[:, None] * X[first]
        yt, Xt = Zy, ZX
    else:
        corr_txt = 'no autocorrelation'
    bt = np.linalg.lstsq(Xt, yt, rcond=None)[0]
    ee = yt - Xt @ bt

    def sigma_from(e):
        if var == 0:
            s = float(np.mean(e ** 2))
            return np.eye(M) * s
        if var == 1:
            return np.diag(np.bincount(gid, weights=e ** 2, minlength=M) / Ti)
        Em = np.zeros((NT, M))
        Em[tid, gid] = e
        return Em.T @ Em / NT

    def gls(S):
        Si = np.linalg.pinv(S) if var == 2 else np.diag(1 / np.diag(S))
        if var == 2:
            k = Xt.shape[1]
            A = np.zeros((k, k))
            c = np.zeros(k)
            for t in range(NT):
                s = tid == t
                g_ = gid[s]
                Xs, ys = Xt[s], yt[s]
                W = Si[np.ix_(g_, g_)]
                A += Xs.T @ W @ Xs
                c += Xs.T @ W @ ys
        else:
            w_ = np.diag(Si)[gid]
            A = Xt.T @ (Xt * w_[:, None])
            c = Xt.T @ (w_ * yt)
        V = np.linalg.pinv(A)
        return V @ c, V
    S = sigma_from(ee)
    b, V = gls(S)
    iters = []
    if igls:
        diff = tol + 1
        itn = 1
        while itn < iterate and diff > tol:
            S = sigma_from(yt - Xt @ b)
            b1, V = gls(S)
            diff = float(np.max(np.abs(b1 - b) / (np.abs(b) + 1)))
            b = b1
            iters.append((itn, diff))
            itn += 1
    ll = None
    if (corr == 0 and igls) or (corr == 0 and var == 0):
        eh = yt - Xt @ b
        if var == 2:
            lndet = NT * math.log(np.linalg.det(S))
        else:
            lndet = float(np.sum(np.log(np.diag(S)[gid])))
        if var == 2:
            Si = np.linalg.pinv(S)
            q = 0.0
            for t in range(NT):
                s = tid == t
                g_ = gid[s]
                q += eh[s] @ Si[np.ix_(g_, g_)] @ eh[s]
        else:
            q = float(np.sum(eh ** 2 / np.diag(S)[gid]))
        ll = -.5 * (n * math.log(2 * math.pi) + lndet + q)
    if p.opts.has('nmk'):
        k = int(np.sum(np.diag(V) > 0))
        V = V * n / (n - k)
    sl = [j for j, c in enumerate(info) if c.kind != 'cons']
    chi2 = float(b[sl] @ np.linalg.solve(V[np.ix_(sl, sl)], b[sl])) if sl else np.nan
    dfm = len(sl)
    n_cv = 1 if var == 0 else (M if var == 1 else M * (M + 1) // 2)
    n_cr = len(rhos)
    n_cf = int(np.linalg.matrix_rank(V))
    vt = {0: 'homoskedastic', 1: 'heteroskedastic', 2: 'heteroskedastic with cross-sectional correlation'}[var]
    sc = {'N': n, 'N_g': M, 'N_t': NT, 'n_cf': n_cf, 'n_cv': n_cv, 'n_cr': n_cr, 'df': dfm, 'chi2': chi2,
          'g_min': int(Ti.min()), 'g_avg': float(Ti.mean()), 'g_max': int(Ti.max()),
          'df_pear': n - n_cv - n_cr - n_cf, 'N_miss': 0, 'rc': 0}
    if ll is not None:
        sc['ll'] = ll
    from .expr import Matrix
    smask = np.zeros(d.n, bool)
    smask[idx] = True
    est = {'cmd': 'xtgls', 'names': names, 'b': b, 'V': V, 'colinfo': info, 'df_r': None, 'depvar': yname,
           'sample': smask, 'stat': 'z', 'vce': 'gls', 'scalars': sc, 'matrices': {'Sigma': Matrix(S)},
           'macros': {'cmd': 'xtgls', 'title': 'Cross-sectional time-series FGLS regression',
                      'coefftype': 'generalized least squares', 'vt': vt, 'corr': corr_txt,
                      'rho': ' '.join(fmt_num(r, '%9.0g').strip() for r in rhos), 'depvar': yname,
                      'ivar': ivar, 'tvar': tvar or '', 'rhotype': rhotype or 'regress', 'chi2type': 'Wald'}}

    def replay(ip_):
        for it_, df_ in iters:
            ip_.txt(f'Iteration {it_}: tolerance = ')
            ip_.res(fmt_num(df_, '%9.0g').strip() + '\n')
        if iters:
            ip_.txt('\n')
        ip_.txt('\nCross-sectional time-series FGLS regression\n\n')
        ip_.txt('Coefficients:  ')
        ip_.res('generalized least squares\n')
        ip_.txt('Panels:        ')
        ip_.res(vt + '\n')
        ip_.txt('Correlation:   ')
        ip_.res(corr_txt)
        if n_cr == 1:
            ip_.txt('  (')
            ip_.res(f'{rhos[0]:9.4f}'.strip())
            ip_.txt(')')
        ip_.txt('\n\n')

        def L(left, val, k, v):
            ip_.txt(f'{left:<27}= ')
            ip_.res(fmt_num(val, '%9.0g'))
            ip_.txt(' ' * (48 - 27 - 2 - 9) + f'{k:<18}= ')
            ip_.res(v + '\n')
        L('Estimated covariances', n_cv, 'Number of obs', f'{n:10,.0f}')
        L('Estimated autocorrelations', n_cr, 'Number of groups', f'{M:10,.0f}')
        if Ti.min() == Ti.max():
            ip_.txt(f'{"Estimated coefficients":<27}= ')
            ip_.res(fmt_num(n_cf, '%9.0g'))
            ip_.txt(' ' * 10 + f'{"Time periods":<18}=  ')
            ip_.res(fmt_num(Ti.max(), '%9.0g') + '\n')
        else:
            ip_.txt(f'{"Estimated coefficients":<27}= ')
            ip_.res(fmt_num(n_cf, '%9.0g'))
            ip_.txt(' ' * 10 + 'Obs per group:\n')
            ip_.txt(' ' * 62 + 'min = ')
            ip_.res(f'{Ti.min():10,.0f}\n')
            ip_.txt(' ' * 62 + 'avg =  ')
            ip_.res(fmt_num(Ti.mean(), '%9.0g') + '\n')
            ip_.txt(' ' * 62 + 'max = ')
            ip_.res(f'{Ti.max():10,.0f}\n')
        cf = '%9.0g' if chi2 > 999999 else '%9.2f'
        lab = f'Wald chi2({dfm})'
        ip_.txt(' ' * 48 + f'{lab:<18}=  ')
        ip_.res(fmt_num(chi2, cf) + '\n')
        if ll is not None:
            ip_.txt(f'{"Log likelihood":<27}= ')
            ip_.res(fmt_num(ll, '%9.0g'))
            ip_.txt(' ' * 10)
        else:
            ip_.txt(' ' * 48)
        ip_.txt(f'{"Prob > chi2":<18}= ')
        ip_.res(f'{sps.chi2.sf(chi2, dfm):10.4f}\n\n')
        coef_table(ip_, est, 'z')
        if var == 2 and M >= NT:
            ip_.txt('Note: when the number of panels is greater than or equal to the number of periods, results\n'
                    '      are based on a generalized inverse of a singular matrix.\n')
        if sc['df_pear'] <= 0:
            ip_.txt('Note: you estimated at least as many quantities as you have observations.\n')
    est['replay'] = replay
    est['predict'] = lambda ip_, kind, m_: _lin_predict(ip_, est, info, b, nocons, kind)
    ip.set_estimation(est)
    replay(ip)


def _lin_predict(ip, est, info, b, nocons, kind):
    d = ip.data
    Xf = np.column_stack([c.values for c in info if c.kind != 'cons'] + ([np.ones(d.n)] if not nocons else []))
    xb = Xf @ b
    if kind in (None, '', 'xb'):
        return xb
    if kind == 'stdp':
        return np.sqrt(np.einsum('ij,jk,ik->i', Xf, est['V'], Xf))
    raise StataError(f'predict option {kind} not allowed', 198)


# ================================================================ xtpcse
@command('xtpcse')
def cmd_xtpcse(ip, cmd, rest, by):
    p = parse_command(rest)
    if not p.body.strip():
        if ip.est and ip.est.get('cmd') == 'xtpcse':
            return ip.est['replay'](ip)
        raise StataError('last estimates not found', 301)
    hetonly, indep = p.opts.has('hetonly', 3), p.opts.has('independent', 3)
    if hetonly and indep:
        raise StataError('options hetonly and independent may not be specified together', 198)
    pairwise = p.opts.has('pairwise', 4)
    cr = (p.opts.get('correlation', 1) or 'independent').strip().lower()
    if 'independent'.startswith(cr):
        corr = 'independent'
    elif cr in ('ar1', 'ar'):
        corr = 'ar1'
    elif cr in ('psar1', 'psar', 'ps'):
        corr = 'psar1'
    else:
        raise StataError('may only specify one from ar1, psar1, and independent in option corr()', 198)
    rhotype = (p.opts.get('rhotype', 3) or '').strip().lower()
    np1 = p.opts.has('np1')
    d, toks, yname, y, X0, names, info, mask, ivar, tvar, nocons = _setup(ip, p)
    if not tvar:
        raise StataError('time variable not set', 459)
    mask_t = mask
    o, pv, tv = _order(d, ivar, tvar, mask_t)
    yy = y[mask_t][o]
    X = X0[o]
    idx = np.flatnonzero(mask_t)[o]
    delta = d.ts.get('delta', 1) or 1
    ug, gid = np.unique(pv, return_inverse=True)
    M = ug.size
    n = yy.size
    prev_ok = np.r_[False, (gid[1:] == gid[:-1]) & (np.round((tv[1:] - tv[:-1]) / delta) == 1)]
    gaps = int(np.sum((gid[1:] == gid[:-1]) & (np.round((tv[1:] - tv[:-1]) / delta) > 1)))
    if gaps and corr != 'independent':
        ip.txt('(note: computations for rho restarted at each gap)\n')
    b0 = np.linalg.lstsq(X, yy, rcond=None)[0]
    e = yy - X @ b0
    k_df = X.shape[1] - (0 if nocons else 1)
    rhos = []
    Xt, yt = X, yy
    if corr != 'independent':
        lag = np.where(prev_ok, np.r_[np.nan, e[:-1]], np.nan)
        bounded = False
        if corr == 'ar1':
            num = den = 0.0
            for g in range(M):
                s = gid == g
                r, nn = _crcar1(e[s], lag[s], rhotype, k_df)
                if r == r:
                    if not p.opts.has('forcerho'):
                        if r > 1 or r < -1:
                            bounded = True
                        r = min(max(r, -1.0), 1.0)
                    num += (nn + np1) * r
                    den += nn + np1
            if den == 0:
                raise StataError('insufficient observations', 2001)
            rho = num / den
            rvec = np.full(M, rho)
            rhos = [rho]
        else:
            rvec = np.zeros(M)
            for g in range(M):
                s = gid == g
                r, nn = _crcar1(e[s], lag[s], rhotype, k_df)
                if r != r:
                    ip.txt(f'(note: rho_i could not be computed for panel {ivar} {ug[g]:g};\n'
                           '       assumed to be 0.)\n')
                    r = 0.0
                elif not p.opts.has('forcerho'):
                    if r > 1 or r < -1:
                        bounded = True
                    r = min(max(r, -1.0), 1.0)
                rvec[g] = r
            rhos = list(rvec)
        if bounded:
            ip.txt('(note: estimates of rho outside [-1,1] bounded to be in the range [-1,1])\n')
        rr = rvec[gid]
        prev = np.flatnonzero(prev_ok)
        yt = yy * np.sqrt(1 - rr ** 2)
        Xt = X * np.sqrt(1 - rr ** 2)[:, None]
        yt[prev] = yy[prev] - rr[prev] * yy[prev - 1]
        Xt[prev] = X[prev] - rr[prev][:, None] * X[prev - 1]
    XtXi = np.linalg.inv(Xt.T @ Xt)
    b = XtXi @ Xt.T @ yt
    e = yt - Xt @ b
    rss = float(e @ e)
    df_r = n - Xt.shape[1]
    if not nocons and corr == 'independent':
        tss = float(np.sum((yt - yt.mean()) ** 2))
    elif not nocons:
        # tsscons: total sum of squares about the mean as if a constant were present
        tss = float(np.sum((yt - yt.mean()) ** 2))
    else:
        tss = float(yt @ yt)
    r2 = 1 - rss / tss
    utimes, tid = np.unique(tv, return_inverse=True)
    NT = utimes.size
    Ti = np.bincount(gid, minlength=M)
    n_sigma = None
    if hetonly:
        sig = np.bincount(gid, weights=e ** 2, minlength=M) / Ti
        XOX = Xt.T @ (Xt * sig[gid][:, None])
        Sigma = np.diag(sig)
        vcetype, n_cv = 'Het-corrected', M
    elif indep:
        s = float(np.mean(e ** 2))
        XOX = s * (Xt.T @ Xt)
        Sigma = np.eye(M) * s
        vcetype, n_cv = 'Indep-corrected', 1
    else:
        Em = np.full((NT, M), np.nan)
        Em[tid, gid] = e
        Sigma = np.zeros((M, M))
        if not pairwise:
            full = ~np.isnan(Em).any(axis=1)
            if not full.any():
                raise StataError('no time periods are common to all panels, cannot estimate disturbance\n'
                                 'covariance matrix using casewise inclusion', 459)
            Ef = Em[full]
            Sigma = Ef.T @ Ef / full.sum()
            n_sigma = int(full.sum())
        else:
            for i in range(M):
                for j in range(i, M):
                    ok = ~np.isnan(Em[:, i]) & ~np.isnan(Em[:, j])
                    v = float(np.sum(Em[ok, i] * Em[ok, j]) / ok.sum()) if ok.any() else 0.0
                    Sigma[i, j] = Sigma[j, i] = v
        k = Xt.shape[1]
        XOX = np.zeros((k, k))
        for t in range(NT):
            s = tid == t
            g_ = gid[s]
            XOX += Xt[s].T @ Sigma[np.ix_(g_, g_)] @ Xt[s]
        vcetype, n_cv = 'Panel-corrected', M * (M + 1) // 2
    V = XtXi @ XOX @ XtXi
    if p.opts.has('nmk'):
        V = n * V / df_r
    V = .5 * (V + V.T)
    sl = [j for j, c in enumerate(info) if c.kind != 'cons']
    chi2 = float(b[sl] @ np.linalg.solve(V[np.ix_(sl, sl)], b[sl])) if sl else np.nan
    dfm = len(sl)
    pchi = float(sps.chi2.sf(chi2, dfm)) if sl else np.nan
    bal = 'balanced' if (np.unique(Ti).size == 1 and np.all(np.bincount(tid) == M)) else 'unbalanced'
    cov = {'Het-corrected': 'heteroskedastic', 'Panel-corrected': 'correlated'}.get(vcetype, 'independent')
    title = ('Linear regression,' if corr == 'independent' else 'Prais-Winsten regression,') + \
        f' {cov} panels corrected standard errors' + (' (PCSEs)' if vcetype == 'Panel-corrected' else '')
    n_cr = 0 if corr == 'independent' else (1 if corr == 'ar1' else M)
    n_cf = Xt.shape[1]
    sc = {'N': n, 'N_g': M, 'N_gaps': gaps, 'df': dfm, 'chi2': chi2, 'p': pchi, 'r2': r2, 'rmse':
          math.sqrt(rss / df_r), 'n_cf': n_cf, 'n_cv': n_cv, 'n_cr': n_cr, 'g_min': int(Ti.min()),
          'g_avg': float(Ti.mean()), 'g_max': int(Ti.max()), 'rc': 0}
    if n_sigma is not None:
        sc['n_sigma'] = n_sigma
    from .expr import Matrix
    smask = np.zeros(d.n, bool)
    smask[idx] = True
    mats = {'Sigma': Matrix(Sigma)}
    if rhos:
        mats['rhomat'] = Matrix([rhos])
    est = {'cmd': 'xtpcse', 'names': names, 'b': b, 'V': V, 'colinfo': info, 'df_r': None, 'depvar': yname,
           'sample': smask, 'stat': 'z', 'vce': 'pcse', 'scalars': sc, 'matrices': mats,
           'macros': {'cmd': 'xtpcse', 'title': title, 'vcetype': vcetype, 'panels': cov, 'corr': corr,
                      'balance': bal, 'ivar': ivar, 'tvar': tvar, 'depvar': yname, 'rhotype': rhotype,
                      'missmeth': 'casewise' if n_sigma is not None else ('pairwise' if pairwise else '')}}
    if n_sigma is not None and n_sigma < Ti.mean() / 2:
        ip.txt(f'(note: the number of observations per panel, e(n_sigma) = {n_sigma},\n'
               '       used to compute the disturbance of covariance matrix e(Sigma)\n'
               '       is less than half of the average number of observations per panel,\n'
               f'       e(n_avg) = {fmt_num(Ti.mean(), "%9.0g").strip()}; you may want to consider the pairwise '
               'option)\n')

    def replay(ip_):
        acorr = {'ar1': 'common AR(1)', 'psar1': 'panel-specific AR(1)'}.get(corr, 'no autocorrelation')
        ip_.txt(f'\n{title}\n\n')

        def LR(lt, lv, rk, rv):
            ip_.txt(f'{lt:<18}')
            ip_.res(lv)
            ip_.txt(' ' * max(48 - 18 - len(lv), 1) if rk else '')
            if rk:
                ip_.txt(f'{rk:<18}= ')
                ip_.res(rv)
            ip_.txt('\n')
        LR('Group variable:', ivar, 'Number of obs', f'{fmt_num(n, "%10.0gc")}')
        LR('Time variable:', tvar, 'Number of groups', f'{M:10,.0f}')
        ip_.txt(f'{"Panels:":<18}')
        s = f'{cov} ({bal})'
        ip_.res(s)
        ip_.txt(' ' * max(48 - 18 - len(s), 1) + 'Obs per group:\n')
        ip_.txt(f'{"Autocorrelation:":<18}')
        ip_.res(acorr)
        ip_.txt(' ' * max(62 - 18 - len(acorr), 1) + 'min = ')
        ip_.res(fmt_num(Ti.min(), '%10.0gc') + '\n')
        if bal == 'unbalanced' and cov == 'correlated':
            ip_.txt(f'{"Sigma computed by":<18}')
            s = f"{'casewise' if not pairwise else 'pairwise'} selection"
            ip_.res(s)
            ip_.txt(' ' * max(62 - 18 - len(s), 1) + 'avg = ')
        else:
            ip_.txt(' ' * 62 + 'avg = ')
        ip_.res(fmt_num(Ti.mean(), '%10.0gc') + '\n')
        ip_.txt(' ' * 62 + 'max = ')
        ip_.res(fmt_num(Ti.max(), '%10.0gc') + '\n')
        ip_.txt(f'{"Estimated covariances":<27}= ')
        ip_.res(fmt_num(n_cv, '%9.0g'))
        ip_.txt(' ' * 10 + f'{"R-squared":<18}= ')
        ip_.res(f'{r2:10.4f}\n')
        ip_.txt(f'{"Estimated autocorrelations":<27}= ')
        ip_.res(fmt_num(n_cr, '%9.0g'))
        lab = f'Wald chi2({dfm})'
        ip_.txt(' ' * 10 + f'{lab:<18}= ')
        ip_.res(f'{chi2:10.2f}\n')
        ip_.txt(f'{"Estimated coefficients":<27}= ')
        ip_.res(fmt_num(n_cf, '%9.0g'))
        ip_.txt(' ' * 10 + f'{"Prob > chi2":<18}= ')
        ip_.res(f'{pchi:10.4f}\n\n')
        coef_table(ip_, est, 'z', robust=vcetype,
                   extra_rows=[('rho', rhos[0], '')] if n_cr == 1 else None)
        if n_cr > 1:
            ip_.txt(f'{"rhos = ":>15}')
            sh = [fmt_num(r, '%9.0g') for r in rhos]
            out = f'{sh[0]:>9}'
            for i_, s_ in enumerate(sh[1:], start=2):
                if len(sh) > 6 and i_ == 6:
                    out += ' ...'
                    break
                out += ' ' + f'{s_:>9}'
            ip_.res(out + '\n')
            ip_.txt('-' * 78 + '\n')
    est['replay'] = replay
    est['predict'] = lambda ip_, kind, m_: _lin_predict(ip_, est, info, b, nocons, kind)
    ip.set_estimation(est)
    replay(ip)
