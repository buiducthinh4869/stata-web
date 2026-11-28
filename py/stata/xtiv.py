"""xtivreg (fe, fd, be, re G2SLS/EC2SLS), ported from Stata's xtivreg.ado."""
import math

import numpy as np
from scipy import stats as sps

from .design import coef_table, build_design, Col
from .errors import StataError
from .estim import _parse_iv, assemble, detect_omitted, full_bV, wald_test, _store_rhs
from .fmt import g_fmt, comma_int, fmt_num
from .interp import command
from .syntax import parse_command


def _tsls(y, X, Z):
    ZtZi = np.linalg.pinv(Z.T @ Z)
    PX = Z @ (ZtZi @ (Z.T @ X))
    A = np.linalg.pinv(PX.T @ X)
    b = A @ (PX.T @ y)
    return b, y - X @ b, A, PX


def _gmean(v, gid, G):
    return (np.bincount(gid, v, minlength=G) / np.bincount(gid, minlength=G))[gid]


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / den if den > 0 else np.nan


def _cols(ip, d, toks, mask):
    cols, mask = build_design(ip, d, toks, mask) if toks else ([], mask)
    return cols, mask


@command('xtivreg')
def cmd_xtivreg(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    if not d.ts or not d.ts.get('panel'):
        raise StataError('panel variable not set; use xtset varname ...', 459)
    ivar, tvar = d.ts['panel'], d.ts.get('time')
    models = [m for m in ('fe', 'fd', 'be', 're') if p.opts.has(m)]
    if len(models) > 1:
        raise StataError('choose only one of re, be, fe, or fd', 198)
    model = models[0] if models else 're'
    if model == 'fd' and not tvar:
        raise StataError('must specify time variable with xtset to use fd', 459)
    small = p.opts.has('small')
    ec2 = p.opts.has('ec2sls', 3)
    vce = (p.opts.get('vce', 1) or '').strip().lower()
    clus = None
    if vce:
        w = vce.split()
        if w[0].startswith('r'):
            clus = ivar
        elif w[0].startswith('cl'):
            clus = d.resolve(w[1])
        elif not w[0].startswith('conv'):
            raise StataError(f'vce({vce}) not allowed', 198)
        if model in ('re', 'be') and clus:
            raise StataError(f'vce({vce}) not allowed with {model} in Stata Web', 198)
    yv_tok, exog, endo, inst = _parse_iv(p.body)
    yv = d.resolve(yv_tok)
    y_full = d.cols[yv].astype(float)
    mask = ip.mark(p) & ~np.isnan(y_full) & ~np.isnan(d.cols[ivar].astype(float))
    cx, mask = _cols(ip, d, exog, mask)
    ce, mask = _cols(ip, d, endo, mask)
    cz, mask = _cols(ip, d, inst, mask)
    nocons = p.opts.has('noconstant', 4)
    if model == 'fd':
        def D(v):
            return v - ip._lag(d, v, 1)
        dy = D(y_full)
        dmask = mask & ~np.isnan(dy) & ~np.isnan(ip._lag(d, np.where(mask, 1.0, np.nan), 1))
        for c in cx + ce + cz:
            dmask &= ~np.isnan(D(np.where(mask, c.values, np.nan)))
        diffs = {}
        for c in cx + ce + cz:
            diffs[id(c)] = D(np.where(mask, c.values, np.nan))
        lev_mask = mask
        mask = dmask
    pid = d.cols[ivar].astype(float)[mask]
    _, gid = np.unique(pid, return_inverse=True)
    G = int(gid.max()) + 1
    n = int(mask.sum())
    Ti = np.bincount(gid)
    tmin, tavg, tmax = int(Ti.min()), float(Ti.mean()), int(Ti.max())

    def mat(cols, use_diff=False):
        if not cols:
            return np.zeros((n, 0)), []
        if use_diff:
            M = np.column_stack([diffs[id(c)][mask] for c in cols])
        else:
            M = np.column_stack([c.values[mask] for c in cols])
        return M, list(cols)
    fd = model == 'fd'
    X1, i1 = mat(cx, fd)
    Y2, i2 = mat(ce, fd)
    Z, iz = mat(cz, fd)
    y = dy[mask] if fd else y_full[mask]
    # drop bases / collinear exog
    keep1 = [j for j, c in enumerate(i1) if not c.base]
    X1, i1 = X1[:, keep1], [i1[j] for j in keep1]
    keep2 = [j for j, c in enumerate(i2) if not c.base]
    Y2, i2 = Y2[:, keep2], [i2[j] for j in keep2]
    keepz = [j for j, c in enumerate(iz) if not c.base]
    Z, iz = Z[:, keepz], [iz[j] for j in keepz]
    if Z.shape[1] < Y2.shape[1]:
        raise StataError('equation not identified; must have at least as many instruments not in the regression as '
                         'there are instrumented variables', 481)
    k = X1.shape[1] + Y2.shape[1]
    names = [c.name for c in i2] + [c.name for c in i1]
    cons = not nocons if fd else True
    X_lev = np.column_stack([c.values[mask] for c in i2 + i1]) if k else np.zeros((n, 0))
    res = {}
    if model == 'fe':
        def dm(v):
            return v - _gmean(v, gid, G) + v.mean()
        yt = dm(y)
        Xt = np.column_stack([np.apply_along_axis(dm, 0, Y2) if Y2.size else Y2,
                              np.apply_along_axis(dm, 0, X1) if X1.size else X1, np.ones(n)])
        Zt = np.column_stack([np.apply_along_axis(dm, 0, X1) if X1.size else X1,
                              np.apply_along_axis(dm, 0, Z) if Z.size else Z, np.ones(n)])
        b, e, A, PX = _tsls(yt, Xt, Zt)
        kreg = k + 1
        rss = float(e @ e)
        df_rz = n - kreg - G + 1
        V = rss / df_rz * A
        Gc = None
        if clus:
            cid = d.cols[clus][mask]
            _, cg = np.unique(cid, return_inverse=True)
            Gc = int(cg.max()) + 1
            S = np.zeros((Gc, kreg))
            np.add.at(S, cg, PX * e[:, None])
            V = A @ (S.T @ S) @ A * (Gc / (Gc - 1)) * ((n - 1) / (n - kreg))
        tss = float(np.sum((yt - yt.mean()) ** 2))
        r2_w = 1 - rss / tss
        XB = X_lev @ b[:k] + b[k]
        u_g = np.bincount(gid, y - XB) / Ti
        sigma_u = float(np.std(u_g, ddof=1)) if G > 1 else np.nan
        corr = _corr(XB, u_g[gid])
        r2_o = _corr(y, XB) ** 2
        r2_b = _corr(np.bincount(gid, y) / Ti, np.bincount(gid, XB) / Ti) ** 2
        sigma_e = math.sqrt(rss / df_rz)
        # F test that all u_i = 0 (fitted endogenous = within first-stage xb + u)
        Ff = np.nan
        if not clus:
            Xexog = X1
            endpb = []
            for j in range(Y2.shape[1]):
                R = np.column_stack([Xexog, Z])
                Rm = R - np.array([_gmean(R[:, q], gid, G) for q in range(R.shape[1])]).T if R.size else R
                yj = Y2[:, j]
                ym = yj - _gmean(yj, gid, G)
                bj = np.linalg.lstsq(Rm, ym, rcond=None)[0] if R.size else np.zeros(0)
                xbt = (R @ bj) if R.size else np.zeros(n)
                c0 = yj.mean() - xbt.mean()
                ui = _gmean(yj - xbt, gid, G) - c0
                endpb.append(xbt + c0 + ui)
            W = np.column_stack(endpb + [X1]) if (endpb or X1.size) else np.zeros((n, 0))
            Wc = np.column_stack([W, np.ones(n)])
            er = y - Wc @ np.linalg.lstsq(Wc, y, rcond=None)[0]
            Wm = W - np.array([_gmean(W[:, q], gid, G) for q in range(W.shape[1])]).T
            ym_ = y - _gmean(y, gid, G)
            eu = ym_ - Wm @ np.linalg.lstsq(Wm, ym_, rcond=None)[0]
            Ff = (float(er @ er) - float(eu @ eu)) / rss * (df_rz / (G - 1))
        allidx = list(range(kreg))
        bfull = b
        if small:
            if clus:
                chi2, _, _ = wald_test(b, V, list(range(k)))
                F = chi2 / k if k else np.nan
                df_r = Gc - 1
            else:
                chi2, _, _ = wald_test(b, V, allidx)
                F = chi2 / k if k else np.nan   # e(F) of _regress scaled (df_r - n_g + 1)/df_r
                df_r = df_rz
            stat = None
        else:
            chi2, _, _ = wald_test(b, V, allidx if not clus else list(range(k)))
            df_r = None
        res.update(r2_w=r2_w, r2_b=r2_b, r2_o=r2_o, sigma_u=sigma_u, sigma_e=sigma_e, corr=corr, F_f=Ff,
                   df_a=G - 1, df_rz=df_rz, rss=rss)
        title = 'Fixed-effects (within) IV regression'
        names_out = names + ['_cons']
    elif model == 'fd':
        Xd = np.column_stack([Y2, X1] + ([np.ones(n)] if cons else []))
        Zd = np.column_stack([X1, Z] + ([np.ones(n)] if cons else []))
        b, e, A, PX = _tsls(y, Xd, Zd)
        kreg = Xd.shape[1]
        rss = float(e @ e)
        df_rz = n - kreg
        V = rss / df_rz * A
        Gc = None
        if clus:
            cid = d.cols[clus][mask]
            _, cg = np.unique(cid, return_inverse=True)
            Gc = int(cg.max()) + 1
            S = np.zeros((Gc, kreg))
            np.add.at(S, cg, PX * e[:, None])
            V = A @ (S.T @ S) @ A * (Gc / (Gc - 1)) * ((n - 1) / (n - kreg))
        ylev = y_full[mask]
        XB = X_lev @ b[:k] + (b[k] if cons else 0)
        u_last = []
        for g in range(G):
            s = gid == g
            u_last.append((ylev[s].sum() - XB[s].sum()) / Ti[g])
        u_g = np.array(u_last)
        sigma_u = float(np.std(u_g, ddof=1)) if G > 1 else np.nan
        corr = _corr(XB, u_g[gid])
        r2_o = _corr(ylev, XB) ** 2
        cumX = np.zeros(n)
        cumY = np.zeros(n)
        for g in range(G):
            s = np.flatnonzero(gid == g)
            cumX[s] = np.cumsum(XB[s] / Ti[g])
            cumY[s] = np.cumsum(ylev[s] / Ti[g])
        r2_b = _corr(cumY, cumX) ** 2
        r2_w = _corr(XB - _gmean(XB, gid, G), ylev - _gmean(ylev, gid, G)) ** 2
        sigma_e = math.sqrt(rss / df_rz)
        if small:
            chi2, _, _ = wald_test(b, V, list(range(k)))
            F = chi2 / k if k else np.nan
            df_r = Gc - 1 if clus else df_rz
        else:
            chi2, _, _ = wald_test(b, V, list(range(k)))
            df_r = None
        res.update(r2_w=r2_w, r2_b=r2_b, r2_o=r2_o, sigma_u=sigma_u, sigma_e=sigma_e, corr=corr, df_rz=df_rz,
                   rss=rss)
        title = 'First-differenced IV regression'
        names_out = [f'D.{nm}' for nm in names] + (['_cons'] if cons else [])
    else:
        # between 2SLS on group means
        first = np.r_[0, np.cumsum(Ti)[:-1]]
        order = np.argsort(gid, kind='stable')

        def means(M):
            if M.size == 0:
                return np.zeros((G, 0))
            return np.column_stack([np.bincount(gid, M[:, q]) / Ti for q in range(M.shape[1])])
        yb = np.bincount(gid, y) / Ti
        Xb = np.column_stack([means(Y2), means(X1), np.ones(G)])
        Zb = np.column_stack([means(X1), means(Z), np.ones(G)])
        bb, eb, Ab, _ = _tsls(yb, Xb, Zb)
        Kb = Xb.shape[1]
        rss_b = float(eb @ eb)
        rmse_b = math.sqrt(rss_b / (G - Kb))
        Gc = None
        if model == 'be':
            b, A = bb, Ab
            V = rss_b / (G - Kb) * Ab
            kreg = Kb
            XB = X_lev @ b[:k] + b[k]
            r2_o = _corr(y, XB) ** 2
            r2_w = _corr(XB - _gmean(XB, gid, G), y - _gmean(y, gid, G)) ** 2
            tssb = float(np.sum((yb - yb.mean()) ** 2))
            r2_b = 1 - rss_b / tssb
            chi2, _, _ = wald_test(b, V, list(range(k)))
            df_rz = G - Kb
            df_r = df_rz if small else None
            F = chi2 / k if k else np.nan
            n_eff = G
            res.update(r2_w=r2_w, r2_b=r2_b, r2_o=r2_o, rmse=rmse_b, df_rz=df_rz, rss=rss_b)
            title = 'Between-effects IV regression:'
            names_out = names + ['_cons']
        else:
            # within IV for sigma_e
            def dm(v):
                return v - _gmean(v, gid, G) + v.mean()
            Xt = np.column_stack([np.apply_along_axis(dm, 0, Y2) if Y2.size else Y2,
                                  np.apply_along_axis(dm, 0, X1) if X1.size else X1, np.ones(n)])
            Zt = np.column_stack([np.apply_along_axis(dm, 0, X1) if X1.size else X1,
                                  np.apply_along_axis(dm, 0, Z) if Z.size else Z, np.ones(n)])
            bw, ew, _, _ = _tsls(dm(y), Xt, Zt)
            rss_w = float(ew @ ew)
            df_rw = n - (k + 1) - G + 1
            s_e = math.sqrt(rss_w / df_rw)
            Tcon = tmin == tmax
            K = k + 2
            sa = p.opts.has('sa')
            if not sa:
                sig_e2 = rss_w / df_rw if Tcon else rss_w / (n - G - K + 1)
            else:
                sig_e2 = rss_w / (n - G)
            K = Kb
            s_u2 = rmse_b ** 2 - s_e ** 2 / tavg
            ubPub = float(np.sum(Ti * eb ** 2))
            if not sa:
                if Tcon:
                    sig_u2 = s_u2
                else:
                    Xm = np.column_stack([means(Y2), means(X1), np.ones(G)])
                    xpx = (Xm * Ti[:, None]).T @ Xm
                    xzx = (Xm * (Ti ** 2)[:, None]).T @ Xm
                    tr = float(np.trace(np.linalg.pinv(xpx) @ xzx))
                    sig_u2 = (ubPub - (G - K) * sig_e2) / (n - tr)
            else:
                sig_u2 = (ubPub - G * sig_e2) / n
            sig_u2 = max(0.0, sig_u2)
            theta = 1 - np.sqrt(sig_e2 / (sig_u2 * Ti + sig_e2))
            th = theta[gid]

            def gls(M):
                if M.size == 0:
                    return M
                return np.column_stack([M[:, q] - th * _gmean(M[:, q], gid, G) for q in range(M.shape[1])])
            yg = y - th * _gmean(y, gid, G)
            Xg = np.column_stack([gls(Y2), gls(X1), 1 - th])
            if not ec2:
                Zg = np.column_stack([gls(X1), gls(Z), 1 - th])
            else:
                ZZ = np.column_stack([X1, Z])
                Zw = np.column_stack([ZZ[:, q] - _gmean(ZZ[:, q], gid, G) for q in range(ZZ.shape[1])])
                Zbm = np.column_stack([_gmean(ZZ[:, q], gid, G) for q in range(ZZ.shape[1])])
                Zg = np.column_stack([Zw, Zbm, 1 - th])
            b, e, A, _ = _tsls(yg, Xg, Zg)
            kreg = Xg.shape[1]
            rss = float(e @ e)
            V = rss / (n - kreg) * A
            XB = X_lev @ b[:k] + b[k]
            r2_o = _corr(y, XB) ** 2
            r2_w = _corr(XB - _gmean(XB, gid, G), y - _gmean(y, gid, G)) ** 2
            r2_b = _corr(np.bincount(gid, XB) / Ti, yb) ** 2
            chi2, _, _ = wald_test(b, V, list(range(k)))
            df_r = (n - kreg) if small else None
            F = chi2 / k if k else np.nan
            sigma_u, sigma_e = math.sqrt(sig_u2), math.sqrt(sig_e2)
            res.update(r2_w=r2_w, r2_b=r2_b, r2_o=r2_o, sigma_u=sigma_u, sigma_e=sigma_e, rss=rss,
                       Tcon=1 if Tcon else 0, theta=float(theta[0]) if Tcon else np.nan,
                       thta_min=float(theta.min()), thta_max=float(theta.max()),
                       thta_5=float(np.percentile(theta, 5)), thta_50=float(np.median(theta)),
                       thta_95=float(np.percentile(theta, 95)))
            title = ('EC2SLS' if ec2 else 'G2SLS') + ' random-effects IV regression'
            names_out = names + ['_cons']
        clus = None
    ncoef = len(names_out)
    info = []
    for nm in names_out:
        if nm.startswith('D.'):
            info.append(Col(name=nm, kind='cont', tsvar=nm[2:], tschain='D', vars=[nm[2:]]))
        else:
            info.append(Col(name=nm, kind='cons' if nm == '_cons' else 'cont', vars=[nm]))
    est = {'nomargins': True, 'names': names_out, 'b': np.asarray(b, float), 'V': V, 'colinfo': info,
           'df_r': df_r, 'depvar': ('D.' + yv) if fd else yv, 'sample': mask, 'cmd': 'xtivreg',
           'stat': 't' if small else 'z'}
    sc = {'N': n if model != 'be' else n, 'N_g': G, 'g_min': tmin, 'g_avg': tavg, 'g_max': tmax,
          'df_m': k, 'chi2': chi2}
    sc.update({kk: v for kk, v in res.items() if v is not None})
    if 'sigma_u' in res and 'sigma_e' in res:
        sig = math.sqrt(res['sigma_u'] ** 2 + res['sigma_e'] ** 2)
        sc['sigma'] = sig
        sc['rho'] = res['sigma_u'] ** 2 / sig ** 2 if sig > 0 else np.nan
    if small:
        sc['F'] = F
        sc['df_r'] = df_r
    if clus:
        sc['N_clust'] = Gc
    ins = ' '.join([c.name for c in i1] + [c.name for c in iz])
    est['scalars'] = sc
    est['macros'] = {'cmd': 'xtivreg', 'model': model, 'depvar': yv, 'ivar': ivar, 'tvar': tvar or '',
                     'instd': ' '.join(c.name for c in i2), 'insts': ins, 'small': 'small' if small else '',
                     'vce': 'cluster' if clus else 'conventional', 'title': title}
    clusname = clus

    def replay(ip_):
        ip_.txt('\n')
        if model == 'fd':
            ip_.txt('First-differenced IV regression\n')
            ip_.txt('Group variable:    ')
            ip_.res(f'{ivar[:12]:<30}')
        else:
            ip_.txt(f'{title:<48}{"Number of obs":<18}= ')
            ip_.res(fmt_num(n, '%10.0fc') + '\n')
            ip_.txt('Group variable: ')
            ip_.res(f'{ivar[:12]:<32}')
            ip_.txt(f'{"Number of groups":<18}= ')
            ip_.res(fmt_num(G, '%10.0fc') + '\n\n')
        if model == 'fd':
            ip_.txt(f'{"Number of obs":<18}= ')
            ip_.res(fmt_num(n, '%10.0fc') + '\n')
            ip_.txt('Time variable:     ')
            ip_.res(f'{(tvar or "")[:12]:<30}')
            ip_.txt(f'{"Number of groups":<18}= ')
            ip_.res(fmt_num(G, '%10.0fc') + '\n\n')
        ip_.txt('R-sq:' + ' ' * 43 + 'Obs per group:\n')
        for lab, key, k2, v2, f2 in (('within ', 'r2_w', 'min', tmin, '%10.0fc'),
                                     ('between', 'r2_b', 'avg', tavg, '%10.1fc'),
                                     ('overall', 'r2_o', 'max', tmax, '%10.0fc')):
            ip_.txt(f'     {lab} = ')
            ip_.res(f'{sc[key]:6.4f}')
            ip_.txt(' ' * 41 + f'{k2} = ')
            ip_.res(fmt_num(v2, f2) + '\n')
        ip_.txt('\n')
        if model in ('fe', 'fd'):
            left2 = ('corr(u_i, Xb)', f'{sc["corr"]:6.4f}', 15)
        elif model == 're':
            left2 = ('corr(u_i, X)', '0', 19)
        else:
            left2 = ('sd(u_i + avg(e_i.))', fmt_num(sc['rmse'], '%9.0g'), 15)
        if clus and not small:
            ip_.txt('\n')
        if small:
            dfm = k
            dfr = df_r
            lab1 = f'F({dfm},{dfr})'
            ip_.txt(' ' * 48 + f'{lab1:<18}=  ')
            ip_.res(f'{F:9.2f}\n')
            pv = sps.f.sf(F, dfm, dfr) if dfm else np.nan
            plab = 'Prob > F'
        else:
            lab1 = f'Wald chi2({k})'
            ip_.txt(' ' * 48 + f'{lab1:<18}=  ')
            ip_.res((f'{chi2:9.2f}' if abs(chi2) < 1e8 else fmt_num(chi2, '%9.2g')) + '\n')
            dfp = (k + 1) if (model == 'fe' and not clus) else k
            pv = sps.chi2.sf(chi2, dfp) if k else np.nan
            plab = 'Prob > chi2'
        l2 = f'{left2[0]:<{left2[2]}}= '
        ip_.txt(l2)
        ip_.res(left2[1])
        tail = ' (assumed)' if model == 're' else ''
        ip_.txt(tail + ' ' * max(1, 48 - len(l2) - len(left2[1]) - len(tail)) + f'{plab:<18}=')
        ip_.res(' ' * 5 + (f'{pv:6.4f}' if pv == pv else '     .') + '\n')
        if model == 're' and p.opts.has('theta', 2):
            if res['Tcon']:
                ip_.txt('theta' + ' ' * 14 + '= ')
                ip_.res(g_fmt(res['theta']) + '\n')
            else:
                ip_.txt('\n' + '-' * 19 + ' theta ' + '-' * 20 + '\n')
                ip_.txt('  min      5%       median        95%      max\n')
                ip_.res(f'{res["thta_min"]:6.4f}{res["thta_5"]:9.4f}{res["thta_50"]:11.4f}'
                        f'{res["thta_95"]:11.4f}{res["thta_max"]:9.4f}\n')
        ip_.txt('\n')
        rob = None
        if clus:
            ip_.txt(f'{f"(Std. err. adjusted for {comma_int(Gc)} clusters in {clusname})":>78}\n')
            rob = 'Robust'
        coef_table(ip_, est, 't' if small else 'z', robust=rob, footer=model == 'be')
        if model != 'be':
            ip_.txt('-' * 13 + '+' + '-' * 64 + '\n')
            for nm_, v_ in (('sigma_u', sc['sigma_u']), ('sigma_e', sc['sigma_e'])):
                ip_.txt(f'{nm_:>12} | ')
                ip_.res(fmt_num(v_, '%10.0g') + '\n')
            ip_.txt(f'{"rho":>12} | ')
            ip_.res(fmt_num(sc['rho'], '%10.0g'))
            ip_.txt('   (fraction of variance due to u_i)\n')
            ip_.txt('-' * 13 + '+' + '-' * 64 + '\n')
            if model == 'fe' and not clus:
                left = 'F  test that all u_i=0:     '
                ip_.txt(left + 'F(')
                ip_.res(str(G - 1))
                ip_.txt(',')
                ip_.res(str(df_rz))
                ip_.txt(') = ')
                ip_.res(f'{Ff:8.2f}')
                used = len(left) + 2 + len(str(G - 1)) + 1 + len(str(df_rz)) + 4 + 8
                ip_.txt(' ' * max(1, 58 - used) + 'Prob > F    = ')
                ip_.res(f'{sps.f.sf(Ff, G - 1, df_rz):6.4f}\n')
                ip_.txt('-' * 78 + '\n')
        instd = ' '.join(c.name for c in i2)
        _wrap(ip_, 'Instrumented:   ', instd)
        _wrap(ip_, 'Instruments:    ', ins)
        ip_.txt('-' * 78 + '\n')

    def predict(ip_, kind, m_):
        from .estim import _predict_linear
        xb = _predict_linear(ip_, est, 'xb')
        if kind in (None, '', 'xb'):
            return xb
        raise StataError(f'predict option {kind} not supported', 198)
    est['replay'] = replay
    est['predict'] = predict
    est['rhs_tokens'] = endo + exog
    try:
        _store_rhs(est, endo + exog, ce + cx, d)
    except Exception:
        pass
    if fd:
        est['predict'] = lambda ip_, kind, m_: (_ for _ in ()).throw(
            StataError('predict after xtivreg, fd is not available in Stata Web', 198))
    ip.set_estimation(est)
    replay(ip)


def _wrap(ip, head, text):
    width = 78
    words = text.split()
    line = head
    first = True
    for w in words:
        if len(line) + len(w) + 1 > width and line.strip() and len(line) > 16:
            ip.txt(line.rstrip() + '\n')
            line = ' ' * 16
        line += w + ' '
    ip.txt(line.rstrip() + '\n')
