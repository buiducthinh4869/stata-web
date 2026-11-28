"""xthtaylor (Hausman-Taylor / Amemiya-MaCurdy), ported from Stata's xthtaylor.ado."""
import math

import numpy as np
from scipy import stats as sps

from .errors import StataError
from .fmt import g_fmt, comma_int, fmt_num
from .interp import command
from .syntax import parse_command, split_top


def _gm(v, gid, G, Ti):
    return (np.bincount(gid, v, minlength=G) / Ti)[gid]


def _iv(y, X, Z):
    ZtZi = np.linalg.pinv(Z.T @ Z)
    PX = Z @ (ZtZi @ (Z.T @ X))
    A = np.linalg.pinv(PX.T @ X)
    b = A @ (PX.T @ y)
    return b, y - X @ b, A, PX


@command('xthtaylor')
def cmd_xthtaylor(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    if not d.ts or not d.ts.get('panel'):
        raise StataError('panel variable not set; use xtset varname ...', 459)
    ivar, tvar = d.ts['panel'], d.ts.get('time')
    toks = split_top(p.body)
    from .tsops import expand_token
    vl = []
    for t in toks:
        vl += d.expand_varlist([t]) if '.' not in t else expand_token(t)
    yname, xz = vl[0], vl[1:]
    endog = []
    for t in split_top(p.opts.get('endog', 3) or ''):
        endog += d.expand_varlist([t]) if '.' not in t else expand_token(t)
    if not endog:
        raise StataError('option endog() required', 198)
    if any(e not in xz for e in endog):
        raise StataError('some variables specified in endog() not included in indepvars', 198)
    amac = p.opts.has('amacurdy', 2)
    small = p.opts.has('small')
    nocons = p.opts.has('noconstant', 4)
    vce = (p.opts.get('vce', 1) or '').strip().lower()
    clusvar = None
    if vce and not vce.startswith('conv'):
        w = vce.split()
        if w[0].startswith('r'):
            clusvar = ivar
        elif w[0].startswith('cl'):
            clusvar = d.resolve(w[1])
        else:
            raise StataError('invalid vce() option', 198)
    from .tsuni import series

    def val(v):
        return series(ip, v)[0] if '.' in v else d.cols[d.resolve(v)].astype(float)
    y_all = val(yname)
    mask = ip.mark(p, d) & ~np.isnan(y_all) & ~np.isnan(d.cols[ivar].astype(float))
    V_all = {v: val(v) for v in xz}
    for v in xz:
        mask &= ~np.isnan(V_all[v])
    if clusvar:
        mask &= ~np.isnan(d.cols[clusvar].astype(float))
    pid = d.cols[ivar].astype(float)[mask]
    _, gid = np.unique(pid, return_inverse=True)
    G = int(gid.max()) + 1
    Ti = np.bincount(gid).astype(float)
    n = int(mask.sum())
    y = y_all[mask]
    cols = {v: V_all[v][mask] for v in xz}
    # time-varying vs invariant
    varying, invariant = [], []
    for v in xz:
        x = cols[v]
        dev = np.max(np.abs(x - _gm(x, gid, G, Ti))) if n else 0
        (varying if dev > 1e-10 * (1 + np.max(np.abs(x))) else invariant).append(v)
    for opt, lst in (('constant', invariant), ('varying', varying)):
        s = p.opts.get(opt, 4)
        if s:
            given = split_top(s)
            if set(given) != set(lst):
                raise StataError(f'variables in {opt}() do not match the time-{"in" if opt == "constant" else ""}'
                                 'variant variables in the data', 198)
    exog = [v for v in xz if v not in endog]
    x1 = [v for v in exog if v in varying]
    x2 = [v for v in endog if v in varying]
    z1 = [v for v in exog if v in invariant]
    z2 = [v for v in endog if v in invariant]
    if not x1 or not z1 and False:
        pass
    if not x1:
        raise StataError('there must be at least one exogenous time-varying variable', 498)
    if len(x1) < len(z2) and not amac:
        raise StataError('the order condition is not satisfied: number of exogenous time-varying variables must be '
                         'at least as large as number of endogenous time-invariant variables', 481)
    tmin, tmax, tavg = int(Ti.min()), int(Ti.max()), float(Ti.mean())
    Tcon = tmin == tmax
    Tbar = tmax if Tcon else 1 / float(np.mean(1 / Ti))

    def M(names):
        return np.column_stack([cols[v] for v in names]) if names else np.zeros((n, 0))
    X1, X2, Z1, Z2 = M(x1), M(x2), M(z1), M(z2)
    Xv = np.column_stack([X1, X2])
    gmXv = np.column_stack([_gm(Xv[:, q], gid, G, Ti) for q in range(Xv.shape[1])])
    ym = _gm(y, gid, G, Ti)
    Xdm = Xv - gmXv
    bw = np.linalg.lstsq(Xdm, y - ym, rcond=None)[0]
    ew = (y - ym) - Xdm @ bw
    sigV2 = float(ew @ ew) / (n - G)
    di = ym - gmXv @ bw
    xbw = Xv @ bw
    Zz = np.column_stack([Z1, Z2, np.ones(n)])
    Zi = np.column_stack([X1, Z1, np.ones(n)])
    gz, _, _, _ = _iv(di, Zz, Zi)
    zg = Zz @ gz
    res = y - xbw - zg
    rbar = np.bincount(gid, res) / Ti
    sig12 = float(np.sum(Ti * rbar ** 2)) / G
    sigU2 = max((sig12 - sigV2) / Tbar, 0.0)
    if sigU2 == 0:
        theta = np.zeros(n)
    elif Tcon:
        theta = np.full(n, 1 - math.sqrt(sigV2 / sig12))
    else:
        theta = (1 - np.sqrt(sigV2 / (Ti * sigU2 + sigV2)))[gid]
    yg = y - theta * ym
    Xg = np.column_stack([Xv - theta[:, None] * gmXv, (1 - theta)[:, None] * np.column_stack([Z1, Z2])])
    names = x1 + x2 + z1 + z2
    if not nocons:
        Xg = np.column_stack([Xg, 1 - theta])
        names = names + ['_cons']
    X1m = np.column_stack([_gm(X1[:, q], gid, G, Ti) for q in range(X1.shape[1])])
    if not amac:
        title = 'Hausman-Taylor'
        Zg = np.column_stack([Xdm, X1m, Z1] + ([1 - theta] if not nocons else []))
    else:
        title = 'Amemiya-MaCurdy'
        if not Tcon:
            raise StataError('Amemiya-MaCurdy estimator requires balanced panels', 498)
        tv = d.cols[tvar].astype(float)[mask]
        xs = []
        for q in range(X1.shape[1]):
            for t in range(tmax):
                col = np.zeros(n)
                for g in range(G):
                    s = np.flatnonzero(gid == g)
                    s = s[np.argsort(tv[s], kind='stable')]
                    col[s] = X1[s[t], q]
                xs.append(col)
        Zg = np.column_stack([Xdm] + xs + [Z1] + ([1 - theta] if not nocons else []))
    b, e, A, PX = _iv(yg, Xg, Zg)
    k = Xg.shape[1]
    rss = float(e @ e)
    df_r = n - k
    s2 = rss / df_r
    V = s2 * A
    Gc = None
    if clusvar:
        cid = d.cols[clusvar][mask]
        _, cg = np.unique(cid, return_inverse=True)
        Gc = int(cg.max()) + 1
        S = np.zeros((Gc, k))
        np.add.at(S, cg, PX * e[:, None])
        V = A @ (S.T @ S) @ A * (Gc / (Gc - 1)) * ((n - 1) / (n - k))
    slopes = list(range(len(names) - (0 if nocons else 1)))
    bs, Vs = b[slopes], V[np.ix_(slopes, slopes)]
    if not clusvar:
        if not nocons:
            W = float(bs @ np.linalg.pinv(Vs) @ bs)
            dfm = len(slopes)
        else:
            W = float(b @ np.linalg.pinv(V) @ b)
            dfm = k
        F = W / dfm
        chi2 = F * dfm
    else:
        chi2 = float(b @ np.linalg.pinv(V) @ b)
        dfm = len(slopes) if not nocons else k
        F = chi2 / dfm
    sigU, sigV = math.sqrt(sigU2), math.sqrt(sigV2)
    rho = sigU2 / (sigU2 + sigV2) if sigU2 + sigV2 > 0 else np.nan
    from .design import Col
    from .estim import _store_rhs
    info = [Col(name=nm, kind='cons' if nm == '_cons' else 'cont', vars=[nm]) for nm in names]
    est = {'nomargins': True, 'names': names, 'b': b, 'V': V, 'colinfo': info, 'df_r': df_r if small else None,
           'depvar': yname, 'sample': mask, 'cmd': 'xthtaylor', 'stat': 't' if small else 'z',
           'scalars': {'N': n, 'N_g': G, 'g_min': tmin, 'g_avg': tavg, 'g_max': tmax, 'chi2': chi2, 'df_m': dfm,
                       'sigma_u': sigU, 'sigma_e': sigV, 'rho': rho, 'Tbar': Tbar, 'Tcon': int(Tcon),
                       'N_clust': Gc or G},
           'macros': {'cmd': 'xthtaylor', 'depvar': yname, 'ivar': ivar, 'title': title,
                      'TVexogenous': ' '.join(x1), 'TVendogenous': ' '.join(x2), 'TIexogenous': ' '.join(z1),
                      'TIendogenous': ' '.join(z2), 'vce': 'cluster' if clusvar else 'conventional',
                      'chi2type': 'Wald'}}
    if small:
        est['scalars']['df_r'] = df_r
    groups = [('TVexogenous', x1), ('TVendogenous', x2), ('TIexogenous', z1), ('TIendogenous', z2)]

    def replay(ip_):
        ip_.txt(f'\n{title + " estimation":<48}{"Number of obs":<18}= ')
        ip_.res(fmt_num(n, '%10.0fc') + '\n')
        ip_.txt('Group variable: ')
        ip_.res(f'{ivar[:12]:<32}')
        ip_.txt(f'{"Number of groups":<18}= ')
        ip_.res(fmt_num(G, '%10.0fc') + '\n\n')
        left = ''
        if amac:
            left = f'Time variable: {tvar[:12]}'
            ip_.txt('Time variable: ')
            ip_.res(tvar[:12])
        ip_.txt(' ' * (48 - len(left)) + 'Obs per group:\n')
        avgfmt = '%10.0f' if float(tavg).is_integer() else '%10.1f'
        for kk, vv, ff in (('min', tmin, '%10.0g'), ('avg', tavg, avgfmt), ('max', tmax, '%10.0g')):
            ip_.txt(' ' * 62 + f'{kk} = ')
            ip_.res(fmt_num(vv, ff) + '\n')
        ip_.txt('\nRandom effects u_i ~ ')
        ip_.res('i.i.d.')
        lab = f'Wald chi2({dfm})'
        ip_.txt(' ' * (48 - 27) + f'{lab:<18}= ')
        ip_.res(f'{chi2:10.2f}\n')
        ip_.txt(' ' * 48 + f'{"Prob > chi2":<18}=')
        ip_.res(' ' * 5 + f'{sps.chi2.sf(abs(chi2), dfm):6.4f}\n\n')
        if clusvar:
            ip_.txt(f'{f"(Std. err. adjusted for {comma_int(Gc)} clusters in {clusvar})":>78}\n')
        tst = 't' if small else 'z'
        ip_.txt('-' * 13 + '+' + '-' * 64 + '\n')
        if clusvar:
            ip_.txt(f'{"":>12} |{"Robust":>21}\n')
        se_lab = 'std. err.' if clusvar else 'Std. err.'
        ip_.txt(f'{yname[:12]:>12} | Coefficient  {se_lab}      {tst}    P>|{tst}|     [95% conf. interval]\n')
        ip_.txt('-' * 13 + '+' + '-' * 64 + '\n')
        se = np.sqrt(np.maximum(np.diag(V), 0))
        crit = sps.t.ppf(.975, df_r) if small else sps.norm.ppf(.975)

        def row(nm, j):
            z = b[j] / se[j] if se[j] > 0 else np.nan
            pv = 2 * (sps.t.sf(abs(z), df_r) if small else sps.norm.sf(abs(z)))
            ip_.txt(f'{nm[:12]:>12} |')
            ip_.res(f'{g_fmt(b[j], 9):>11}{g_fmt(se[j], 9):>11}{z:9.2f}{pv:8.3f}'
                    f'{g_fmt(b[j] - crit * se[j], 9):>13}{g_fmt(b[j] + crit * se[j], 9):>12}\n')
        j = 0
        for gname, lst in groups:
            if not lst:
                continue
            ip_.res(f'{gname:<12}')
            ip_.txt(' |\n')
            for v in lst:
                row(v, names.index(v))
        if not nocons:
            ip_.txt(f'{"":>12} |\n')
            row('_cons', len(names) - 1)
        ip_.txt('-' * 13 + '+' + '-' * 64 + '\n')
        for nm_, v_ in (('sigma_u', sigU), ('sigma_e', sigV)):
            ip_.txt(f'{nm_:>12} | ')
            ip_.res(fmt_num(v_, '%10.0g') + '\n')
        ip_.txt(f'{"rho":>12} | ')
        ip_.res(fmt_num(rho, '%10.0g'))
        ip_.txt('   (fraction of variance due to u_i)\n')
        ip_.txt('-' * 78 + '\n')
        ip_.txt('Note: ')
        ip_.res('TV')
        ip_.txt(' refers to time varying; ')
        ip_.res('TI')
        ip_.txt(' refers to time invariant.\n')

    def predict(ip_, kind, m_):
        if kind not in (None, '', 'xb'):
            raise StataError(f'predict option {kind} not supported', 198)
        out = np.zeros(d.N) + (b[-1] if not nocons else 0)
        for j, nm in enumerate(names):
            if nm != '_cons':
                out = out + b[j] * val(nm)
        return out
    est['replay'] = replay
    est['predict'] = predict
    ip.set_estimation(est)
    replay(ip)
