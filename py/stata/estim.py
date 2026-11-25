"""Linear estimation commands and post-estimation."""
import math
import re

import numpy as np
from scipy import stats as sps

from . import expr as E
from .design import build_design, mark_collinear, coef_table, right_lines, Col
from .errors import StataError
from .fmt import g_fmt, fmt_num, comma_int, abbrev, num_to_macro
from .interp import command, alias, canon_coef
from .syntax import parse_command, split_top, unquote, Options, parse_terms, parse_numlist

alias('regress', 'reg', 'regr', 'regre', 'regres')


# ================================================================ helpers
def parse_vce(opts):
    """Return (type, clustvar) with type in ols/robust/cluster."""
    v = opts.get('vce')
    if v is not None:
        parts = v.split()
        k = parts[0].lower() if parts else ''
        if k.startswith('r'):
            return 'robust', None
        if k.startswith('cl'):
            return 'cluster', parts[1] if len(parts) > 1 else None
        if k in ('ols', 'unadjusted', 'conventional', 'oim', 'gls', 'eim', 'opg'):
            return 'ols', None
        if k.startswith('boot'):
            return 'bootstrap', None
        return 'ols', None
    cl = opts.get('cluster', 2)
    if cl:
        return 'cluster', cl.split()[0]
    if opts.has('robust', 1):
        return 'robust', None
    return 'ols', None


def get_weights(ip, p, d):
    if not p.weight:
        return None, None
    w = np.asarray(E.eval_expr(p.weight[1], E.Ctx(ip, d)), dtype=float) * np.ones(d.n)
    return w, p.weight[0]


def cluster_ids(d, var, mask):
    v = d.resolve(var)
    x = d.cols[v]
    if d.isstr(v):
        _, inv = np.unique(np.array([str(s) for s in x[mask]]), return_inverse=True)
    else:
        _, inv = np.unique(x[mask], return_inverse=True)
    return inv, v


def sandwich(Xs, scores, bread, groups=None):
    """bread @ meat @ bread with scores (n x k) optionally summed within groups."""
    if groups is not None:
        G = int(groups.max()) + 1
        S = np.zeros((G, scores.shape[1]))
        np.add.at(S, groups, scores)
    else:
        S = scores
    meat = S.T @ S
    return bread @ meat @ bread


class Fit:
    pass


def prepare_linear(ip, p, d, rhs_tokens, depvar_tok, extra_mask=None, nocons=False, override=None):
    y_var = d.resolve(depvar_tok)
    y = d.cols[y_var].astype(float)
    mask = ip.mark(p, d)
    if extra_mask is not None:
        mask &= extra_mask
    mask &= ~np.isnan(y)
    w, wtype = get_weights(ip, p, d)
    if w is not None:
        mask &= ~np.isnan(w) & (w > 0)
    cols, mask = build_design(ip, d, rhs_tokens, mask, override=override)
    return y_var, y, cols, mask, w, wtype


def assemble(cols, mask, cons=True, demean_groups=None, w=None):
    """Return X (n_sample x k_active) with constant last, list of names, colinfo, active idx."""
    Xall = np.column_stack([c.values[mask] for c in cols]) if cols else np.zeros((int(mask.sum()), 0))
    names = [c.name for c in cols]
    info = list(cols)
    if cons:
        names = names + ['_cons']
        info = info + [Col(name='_cons', values=None, term=-1, group='_cons', level='', kind='cons', vars=[])]
        Xall = np.column_stack([Xall, np.ones(Xall.shape[0])])
    return Xall, names, info


def detect_omitted(Xall, info, cons=True, Xcheck=None):
    """Mark base and collinear columns; returns active index list."""
    Xc = Xcheck if Xcheck is not None else Xall
    k = Xall.shape[1]
    order = list(range(k))
    if cons:
        order = [k - 1] + list(range(k - 1))
    cand = [j for j in order if not info[j].base]
    sub = Xc[:, cand]
    om = mark_collinear(sub, None)
    for j, o in zip(cand, om):
        if o:
            info[j].omitted = True
            if info[j].kind == 'factor':
                info[j].name = _omit_name(info[j].name)
            elif info[j].kind == 'cons':
                pass
            else:
                info[j].name = 'o.' + info[j].name
    active = [j for j in range(k) if not info[j].base and not info[j].omitted]
    return active


def _omit_name(name):
    parts = name.split('#')
    out = []
    for p in parts:
        m = re.match(r'^(\d+(?:\.\d+)?)(b?)\.(.+)$', p)
        if m and 'o' not in m.group(2):
            out.append(f'{m.group(1)}{m.group(2)}o.{m.group(3)}')
        else:
            out.append(p)
    return '#'.join(out)


def full_bV(k, active, b_act, V_act):
    b = np.zeros(k)
    V = np.zeros((k, k))
    b[active] = b_act
    V[np.ix_(active, active)] = V_act
    return b, V


def wald_test(b, V, idx, df_r=None):
    """Wald test that b[idx]==0. Returns (stat, df1, p, kind)."""
    if not idx:
        return np.nan, 0, np.nan
    bb = b[idx]
    VV = V[np.ix_(idx, idx)]
    try:
        Vi = np.linalg.pinv(VV)
        q = np.linalg.matrix_rank(VV)
    except np.linalg.LinAlgError:
        return np.nan, len(idx), np.nan
    chi2 = float(bb @ Vi @ bb)
    if df_r is not None:
        F = chi2 / q if q else np.nan
        return F, q, sps.f.sf(F, q, df_r) if q else np.nan
    return chi2, q, sps.chi2.sf(chi2, q) if q else np.nan


# ================================================================ regress
@command('regress')
def cmd_regress(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    toks = split_top(p.body)
    if not toks:
        if ip.est and ip.est.get('cmd') == 'regress':
            ip.est['replay'](ip)
            return
        raise StataError('varlist required', 100)
    nocons = p.opts.has('noconstant', 4)
    yv, y, cols, mask, w, wtype = prepare_linear(ip, p, d, toks[1:], toks[0])
    vce, clv = parse_vce(p.opts)
    if wtype == 'pweight' and vce == 'ols':
        vce = 'robust'
    if clv:
        cv = d.resolve(clv)
        cx = d.cols[cv]
        mask &= (cx != '') if d.isstr(cv) else ~np.isnan(cx)
    fit = ols_core(ip, d, y, cols, mask, w, wtype, vce, clv, nocons)
    fit.depvar = yv
    est = make_ols_est(ip, fit, 'regress', p)
    if p.opts.has('beta'):
        X = fit.Xfull
        sdx = np.array([np.sqrt(np.cov(X[:, j], aweights=fit.wn)) if fit.info[j].kind != 'cons' else np.nan
                        for j in range(X.shape[1])])
        sdy = np.sqrt(np.cov(fit.y, aweights=fit.wn))
        est['beta'] = fit.b * sdx / sdy
    ip.set_estimation(est)
    est['replay'](ip)


def ols_core(ip, d, y, cols, mask, w, wtype, vce, clv, nocons=False, absorb_dfa=0, Xpre=None, ypre=None,
             df_extra=0):
    f = Fit()
    Xall, names, info = assemble(cols, mask, cons=not nocons)
    yy = y[mask] if ypre is None else ypre
    n = len(yy)
    if n == 0:
        raise StataError('no observations', 2000)
    if w is not None:
        ws = w[mask]
        if wtype in ('aweight', 'pweight'):
            wn = ws * n / ws.sum()
        else:
            wn = ws
    else:
        ws = None
        wn = np.ones(n)
    Xc = Xall if Xpre is None else Xpre
    sw = np.sqrt(wn)
    active = detect_omitted(Xc * sw[:, None], info, cons=not nocons and Xpre is None)
    X = Xc[:, active]
    XtWX = X.T @ (X * wn[:, None])
    try:
        XtXi = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        XtXi = np.linalg.pinv(XtWX)
    b_act = XtXi @ (X.T @ (wn * yy))
    e = yy - X @ b_act
    N = int(wn.sum()) if wtype == 'fweight' else n
    k = len(active)
    df_r = N - k - absorb_dfa - df_extra
    rss = float(np.sum(wn * e ** 2))
    if not nocons:
        ybar = np.sum(wn * yy) / wn.sum()
        tss = float(np.sum(wn * (yy - ybar) ** 2))
    else:
        tss = float(np.sum(wn * yy ** 2))
    mss = tss - rss
    s2 = rss / df_r if df_r > 0 else np.nan
    G = None
    if vce == 'ols':
        V_act = s2 * XtXi
    elif vce == 'robust':
        scores = X * (wn * e)[:, None] if wtype != 'pweight' else X * (wn * e)[:, None]
        V_act = sandwich(X, scores, XtXi) * (N / (N - k - absorb_dfa - df_extra) if N - k > 0 else 1)
    elif vce == 'cluster':
        gid, cname = cluster_ids(d, clv, mask)
        G = int(gid.max()) + 1
        scores = X * (wn * e)[:, None]
        qc = G / (G - 1) * (N - 1) / (N - k - absorb_dfa - df_extra)
        V_act = sandwich(X, scores, XtXi, gid) * qc
        f.clustvar = cname
    else:
        V_act = s2 * XtXi
    b, V = full_bV(len(names), active, b_act, V_act)
    f.names = [c.name for c in info]
    f.info = info
    f.b, f.V = b, V
    f.N, f.k, f.df_r = N, k, df_r
    f.rss, f.tss, f.mss = rss, tss, mss
    f.e, f.y, f.Xfull, f.wn = e, yy, Xall, wn
    f.mask = mask
    f.vce, f.G = vce, G
    f.nocons = nocons
    f.active = active
    f.s2 = s2
    f.wtype = wtype
    df_m = k - (0 if nocons else 1)
    f.df_m = df_m
    if vce == 'cluster':
        f.df_r_test = G - 1
    else:
        f.df_r_test = df_r
    return f


def make_ols_est(ip, f, cmdname, p, title='Linear regression'):
    N, k, df_r = f.N, f.k, f.df_r
    r2 = f.mss / f.tss if f.tss > 0 else np.nan
    r2a = 1 - (1 - r2) * (N - (0 if f.nocons else 1)) / df_r if df_r > 0 else np.nan
    rmse = math.sqrt(f.s2) if f.s2 == f.s2 and f.s2 >= 0 else np.nan
    slopes = [j for j in f.active if f.info[j].kind != 'cons']
    df_rt = f.df_r_test
    if f.vce == 'ols':
        df_m = f.df_m
        F = (f.mss / df_m) / (f.rss / df_r) if df_m > 0 and df_r > 0 and f.rss > 0 else np.nan
        pF = sps.f.sf(F, df_m, df_r) if F == F else np.nan
    else:
        F, df_m, pF = wald_test(f.b, f.V, slopes, df_rt)
        if f.vce == 'cluster' and df_m > df_rt:
            F, pF = np.nan, np.nan
    est = {'names': f.names, 'b': f.b, 'V': f.V, 'colinfo': f.info, 'df_r': df_rt, 'depvar': f.depvar,
           'sample': f.mask, 'cmd': cmdname, 'stat': 't', 'vce': f.vce, 'fit': f,
           'scalars': {'N': N, 'df_m': df_m, 'df_r': df_rt, 'F': F, 'r2': r2, 'r2_a': r2a, 'rmse': rmse,
                       'rss': f.rss, 'mss': f.mss, 'rank': k, 'll': _ols_ll(f)},
           'macros': {'cmd': cmdname, 'depvar': f.depvar, 'vce': f.vce if f.vce != 'ols' else 'ols',
                      'title': title, 'clustvar': getattr(f, 'clustvar', '') or '',
                      'vcetype': {'robust': 'Robust', 'cluster': 'Robust'}.get(f.vce, '')}}
    if f.vce == 'cluster':
        est['scalars']['N_clust'] = f.G
    est['tokens'] = None
    est['mask'] = f.mask

    def replay(ip_):
        ip_.txt('\n')
        if f.vce == 'ols' and cmdname == 'regress':
            _anova_header(ip_, f, F, pF, r2, r2a, rmse)
        else:
            right = [('Number of obs', comma_int(N)),
                     (f'F({df_m}, {df_rt})', fmt_num(F, '%10.2f').strip() if F == F else '.'),
                     ('Prob > F', fmt_num(pF, '%10.4f').strip() if pF == pF else '.'),
                     ('R-squared', f'{r2:.4f}'),
                     ('Root MSE', g_fmt(rmse, 9) if rmse == rmse else '.')]
            if f.vce == 'ols':
                right.insert(4, ('Adj R-squared', f'{r2a:.4f}'))
            right_lines(ip_, [title], right)
            ip_.txt('\n')
        if f.vce == 'cluster':
            msg = f'(Std. err. adjusted for {comma_int(f.G)} clusters in {f.clustvar})'
            ip_.txt(f'{msg:>78}\n')
        coef_table(ip_, est, 't', robust='Robust' if f.vce in ('robust', 'cluster') else None,
                   beta=est.get('beta'))
    est['replay'] = replay
    est['predict'] = lambda ip_, kind, mask_: _predict_linear(ip_, est, kind)
    return est


def _ols_ll(f):
    n = f.N
    if f.rss <= 0:
        return np.nan
    return -0.5 * n * (math.log(2 * math.pi) + math.log(f.rss / n) + 1)


def _anova_header(ip, f, F, pF, r2, r2a, rmse):
    N, df_m, df_r = f.N, f.df_m, f.df_r
    ip.txt('      Source |       SS           df       MS      Number of obs   = ')
    ip.res(f'{comma_int(N):>9}\n')
    ip.txt('-------------+----------------------------------   ')
    lab = f'F({df_m}, {df_r})'
    ip.txt(f'{lab:<16}= ')
    ip.res(f'{fmt_num(F, "%9.2f").strip() if F == F else ".":>9}\n')
    ms_m = f.mss / df_m if df_m > 0 else np.nan
    ip.txt('       Model |')
    ip.res(f' {g_fmt(f.mss, 10):>11}{df_m:>11}{g_fmt(ms_m, 10):>12}')
    ip.txt('   Prob > F        = ')
    ip.res(f'{fmt_num(pF, "%9.4f").strip() if pF == pF else ".":>9}\n')
    ip.txt('    Residual |')
    ip.res(f' {g_fmt(f.rss, 10):>11}{df_r:>11}{g_fmt(f.rss / df_r if df_r > 0 else np.nan, 10):>12}')
    ip.txt('   R-squared       = ')
    ip.res(f'{r2:>9.4f}\n')
    ip.txt('-------------+----------------------------------   Adj R-squared   = ')
    ip.res(f'{r2a:>9.4f}\n')
    tdf = N - 1 if not f.nocons else N
    ip.txt('       Total |')
    ip.res(f' {g_fmt(f.tss, 10):>11}{tdf:>11}{g_fmt(f.tss / tdf, 10):>12}')
    ip.txt('   Root MSE        = ')
    ip.res(f'{g_fmt(rmse, 9):>9}\n\n')


def _predict_linear(ip, est, kind, override=None):
    """Linear prediction for any observation (uses current data)."""
    d = ip.data
    toks = est.get('rhs_tokens')
    if toks is None:
        raise StataError('prediction not available', 198)
    m = np.ones(d.n, dtype=bool)
    levels = est.get('levels')
    cols, _ = build_design(ip, d, toks, m, override=override, levels_from=levels)
    names = [c.name for c in cols]
    b = est['b']
    xb = np.zeros(d.n)
    bmap = {canon_coef(n): bb for n, bb in zip(est['names'], b)}
    for c in cols:
        key = canon_coef(c.name)
        if key in bmap:
            xb += np.nan_to_num(c.values) * bmap[key]
    if '_cons' in est['names']:
        xb += bmap['_cons']
    # missing where any regressor missing
    miss = np.zeros(d.n, dtype=bool)
    for t in parse_terms(toks):
        for c in t:
            v = d.resolve(c.var)
            if c.tsop:
                vals = ip.ts_op(E.Ctx(ip, d), c.tsop[0], c.tsop[1], v)
            else:
                vals = d.cols[v] if (override is None or v not in override) else override[v]
            miss |= np.isnan(np.asarray(vals, dtype=float))
    xb[miss] = np.nan
    if est.get('xb_offset') is not None:
        xb = xb + est['xb_offset']
    if kind in ('xb', None, ''):
        return xb
    if kind in ('residuals', 'resid', 'res', 'r'):
        y = d.cols[est['depvar']]
        return y - xb
    raise StataError(f'predict option {kind} not supported', 198)


def _store_rhs(est, toks, cols, d):
    est['rhs_tokens'] = toks
    levels = {}
    for t in parse_terms(toks):
        for c in t:
            if c.kind == 'i':
                v = d.resolve(c.var)
                ok = d.cols[v][est['sample']]
                levels[v] = sorted(np.unique(ok[~np.isnan(ok)]).tolist())
    est['levels'] = levels


_orig_regress = cmd_regress


@command('regress')
def cmd_regress2(ip, cmd, rest, by):
    from .cmds_stats import by_loop
    if by:
        def run(extra):
            _regress_impl(ip, rest, extra)
        by_loop(ip, by, run)
        return
    _regress_impl(ip, rest, None)


def _regress_impl(ip, rest, extra):
    p = parse_command(rest)
    d = ip.data
    toks = split_top(p.body)
    if not toks:
        if ip.est and ip.est.get('replay'):
            ip.est['replay'](ip)
            return
        raise StataError('varlist required', 100)
    nocons = p.opts.has('noconstant', 4)
    yv, y, cols, mask, w, wtype = prepare_linear(ip, p, d, toks[1:], toks[0], extra_mask=extra)
    vce, clv = parse_vce(p.opts)
    if wtype == 'pweight' and vce == 'ols':
        vce = 'robust'
    if clv:
        cv = d.resolve(clv)
        cx = d.cols[cv]
        mask &= (cx != '') if d.isstr(cv) else ~np.isnan(cx)
    fit = ols_core(ip, d, y, cols, mask, w, wtype, vce, clv, nocons)
    fit.depvar = yv
    est = make_ols_est(ip, fit, 'regress', p)
    _store_rhs(est, toks[1:], cols, d)
    if p.opts.has('beta'):
        X = fit.Xfull
        sdx = np.array([np.sqrt(np.cov(X[:, j])) if fit.info[j].kind != 'cons' else np.nan
                        for j in range(X.shape[1])])
        sdy = np.sqrt(np.cov(fit.y))
        est['beta'] = fit.b * sdx / sdy
    ip.set_estimation(est)
    est['replay'](ip)


# ================================================================ predict
@command('predict')
def cmd_predict(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    if ip.est is None:
        raise StataError('last estimates not found', 301)
    toks = p.body.split()
    typ = None
    if toks and toks[0] in ('double', 'float'):
        typ = toks.pop(0)
    name = toks[0]
    kinds = [n for n in p.opts.names()]
    kind = kinds[0] if kinds else None
    fn = ip.est.get('predict')
    if fn is None:
        raise StataError('predict not available after ' + ip.est.get('cmd', ''), 198)
    vals = fn(ip, kind, None)
    mask = ip.mark(p)
    vals = np.where(mask, vals, np.nan)
    d.add_var(name, vals, typ or 'float')
    lab = {'xb': 'Linear prediction', None: 'Fitted values', 'residuals': 'Residuals', 'pr': 'Pr(' + ip.est.get('depvar', '') + ')'}
    if kind in (None, 'xb'):
        d.varlab[name] = 'Fitted values' if ip.est.get('cmd') in ('regress',) else 'Linear prediction'
    elif kind and kind.startswith('r'):
        d.varlab[name] = 'Residuals'
    elif kind == 'pr':
        d.varlab[name] = f'Pr({ip.est.get("depvar", "")})'
    nm = int(np.isnan(d.cols[name]).sum())
    if nm:
        ip.txt(f'({comma_int(nm)} missing value{"s" if nm != 1 else ""} generated)\n')


# ================================================================ xtreg
@command('xtreg')
def cmd_xtreg(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    toks = split_top(p.body)
    if not d.ts or not d.ts.get('panel'):
        raise StataError('panel variable not set; use xtset varname ...', 459)
    ivar = d.ts['panel']
    model = 'fe' if p.opts.has('fe') else ('be' if p.opts.has('be') else ('mle' if p.opts.has('mle') else 're'))
    yv, y, cols, mask, w, wtype = prepare_linear(ip, p, d, toks[1:], toks[0])
    mask &= ~np.isnan(d.cols[ivar])
    vce, clv = parse_vce(p.opts)
    if vce == 'robust':
        vce, clv = 'cluster', ivar
    if wtype == 'pweight' and vce == 'ols':
        vce, clv = 'cluster', ivar
    if model == 'fe':
        est = _xtreg_fe(ip, d, y, yv, cols, mask, w, wtype, vce, clv, ivar)
    elif model == 'be':
        est = _xtreg_be(ip, d, y, yv, cols, mask, ivar)
    else:
        est = _xtreg_re(ip, d, y, yv, cols, mask, vce, clv, ivar)
    _store_rhs(est, toks[1:], cols, d)
    ip.set_estimation(est)
    est['replay'](ip)


def _group_index(d, ivar, mask):
    g = d.cols[ivar][mask]
    _, inv = np.unique(g, return_inverse=True)
    return inv


def _gmeans(x, gid, w=None):
    G = int(gid.max()) + 1
    if w is None:
        w = np.ones(len(gid))
    sw = np.bincount(gid, weights=w, minlength=G)
    if x.ndim == 1:
        return np.bincount(gid, weights=w * x, minlength=G) / sw
    out = np.zeros((G, x.shape[1]))
    for j in range(x.shape[1]):
        out[:, j] = np.bincount(gid, weights=w * x[:, j], minlength=G) / sw
    return out


def _xtreg_fe(ip, d, y, yv, cols, mask, w, wtype, vce, clv, ivar):
    gid = _group_index(d, ivar, mask)
    Xall, names, info = assemble(cols, mask, cons=True)
    yy = y[mask]
    n = len(yy)
    ww = w[mask] if w is not None else np.ones(n)
    if wtype in ('aweight', 'pweight'):
        ww = ww * n / ww.sum()
    G = int(gid.max()) + 1
    ybar_i = _gmeans(yy, gid, ww)
    Xbar_i = _gmeans(Xall, gid, ww)
    ybar = np.sum(ww * yy) / ww.sum()
    Xbar = (ww @ Xall) / ww.sum()
    yt = yy - ybar_i[gid] + ybar
    Xt = Xall - Xbar_i[gid] + Xbar
    # collinearity on demeaned (constant handled separately)
    Xchk = Xall - Xbar_i[gid]
    k_all = Xall.shape[1]
    cand = [j for j in range(k_all - 1) if not info[j].base]
    om = mark_collinear(Xchk[:, cand] * np.sqrt(ww)[:, None], None)
    for j, o in zip(cand, om):
        if o:
            info[j].omitted = True
            info[j].name = _omit_name(info[j].name) if info[j].kind == 'factor' else 'o.' + info[j].name
    active = [j for j in range(k_all) if not info[j].base and not info[j].omitted]
    X = Xt[:, active]
    XtWX = X.T @ (X * ww[:, None])
    XtXi = np.linalg.pinv(XtWX)
    b_act = XtXi @ (X.T @ (ww * yt))
    e = yt - X @ b_act
    K = len(active) - 1
    df_r = n - G - K
    rss = float(np.sum(ww * e ** 2))
    sig_e2 = rss / df_r
    Gc = None
    if vce == 'ols':
        V_act = sig_e2 * XtXi
        df_test = df_r
    else:
        cgid, cname = cluster_ids(d, clv, mask)
        Gc = int(cgid.max()) + 1
        scores = X * (ww * e)[:, None]
        qc = Gc / (Gc - 1) * (n - 1) / (n - K - 1)
        V_act = sandwich(X, scores, XtXi, cgid) * qc
        df_test = Gc - 1
    b, V = full_bV(len(names), active, b_act, V_act)
    names = [c.name for c in info]
    slopes = [j for j in active if info[j].kind != 'cons']
    if vce == 'ols':
        F, dfm, pF = wald_test(b, V, slopes, df_r)
    else:
        F, dfm, pF = wald_test(b, V, slopes, df_test)
    xb = Xall[:, active] @ b_act
    xb_noc = xb - b[-1]
    yw = yy - ybar_i[gid]
    xw = xb - (_gmeans(xb, gid))[gid]
    r2_w = _corr2(xw, yw)
    r2_b = _corr2(_gmeans(xb, gid), ybar_i)
    r2_o = _corr2(xb, yy)
    u = ybar_i - _gmeans(xb, gid)
    sigma_u = float(np.std(u, ddof=1)) if G > 1 else 0.0
    sigma_e = math.sqrt(sig_e2)
    rho = sigma_u ** 2 / (sigma_u ** 2 + sigma_e ** 2)
    corr_u = float(np.corrcoef(u[gid], xb_noc)[0, 1]) if np.std(xb_noc) > 0 else np.nan
    # F test u_i = 0
    Xp = Xall[:, active]
    bp = np.linalg.lstsq(Xp * np.sqrt(ww)[:, None], yy * np.sqrt(ww), rcond=None)[0]
    rss_p = float(np.sum(ww * (yy - Xp @ bp) ** 2))
    Fu = ((rss_p - rss) / (G - 1)) / (rss / df_r) if G > 1 else np.nan
    pFu = sps.f.sf(Fu, G - 1, df_r) if G > 1 else np.nan
    Ti = np.bincount(gid)
    est = {'names': names, 'b': b, 'V': V, 'colinfo': info, 'df_r': df_test, 'depvar': yv, 'sample': mask,
           'cmd': 'xtreg', 'stat': 't', 'vce': vce, 'model': 'fe',
           'scalars': {'N': n, 'N_g': G, 'df_m': dfm, 'df_r': df_test, 'F': F, 'r2': r2_w, 'r2_w': r2_w,
                       'r2_b': r2_b, 'r2_o': r2_o, 'sigma_u': sigma_u, 'sigma_e': sigma_e, 'rho': rho,
                       'corr': corr_u, 'F_f': Fu, 'g_min': Ti.min(), 'g_max': Ti.max(), 'g_avg': Ti.mean(),
                       'rss': rss, 'rmse': math.sqrt(rss / df_r),
                       'r2_a': 1 - (1 - r2_w) * (n - 1) / df_r},
           'macros': {'cmd': 'xtreg', 'model': 'fe', 'depvar': yv, 'ivar': ivar, 'vce': vce,
                      'clustvar': clv or '', 'title': 'Fixed-effects (within) regression',
                      'vcetype': 'Robust' if vce != 'ols' else ''}}
    if Gc:
        est['scalars']['N_clust'] = Gc

    def replay(ip_):
        ip_.txt('\n')
        xt_header(ip_, 'Fixed-effects (within) regression', ivar, n, G, r2_w, r2_b, r2_o, Ti,
                  f'F({dfm},{df_test})', f'{F:.2f}', 'Prob > F', f'{pF:.4f}',
                  f'corr(u_i, Xb) = {corr_u:.4f}' if corr_u == corr_u else 'corr(u_i, Xb) = .')
        if vce != 'ols':
            msg = f'(Std. err. adjusted for {comma_int(Gc)} clusters in {clv})'
            ip_.txt(f'{msg:>78}\n')
        coef_table(ip_, est, 't', robust='Robust' if vce != 'ols' else None,
                   extra_rows=[('sigma_u', sigma_u, ''), ('sigma_e', sigma_e, ''),
                               ('rho', rho, '(fraction of variance due to u_i)')])
        if vce == 'ols':
            ip_.txt(f'F test that all u_i=0: F({G - 1}, {df_r}) = ')
            ip_.res(f'{Fu:.2f}')
            ip_.txt(f'{"Prob > F = ":>25}')
            ip_.res(f'{pFu:.4f}\n')
    est['replay'] = replay
    est['predict'] = lambda ip_, kind, m_: _xt_predict(ip_, est, kind, u, gid, mask)
    return est


def xt_header(ip, title, ivar, n, G, r2w, r2b, r2o, Ti, tlab, tval, plab, pval, corrline):
    def R(k, v):
        ip.txt(f'{k:<18}= ')
        ip.res(f'{v:>10}\n')
    f4 = lambda x: f'{x:.4f}' if x == x else '.'
    ip.txt(f'{title:<48}')
    R('Number of obs', comma_int(n))
    ip.txt(f'{"Group variable: " + ivar:<48}')
    R('Number of groups', comma_int(G))
    ip.txt('\n')
    ip.txt(f'{"R-squared:":<48}Obs per group:\n')
    ip.txt(f'{"     Within  = " + f4(r2w):<48}{"min":>17} = ')
    ip.res(f'{comma_int(Ti.min()):>10}\n')
    ip.txt(f'{"     Between = " + f4(r2b):<48}{"avg":>17} = ')
    ip.res(f'{Ti.mean():>10.1f}\n')
    ip.txt(f'{"     Overall = " + f4(r2o):<48}{"max":>17} = ')
    ip.res(f'{comma_int(Ti.max()):>10}\n\n')
    ip.txt(f'{"":<48}')
    R(tlab, tval)
    ip.txt(f'{corrline:<48}')
    R(plab, pval)
    ip.txt('\n')


def _xt_predict(ip, est, kind, u, gid, mask):
    xb = _predict_linear(ip, est, 'xb')
    if kind in (None, 'xb', ''):
        return xb
    d = ip.data
    full_u = np.full(d.n, np.nan)
    full_u[np.flatnonzero(mask)] = u[gid]
    if kind == 'u':
        return full_u
    if kind == 'ue':
        return d.cols[est['depvar']] - xb
    if kind == 'e':
        return d.cols[est['depvar']] - xb - full_u
    if kind == 'xbu':
        return xb + full_u
    return _predict_linear(ip, est, kind)


def _corr2(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def _xtreg_be(ip, d, y, yv, cols, mask, ivar):
    gid = _group_index(d, ivar, mask)
    Xall, names, info = assemble(cols, mask, cons=True)
    yy = y[mask]
    G = int(gid.max()) + 1
    ym = _gmeans(yy, gid)
    Xm = _gmeans(Xall, gid)
    active = detect_omitted(Xm, info, cons=True)
    X = Xm[:, active]
    XtXi = np.linalg.pinv(X.T @ X)
    b_act = XtXi @ X.T @ ym
    e = ym - X @ b_act
    K = len(active)
    df_r = G - K
    rss = float(e @ e)
    s2 = rss / df_r
    V_act = s2 * XtXi
    b, V = full_bV(len(names), active, b_act, V_act)
    names = [c.name for c in info]
    slopes = [j for j in active if info[j].kind != 'cons']
    F, dfm, pF = wald_test(b, V, slopes, df_r)
    xb = Xall[:, active] @ b_act
    r2_b = _corr2(X @ b_act, ym)
    r2_o = _corr2(xb, yy)
    r2_w = _corr2(xb - _gmeans(xb, gid)[gid], yy - ym[gid])
    Ti = np.bincount(gid)
    est = {'names': names, 'b': b, 'V': V, 'colinfo': info, 'df_r': df_r, 'depvar': yv, 'sample': mask,
           'cmd': 'xtreg', 'stat': 't', 'vce': 'ols', 'model': 'be',
           'scalars': {'N': len(yy), 'N_g': G, 'df_m': dfm, 'df_r': df_r, 'F': F, 'r2': r2_b,
                       'r2_w': r2_w, 'r2_b': r2_b, 'r2_o': r2_o, 'rmse': math.sqrt(s2)},
           'macros': {'cmd': 'xtreg', 'model': 'be', 'depvar': yv, 'ivar': ivar}}

    def replay(ip_):
        ip_.txt('\n')
        xt_header(ip_, 'Between regression (regression on group means)', ivar, len(yy), G, r2_w, r2_b, r2_o, Ti,
                  f'F({dfm},{df_r})', f'{F:.2f}', 'Prob > F', f'{pF:.4f}',
                  f'sd(u_i + avg(e_i.)) = {g_fmt(math.sqrt(s2), 9)}')
        ip_.txt('\n')
        coef_table(ip_, est, 't')
    est['replay'] = replay
    est['predict'] = lambda ip_, kind, m_: _predict_linear(ip_, est, kind)
    return est


def _xtreg_re(ip, d, y, yv, cols, mask, vce, clv, ivar):
    gid = _group_index(d, ivar, mask)
    Xall, names, info = assemble(cols, mask, cons=True)
    yy = y[mask]
    n = len(yy)
    G = int(gid.max()) + 1
    Ti = np.bincount(gid).astype(float)
    ym = _gmeans(yy, gid)
    Xm = _gmeans(Xall, gid)
    # collinearity on full X
    active = detect_omitted(Xall, info, cons=True)
    # within regression for sigma_e
    Xw = Xall[:, active] - Xm[gid][:, active]
    tv = [k for k, j in enumerate(active) if info[j].kind != 'cons' and np.any(np.abs(Xw[:, k]) > 1e-10)]
    yw = yy - ym[gid]
    Xw_tv = Xw[:, tv]
    if Xw_tv.shape[1]:
        bw = np.linalg.lstsq(Xw_tv, yw, rcond=None)[0]
        ew = yw - Xw_tv @ bw
    else:
        ew = yw
    Kw = len(tv)
    sig_e2 = float(ew @ ew) / (n - G - Kw)
    # between regression
    Xb = Xm[:, active]
    bb = np.linalg.lstsq(Xb, ym, rcond=None)[0]
    eb = ym - Xb @ bb
    Kb = len(active)
    T_bar = G / np.sum(1.0 / Ti)
    sig_u2 = float(eb @ eb) / (G - Kb) - sig_e2 / T_bar
    if sig_u2 < 0:
        sig_u2 = 0.0
    theta = 1 - np.sqrt(sig_e2 / (Ti * sig_u2 + sig_e2))
    ys = yy - theta[gid] * ym[gid]
    Xs = Xall[:, active] - theta[gid][:, None] * Xm[gid][:, active]
    XtXi = np.linalg.pinv(Xs.T @ Xs)
    b_act = XtXi @ Xs.T @ ys
    es = ys - Xs @ b_act
    K = len(active)
    Gc = None
    if vce == 'ols':
        V_act = sig_e2 * XtXi
    else:
        cgid, cname = cluster_ids(d, clv, mask)
        Gc = int(cgid.max()) + 1
        V_act = sandwich(Xs, Xs * es[:, None], XtXi, cgid) * (Gc / (Gc - 1))
    b, V = full_bV(len(names), active, b_act, V_act)
    names = [c.name for c in info]
    slopes = [j for j in active if info[j].kind != 'cons']
    chi2, dfm, pchi = wald_test(b, V, slopes)
    xb = Xall[:, active] @ b_act
    r2_w = _corr2(xb - _gmeans(xb, gid)[gid], yy - ym[gid])
    r2_b = _corr2(_gmeans(xb, gid), ym)
    r2_o = _corr2(xb, yy)
    sigma_u, sigma_e = math.sqrt(sig_u2), math.sqrt(sig_e2)
    rho = sig_u2 / (sig_u2 + sig_e2)
    est = {'names': names, 'b': b, 'V': V, 'colinfo': info, 'df_r': None, 'depvar': yv, 'sample': mask,
           'cmd': 'xtreg', 'stat': 'z', 'vce': vce, 'model': 're',
           'scalars': {'N': n, 'N_g': G, 'df_m': dfm, 'chi2': chi2, 'r2_w': r2_w, 'r2_b': r2_b, 'r2_o': r2_o,
                       'sigma_u': sigma_u, 'sigma_e': sigma_e, 'rho': rho, 'r2': r2_o},
           'macros': {'cmd': 'xtreg', 'model': 're', 'depvar': yv, 'ivar': ivar, 'vce': vce,
                      'title': 'Random-effects GLS regression', 'vcetype': 'Robust' if vce != 'ols' else ''}}
    if Gc:
        est['scalars']['N_clust'] = Gc

    def replay(ip_):
        ip_.txt('\n')
        xt_header(ip_, 'Random-effects GLS regression', ivar, n, G, r2_w, r2_b, r2_o, Ti,
                  f'Wald chi2({dfm})', f'{chi2:.2f}', 'Prob > chi2', f'{pchi:.4f}', 'corr(u_i, X) = 0 (assumed)')
        if vce != 'ols':
            msg = f'(Std. err. adjusted for {comma_int(Gc)} clusters in {clv})'
            ip_.txt(f'{msg:>78}\n')
        coef_table(ip_, est, 'z', robust='Robust' if vce != 'ols' else None,
                   extra_rows=[('sigma_u', sigma_u, ''), ('sigma_e', sigma_e, ''),
                               ('rho', rho, '(fraction of variance due to u_i)')])
    est['replay'] = replay
    est['predict'] = lambda ip_, kind, m_: _predict_linear(ip_, est, kind)
    return est


# ================================================================ reghdfe
@command('reghdfe', 'areg')
def cmd_reghdfe(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    toks = split_top(p.body)
    absorb = p.opts.get('absorb', 1)
    if absorb is None:
        raise StataError('option absorb() required', 198)
    fe_vars = d.expand_varlist([t.replace('i.', '') for t in absorb.split()])
    yv, y, cols, mask, w, wtype = prepare_linear(ip, p, d, toks[1:], toks[0])
    for v in fe_vars:
        mask &= ~np.isnan(d.cols[v]) if not d.isstr(v) else (d.cols[v] != '')
    vce, clv = parse_vce(p.opts)
    if clv:
        cv = d.resolve(clv)
        mask &= ~np.isnan(d.cols[cv])
    # drop singletons iteratively
    nsing = 0
    if cmd == 'reghdfe':
        while True:
            drop = np.zeros(d.n, dtype=bool)
            for v in fe_vars:
                vals = d.cols[v]
                idx = np.flatnonzero(mask)
                u, inv, cnt = np.unique(vals[idx], return_inverse=True, return_counts=True)
                single = cnt[inv] == 1
                drop[idx[single]] = True
            k = int(drop.sum())
            if k == 0:
                break
            nsing += k
            mask &= ~drop
    n = int(mask.sum())
    yy = y[mask]
    Xall, names, info = assemble(cols, mask, cons=True)
    fe_ids = []
    for v in fe_vars:
        _, inv = np.unique(d.cols[v][mask], return_inverse=True)
        fe_ids.append(inv)

    def demean(z):
        z = z.copy()
        for it in range(10000):
            old = z.copy()
            for g in fe_ids:
                m = np.bincount(g, weights=z, minlength=g.max() + 1) / np.bincount(g, minlength=g.max() + 1)
                z = z - m[g]
            if np.max(np.abs(z - old)) < 1e-12 * (1 + np.max(np.abs(z))):
                break
            if len(fe_ids) == 1:
                break
        return z
    yt = demean(yy)
    k_all = Xall.shape[1]
    Xt = np.column_stack([demean(Xall[:, j]) for j in range(k_all - 1)]) if k_all > 1 else np.zeros((n, 0))
    cand = [j for j in range(k_all - 1) if not info[j].base]
    om = mark_collinear(Xt[:, cand], None)
    for j, o in zip(cand, om):
        if o:
            info[j].omitted = True
            info[j].name = _omit_name(info[j].name) if info[j].kind == 'factor' else 'o.' + info[j].name
    active = [j for j in range(k_all - 1) if not info[j].base and not info[j].omitted]
    X = Xt[:, active]
    XtXi = np.linalg.pinv(X.T @ X)
    b_act = XtXi @ X.T @ yt
    e = yt - X @ b_act
    K = len(active)
    # absorbed df
    cgid = None
    Gc = None
    if vce == 'cluster':
        cgid, cname = cluster_ids(d, clv, mask)
        Gc = int(cgid.max()) + 1
    dfa_rows = []
    dfa = 0
    for v, g in zip(fe_vars, fe_ids):
        L = int(g.max()) + 1
        nested = False
        if cgid is not None:
            # FE nested within cluster: each FE level maps to one cluster
            pairs = np.unique(np.column_stack([g, cgid]), axis=0)
            nested = len(pairs) == L
        if nested:
            red = L
        else:
            red = 1 if dfa_rows else 0
        coefs = L - red
        dfa += coefs
        dfa_rows.append((v, L, red, coefs, nested))
    df_r = n - K - dfa - 1
    rss = float(e @ e)
    s2 = rss / df_r
    if vce == 'ols':
        V_act = s2 * XtXi
        df_test = df_r
    elif vce == 'robust':
        V_act = sandwich(X, X * e[:, None], XtXi) * (n / df_r) if False else sandwich(X, X * e[:, None], XtXi) * ((n - 1) / (n - K - dfa - 1))
        df_test = df_r
    else:
        qc = Gc / (Gc - 1) * (n - 1) / (n - K - dfa - 1)
        V_act = sandwich(X, X * e[:, None], XtXi, cgid) * qc
        df_test = Gc - 1
    # constant
    cons = float(yy.mean() - Xall[:, active].mean(axis=0) @ b_act)
    act_full = active + [k_all - 1]
    b = np.zeros(k_all)
    V = np.zeros((k_all, k_all))
    b[active] = b_act
    b[k_all - 1] = cons
    V[np.ix_(active, active)] = V_act
    xm = Xall[:, active].mean(axis=0)
    V[k_all - 1, k_all - 1] = float(xm @ V_act @ xm) if K else s2 / n
    V[active, k_all - 1] = V[k_all - 1, active] = -(V_act @ xm) if K else 0
    names = [c.name for c in info]
    F, dfm, pF = wald_test(b, V, active, df_test)
    tss = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1 - rss / tss
    r2a = 1 - (1 - r2) * (n - 1) / df_r
    tssw = float(yt @ yt)
    r2w = 1 - rss / tssw
    est = {'names': names, 'b': b, 'V': V, 'colinfo': info, 'df_r': df_test, 'depvar': yv, 'sample': mask,
           'cmd': cmd, 'stat': 't', 'vce': vce,
           'scalars': {'N': n, 'df_m': dfm, 'df_r': df_test, 'F': F, 'r2': r2, 'r2_a': r2a, 'r2_within': r2w,
                       'rmse': math.sqrt(s2), 'rss': rss, 'df_a': dfa, 'num_singletons': nsing},
           'macros': {'cmd': cmd, 'depvar': yv, 'absvars': ' '.join(fe_vars), 'vce': vce,
                      'clustvar': clv or '', 'vcetype': 'Robust' if vce != 'ols' else ''}}
    if Gc:
        est['scalars']['N_clust'] = Gc

    def replay(ip_):
        if nsing:
            ip_.txt(f'(dropped {nsing} singleton observations)\n')
        ip_.txt('\n')
        left = ['HDFE Linear regression', f'Absorbing {len(fe_vars)} HDFE group{"s" if len(fe_vars) > 1 else ""}',
                {'ols': '', 'robust': 'Statistics robust to heteroskedasticity',
                 'cluster': 'Statistics robust to heteroskedasticity'}[vce], '', '', '']
        right = [('Number of obs', comma_int(n)), (f'F({dfm:>4},{df_test:>6})', f'{F:.2f}'),
                 ('Prob > F', f'{pF:.4f}'), ('R-squared', f'{r2:.4f}'), ('Adj R-squared', f'{r2a:.4f}'),
                 ('Within R-sq.', f'{r2w:.4f}')]
        if Gc:
            left.append(f'Number of clusters ({clv}) = {comma_int(Gc):>10}')
        else:
            left.append('')
        right.append(('Root MSE', g_fmt(math.sqrt(s2), 9)))
        right_lines(ip_, left, right)
        ip_.txt('\n')
        if Gc:
            msg = f'(Std. err. adjusted for {comma_int(Gc)} clusters in {clv})'
            ip_.txt(f'{msg:>78}\n')
        coef_table(ip_, est, 't', robust='Robust' if vce != 'ols' else None)
        if cmd == 'reghdfe':
            ip_.txt('\nAbsorbed degrees of freedom:\n')
            ip_.txt('-----------------------------------------------------+\n')
            ip_.txt(' Absorbed FE | Categories  - Redundant  = Num. Coefs |\n')
            ip_.txt('-------------+---------------------------------------|\n')
            for v, L, red, cf, nested in dfa_rows:
                ip_.txt(f'{abbrev(v):>12} |')
                ip_.res(f'{L:>11}{red:>12}{cf:>13}')
                ip_.txt(f'    {"*" if nested else " "}|\n')
            ip_.txt('-----------------------------------------------------+\n')
            if any(r[4] for r in dfa_rows):
                ip_.txt('* = FE nested within cluster; treated as redundant for DoF computation\n')
    est['replay'] = replay
    est['predict'] = lambda ip_, kind, m_: _predict_linear(ip_, est, kind)
    _store_rhs(est, toks[1:], cols, d)
    ip.set_estimation(est)
    replay(ip)


# ================================================================ IV
def _parse_iv(body):
    """Split 'y x1 (endog = instr) x2' into parts."""
    k = body.find('(')
    j = body.rfind(')')
    if k < 0 or j < 0:
        raise StataError('equation not found', 198)
    inner = body[k + 1:j]
    if '=' not in inner:
        raise StataError('invalid IV syntax', 198)
    endo, inst = inner.split('=', 1)
    outside = (body[:k] + ' ' + body[j + 1:]).split()
    return outside[0], outside[1:], endo.split(), inst.split()


class IVData:
    pass


def iv_setup(ip, p, d, method_tokens=None):
    yv_tok, exog, endo, inst = _parse_iv(p.body)
    yv = d.resolve(yv_tok)
    y = d.cols[yv].astype(float)
    mask = ip.mark(p) & ~np.isnan(y)
    w, wtype = get_weights(ip, p, d)
    cols_x1, mask = build_design(ip, d, exog, mask)
    cols_y2, mask = build_design(ip, d, endo, mask)
    cols_z, mask = build_design(ip, d, inst, mask)
    vce, clv = parse_vce(p.opts)
    if clv:
        cv = d.resolve(clv)
        mask &= ~np.isnan(d.cols[cv])
    nocons = p.opts.has('noconstant', 4)
    # rebuild with final mask
    cols_x1, _ = build_design(ip, d, exog, mask)
    cols_y2, _ = build_design(ip, d, endo, mask)
    cols_z, _ = build_design(ip, d, inst, mask)
    iv = IVData()
    iv.yv, iv.y = yv, y[mask]
    iv.mask = mask
    iv.vce, iv.clv = vce, clv
    iv.n = int(mask.sum())
    X1, n1, i1 = assemble(cols_x1, mask, cons=not nocons)
    Y2, n2, i2 = assemble(cols_y2, mask, cons=False)
    Z, nz, iz = assemble(cols_z, mask, cons=False)
    # omit base / collinear among exog
    a1 = detect_omitted(X1, i1, cons=not nocons)
    a2 = [j for j in range(Y2.shape[1]) if not i2[j].base]
    az = [j for j in range(Z.shape[1]) if not iz[j].base]
    iv.X1, iv.Y2, iv.Z = X1[:, a1], Y2[:, a2], Z[:, az]
    iv.i1, iv.i2, iv.iz = i1, i2, iz
    iv.a1, iv.a2, iv.az = a1, a2, az
    iv.names_x1 = [i1[j].name for j in range(len(i1))]
    iv.names_y2 = [i2[j].name for j in range(len(i2))]
    iv.names_z = [iz[j].name for j in az]
    iv.exog_names = [i1[j].name for j in a1 if i1[j].kind != 'cons']
    iv.nocons = nocons
    iv.toks = (exog, endo, inst)
    iv.cluster = None
    if vce == 'cluster':
        iv.cluster, iv.clname = cluster_ids(d, clv, mask)
    return iv


def tsls(y, X, Zf):
    ZtZi = np.linalg.pinv(Zf.T @ Zf)
    Pz_X = Zf @ (ZtZi @ (Zf.T @ X))
    A = np.linalg.pinv(Pz_X.T @ X)
    b = A @ (Pz_X.T @ y)
    e = y - X @ b
    return b, e, Pz_X, A


@command('ivregress')
def cmd_ivregress(ip, cmd, rest, by):
    rest = rest.strip()
    method = rest.split()[0]
    p = parse_command(rest[len(method):])
    d = ip.data
    iv = iv_setup(ip, p, d)
    small = p.opts.has('small')
    X = np.column_stack([iv.Y2, iv.X1])
    Zf = np.column_stack([iv.Z, iv.X1])
    n = iv.n
    k = X.shape[1]
    b_act, e, Xh, A = tsls(iv.y, X, Zf)
    rss = float(e @ e)
    if iv.vce == 'ols':
        s2 = rss / (n - k) if small else rss / n
        V_act = s2 * A
    elif iv.vce == 'robust':
        V_act = sandwich(Xh, Xh * e[:, None], A) * (n / (n - k) if small else 1)
    else:
        G = int(iv.cluster.max()) + 1
        qc = (G / (G - 1)) * ((n - 1) / (n - k) if small else 1)
        V_act = sandwich(Xh, Xh * e[:, None], A, iv.cluster) * qc
    # names: endog first then exog (with bases/omitted from exog)
    info = [iv.i2[j] for j in iv.a2] + list(iv.i1)
    names = [c.name for c in info]
    active = list(range(len(iv.a2))) + [len(iv.a2) + j for j in iv.a1]
    b, V = full_bV(len(names), active, b_act, V_act)
    tss = float(np.sum((iv.y - iv.y.mean()) ** 2)) if not iv.nocons else float(iv.y @ iv.y)
    r2 = 1 - rss / tss
    rmse = math.sqrt(rss / (n - k) if small else rss / n)
    slopes = [a for a, c in zip(active, [info[a] for a in active]) if c.kind != 'cons']
    G = int(iv.cluster.max()) + 1 if iv.cluster is not None else None
    if small:
        dft = (G - 1) if G else n - k
        stat, dfm, pv = wald_test(b, V, slopes, dft)
    else:
        dft = None
        stat, dfm, pv = wald_test(b, V, slopes)
    est = {'names': names, 'b': b, 'V': V, 'colinfo': info, 'df_r': dft, 'depvar': iv.yv, 'sample': iv.mask,
           'cmd': 'ivregress', 'stat': 't' if small else 'z', 'vce': iv.vce, 'iv': iv, 'small': small,
           'scalars': {'N': n, 'r2': r2 if r2 > 0 else np.nan, 'rmse': rmse, 'rss': rss, 'df_m': dfm,
                       'chi2' if not small else 'F': stat},
           'macros': {'cmd': 'ivregress', 'depvar': iv.yv, 'instd': ' '.join(iv.names_y2[j] for j in iv.a2),
                      'insts': ' '.join(iv.exog_names + iv.names_z), 'exogr': ' '.join(iv.exog_names),
                      'vce': iv.vce, 'clustvar': iv.clv or '', 'vcetype': 'Robust' if iv.vce != 'ols' else ''}}
    if G:
        est['scalars']['N_clust'] = G
    if not small:
        est['scalars']['df_r'] = np.nan

    def replay(ip_):
        ip_.txt('\n')
        right = [('Number of obs', comma_int(n)),
                 (f'Wald chi2({dfm})' if not small else f'F({dfm}, {dft})', f'{stat:.2f}'),
                 ('Prob > chi2' if not small else 'Prob > F', f'{pv:.4f}'),
                 ('R-squared', f'{r2:.4f}' if r2 > 0 else '.'), ('Root MSE', g_fmt(rmse, 9))]
        right_lines(ip_, ['Instrumental variables 2SLS regression'], right)
        ip_.txt('\n')
        if G:
            msg = f'(Std. err. adjusted for {comma_int(G)} clusters in {iv.clv})'
            ip_.txt(f'{msg:>78}\n')
        coef_table(ip_, est, 't' if small else 'z', robust='Robust' if iv.vce != 'ols' else None)
        ip_.txt(f'Endogenous: {est["macros"]["instd"]}\n')
        ip_.txt(f'Exogenous:  {est["macros"]["insts"]}\n')
    est['replay'] = replay
    est['rhs_tokens'] = iv.toks[1] + iv.toks[0]
    est['levels'] = None
    est['predict'] = lambda ip_, kind, m_: _predict_linear(ip_, est, kind)
    ip.set_estimation(est)
    replay(ip)


def _partial(M, X1):
    if X1.shape[1] == 0:
        return M
    B = np.linalg.lstsq(X1, M, rcond=None)[0]
    return M - X1 @ B


def first_stage_stats(iv, vce, small=True):
    """Per-endogenous first-stage statistics."""
    out = []
    Zf = np.column_stack([iv.Z, iv.X1])
    n = iv.n
    kz = Zf.shape[1]
    L = iv.Z.shape[1]
    Zp = _partial(iv.Z, iv.X1)
    for j in range(iv.Y2.shape[1]):
        y2 = iv.Y2[:, j]
        bfull = np.linalg.lstsq(Zf, y2, rcond=None)[0]
        e = y2 - Zf @ bfull
        rss = float(e @ e)
        tss = float(np.sum((y2 - y2.mean()) ** 2))
        r2 = 1 - rss / tss
        r2a = 1 - (1 - r2) * (n - 1) / (n - kz)
        y2p = _partial(y2[:, None], iv.X1)[:, 0]
        pr2 = 1 - rss / float(y2p @ y2p)
        if vce == 'ols':
            F = (pr2 / L) / ((1 - pr2) / (n - kz))
            df2 = n - kz
        else:
            ZtZi = np.linalg.pinv(Zf.T @ Zf)
            groups = iv.cluster if vce == 'cluster' else None
            Vr = sandwich(Zf, Zf * e[:, None], ZtZi, groups)
            if vce == 'cluster':
                G = int(groups.max()) + 1
                Vr *= G / (G - 1) * (n - 1) / (n - kz)
                df2 = G - 1
            else:
                Vr *= n / (n - kz)
                df2 = n - kz
            idx = list(range(L))
            bz = bfull[idx]
            F = float(bz @ np.linalg.pinv(Vr[np.ix_(idx, idx)]) @ bz) / L
        out.append({'r2': r2, 'r2a': r2a, 'pr2': pr2, 'F': F, 'df1': L, 'df2': df2,
                    'p': sps.f.sf(F, L, df2), 'b': bfull, 'e': e})
    return out


_SY_SIZE = {1: (16.38, 8.96, 6.66, 5.53), 2: (19.93, 11.59, 8.75, 7.25), 3: (22.30, 12.83, 9.54, 7.80),
            4: (24.58, 13.96, 10.26, 8.31), 5: (26.87, 15.09, 10.98, 8.84)}
_SY_LIML = {1: (16.38, 8.96, 6.66, 5.53), 2: (8.68, 5.33, 4.42, 3.92), 3: (6.46, 4.36, 3.69, 3.32),
            4: (5.44, 3.87, 3.30, 2.98), 5: (4.84, 3.56, 3.05, 2.77)}
_SY_BIAS = {3: (13.91, 9.08, 6.46, 5.39), 4: (16.85, 10.27, 6.71, 5.34), 5: (18.37, 10.83, 6.77, 5.25)}


def estat_firststage(ip):
    est = ip.est
    iv = est['iv']
    fs = first_stage_stats(iv, iv.vce)
    L = iv.Z.shape[1]
    ip.txt('\n  First-stage regression summary statistics\n')
    ip.txt('  ' + '-' * 76 + '\n')
    rob = iv.vce != 'ols'
    ip.txt(f'  {"":>12} |{"Adjusted":>20}{"Partial":>13}{"Robust" if rob else "":>12}\n')
    ip.txt(f'  {"Variable":>12} |{"R-sq.":>10}{"R-sq.":>10}{"R-sq.":>13}{"F(" + str(L) + "," + str(fs[0]["df2"]) + ")":>14}{"Prob > F":>11}\n')
    ip.txt('  ' + '-' * 13 + '+' + '-' * 62 + '\n')
    for j, s in zip(iv.a2, fs):
        ip.txt(f'  {abbrev(iv.names_y2[j]):>12} |')
        ip.res(f'{s["r2"]:>10.4f}{s["r2a"]:>10.4f}{s["pr2"]:>13.4f}{s["F"]:>14.4f}{s["p"]:>11.4f}\n')
    ip.txt('  ' + '-' * 76 + '\n')
    if rob:
        ip.txt('\n  (no minimum eigenvalue statistic reported with robust or cluster VCE)\n')
    else:
        mineig = fs[0]['F'] if len(fs) == 1 else min(s['F'] for s in fs)
        ip.txt('\n  Minimum eigenvalue statistic = ')
        ip.res(f'{mineig:.4f}\n\n')
        ip.txt('  Critical Values                      # of endogenous regressors:    ')
        ip.res(f'{len(fs)}\n')
        ip.txt('  H0: Instruments are weak             # of excluded instruments:     ')
        ip.res(f'{L}\n')
        ip.txt('  ' + '-' * 69 + '\n')
        ip.txt('                                     |    5%     10%     20%     30%\n')
        ip.txt('  2SLS relative bias                 |')
        if L in _SY_BIAS:
            ip.res(''.join(f'{v:>8.2f}' for v in _SY_BIAS[L]) + '\n')
        else:
            ip.txt('         (not available)\n')
        ip.txt('  -----------------------------------+' + '-' * 33 + '\n')
        ip.txt('                                     |   10%     15%     20%     25%\n')
        ip.txt('  2SLS size of nominal 5% Wald test  |')
        ip.res(''.join(f'{v:>8.2f}' for v in _SY_SIZE.get(L, (np.nan,) * 4)) + '\n')
        ip.txt('  LIML size of nominal 5% Wald test  |')
        ip.res(''.join(f'{v:>8.2f}' for v in _SY_LIML.get(L, (np.nan,) * 4)) + '\n')
        ip.txt('  ' + '-' * 69 + '\n')
    ip.set_r(scalars={'mineig': fs[0]['F']} if fs else {},
             matrices={'singleresults': E.Matrix(np.array([[s['r2'], s['r2a'], s['pr2'], s['F'], s['df1'],
                                                             s['df2'], s['p']] for s in fs]))})


def estat_endogenous(ip):
    est = ip.est
    iv = est['iv']
    n = iv.n
    fs = first_stage_stats(iv, 'ols')
    V2 = np.column_stack([s['e'] for s in fs])
    ke = V2.shape[1]
    X = np.column_stack([iv.Y2, iv.X1])
    if iv.vce == 'ols':
        Xr = X
        br = np.linalg.lstsq(Xr, iv.y, rcond=None)[0]
        ssr_r = float(np.sum((iv.y - Xr @ br) ** 2))
        Xu = np.column_stack([X, V2])
        bu = np.linalg.lstsq(Xu, iv.y, rcond=None)[0]
        ssr_u = float(np.sum((iv.y - Xu @ bu) ** 2))
        # Durbin uses the IV-based statistic
        Zf = np.column_stack([iv.Z, iv.X1])
        b2, e2, Xh, A = tsls(iv.y, X, Zf)
        Pz = lambda v: Zf @ np.linalg.lstsq(Zf, v, rcond=None)[0]
        e_ols = iv.y - X @ br
        num = float(e2 @ Pz(e2)) - float(e_ols @ Pz(e_ols))
        D = n * (ssr_r - ssr_u) / ssr_r
        WH = ((ssr_r - ssr_u) / ke) / (ssr_u / (n - Xu.shape[1]))
        df2 = n - Xu.shape[1]
        ip.txt('\n  Tests of endogeneity\n  H0: Variables are exogenous\n\n')
        ip.txt(f'  Durbin (score) chi2({ke})          =  ')
        ip.res(f'{D:.5g}'.rjust(8))
        ip.txt('  (p = ')
        ip.res(f'{sps.chi2.sf(D, ke):.4f}')
        ip.txt(')\n')
        ip.txt(f'  Wu-Hausman F({ke},{df2})'.ljust(36) + '=  ')
        ip.res(f'{WH:.5g}'.rjust(8))
        ip.txt('  (p = ')
        ip.res(f'{sps.f.sf(WH, ke, df2):.4f}')
        ip.txt(')\n')
        ip.set_r(scalars={'durbin': D, 'p_durbin': sps.chi2.sf(D, ke), 'wu': WH, 'p_wu': sps.f.sf(WH, ke, df2),
                          'df': ke, 'wudf_r': df2})
    else:
        Xu = np.column_stack([X, V2])
        XtXi = np.linalg.pinv(Xu.T @ Xu)
        bu = XtXi @ Xu.T @ iv.y
        eu = iv.y - Xu @ bu
        groups = iv.cluster if iv.vce == 'cluster' else None
        Vr = sandwich(Xu, Xu * eu[:, None], XtXi, groups)
        k = Xu.shape[1]
        if groups is not None:
            G = int(groups.max()) + 1
            Vr *= G / (G - 1) * (n - 1) / (n - k)
            df2 = G - 1
        else:
            Vr *= n / (n - k)
            df2 = n - k
        idx = list(range(k - ke, k))
        bb = bu[idx]
        Fr = float(bb @ np.linalg.pinv(Vr[np.ix_(idx, idx)]) @ bb) / ke
        # robust score (Wooldridge 1995)
        Zf = np.column_stack([iv.Z, iv.X1])
        b2, e2, Xh, A = tsls(iv.y, X, Zf)
        Yhat = Xh[:, :iv.Y2.shape[1]]
        R = _partial(V2, np.column_stack([iv.X1, Yhat]))
        U = R * e2[:, None]
        if groups is not None:
            Gs = int(groups.max()) + 1
            S = np.zeros((Gs, U.shape[1]))
            np.add.at(S, groups, U)
            U = S
        ones = np.ones(U.shape[0])
        coef = np.linalg.lstsq(U, ones, rcond=None)[0]
        ssr = float(np.sum((ones - U @ coef) ** 2))
        score = U.shape[0] - ssr
        ip.txt('\n  Tests of endogeneity\n  H0: Variables are exogenous\n\n')
        ip.txt(f'  Robust score chi2({ke})'.ljust(36) + '=  ')
        ip.res(f'{score:.5g}'.rjust(8))
        ip.txt('  (p = ')
        ip.res(f'{sps.chi2.sf(score, ke):.4f}')
        ip.txt(')\n')
        ip.txt(f'  Robust regression F({ke},{df2})'.ljust(36) + '=  ')
        ip.res(f'{Fr:.5g}'.rjust(8))
        ip.txt('  (p = ')
        ip.res(f'{sps.f.sf(Fr, ke, df2):.4f}')
        ip.txt(')\n')
        if groups is not None:
            ip.txt(f'  (Adjusted for {G} clusters in {iv.clv})\n')
        ip.set_r(scalars={'score': score, 'p_score': sps.chi2.sf(score, ke), 'regF': Fr,
                          'p_regF': sps.f.sf(Fr, ke, df2), 'df': ke, 'regFdf_n': ke, 'regFdf_r': df2})


# ---------------------------------------------------------------- ivreg2
@command('ivreg2', 'ivreg')
def cmd_ivreg2(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    iv = iv_setup(ip, p, d)
    small = p.opts.has('small')
    n = iv.n
    X = np.column_stack([iv.Y2, iv.X1])
    Zf = np.column_stack([iv.Z, iv.X1])
    k = X.shape[1]
    L1 = iv.Z.shape[1]
    kz = Zf.shape[1]
    b_act, e, Xh, A = tsls(iv.y, X, Zf)
    rss = float(e @ e)
    groups = iv.cluster
    G = int(groups.max()) + 1 if groups is not None else None
    if iv.vce == 'ols':
        s2 = rss / (n - k) if small else rss / n
        V_act = s2 * A
    else:
        V_act = sandwich(Xh, Xh * e[:, None], A, groups)
        if small:
            V_act *= (G / (G - 1) * (n - 1) / (n - k)) if G else n / (n - k)
    info = [iv.i2[j] for j in iv.a2] + list(iv.i1)
    names = [c.name for c in info]
    active = list(range(len(iv.a2))) + [len(iv.a2) + j for j in iv.a1]
    b, V = full_bV(len(names), active, b_act, V_act)
    tss = float(np.sum((iv.y - iv.y.mean()) ** 2))
    utss = float(iv.y @ iv.y)
    r2c = 1 - rss / tss
    r2u = 1 - rss / utss
    rmse = math.sqrt(rss / (n - k) if small else rss / n)
    slopes = [a for a in active if info[a].kind != 'cons']
    df2 = (G - 1) if G else (n - k)
    Fw, dfm, pF = wald_test(b, V, slopes, df2)
    if not small:
        # ivreg2 reports F with large-sample V rescaled
        chi2, _, _ = wald_test(b, V, slopes)
        Fw = chi2 / dfm * ((n - k) / n if not G else (G - 1) / G * (n - k) / (n - 1)) if dfm else np.nan
        pF = sps.f.sf(Fw, dfm, df2)
    # identification statistics
    Zp = _partial(iv.Z, iv.X1)
    Y2p = _partial(iv.Y2, iv.X1)
    # Cragg-Donald (min eigenvalue)
    try:
        SZZ = Zp.T @ Zp
        Pz_Y = Zp @ np.linalg.lstsq(Zp, Y2p, rcond=None)[0]
        Mz_Y = Y2p - Pz_Y
        Sv = Mz_Y.T @ Mz_Y / (n - kz)
        Sp = Pz_Y.T @ Pz_Y
        Lc = np.linalg.cholesky(np.linalg.inv(Sv))
        cd_mat = Lc.T @ Sp @ Lc
        cd_eig = float(np.min(np.linalg.eigvalsh(cd_mat)))
        cdf = cd_eig / L1
        # Anderson canonical corr LM
        Syy = Y2p.T @ Y2p
        can = np.linalg.eigvals(np.linalg.solve(Syy, Sp)).real
        anderson = n * float(np.min(can))
    except np.linalg.LinAlgError:
        cdf = anderson = np.nan
    # Kleibergen-Paap (single endogenous regressor: robust tests on pi)
    kp_lm = kp_f = np.nan
    if iv.vce != 'ols' and iv.Y2.shape[1] >= 1:
        y2 = Y2p[:, 0]
        pi = np.linalg.lstsq(Zp, y2, rcond=None)[0]
        v = y2 - Zp @ pi
        ZZi = np.linalg.pinv(Zp.T @ Zp)
        Vpi = sandwich(Zp, Zp * v[:, None], ZZi, groups)
        wald = float(pi @ np.linalg.pinv(Vpi) @ pi)
        if G:
            kp_f = wald / L1 * (n - kz) / n * (G - 1) / G
        else:
            kp_f = wald / L1 * (n - kz) / n
        # LM: scores under H0 (pi=0) use y2 itself
        S0 = Zp * y2[:, None]
        if groups is not None:
            SS = np.zeros((G, S0.shape[1]))
            np.add.at(SS, groups, S0)
            S0 = SS
        g = (Zp.T @ y2)
        kp_lm = float(g @ np.linalg.pinv(S0.T @ S0) @ g)
    # overidentification
    if L1 > iv.Y2.shape[1]:
        if iv.vce == 'ols':
            ePz = Zf @ np.linalg.lstsq(Zf, e, rcond=None)[0]
            J = float(e @ ePz) / (rss / n)
            jname = 'Sargan statistic (overidentification test of all instruments):'
        else:
            Ze = Zf * e[:, None]
            if groups is not None:
                SS = np.zeros((G, Ze.shape[1]))
                np.add.at(SS, groups, Ze)
                Ze = SS
            S = Ze.T @ Ze / n
            Wm = np.linalg.pinv(S)
            # efficient GMM
            ZX = Zf.T @ X / n
            Zy = Zf.T @ iv.y / n
            bg = np.linalg.pinv(ZX.T @ Wm @ ZX) @ ZX.T @ Wm @ Zy
            eg = iv.y - X @ bg
            gbar = Zf.T @ eg / n
            J = float(n * gbar @ Wm @ gbar)
            jname = 'Hansen J statistic (overidentification test of all instruments):'
        Jdf = L1 - iv.Y2.shape[1]
        Jp = sps.chi2.sf(J, Jdf)
    else:
        J, Jdf, Jp = 0.0, 0, np.nan
        jname = ('Hansen J statistic' if iv.vce != 'ols' else 'Sargan statistic') + ' (overidentification test of all instruments):'
    est = {'names': names, 'b': b, 'V': V, 'colinfo': info, 'df_r': df2 if small else None, 'depvar': iv.yv,
           'sample': iv.mask, 'cmd': 'ivreg2', 'stat': 't' if small else 'z', 'vce': iv.vce, 'iv': iv,
           'scalars': {'N': n, 'r2': r2c, 'r2c': r2c, 'r2u': r2u, 'rmse': rmse, 'rss': rss, 'F': Fw, 'df_m': dfm,
                       'cdf': cdf, 'widstat': kp_f if iv.vce != 'ols' else cdf, 'idstat': kp_lm if iv.vce != 'ols' else anderson,
                       'j': J, 'jdf': Jdf, 'jp': Jp, 'N_clust': G if G else np.nan, 'df_r': df2},
           'macros': {'cmd': 'ivreg2', 'depvar': iv.yv, 'instd': ' '.join(iv.names_y2[j] for j in iv.a2),
                      'inexog': ' '.join(iv.exog_names), 'exexog': ' '.join(iv.names_z),
                      'vce': iv.vce, 'clustvar': iv.clv or '', 'vcetype': 'Robust' if iv.vce != 'ols' else ''}}

    def replay(ip_):
        if p.opts.has('first') or p.opts.has('ffirst'):
            _ivreg2_first(ip_, iv, small)
        ip_.txt('\nIV (2SLS) estimation\n--------------------\n\n')
        ip_.txt('Estimates efficient for homoskedasticity only\n')
        if iv.vce == 'robust':
            ip_.txt('Statistics robust to heteroskedasticity\n')
        elif iv.vce == 'cluster':
            ip_.txt(f'Statistics robust to heteroskedasticity and clustering on {iv.clv}\n')
        else:
            ip_.txt('Statistics consistent for homoskedasticity only\n')
        ip_.txt('\n')
        if G:
            ip_.txt(f'Number of clusters ({iv.clv}) = {G:>11}{"":>15}Number of obs = {n:>8}\n')
        else:
            ip_.txt(f'{"":>54}Number of obs = {n:>8}\n')
        ip_.txt(f'{"":>54}F({dfm:>3},{df2:>6}) = {Fw:>8.2f}\n')
        ip_.txt(f'{"":>54}Prob > F      = {pF:>8.4f}\n')
        ip_.txt(f'Total (centered) SS     = {g_fmt(tss, 12):>12}{"":>17}Centered R2   = {r2c:>8.4f}\n')
        ip_.txt(f'Total (uncentered) SS   = {g_fmt(utss, 12):>12}{"":>17}Uncentered R2 = {r2u:>8.4f}\n')
        ip_.txt(f'Residual SS             = {g_fmt(rss, 12):>12}{"":>17}Root MSE      = {g_fmt(rmse, 8):>8}\n\n')
        coef_table(ip_, est, 't' if small else 'z', robust='Robust' if iv.vce != 'ols' else None)
        if iv.vce != 'ols':
            ip_.txt('Underidentification test (Kleibergen-Paap rk LM statistic):'.ljust(66))
            ip_.res(f'{kp_lm:>12.3f}\n')
        else:
            ip_.txt('Underidentification test (Anderson canon. corr. LM statistic):'.ljust(66))
            ip_.res(f'{anderson:>12.3f}\n')
        idv = kp_lm if iv.vce != 'ols' else anderson
        ip_.txt(f'{"Chi-sq(" + str(L1 - iv.Y2.shape[1] + 1) + ") P-val =":>63}')
        ip_.res(f'{sps.chi2.sf(idv, L1 - iv.Y2.shape[1] + 1):>15.4f}\n')
        ip_.txt('-' * 78 + '\n')
        ip_.txt('Weak identification test (Cragg-Donald Wald F statistic):'.ljust(66))
        ip_.res(f'{cdf:>12.3f}\n')
        if iv.vce != 'ols':
            ip_.txt('                         (Kleibergen-Paap rk Wald F statistic):'.ljust(66))
            ip_.res(f'{kp_f:>12.3f}\n')
        sy = _SY_SIZE.get(L1) if iv.Y2.shape[1] == 1 else None
        if sy:
            labs = ['10%', '15%', '20%', '25%']
            for kk, (lb, v) in enumerate(zip(labs, sy)):
                pre = 'Stock-Yogo weak ID test critical values: ' if kk == 0 else ' ' * 41
                ip_.txt(f'{pre}{lb} maximal IV size'.ljust(66))
                ip_.res(f'{v:>12.2f}\n')
            ip_.txt('Source: Stock-Yogo (2005).  Reproduced by permission.\n')
            if iv.vce != 'ols':
                ip_.txt('NB: Critical values are for Cragg-Donald F statistic and i.i.d. errors.\n')
        ip_.txt('-' * 78 + '\n')
        ip_.txt(jname.ljust(66))
        ip_.res(f'{J:>12.3f}\n')
        if Jdf:
            ip_.txt(f'{"Chi-sq(" + str(Jdf) + ") P-val =":>63}')
            ip_.res(f'{Jp:>15.4f}\n')
        else:
            ip_.txt(f'{"(equation exactly identified)":>78}\n')
        for opt in ('endog', 'orthog'):
            if p.opts.has(opt):
                _ivreg2_ctest(ip_, iv, p.opts.get(opt), opt, b_act, e)
        ip_.txt('-' * 78 + '\n')
        ip_.txt(f'Instrumented:         {est["macros"]["instd"]}\n')
        ip_.txt(f'Included instruments: {est["macros"]["inexog"]}\n')
        ip_.txt(f'Excluded instruments: {est["macros"]["exexog"]}\n')
        ip_.txt('-' * 78 + '\n')
    est['replay'] = replay
    est['rhs_tokens'] = iv.toks[1] + iv.toks[0]
    est['levels'] = None
    est['predict'] = lambda ip_, kind, m_: _predict_linear(ip_, est, kind)
    ip.set_estimation(est)
    replay(ip)


def _ivreg2_first(ip, iv, small):
    fs = first_stage_stats(iv, iv.vce)
    Zf = np.column_stack([iv.Z, iv.X1])
    names_z = iv.names_z + [iv.i1[j].name for j in iv.a1]
    for j, s in zip(iv.a2, fs):
        ip.txt(f'\nFirst-stage regressions\n-----------------------\n\nFirst-stage regression of {iv.names_y2[j]}:\n\n')
        n = iv.n
        ZtZi = np.linalg.pinv(Zf.T @ Zf)
        groups = iv.cluster if iv.vce == 'cluster' else None
        if iv.vce == 'ols':
            Vz = float(s['e'] @ s['e']) / (n - Zf.shape[1]) * ZtZi
        else:
            Vz = sandwich(Zf, Zf * s['e'][:, None], ZtZi, groups)
        est = {'names': names_z, 'b': s['b'], 'V': Vz, 'colinfo': None, 'depvar': iv.names_y2[j]}
        coef_table(ip, est, 'z')
        ip.txt(f'F test of excluded instruments:\n  F({s["df1"]},{s["df2"]}) = ')
        ip.res(f'{s["F"]:8.2f}\n')
        ip.txt('  Prob > F      = ')
        ip.res(f'{s["p"]:8.4f}\n')
    ip.txt('\nSummary results for first-stage regressions\n-------------------------------------------\n\n')
    ip.txt('Variable     | F(' + f'{fs[0]["df1"]},{fs[0]["df2"]}' + ')  P-val | Partial R2\n')
    for j, s in zip(iv.a2, fs):
        ip.txt(f'{abbrev(iv.names_y2[j]):<12} |')
        ip.res(f'{s["F"]:>10.2f}{s["p"]:>8.4f} |{s["pr2"]:>10.4f}\n')


def _ivreg2_ctest(ip, iv, varlist, kind, b, e):
    ip.txt('-' * 78 + '\n')
    if kind == 'endog':
        ip.txt(f'Endogeneity test of endogenous regressors:'.ljust(66))
    else:
        ip.txt(f'-orthog- option:\nC statistic (exogeneity/orthogonality of specified instruments):'.ljust(66))
    # difference-in-Sargan with homoskedastic approximation
    n = iv.n
    X = np.column_stack([iv.Y2, iv.X1])
    Zf = np.column_stack([iv.Z, iv.X1])
    if kind == 'endog':
        Zbig = np.column_stack([Zf, iv.Y2])
    else:
        Zbig = Zf
    def J_of(Zm, Xm):
        bb, ee, _, _ = tsls(iv.y, Xm, Zm)
        Pe = Zm @ np.linalg.lstsq(Zm, ee, rcond=None)[0]
        return float(ee @ Pe) / (float(ee @ ee) / n)
    Ju = J_of(Zf, X)
    Jr = J_of(Zbig, X)
    C = max(Jr - Ju, 0.0)
    ip.res(f'{C:>12.3f}\n')
    ip.txt(f'{"Chi-sq(1) P-val =":>63}')
    ip.res(f'{sps.chi2.sf(C, 1):>15.4f}\n')
    ip.txt(f'Regressors tested:    {varlist}\n' if kind == 'endog' else f'Instruments tested:   {varlist}\n')


@command('weakivtest')
def cmd_weakivtest(ip, cmd, rest, by):
    est = ip.est
    if est is None or 'iv' not in est:
        raise StataError('weakivtest requires ivreg2 or ivregress estimates', 301)
    iv = est['iv']
    if iv.Y2.shape[1] != 1:
        raise StataError('weakivtest supports one endogenous regressor', 198)
    n = iv.n
    Zp = _partial(iv.Z, iv.X1)
    y2 = _partial(iv.Y2, iv.X1)[:, 0]
    yp = _partial(iv.y[:, None], iv.X1)[:, 0]
    L = Zp.shape[1]
    # normalise instruments: Q_zz = Z'Z/n
    Qzz = Zp.T @ Zp / n
    pi = np.linalg.lstsq(Zp, y2, rcond=None)[0]
    v2 = y2 - Zp @ pi
    gam = np.linalg.lstsq(Zp, yp, rcond=None)[0]
    v1 = yp - Zp @ gam
    groups = iv.cluster if iv.vce == 'cluster' else None
    Qi = np.linalg.inv(Qzz)

    def avar(a, b_):
        S1 = Zp * a[:, None]
        S2 = Zp * b_[:, None]
        if groups is not None:
            G = int(groups.max()) + 1
            T1 = np.zeros((G, L))
            T2 = np.zeros((G, L))
            np.add.at(T1, groups, S1)
            np.add.at(T2, groups, S2)
            S1, S2 = T1, T2
        return Qi @ (S1.T @ S2 / n) @ Qi
    W2 = avar(v2, v2)
    Feff = float(n * pi @ Qzz @ pi / np.trace(W2 @ Qzz))
    # critical values via Patnaik approximation (simplified, K_eff = L for one instrument)
    cv = {5: 37.418, 10: 23.109, 20: 15.062, 30: 12.039} if L == 1 else None
    ip.txt('\nMontiel-Pflueger robust weak instrument test\n')
    ip.txt('-' * 46 + '\n')
    ip.txt(f'Effective F statistic:{Feff:>24.3f}\n')
    ip.txt(f'Confidence level alpha:{"5%":>23}\n')
    ip.txt('-' * 46 + '\n\n')
    ip.txt('-' * 46 + '\n')
    ip.txt('Critical Values           TSLS        LIML\n')
    ip.txt('-' * 46 + '\n')
    if cv:
        for tau, val in cv.items():
            ip.txt(f'% of Worst Case Bias\ntau={tau}%{val:>25.3f}{val:>12.3f}\n')
    else:
        ip.txt('(critical values for more than one instrument not computed in web emulator)\n')
    ip.txt('-' * 46 + '\n')
    ip.set_r(scalars={'F_eff': Feff, 'c_TSLS_10': cv[10] if cv else np.nan})


# ================================================================ logit / probit
@command('logit', 'probit', 'logistic')
def cmd_logit(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    toks = split_top(p.body)
    yv, y, cols, mask, w, wtype = prepare_linear(ip, p, d, toks[1:], toks[0])
    yy = (y[mask] != 0).astype(float)
    Xall, names, info = assemble(cols, mask, cons=not p.opts.has('noconstant', 4))
    active = detect_omitted(Xall, info, cons=True)
    X = Xall[:, active]
    n = len(yy)
    probit = cmd == 'probit'

    def ll_grad_hess(bb):
        xb = X @ bb
        if probit:
            Pp = sps.norm.cdf(xb)
            Pp = np.clip(Pp, 1e-300, 1 - 1e-16)
            ll = np.sum(yy * np.log(Pp) + (1 - yy) * np.log(1 - Pp))
            phi = sps.norm.pdf(xb)
            lam = np.where(yy == 1, phi / Pp, -phi / (1 - Pp))
            g = X.T @ lam
            hw = lam * (lam + xb)
            H = -(X.T @ (X * hw[:, None]))
        else:
            Pp = 1 / (1 + np.exp(-xb))
            ll = np.sum(yy * xb - np.logaddexp(0, xb))
            g = X.T @ (yy - Pp)
            H = -(X.T @ (X * (Pp * (1 - Pp))[:, None]))
        return ll, g, H
    pbar = yy.mean()
    ll0 = n * (pbar * math.log(pbar) + (1 - pbar) * math.log(1 - pbar)) if 0 < pbar < 1 else 0.0
    bb = np.zeros(X.shape[1])
    if not p.opts.has('noconstant', 4):
        ci = [k for k, j in enumerate(active) if info[j].kind == 'cons']
        if ci:
            bb[ci[0]] = math.log(pbar / (1 - pbar)) if not probit else sps.norm.ppf(pbar)
    log_lines = [f'Iteration 0:  Log likelihood = {ll0:>10.0f}'.replace(f'{ll0:>10.0f}', g_fmt(ll0, 10))]
    log_lines = [ll0]
    ll = ll0
    for it in range(100):
        ll_new, g, H = ll_grad_hess(bb)
        step = np.linalg.solve(H, g)
        bnew = bb - step
        ll_try, _, _ = ll_grad_hess(bnew)
        s = 1.0
        while ll_try < ll_new - 1e-10 and s > 1e-6:
            s /= 2
            bnew = bb - s * step
            ll_try, _, _ = ll_grad_hess(bnew)
        bb = bnew
        log_lines.append(ll_try)
        if abs(ll_try - ll_new) < 1e-10 and np.max(np.abs(step)) < 1e-8:
            break
    ll, g, H = ll_grad_hess(bb)
    Vact = np.linalg.inv(-H)
    vce, clv = parse_vce(p.opts)
    if vce in ('robust', 'cluster'):
        xb = X @ bb
        if probit:
            Pp = sps.norm.cdf(xb)
            phi = sps.norm.pdf(xb)
            sc = np.where(yy == 1, phi / Pp, -phi / (1 - Pp))
        else:
            sc = yy - 1 / (1 + np.exp(-xb))
        S = X * sc[:, None]
        if vce == 'cluster':
            gid, _ = cluster_ids(d, clv, mask)
            G = int(gid.max()) + 1
            Vact = sandwich(X, S, Vact, gid) * G / (G - 1)
        else:
            Vact = sandwich(X, S, Vact) * n / (n - 1)
    b, V = full_bV(len(names), active, bb, Vact)
    names = [c.name for c in info]
    slopes = [j for j in active if info[j].kind != 'cons']
    lr = 2 * (ll - ll0)
    k = len(slopes)
    r2p = 1 - ll / ll0 if ll0 != 0 else np.nan
    if vce != 'ols':
        stat, _, pv = wald_test(b, V, slopes)
        statlab = f'Wald chi2({k})'
    else:
        stat, pv = lr, sps.chi2.sf(lr, k)
        statlab = f'LR chi2({k})'
    title = 'Logistic regression' if not probit else 'Probit regression'
    est = {'names': names, 'b': b, 'V': V, 'colinfo': info, 'df_r': None, 'depvar': yv, 'sample': mask,
           'cmd': cmd if cmd != 'logistic' else 'logit', 'stat': 'z', 'vce': vce,
           'scalars': {'N': n, 'll': ll, 'll_0': ll0, 'chi2': stat, 'r2_p': r2p, 'df_m': k, 'p': pv},
           'macros': {'cmd': cmd, 'depvar': yv, 'title': title, 'vce': vce}}
    nolog = p.opts.has('nolog') or ip.quiet

    def replay(ip_, show_iter=False):
        if show_iter and not nolog:
            ip_.txt('\n')
            for i, l in enumerate(log_lines):
                ip_.txt(f'Iteration {i}:  Log likelihood = ')
                ip_.res(f'{g_fmt(l, 10)}\n')
        ip_.txt('\n')
        right_lines(ip_, [title, '', '', f'Log likelihood = {g_fmt(ll, 10)}'],
                    [('Number of obs', comma_int(n)), (statlab, f'{stat:.2f}'),
                     ('Prob > chi2', f'{pv:.4f}'), ('Pseudo R2', f'{r2p:.4f}')])
        ip_.txt('\n')
        coef_table(ip_, est, 'z', robust='Robust' if vce != 'ols' else None)

    def predict(ip_, kind, m_):
        xb = _predict_linear(ip_, est, 'xb')
        if kind in (None, 'pr', 'p', ''):
            return sps.norm.cdf(xb) if probit else 1 / (1 + np.exp(-xb))
        if kind == 'xb':
            return xb
        raise StataError(f'predict option {kind} not supported', 198)
    est['replay'] = lambda ip_: replay(ip_, False)
    est['predict'] = predict
    _store_rhs(est, toks[1:], cols, d)
    ip.set_estimation(est)
    replay(ip, True)


# ================================================================ test / testparm / lincom
def _parse_lincomb(ip, s, names):
    """Parse 'a + b - 2*c' into a coefficient vector R."""
    s = s.replace(' ', '')
    terms = re.findall(r'([+-]?)([^+-]+)', s)
    R = np.zeros(len(names))
    for sign, t in terms:
        coef = 1.0
        if '*' in t:
            a, bnm = t.split('*', 1)
            try:
                coef = float(a)
                t = bnm
            except ValueError:
                coef = float(bnm)
                t = a
        t = t.strip()
        t = re.sub(r'^_b\[(.*)\]$', r'\1', t)
        key = ip.match_coef(t)
        if key is None:
            raise StataError(f'{t} not found', 111)
        R[names.index(key)] += (-coef if sign == '-' else coef)
    return R


@command('test')
def cmd_test(ip, cmd, rest, by):
    est = ip.est
    if est is None:
        raise StataError('last estimates not found', 301)
    names = est['names']
    s = rest.strip()
    rows = []
    rhs = []
    labels = []
    if '(' in s and s.startswith('('):
        eqs = re.findall(r'\(([^()]*)\)', s)
    elif '=' in s:
        eqs = [s]
    else:
        eqs = None
    if eqs:
        for eq in eqs:
            if '=' in eq:
                l, r = eq.split('=', 1)
                R = _parse_lincomb(ip, l, names)
                try:
                    q = float(r)
                except ValueError:
                    R -= _parse_lincomb(ip, r, names)
                    q = 0.0
            else:
                R = _parse_lincomb(ip, eq, names)
                q = 0.0
            rows.append(R)
            rhs.append(q)
            labels.append(eq.strip() if '=' in eq else f'{eq.strip()} = 0')
    else:
        toks = split_top(s)
        for t in toks:
            key = ip.match_coef(t)
            if key is None:
                raise StataError(f'{t} not found', 111)
            R = np.zeros(len(names))
            R[names.index(key)] = 1
            rows.append(R)
            rhs.append(0.0)
            labels.append(f'{key} = 0')
    _wald_report(ip, est, rows, rhs, labels)


def _wald_report(ip, est, rows, rhs, labels):
    R = np.array(rows)
    q = np.array(rhs)
    b, V = est['b'], est['V']
    diff = R @ b - q
    RVR = R @ V @ R.T
    rank = np.linalg.matrix_rank(RVR)
    W = float(diff @ np.linalg.pinv(RVR) @ diff)
    ip.txt('\n')
    for i, l in enumerate(labels, 1):
        l = re.sub(r'\s*([+-=])\s*', r' \1 ', l).replace('  ', ' ')
        ip.txt(f' ({i:>2})  {l.strip()}\n')
    df_r = est.get('df_r')
    if est.get('stat') == 't' and df_r is not None:
        F = W / rank if rank else np.nan
        pv = sps.f.sf(F, rank, df_r)
        ip.txt(f'\n       F({rank:>3},{df_r:>6}) = ')
        ip.res(f'{F:>8.2f}\n')
        ip.txt('            Prob > F = ')
        ip.res(f'{pv:>9.4f}\n')
        ip.set_r(scalars={'F': F, 'p': pv, 'df': rank, 'df_r': df_r, 'drop': 0})
    else:
        pv = sps.chi2.sf(W, rank)
        ip.txt(f'\n           chi2({rank:>3}) = ')
        ip.res(f'{W:>8.2f}\n')
        ip.txt('         Prob > chi2 = ')
        ip.res(f'{pv:>9.4f}\n')
        ip.set_r(scalars={'chi2': W, 'p': pv, 'df': rank, 'drop': 0})


@command('testparm')
def cmd_testparm(ip, cmd, rest, by):
    est = ip.est
    names = est['names']
    info = est.get('colinfo')
    toks = split_top(rest.split(',')[0])
    terms = parse_terms(toks)
    d = ip.data
    want = set()
    for t in terms:
        key = '#'.join(sorted(d.resolve(c.var) for c in t))
        want.add(key)
    rows, labels = [], []
    for j, nm in enumerate(names):
        if nm == '_cons':
            continue
        ci = info[j] if info else None
        if ci is not None and (ci.base or ci.omitted):
            continue
        parts = nm.split('#')
        vs = '#'.join(sorted(re.sub(r'^.*\.', '', p_) for p_ in parts))
        if vs in want:
            R = np.zeros(len(names))
            R[j] = 1
            rows.append(R)
            labels.append(f'{nm} = 0')
    if not rows:
        raise StataError('no such variables', 111)
    _wald_report(ip, est, rows, [0.0] * len(rows), labels)


@command('lincom')
def cmd_lincom(ip, cmd, rest, by):
    est = ip.est
    s = rest.split(',')[0]
    R = _parse_lincomb(ip, s, est['names'])
    val = float(R @ est['b'])
    se = math.sqrt(float(R @ est['V'] @ R))
    e2 = {'names': ['(1)'], 'b': np.array([val]), 'V': np.array([[se * se]]), 'colinfo': None,
          'df_r': est.get('df_r'), 'depvar': est.get('depvar', '')}
    ip.txt(f'\n ( 1)  {s.strip()} = 0\n\n')
    coef_table(ip, e2, est.get('stat', 't'))
    ip.set_r(scalars={'estimate': val, 'se': se})


# ================================================================ hausman
@command('hausman')
def cmd_hausman(ip, cmd, rest, by):
    p = parse_command(rest)
    names = p.body.split()
    e1 = ip.stored.get(names[0]) if names[0] != '.' else ip.est
    e2 = ip.stored.get(names[1]) if names[1] != '.' else ip.est
    if e1 is None or e2 is None:
        raise StataError('estimation results not found', 111)
    common = [n for n in e1['names'] if n in e2['names'] and n != '_cons']
    idx1, idx2 = [], []
    for nmm in common:
        i1, i2 = e1['names'].index(nmm), e2['names'].index(nmm)
        if e1['V'][i1, i1] > 0 and e2['V'][i2, i2] > 0:
            idx1.append(i1)
            idx2.append(i2)
    b1 = e1['b'][idx1]
    b2 = e2['b'][idx2]
    Vd = e1['V'][np.ix_(idx1, idx1)] - e2['V'][np.ix_(idx2, idx2)]
    diff = b1 - b2
    chi2 = float(diff @ np.linalg.pinv(Vd) @ diff)
    df = np.linalg.matrix_rank(Vd)
    pv = sps.chi2.sf(chi2, df)
    ip.txt(f'\n{"---- Coefficients ----":>35}\n')
    ip.txt(f'{"":>12} |{"(b)":>11}{"(B)":>13}{"(b-B)":>16}{"sqrt(diag(V_b-V_B))":>22}\n')
    ip.txt(f'{"":>12} |{names[0][:10]:>11}{names[1][:10]:>13}{"Difference":>16}{"Std. err.":>17}\n')
    ip.txt('-' * 13 + '+' + '-' * 64 + '\n')
    sd = np.sqrt(np.maximum(np.diag(Vd), 0))
    for k, j in enumerate(idx1):
        ip.txt(f'{abbrev(e1["names"][j]):>12} |')
        ip.res(f'{g_fmt(b1[k], 9):>11}{g_fmt(b2[k], 9):>13}{g_fmt(diff[k], 9):>16}{g_fmt(sd[k], 9) if sd[k] > 0 else ".":>17}\n')
    ip.txt('-' * 78 + '\n')
    ip.txt(f'{"b = Consistent under H0 and Ha; obtained from " + e1.get("cmd", "") + ".":>78}\n')
    ip.txt(f'{"B = Inconsistent under Ha, efficient under H0; obtained from " + e2.get("cmd", "") + ".":>78}\n\n')
    ip.txt('Test of H0: Difference in coefficients not systematic\n\n')
    ip.txt(f"    chi2({df}) = (b-B)'[(V_b-V_B)^(-1)](b-B)\n")
    ip.txt('            = ')
    ip.res(f'{chi2:8.2f}\n')
    ip.txt('Prob > chi2 = ')
    ip.res(f'{pv:8.4f}\n')
    if np.any(np.linalg.eigvalsh((Vd + Vd.T) / 2) < -1e-12):
        ip.txt('(V_b-V_B is not positive definite)\n')
    ip.set_r(scalars={'chi2': chi2, 'df': df, 'p': pv})


# ================================================================ estat
@command('estat')
def cmd_estat(ip, cmd, rest, by):
    est = ip.est
    if est is None:
        raise StataError('last estimates not found', 301)
    p = parse_command(rest)
    parts = p.body.split()
    sub = parts[0] if parts else ''
    c = est.get('cmd')
    if c in ('ivregress',) and sub.startswith('first'):
        return estat_firststage(ip)
    if c in ('ivregress',) and sub.startswith('endog'):
        return estat_endogenous(ip)
    if c in ('ivregress',) and sub.startswith('overid'):
        return _estat_overid(ip)
    if sub.startswith('hett'):
        return _estat_hettest(ip)
    if sub.startswith('ovt'):
        return _estat_ovtest(ip)
    if sub.startswith('vif'):
        return cmd_vif(ip, 'vif', '', None)
    if sub.startswith('ic'):
        ll = est['scalars'].get('ll', np.nan)
        k = len([j for j in range(len(est['names'])) if est['V'][j, j] > 0])
        n = est['scalars']['N']
        ip.txt('\nAkaike\'s information criterion and Bayesian information criterion\n\n')
        ip.txt('-----------------------------------------------------------------------------\n')
        ip.txt('       Model |          N   ll(null)  ll(model)      df        AIC        BIC\n')
        ip.txt('-------------+---------------------------------------------------------------\n')
        ip.txt(f'{".":>12} |')
        ip.res(f'{comma_int(n):>11}{"."[:]:>11}{g_fmt(ll, 9):>11}{k:>8}{g_fmt(-2 * ll + 2 * k, 9):>11}{g_fmt(-2 * ll + k * math.log(n), 9):>11}\n')
        ip.txt('-----------------------------------------------------------------------------\n')
        return
    if sub.startswith('sum'):
        return _estat_summarize(ip, est)
    hook = est.get('estat')
    if hook:
        return hook(ip, sub, p)
    raise StataError(f'estat {sub} not valid after {c}', 321)


def _estat_overid(ip):
    est = ip.est
    iv = est['iv']
    X = np.column_stack([iv.Y2, iv.X1])
    Zf = np.column_stack([iv.Z, iv.X1])
    b, e, _, _ = tsls(iv.y, X, Zf)
    L1, ke = iv.Z.shape[1], iv.Y2.shape[1]
    if L1 <= ke:
        ip.txt('\n  Tests of overidentifying restrictions:\n  (model is exactly identified)\n')
        return
    n = iv.n
    Pe = Zf @ np.linalg.lstsq(Zf, e, rcond=None)[0]
    S = n * float(e @ Pe) / float(e @ e)
    df = L1 - ke
    ip.txt('\n  Tests of overidentifying restrictions:\n\n  Sargan (score) chi2(' + str(df) + ') = ')
    ip.res(f'{S:.5g}')
    ip.txt('  (p = ')
    ip.res(f'{sps.chi2.sf(S, df):.4f}')
    ip.txt(')\n')


def _estat_summarize(ip, est):
    d = ip.data
    m = est['sample']
    vars_ = [est['depvar']]
    for nm in est['names']:
        if nm != '_cons' and '#' not in nm and '.' not in nm and nm in d.cols:
            vars_.append(nm)
    ip.txt('\n  Estimation sample ' + est.get('cmd', '') + f'{"Number of obs = ":>40}')
    ip.res(f'{comma_int(int(m.sum()))}\n\n')
    ip.txt('  -------------------------------------------------------------\n')
    ip.txt('      Variable |         Mean      Std. dev.         Min          Max\n')
    ip.txt('  -------------+-----------------------------------------------\n')
    for v in vars_:
        x = d.cols[v][m]
        x = x[~np.isnan(x)]
        ip.txt(f'  {abbrev(v):>12} |')
        ip.res(f'{g_fmt(x.mean(), 9):>13}{g_fmt(x.std(ddof=1), 9):>15}{g_fmt(x.min(), 9):>12}{g_fmt(x.max(), 9):>13}\n')
    ip.txt('  -------------------------------------------------------------\n')


def _estat_hettest(ip):
    est = ip.est
    f = est.get('fit')
    if f is None:
        raise StataError('estat hettest only after regress', 321)
    e = f.e
    n = len(e)
    yhat = f.y - e
    s2 = float(e @ e) / n
    z = e ** 2 / s2
    Xz = np.column_stack([yhat, np.ones(n)])
    bz = np.linalg.lstsq(Xz, z, rcond=None)[0]
    fit_ = Xz @ bz
    ess = float(np.sum((fit_ - z.mean()) ** 2))
    chi2 = ess / 2
    pv = sps.chi2.sf(chi2, 1)
    ip.txt('\nBreusch–Pagan/Cook–Weisberg test for heteroskedasticity \nAssumption: Normal error terms\n')
    ip.txt(f'Variable: Fitted values of {est["depvar"]}\n\nH0: Constant variance\n\n')
    ip.txt('    chi2(1) = ')
    ip.res(f'{chi2:7.2f}\n')
    ip.txt('Prob > chi2 = ')
    ip.res(f'{pv:7.4f}\n')
    ip.set_r(scalars={'chi2': chi2, 'df': 1, 'p': pv})


def _estat_ovtest(ip):
    est = ip.est
    f = est.get('fit')
    e = f.e
    yhat = f.y - e
    Xa = f.Xfull[:, f.active]
    s = (yhat - yhat.mean()) / yhat.std()
    Xu = np.column_stack([Xa, s ** 2, s ** 3, s ** 4])
    bu = np.linalg.lstsq(Xu, f.y, rcond=None)[0]
    ssr_u = float(np.sum((f.y - Xu @ bu) ** 2))
    ssr_r = float(e @ e)
    n = len(e)
    df2 = n - np.linalg.matrix_rank(Xu)
    F = ((ssr_r - ssr_u) / 3) / (ssr_u / df2)
    pv = sps.f.sf(F, 3, df2)
    ip.txt(f'\nRamsey RESET test for omitted variables\nOmitted: Powers of fitted values of {est["depvar"]}\n\n')
    ip.txt('H0: Model has no omitted variables\n\n')
    ip.txt(f'F(3, {df2}) = ')
    ip.res(f'{F:6.2f}\n')
    ip.txt('   Prob > F = ')
    ip.res(f'{pv:6.4f}\n')
    ip.set_r(scalars={'F': F, 'df': 3, 'df_r': df2, 'p': pv})


@command('vif')
def cmd_vif(ip, cmd, rest, by):
    est = ip.est
    f = est.get('fit')
    if f is None:
        raise StataError('vif only after regress', 321)
    act = [j for j in f.active if f.info[j].kind != 'cons']
    X = f.Xfull[:, act]
    rows = []
    for k, j in enumerate(act):
        others = np.column_stack([np.delete(X, k, axis=1), np.ones(X.shape[0])])
        b_ = np.linalg.lstsq(others, X[:, k], rcond=None)[0]
        r = X[:, k] - others @ b_
        r2 = 1 - float(r @ r) / float(np.sum((X[:, k] - X[:, k].mean()) ** 2))
        v = 1 / (1 - r2)
        rows.append((f.names[j], v))
    rows.sort(key=lambda t: -t[1])
    ip.txt('\n    Variable |       VIF       1/VIF  \n-------------+----------------------\n')
    for nm, v in rows:
        ip.txt(f'{abbrev(nm):>12} |')
        ip.res(f'{v:>10.2f}{1 / v:>12.6f}\n')
    ip.txt('-------------+----------------------\n')
    ip.txt(f'{"Mean VIF":>12} |')
    ip.res(f'{np.mean([v for _, v in rows]):>10.2f}\n')


# ================================================================ margins
@command('margins')
def cmd_margins(ip, cmd, rest, by):
    est = ip.est
    if est is None or est.get('rhs_tokens') is None:
        raise StataError('margins not available after this command', 321)
    p = parse_command(rest)
    d = ip.data
    sample = est['sample']
    over_terms = split_top(p.body)
    dydx = p.opts.get('dydx')
    at = p.opts.get('at')
    b, V = est['b'], est['V']
    names = est['names']
    df_r = est.get('df_r')
    scen = [({}, '')]
    at_list = []
    if at:
        m = re.match(r'^\s*(\w+)\s*=\s*\((.*)\)\s*$', at)
        if m:
            var = d.resolve(m.group(1))
            vals = parse_numlist(m.group(2))
            scen = [({var: np.full(d.n, v)}, f'{var} = {g_fmt(v)}') for v in vals]
            at_list = [(var, v) for v in vals]
        else:
            m = re.match(r'^\s*(\w+)\s*=\s*(\S+)\s*$', at)
            var = d.resolve(m.group(1))
            v = float(m.group(2))
            scen = [({var: np.full(d.n, v)}, f'{var} = {g_fmt(v)}')]
            at_list = [(var, v)]
    over_var = None
    if over_terms:
        over_var = d.resolve(over_terms[0].replace('i.', ''))
        lv = est.get('levels', {}).get(over_var) or sorted(np.unique(d.cols[over_var][sample]).tolist())
        scen = [({over_var: np.full(d.n, l)}, l) for l in lv]
    xvar = d.resolve(dydx) if dydx else None
    bmap = {canon_coef(n): i for i, n in enumerate(names)}

    def grad_rows(override):
        cols, _ = build_design(ip, d, est['rhs_tokens'], np.ones(d.n, dtype=bool), override=override,
                               levels_from=est.get('levels'))
        G = np.zeros((d.n, len(names)))
        for c in cols:
            key = canon_coef(c.name)
            if key in bmap:
                G[:, bmap[key]] = np.nan_to_num(c.values)
        if '_cons' in bmap:
            G[:, bmap['_cons']] = 1.0
        return G
    results = []
    for ov, lab in scen:
        if xvar:
            x0 = d.cols[xvar].astype(float)
            h = 1e-4 * (np.nanstd(x0[sample]) + 1e-8)
            ov1 = dict(ov)
            ov2 = dict(ov)
            ov1[xvar] = (ov.get(xvar, x0)) + h
            ov2[xvar] = (ov.get(xvar, x0)) - h
            Gd = (grad_rows(ov1) - grad_rows(ov2)) / (2 * h)
        else:
            Gd = grad_rows(ov)
        g = Gd[sample].mean(axis=0)
        val = float(g @ b)
        se = math.sqrt(max(float(g @ V @ g), 0))
        results.append((lab, val, se))
    stat = est.get('stat', 't')
    crit = sps.t.ppf(0.975, df_r) if stat == 't' and df_r else sps.norm.ppf(0.975)
    N = int(sample.sum())
    ip.txt('\n' + ('Average marginal effects' if xvar else ('Predictive margins' if (over_var or at_list) else 'Predictive margins')))
    ip.txt(f'{"Number of obs = ":>41}')
    ip.res(f'{comma_int(N)}\n')
    if est.get('vce') in ('robust', 'cluster'):
        ip.txt('Model VCE: Robust\n')
    else:
        ip.txt('Model VCE: OLS\n')
    ip.txt('\nExpression: Linear prediction, predict()\n')
    if xvar:
        ip.txt(f'dy/dx wrt:  {xvar}\n')
    if at_list:
        ip.txt('\n')
        for k, (v, val) in enumerate(at_list, 1):
            ip.txt(f'{k}._at: {v} = {g_fmt(val)}\n')
    ip.txt('\n' + '-' * 78 + '\n')
    ip.txt(f'{"":>12} |{"Delta-method":>24}\n')
    ip.txt(f'{"":>12} |{"dy/dx" if xvar else "Margin":>11}   std. err.{"      t    P>|t|" if stat == "t" else "      z    P>|z|"}     [95% conf. interval]\n')
    ip.txt('-' * 13 + '+' + '-' * 64 + '\n')
    if xvar and (over_var or at_list):
        ip.txt(f'{abbrev(xvar):<12} |\n')
    if over_var:
        ip.txt(f'{abbrev(over_var):>12} |\n')
    if at_list and not over_var:
        ip.txt(f'{"_at":>12} |\n')
    tab = []
    for k, (lab, val, se) in enumerate(results, 1):
        if over_var:
            rl = d.value_text(over_var, lab) or g_fmt(lab)
            rl = f'{rl[:10]:>11} '
        elif at_list:
            rl = f'{k:>11} '
        else:
            rl = abbrev(xvar, 12) if xvar else '_cons'
        t = val / se if se > 0 else np.nan
        pv = 2 * (sps.t.sf(abs(t), df_r) if stat == 't' and df_r else sps.norm.sf(abs(t)))
        ip.txt(f'{rl:>12} |')
        ip.res(f'{g_fmt(val, 9):>11}{g_fmt(se, 9):>11}{t:9.2f}{pv:8.3f}{g_fmt(val - crit * se, 9):>13}{g_fmt(val + crit * se, 9):>12}\n')
        tab.append((lab, val, se, val - crit * se, val + crit * se))
    ip.txt('-' * 78 + '\n')
    ip.margins_last = {'rows': tab, 'xvar': xvar, 'over': over_var, 'at': at_list,
                       'labels': [d.value_text(over_var, r[0]) or g_fmt(r[0]) for r in tab] if over_var else
                       ([g_fmt(v) for _, v in at_list] if at_list else [''])}
    ip.set_r(matrices={'table': E.Matrix(np.array([[r[1] for r in tab], [r[2] for r in tab]]), ['b', 'se'])})
