"""Additional everyday commands: levelsof, ds, lookfor, clonevar, xtile, ci, sample,
oneway, prtest, ranksum, signrank, kwallis, swilk, sktest, mvencode/mvdecode,
mkmat, matlist, tsline, xtline."""
import re
import numpy as np
from scipy import stats as st

from . import expr as E
from .errors import StataError
from .fmt import g_fmt, comma_int, abbrev, num_to_macro
from .interp import command
from .syntax import parse_command, parse_numlist


def _vals(ip, p, v, mask=None):
    d = ip.data
    m = ip.mark(p) if mask is None else mask
    x = d.cols[v].astype(float)
    return x[m & ~np.isnan(x)]


def _groups(ip, p, gname, y=None):
    d = ip.data
    g = d.resolve(gname)
    m = ip.mark(p)
    if y is not None:
        m &= ~np.isnan(d.cols[y].astype(float))
    gc = d.cols[g]
    if not d.isstr(g):
        m &= ~np.isnan(gc.astype(float))
    lv = sorted(set(gc[m].tolist()), key=lambda t: (0, t) if isinstance(t, str) else (1, t))
    lab = lambda l: l if d.isstr(g) else (d.value_text(g, l) or g_fmt(l).strip())
    return g, m, lv, lab


@command('levelsof', 'levels')
def cmd_levelsof(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    v = d.resolve(p.body.split()[0])
    m = ip.mark(p)
    col = d.cols[v]
    if d.isstr(v):
        lv = sorted(set(x for x in col[m] if x != '' or p.opts.has('missing')))
        clean = p.opts.has('clean')
        items = [x if clean and ' ' not in x else '`"%s"\'' % x for x in lv]
    else:
        x = col[m].astype(float)
        if not p.opts.has('missing'):
            x = x[~np.isnan(x)]
        lv = sorted(set(x.tolist()), key=lambda t: np.inf if np.isnan(t) else t)
        items = ['.' if np.isnan(t) else num_to_macro(t) for t in lv]
    sep = p.opts.get('separate') or ' '
    s = sep.join(items)
    if p.opts.has('local'):
        ip.locals[p.opts.get('local').strip()] = s
    if not ip.quiet:
        ip.txt(s + '\n')
    ip.set_r(scalars={'N': int(m.sum()), 'r': len(lv)}, macros={'levels': s})


@command('ds')
def cmd_ds(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    vl = d.expand_varlist(p.body.split()) if p.body.strip() else list(d.vars)
    if p.opts.has('has') or p.opts.has('not'):
        spec = (p.opts.get('has') or p.opts.get('not') or '').split()
        kind = spec[0] if spec else ''
        if kind.startswith('type'):
            want = spec[1] if len(spec) > 1 else 'numeric'
            ok = lambda v: d.isstr(v) if want == 'string' else (not d.isstr(v) if want == 'numeric' else d.types[v] == want)
        elif kind.startswith('varl'):
            ok = lambda v: bool(d.varlab.get(v))
        elif kind.startswith('vall'):
            ok = lambda v: bool(d.vallab.get(v))
        else:
            ok = lambda v: True
        neg = p.opts.has('not')
        vl = [v for v in vl if ok(v) != neg]
    if not ip.quiet and vl:
        w = max(len(v) for v in vl) + 2
        w = max(w, 10)
        per = max(1, 78 // w)
        for i in range(0, len(vl), per):
            ip.txt(''.join(v.ljust(w) for v in vl[i:i + per]).rstrip() + '\n')
    ip.set_r(macros={'varlist': ' '.join(vl)})


@command('lookfor')
def cmd_lookfor(ip, cmd, rest, by):
    d = ip.data
    keys = [k.lower() for k in rest.split()]
    hits = [v for v in d.vars if any(k in v.lower() or k in d.varlab.get(v, '').lower() for k in keys)]
    if hits:
        from .interp import COMMANDS
        COMMANDS['describe'](ip, 'describe', ' '.join(hits), None)
    ip.set_r(macros={'varlist': ' '.join(hits)})


@command('clonevar')
def cmd_clonevar(ip, cmd, rest, by):
    p = parse_command(rest)
    m = re.match(r'^\s*(\w+)\s*=\s*(\w+)\s*$', p.body)
    if not m:
        raise StataError('syntax: clonevar newvar = oldvar', 198)
    d = ip.data
    new, old = m.group(1), d.resolve(m.group(2))
    if d.has(new):
        raise StataError('variable %s already defined' % new, 110)
    mask = ip.mark(p)
    col = d.cols[old].copy()
    if d.isstr(old):
        col = np.where(mask, col, '')
    else:
        col = np.where(mask, col, np.nan)
    d.add_var(new, col, d.types[old])
    for a in ('varlab', 'vallab', 'formats'):
        if old in getattr(d, a):
            getattr(d, a)[new] = getattr(d, a)[old]


def _pctiles(x, w, ps):
    """Stata default percentile definition (no altdef)."""
    o = np.argsort(x, kind='stable')
    x, w = x[o], w[o]
    W = w.sum()
    cw = np.cumsum(w)
    out = []
    for pp in ps:
        P = W * pp / 100.0
        i = np.searchsorted(cw, P, side='left')
        if i < len(x) and np.isclose(cw[i], P) and i + 1 < len(x):
            out.append((x[i] + x[i + 1]) / 2)
        else:
            out.append(x[min(i, len(x) - 1)])
    return np.array(out)


@command('xtile')
def cmd_xtile(ip, cmd, rest, by):
    p = parse_command(rest)
    m = re.match(r'^\s*(\w+)\s*=\s*(.+)$', p.body)
    if not m:
        raise StataError('syntax: xtile newvar = exp, nquantiles(#)', 198)
    d = ip.data
    new = m.group(1)
    x = np.asarray(E.eval_expr(m.group(2), E.Ctx(ip, d)), dtype=float) * np.ones(d.n)
    mask = ip.mark(p) & ~np.isnan(x)
    w = np.ones(d.n)
    if p.weight:
        w = np.asarray(E.eval_expr(p.weight[1], E.Ctx(ip, d)), dtype=float) * np.ones(d.n)
    if p.opts.has('cutpoints'):
        cv = d.resolve(p.opts.get('cutpoints'))
        cuts = np.sort(d.cols[cv][~np.isnan(d.cols[cv].astype(float))].astype(float))
    else:
        nq = int(p.opts.get('nquantiles') or p.opts.get('nq') or 2)
        cuts = _pctiles(x[mask], w[mask], [100.0 * k / nq for k in range(1, nq)])
    out = np.full(d.n, np.nan)
    xm = x[mask]
    cat = np.searchsorted(cuts, xm, side='left') + 1.0
    out[mask] = cat
    d.add_var(new, out, 'byte' if len(cuts) < 100 else 'int')


@command('pctile', '_pctile')
def cmd_pctile(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    if cmd == '_pctile':
        v = d.resolve(p.body.split()[0])
        x = _vals(ip, p, v)
        ps = parse_numlist(p.opts.get('percentiles')) if p.opts.has('percentiles') else \
            [100.0 * k / int(p.opts.get('nquantiles') or 2) for k in range(1, int(p.opts.get('nquantiles') or 2))]
        r = _pctiles(x, np.ones(len(x)), ps)
        ip.set_r(scalars={'r%d' % (k + 1): r[k] for k in range(len(r))})
        return
    m = re.match(r'^\s*(\w+)\s*=\s*(\w+)', p.body)
    new, v = m.group(1), d.resolve(m.group(2))
    x = _vals(ip, p, v)
    nq = int(p.opts.get('nquantiles') or 2)
    r = _pctiles(x, np.ones(len(x)), [100.0 * k / nq for k in range(1, nq)])
    col = np.full(d.n, np.nan)
    col[:len(r)] = r
    d.add_var(new, col, 'float')


@command('ci', 'cii')
def cmd_ci(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    body = p.body.split()
    if body and body[0] in ('means', 'mean'):
        body = body[1:]
    elif body and body[0] in ('proportions', 'variances'):
        raise StataError('ci %s not supported in Stata Web; use ci means' % body[0], 199)
    level = float(p.opts.get('level') or 95)
    vl = d.expand_varlist(body) if body else [v for v in d.vars if not d.isstr(v)]
    ip.txt('\n    Variable |        Obs        Mean    Std. err.       [%g%% conf. interval]\n' % level)
    ip.txt('-' * 13 + '+' + '-' * 63 + '\n')
    for v in vl:
        x = _vals(ip, p, v)
        n = len(x)
        mu = x.mean() if n else np.nan
        se = x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        t = st.t.ppf(0.5 + level / 200, n - 1) if n > 1 else np.nan
        ip.txt('%12s |' % abbrev(v, 12))
        ip.res('%11s%12s%12s%15s%12s\n' % (comma_int(n), g_fmt(mu, 9), g_fmt(se, 9), g_fmt(mu - t * se, 9), g_fmt(mu + t * se, 9)))
        ip.set_r(scalars={'N': n, 'mean': mu, 'se': se, 'lb': mu - t * se, 'ub': mu + t * se, 'level': level})


@command('sample')
def cmd_sample(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    k = float(p.body.split()[0])
    m = ip.mark(p)
    idx = np.where(m)[0]
    keep_n = int(k) if p.opts.has('count') else int(round(len(idx) * k / 100))
    chosen = ip.rng.permutation(idx)[:keep_n]
    keep = ~m
    keep[chosen] = True
    dropped = int((~keep).sum())
    d.keep_obs(keep)
    ip.txt('(%s observation%s deleted)\n' % (comma_int(dropped), '' if dropped == 1 else 's'))


@command('oneway')
def cmd_oneway(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    yv, gv = p.body.split()[:2]
    y = d.resolve(yv)
    g, m, lv, lab = _groups(ip, p, gv, y)
    yy = d.cols[y].astype(float)
    groups = [yy[m & (d.cols[g] == l)] for l in lv]
    if p.opts.has('tabulate'):
        ip.txt('\n%12s |          Summary of %s\n' % (abbrev(g, 12), d.varlab.get(y, y)))
        ip.txt('%12s |        Mean   Std. dev.       Freq.\n' % '')
        ip.txt('-' * 13 + '+' + '-' * 36 + '\n')
        for l, x in zip(lv, groups):
            ip.txt('%12s |' % abbrev(str(lab(l)), 12)); ip.res('%12s%12s%12s\n' % (g_fmt(x.mean(), 9), g_fmt(x.std(ddof=1), 9), comma_int(len(x))))
        ip.txt('-' * 13 + '+' + '-' * 36 + '\n')
        a = np.concatenate(groups)
        ip.txt('%12s |' % 'Total'); ip.res('%12s%12s%12s\n' % (g_fmt(a.mean(), 9), g_fmt(a.std(ddof=1), 9), comma_int(len(a))))
    a = np.concatenate(groups)
    N, k = len(a), len(groups)
    gm = a.mean()
    ssb = sum(len(x) * (x.mean() - gm) ** 2 for x in groups)
    ssw = sum(((x - x.mean()) ** 2).sum() for x in groups)
    dfb, dfw = k - 1, N - k
    F = (ssb / dfb) / (ssw / dfw)
    pF = st.f.sf(F, dfb, dfw)
    ip.txt('\n                        Analysis of variance\n')
    ip.txt('    Source              SS         df      MS            F     Prob > F\n')
    ip.txt('-' * 72 + '\n')
    ip.txt('Between groups   '); ip.res('%12s%8d%12s%12.2f%11.4f\n' % (g_fmt(ssb, 9), dfb, g_fmt(ssb / dfb, 9), F, pF))
    ip.txt(' Within groups   '); ip.res('%12s%8d%12s\n' % (g_fmt(ssw, 9), dfw, g_fmt(ssw / dfw, 9)))
    ip.txt('-' * 72 + '\n')
    ip.txt('    Total        '); ip.res('%12s%8d%12s\n' % (g_fmt(ssb + ssw, 9), N - 1, g_fmt((ssb + ssw) / (N - 1), 9)))
    if all(len(x) > 1 for x in groups):
        b, pb = st.bartlett(*groups)
        ip.txt("\nBartlett's equal-variances test: chi2(%d) = " % dfb); ip.res('%8.4f' % b); ip.txt('   Prob>chi2 = '); ip.res('%.3f\n' % pb)
    ip.set_r(scalars={'N': N, 'F': F, 'df_m': dfb, 'df_r': dfw, 'mss': ssb, 'rss': ssw, 'chi2bart': b if all(len(x) > 1 for x in groups) else None})


def _ztail(z):
    return 2 * st.norm.sf(abs(z))


@command('prtest')
def cmd_prtest(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    if p.opts.has('by'):
        y = d.resolve(p.body.split()[0])
        g, m, lv, lab = _groups(ip, p, p.opts.get('by'), y)
        if len(lv) != 2:
            raise StataError('more than 2 groups found, only 2 allowed', 420)
        xs = [d.cols[y].astype(float)[m & (d.cols[g] == l)] for l in lv]
        names = [str(lab(l)) for l in lv]
    else:
        mm = re.match(r'^\s*(\w+)\s*==\s*(\S+)\s*$', p.body)
        y = d.resolve(mm.group(1))
        x = _vals(ip, p, y)
        p0 = float(mm.group(2))
        n, ph = len(x), x.mean()
        se = np.sqrt(ph * (1 - ph) / n)
        z = (ph - p0) / np.sqrt(p0 * (1 - p0) / n)
        zc = st.norm.ppf(0.975)
        ip.txt('\nOne-sample test of proportion%sNumber of obs = ' % (' ' * 19)); ip.res('%10s\n' % comma_int(n))
        ip.txt('\n' + '-' * 78 + '\n    Variable |       Mean   Std. err.                     [95% conf. interval]\n' + '-' * 13 + '+' + '-' * 64 + '\n')
        ip.txt('%12s |' % abbrev(y, 12)); ip.res('%11s%12s%32s%12s\n' % (g_fmt(ph, 9), g_fmt(se, 9), g_fmt(ph - zc * se, 9), g_fmt(ph + zc * se, 9)))
        ip.txt('-' * 78 + '\n    p = proportion(%s)%sz = ' % (y, ' ' * max(1, 46 - len(y)))); ip.res('%9.4f\n' % z)
        ip.txt('H0: p = %s\n\n     Ha: p < %s             Ha: p != %s               Ha: p > %s\n' % (g_fmt(p0).strip(), g_fmt(p0).strip(), g_fmt(p0).strip(), g_fmt(p0).strip()))
        ip.txt(' Pr(Z < z) = '); ip.res('%6.4f' % st.norm.cdf(z)); ip.txt('         Pr(|Z| > |z|) = '); ip.res('%6.4f' % _ztail(z)); ip.txt('          Pr(Z > z) = '); ip.res('%6.4f\n' % st.norm.sf(z))
        ip.set_r(scalars={'N_1': n, 'P_1': ph, 'z': z, 'p': _ztail(z)})
        return
    n1, n2 = len(xs[0]), len(xs[1])
    p1, p2 = xs[0].mean(), xs[1].mean()
    pp = (xs[0].sum() + xs[1].sum()) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    zc = st.norm.ppf(0.975)
    ip.txt('\nTwo-sample test of proportions%s%s: Number of obs = ' % (' ' * max(1, 20 - len(names[0])), names[0])); ip.res('%10s\n' % comma_int(n1))
    ip.txt('%s%s: Number of obs = ' % (' ' * (50 - len(names[1])), names[1])); ip.res('%10s\n' % comma_int(n2))
    ip.txt('-' * 78 + '\n       Group |       Mean   Std. err.      z    P>|z|     [95% conf. interval]\n' + '-' * 13 + '+' + '-' * 64 + '\n')
    for nm, ph, n in ((names[0], p1, n1), (names[1], p2, n2)):
        se = np.sqrt(ph * (1 - ph) / n)
        ip.txt('%12s |' % abbrev(nm, 12)); ip.res('%11s%12s%32s%12s\n' % (g_fmt(ph, 9), g_fmt(se, 9), g_fmt(ph - zc * se, 9), g_fmt(ph + zc * se, 9)))
    ip.txt('-' * 13 + '+' + '-' * 64 + '\n')
    sed = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ip.txt('%12s |' % 'diff'); ip.res('%11s%12s%32s%12s\n' % (g_fmt(p1 - p2, 9), g_fmt(sed, 9), g_fmt(p1 - p2 - zc * sed, 9), g_fmt(p1 - p2 + zc * sed, 9)))
    ip.txt('%12s |  under H0:%12s%8.2f%9.3f\n' % ('', g_fmt(np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2)), 9), z, _ztail(z)))
    ip.txt('-' * 78 + '\n        diff = prop(%s) - prop(%s)%sz = ' % (names[0], names[1], ' ' * max(1, 36 - len(names[0]) - len(names[1])))); ip.res('%9.4f\n' % z)
    ip.txt('    H0: diff = 0\n\n    Ha: diff < 0                 Ha: diff != 0                 Ha: diff > 0\n')
    ip.txt(' Pr(Z < z) = '); ip.res('%6.4f' % st.norm.cdf(z)); ip.txt('         Pr(|Z| > |z|) = '); ip.res('%6.4f' % _ztail(z)); ip.txt('          Pr(Z > z) = '); ip.res('%6.4f\n' % st.norm.sf(z))
    ip.set_r(scalars={'N_1': n1, 'N_2': n2, 'P_1': p1, 'P_2': p2, 'z': z, 'p': _ztail(z)})


@command('ranksum')
def cmd_ranksum(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    y = d.resolve(p.body.split()[0])
    g, m, lv, lab = _groups(ip, p, p.opts.get('by'), y)
    if len(lv) != 2:
        raise StataError('more than 2 groups found, only 2 allowed', 420)
    yy = d.cols[y].astype(float)
    x1, x2 = yy[m & (d.cols[g] == lv[0])], yy[m & (d.cols[g] == lv[1])]
    a = np.concatenate([x1, x2])
    r = st.rankdata(a)
    n1, n2, N = len(x1), len(x2), len(a)
    R1 = r[:n1].sum()
    e1 = n1 * (N + 1) / 2
    _, cnt = np.unique(a, return_counts=True)
    v_un = n1 * n2 * (N + 1) / 12
    adj = -n1 * n2 * (cnt ** 3 - cnt).sum() / (12 * N * (N - 1))
    z = (R1 - e1) / np.sqrt(v_un + adj)
    ip.txt('\nTwo-sample Wilcoxon rank-sum (Mann–Whitney) test\n\n')
    ip.txt('%12s |      Obs    Rank sum    Expected\n' % abbrev(g, 12) + '-' * 13 + '+' + '-' * 33 + '\n')
    for l, n, R in ((lv[0], n1, R1), (lv[1], n2, r[n1:].sum())):
        ip.txt('%12s |' % abbrev(str(lab(l)), 12)); ip.res('%9s%12s%12s\n' % (comma_int(n), g_fmt(R, 9), g_fmt(n * (N + 1) / 2, 9)))
    ip.txt('-' * 13 + '+' + '-' * 33 + '\n%12s |' % 'Combined'); ip.res('%9s%12s%12s\n' % (comma_int(N), g_fmt(r.sum(), 9), g_fmt(r.sum(), 9)))
    ip.txt('\nUnadjusted variance  '); ip.res('%12s\n' % g_fmt(v_un, 9))
    ip.txt('Adjustment for ties  '); ip.res('%12s\n' % g_fmt(adj, 9))
    ip.txt('%21s------------\nAdjusted variance    ' % ''); ip.res('%12s\n' % g_fmt(v_un + adj, 9))
    ip.txt('\nH0: %s(%s==%s) = %s(%s==%s)\n' % (y, g, lab(lv[0]), y, g, lab(lv[1])))
    ip.txt('         z = '); ip.res('%7.3f\n' % z)
    ip.txt('Prob > |z| = '); ip.res('%7.4f\n' % _ztail(z))
    ip.set_r(scalars={'N_1': n1, 'N_2': n2, 'z': z, 'p': _ztail(z), 'sum_obs': R1, 'sum_exp': e1, 'Var_a': v_un + adj})


@command('signrank')
def cmd_signrank(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    mm = re.match(r'^\s*(\w+)\s*=\s*(.+)$', p.body)
    y = d.resolve(mm.group(1))
    rhs = np.asarray(E.eval_expr(mm.group(2), E.Ctx(ip, d)), dtype=float) * np.ones(d.n)
    m = ip.mark(p)
    dd = (d.cols[y].astype(float) - rhs)[m]
    dd = dd[~np.isnan(dd)]
    N = len(dd)
    nz = dd[dd != 0]
    r = st.rankdata(np.abs(dd))
    rp, rn = r[dd > 0].sum(), r[dd < 0].sum()
    e = N * (N + 1) / 4
    v = N * (N + 1) * (2 * N + 1) / 24
    _, cnt = np.unique(np.abs(dd[dd != 0]), return_counts=True)
    adj_t = -(cnt ** 3 - cnt).sum() / 48
    nzero = int((dd == 0).sum())
    adj_z = -nzero * (nzero + 1) * (2 * nzero + 1) / 24
    var = v + adj_t + adj_z
    z = (rp - e) / np.sqrt(var)
    ip.txt('\nWilcoxon signed-rank test\n\n        Sign |      Obs   Sum ranks    Expected\n' + '-' * 13 + '+' + '-' * 33 + '\n')
    for nm, c, s in (('Positive', int((dd > 0).sum()), rp), ('Negative', int((dd < 0).sum()), rn), ('Zero', nzero, r[dd == 0].sum())):
        ip.txt('%12s |' % nm); ip.res('%9s%12s%12s\n' % (comma_int(c), g_fmt(s, 9), g_fmt(e if nm != 'Zero' else s, 9)))
    ip.txt('-' * 13 + '+' + '-' * 33 + '\n%12s |' % 'All'); ip.res('%9s%12s%12s\n' % (comma_int(N), g_fmt(r.sum(), 9), g_fmt(r.sum(), 9)))
    ip.txt('\nUnadjusted variance  '); ip.res('%12s\n' % g_fmt(v, 9))
    ip.txt('Adjustment for ties  '); ip.res('%12s\n' % g_fmt(adj_t, 9))
    ip.txt('Adjustment for zeros '); ip.res('%12s\n' % g_fmt(adj_z, 9))
    ip.txt('%21s------------\nAdjusted variance    ' % ''); ip.res('%12s\n' % g_fmt(var, 9))
    ip.txt('\nH0: %s = %s\n         z = ' % (y, mm.group(2).strip())); ip.res('%7.3f\n' % z)
    ip.txt('Prob > |z| = '); ip.res('%7.4f\n' % _ztail(z))
    ip.set_r(scalars={'N_pos': int((dd > 0).sum()), 'N_neg': int((dd < 0).sum()), 'N_tie': nzero, 'z': z, 'p': _ztail(z), 'sum_pos': rp, 'sum_neg': rn, 'Var_a': var})


@command('kwallis')
def cmd_kwallis(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    y = d.resolve(p.body.split()[0])
    g, m, lv, lab = _groups(ip, p, p.opts.get('by'), y)
    yy = d.cols[y].astype(float)
    groups = [yy[m & (d.cols[g] == l)] for l in lv]
    a = np.concatenate(groups)
    r = st.rankdata(a)
    N = len(a)
    H = 0
    pos = 0
    w = max(len(abbrev(g, 12)), max(len(str(lab(l))[:12]) for l in lv), 5)
    bar = '  +' + '-' * (w + 2) + '+-----+----------+\n'
    ip.txt('\nKruskal–Wallis equality-of-populations rank test\n\n' + bar)
    ip.txt('  | %s | Obs | Rank sum |\n' % abbrev(g, 12).rjust(w))
    ip.txt('  |' + '-' * (w + 2) + '+-----+----------|\n')
    for l, x in zip(lv, groups):
        R = r[pos:pos + len(x)].sum()
        pos += len(x)
        H += R ** 2 / len(x)
        ip.txt('  | %s | ' % str(lab(l))[:12].rjust(w)); ip.res('%3d' % len(x)); ip.txt(' | '); ip.res('%8.2f' % R); ip.txt(' |\n')
    ip.txt(bar)
    H = 12 / (N * (N + 1)) * H - 3 * (N + 1)
    _, cnt = np.unique(a, return_counts=True)
    Ht = H / (1 - (cnt ** 3 - cnt).sum() / (N ** 3 - N))
    df = len(groups) - 1
    ip.txt('\nchi2(%d) = ' % df); ip.res('%8.3f' % H); ip.txt('\nProbability = '); ip.res('%9.4f\n' % st.chi2.sf(H, df))
    ip.txt('\nchi2(%d) with ties = ' % df); ip.res('%8.3f' % Ht); ip.txt('\nProbability = '); ip.res('%9.4f\n' % st.chi2.sf(Ht, df))
    ip.set_r(scalars={'df': df, 'chi2': H, 'chi2_adj': Ht})


@command('swilk')
def cmd_swilk(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    vl = d.expand_varlist(p.body.split())
    ip.txt('\n%sShapiro–Wilk W test for normal data\n\n' % (' ' * 18))
    ip.txt('    Variable |        Obs       W           V         z       Prob>z\n' + '-' * 13 + '+' + '-' * 54 + '\n')
    for v in vl:
        x = _vals(ip, p, v)
        n = len(x)
        W, pv = st.shapiro(x)
        u = np.log(n)
        mu = 0.0038915 * u ** 3 - 0.083751 * u ** 2 - 0.31082 * u - 1.5861
        V = (1 - W) / np.exp(mu) if n >= 12 else np.nan
        z = st.norm.isf(pv)
        ip.txt('%12s |' % abbrev(v, 12)); ip.res('%11s%10.5f%11.3f%10.3f%11.5f\n' % (comma_int(n), W, V, z, pv))
        ip.set_r(scalars={'N': n, 'W': W, 'V': V, 'z': z, 'p': pv})


@command('sktest')
def cmd_sktest(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    vl = d.expand_varlist(p.body.split())
    ip.txt('\n%sSkewness and kurtosis tests for normality\n%s----- Joint test -----\n' % (' ' * 18, ' ' * 52))
    ip.txt('    Variable |       Obs   Pr(skewness)   Pr(kurtosis)   chi2(2)    Prob>chi2\n' + '-' * 13 + '+' + '-' * 65 + '\n')
    for v in vl:
        x = _vals(ip, p, v)
        zs, ps = st.skewtest(x)
        zk, pk = st.kurtosistest(x)
        chi = zs ** 2 + zk ** 2
        ip.txt('%12s |' % abbrev(v, 12)); ip.res('%10s%15.4f%15.4f%10.2f%13.4f\n' % (comma_int(len(x)), ps, pk, chi, st.chi2.sf(chi, 2)))
        ip.set_r(scalars={'N': len(x), 'P_skew': ps, 'P_kurt': pk, 'chi2': chi, 'P_chi2': st.chi2.sf(chi, 2)})
    ip.txt('(chi2 kết hợp chưa áp dụng hiệu chỉnh Royston của Stata)\n')


@command('mvencode', 'mvdecode')
def cmd_mv(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    vl = [v for v in d.expand_varlist(p.body.split() or ['_all']) if not d.isstr(v)]
    m = ip.mark(p)
    spec = p.opts.get('mv')
    if not spec:
        raise StataError('option mv() required', 198)
    for v in vl:
        x = d.cols[v].astype(float).copy()
        if cmd == 'mvencode':
            hit = m & np.isnan(x)
            x[hit] = float(spec.split('=')[-1])
        else:
            vals = parse_numlist(spec.split('=')[0])
            hit = m & np.isin(x, vals)
            x[hit] = np.nan
        if hit.any():
            d.set_col(v, x)
            ip.txt('%12s: %d %s\n' % (abbrev(v, 12), int(hit.sum()), ('missing value generated' if hit.sum() == 1 else 'missing values generated') if cmd == 'mvdecode' else ('missing value recoded' if hit.sum() == 1 else 'missing values recoded')))


@command('mkmat')
def cmd_mkmat(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    vl = d.expand_varlist(p.body.split())
    m = ip.mark(p)
    A = np.column_stack([d.cols[v].astype(float)[m] for v in vl])
    if p.opts.has('matrix'):
        ip.matrices[p.opts.get('matrix').strip()] = E.Matrix(A, colnames=vl)
    else:
        for k, v in enumerate(vl):
            ip.matrices[v] = E.Matrix(A[:, [k]], colnames=[v])


@command('matlist')
def cmd_matlist(ip, cmd, rest, by):
    from .cmds_core import _list_matrix
    name = rest.split(',')[0].strip()
    if name not in ip.matrices:
        raise StataError('matrix %s not found' % name, 111)
    _list_matrix(ip, name, ip.matrices[name])


@command('tsline', 'tsrline')
def cmd_tsline(ip, cmd, rest, by):
    from .interp import COMMANDS
    d = ip.data
    if not d.ts:
        raise StataError('time variable not set, use tsset varname ...', 111)
    COMMANDS['twoway'](ip, 'twoway', 'tsline ' + rest, by)


@command('xtline')
def cmd_xtline(ip, cmd, rest, by):
    from .interp import COMMANDS
    d = ip.data
    if not d.ts or not d.ts.get('panel'):
        raise StataError('panel variable not set; use xtset varname ...', 459)
    pv, tv = d.ts['panel'], d.ts['time']
    p = parse_command(rest)
    ys = d.expand_varlist(p.body.split())
    other = re.sub(r'\boverlay\b', '', p.opts.raw).strip()
    cond = (' if ' + p.if_exp) if p.if_exp else ''
    if p.opts.has('overlay'):
        lv = sorted(set(d.cols[pv][~np.isnan(d.cols[pv].astype(float))].tolist()))
        plots = ' '.join('(line %s %s if %s==%s%s, sort)' % (ys[0], tv, pv, num_to_macro(l), cond.replace(' if ', ' & ')) for l in lv)
        leg = ' '.join('%d "%s"' % (k + 1, d.value_text(pv, l) or num_to_macro(l)) for k, l in enumerate(lv))
        COMMANDS['twoway'](ip, 'twoway', '%s, legend(order(%s)) %s' % (plots, leg, other), by)
    else:
        COMMANDS['twoway'](ip, 'twoway', 'line %s %s%s, sort by(%s) %s' % (' '.join(ys), tv, cond, pv, other), by)
