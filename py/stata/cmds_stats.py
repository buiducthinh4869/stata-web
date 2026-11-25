"""Descriptive statistics commands."""
import math
import re

import numpy as np
from scipy import stats as sps

from . import expr as E
from .cmds_data import _stata_pctile, _valtxt
from .errors import StataError
from .fmt import fmt_num, comma_int, abbrev, g_fmt
from .interp import command, alias
from .syntax import parse_command, split_top, unquote

alias('tabulate', 'ta', 'tab', 'tabu', 'tabul', 'tabula', 'tabulat')
alias('correlate', 'cor', 'corr', 'corre', 'correl', 'correla', 'correlat')


def by_loop(ip, by, fn):
    """Run fn(extra_mask) once per by-group, printing Stata's separators."""
    if not by:
        return fn(np.ones(ip.data.n, dtype=bool))
    d = ip.data
    gid = ip.by_groups(by)[0]
    ng = int(gid.max()) + 1 if len(gid) else 0
    for g in range(ng):
        sel = gid == g
        first = np.flatnonzero(sel)[0]
        hdr = ', '.join(f'{v} = {_valtxt(d, v, d.cols[v][first])}' for v in by)
        ip.txt('\n' + '-' * 79 + f'\n-> {hdr}\n')
        fn(sel)


def _weights(ip, p, d):
    if not p.weight:
        return None
    w = np.asarray(E.eval_expr(p.weight[1], E.Ctx(ip, d)), dtype=float) * np.ones(d.n)
    return w


# ------------------------------------------------------------------ summarize
@command('summarize')
def cmd_summarize(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    vars_ = d.expand_varlist(split_top(p.body)) if p.body.strip() else list(d.vars)
    detail = p.opts.has('detail', 1)
    meanonly = p.opts.has('meanonly', 4)
    sep = p.opts.get('separator', 3)
    sep = int(sep) if sep not in (None, '') else 5
    w = _weights(ip, p, d)

    def run(extra):
        base = ip.mark(p) & extra
        first = True
        for k, v in enumerate(vars_):
            if d.isstr(v):
                x = None
            else:
                x = d.cols[v]
            if x is None:
                ok = np.zeros(0)
                ww = None
            else:
                sel = base & ~np.isnan(x)
                if w is not None:
                    sel &= ~np.isnan(w) & (w != 0)
                ok = x[sel]
                ww = w[sel] if w is not None else None
            res = _sumstats(ok, ww, p.weight[0] if p.weight else None, detail)
            ip.set_r(scalars=res)
            if meanonly:
                continue
            if detail:
                _print_detail(ip, d, v, ok, res)
            else:
                if first:
                    ip.txt('\n    Variable |        Obs        Mean    Std. dev.       Min        Max\n')
                    ip.txt('-------------+---------------------------------------------------------\n')
                    first = False
                elif sep and k % sep == 0:
                    ip.txt('-------------+---------------------------------------------------------\n')
                ip.txt(f'{abbrev(v):>12} |')
                if res['N'] == 0:
                    ip.res(f'{0:>11}\n')
                    continue
                ip.res(f'{comma_int(res["N"]):>11}{g_fmt(res["mean"], 9):>12}{g_fmt(res["sd"], 9):>12}'
                       f'{g_fmt(res["min"], 9):>11}{g_fmt(res["max"], 9):>11}\n')
    by_loop(ip, by, run)


def _sumstats(x, w, wt, detail):
    n = x.size
    if n == 0:
        return {'N': 0, 'sum_w': 0, 'sum': 0}
    if w is None:
        mean = x.mean()
        sd = x.std(ddof=1) if n > 1 else np.nan
        sw = float(n)
        var = sd ** 2 if n > 1 else np.nan
        sm = x.sum()
    else:
        sw = w.sum()
        mean = np.sum(w * x) / sw
        var = np.sum(w * (x - mean) ** 2) / sw * n / (n - 1) if n > 1 else np.nan
        sd = math.sqrt(var) if n > 1 else np.nan
        sm = np.sum(w * x)
    r = {'N': float(n), 'sum_w': float(sw), 'mean': float(mean), 'Var': float(var) if var == var else np.nan,
         'sd': float(sd), 'min': float(x.min()), 'max': float(x.max()), 'sum': float(sm)}
    if detail:
        for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
            r[f'p{q}'] = float(_stata_pctile(x, q))
        if n > 2 and sd > 0:
            m2 = np.mean((x - x.mean()) ** 2)
            r['skewness'] = float(np.mean((x - x.mean()) ** 3) / m2 ** 1.5)
            r['kurtosis'] = float(np.mean((x - x.mean()) ** 4) / m2 ** 2)
        else:
            r['skewness'] = np.nan
            r['kurtosis'] = np.nan
    return r


def _print_detail(ip, d, v, x, r):
    lab = d.varlab.get(v, v)
    ip.txt('\n' + lab.center(61).rstrip() + '\n')
    ip.txt('-' * 61 + '\n')
    srt = np.sort(x)
    sm = list(srt[:4]) + [np.nan] * 4
    lg = list(srt[::-1][:4]) + [np.nan] * 4
    G = lambda z: g_fmt(z, 9)
    ip.txt('      Percentiles      Smallest\n')
    ip.res(f' 1%  {G(r.get("p1")):>11}  {G(sm[0]):>13}\n')
    ip.res(f' 5%  {G(r.get("p5")):>11}  {G(sm[1]):>13}\n')
    ip.res(f'10%  {G(r.get("p10")):>11}  {G(sm[2]):>13}')
    ip.txt(f'       Obs        ')
    ip.res(f'{comma_int(r["N"]):>11}\n')
    ip.res(f'25%  {G(r.get("p25")):>11}  {G(sm[3]):>13}')
    ip.txt(f'       Sum of wgt.')
    ip.res(f'{comma_int(r["sum_w"]):>11}\n\n')
    ip.res(f'50%  {G(r.get("p50")):>11}  {"":>13}')
    ip.txt('       Mean       ')
    ip.res(f'{G(r["mean"]):>11}\n')
    ip.txt(f'{"Largest":>31}       Std. dev.  ')
    ip.res(f'{G(r["sd"]):>11}\n')
    ip.res(f'75%  {G(r.get("p75")):>11}  {G(lg[3]):>13}\n')
    ip.res(f'90%  {G(r.get("p90")):>11}  {G(lg[2]):>13}')
    ip.txt('       Variance   ')
    ip.res(f'{G(r["Var"]):>11}\n')
    ip.res(f'95%  {G(r.get("p95")):>11}  {G(lg[1]):>13}')
    ip.txt('       Skewness   ')
    ip.res(f'{G(r["skewness"]):>11}\n')
    ip.res(f'99%  {G(r.get("p99")):>11}  {G(lg[0]):>13}')
    ip.txt('       Kurtosis   ')
    ip.res(f'{G(r["kurtosis"]):>11}\n')


# ------------------------------------------------------------------ tabulate
@command('tabulate', 'tab1', 'tab2')
def cmd_tabulate(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    vars_ = d.expand_varlist(split_top(p.body))

    def run(extra):
        base = ip.mark(p) & extra
        if p.opts.has('summarize', 2):
            sv = d.resolve(p.opts.get('summarize', 2))
            _tab_summ(ip, d, vars_, sv, base, p)
            return
        if len(vars_) == 1 or cmd == 'tab1':
            for v in vars_:
                _tab1(ip, d, v, base, p)
        else:
            _tab2(ip, d, vars_[0], vars_[1], base, p)
    by_loop(ip, by, run)


def _levels(d, v, x, miss):
    if d.isstr(v):
        vals = sorted(set(x) - ({''} if not miss else set()))
        return vals
    ok = x[~np.isnan(x)]
    lv = sorted(np.unique(ok).tolist())
    if miss and np.isnan(x).any():
        lv.append(np.nan)
    return lv


def _lab(d, v, val, nolabel=False):
    if d.isstr(v):
        return str(val)
    if isinstance(val, float) and np.isnan(val):
        return '.'
    t = None if nolabel else d.value_text(v, val)
    return t if t is not None else g_fmt(val)


def _tab1(ip, d, v, base, p):
    x = d.cols[v][base]
    miss = p.opts.has('missing', 1)
    if not d.isstr(v) and not miss:
        x = x[~np.isnan(x)]
    if d.isstr(v) and not miss:
        x = x[x != '']
    lv = _levels(d, v, x, miss)
    tot = len(x)
    name = d.varlab.get(v, v) if False else v
    labw = max(10, min(30, max([len(_lab(d, v, l, p.opts.has('nolabel'))) for l in lv] + [len(abbrev(v, 30))]) + 1))
    ip.txt('\n' + f'{abbrev(v, labw):>{labw}} |      Freq.     Percent        Cum.\n')
    ip.txt('-' * (labw + 1) + '+-----------------------------------\n')
    cum = 0
    counts = []
    for l in lv:
        if isinstance(l, float) and np.isnan(l):
            c = int(np.isnan(x).sum())
        else:
            c = int(np.sum(x == l))
        counts.append(c)
        cum += c
        ip.txt(f'{_lab(d, v, l, p.opts.has("nolabel"))[:labw]:>{labw}} |')
        ip.res(f'{comma_int(c):>11}{100 * c / tot:>12.2f}{100 * cum / tot:>12.2f}\n')
    ip.txt('-' * (labw + 1) + '+-----------------------------------\n')
    ip.txt(f'{"Total":>{labw}} |')
    ip.res(f'{comma_int(tot):>11}{100.0:>12.2f}\n')
    ip.set_r(scalars={'N': tot, 'r': len(lv)})
    if p.opts.has('generate', 3):
        stub = p.opts.get('generate', 3)
        allx = d.cols[v]
        for k, l in enumerate(lv, 1):
            d.add_var(f'{stub}{k}', np.where(np.isnan(allx) if not d.isstr(v) else allx == '', np.nan,
                                              (allx == l).astype(float)), 'byte')


def _tab2(ip, d, v1, v2, base, p):
    x1 = d.cols[v1]
    x2 = d.cols[v2]
    miss = p.opts.has('missing', 1)
    sel = base.copy()
    if not miss:
        if not d.isstr(v1):
            sel &= ~np.isnan(x1)
        if not d.isstr(v2):
            sel &= ~np.isnan(x2)
    a, b = x1[sel], x2[sel]
    l1 = _levels(d, v1, a, miss)
    l2 = _levels(d, v2, b, miss)
    T = np.zeros((len(l1), len(l2)))
    for i, u in enumerate(l1):
        for j, w in enumerate(l2):
            T[i, j] = np.sum((a == u) & (b == w))
    row = p.opts.has('row', 1)
    col = p.opts.has('column', 3)
    cell = p.opts.has('cell', 2)
    nol = p.opts.has('nolabel')
    labw = max(10, min(20, max([len(_lab(d, v1, l, nol)) for l in l1] + [len(v1)]) + 1))
    cw = max(10, min(14, max([len(_lab(d, v2, l, nol)) for l in l2] + [8]) + 2))
    tot = T.sum()
    rs = T.sum(axis=1)
    cs = T.sum(axis=0)
    hdr_title = abbrev(v2, cw * len(l2))
    ip.txt('\n')
    if row or col or cell:
        ip.txt('+-------------------+\n| Key               |\n|-------------------|\n')
        if True:
            ip.txt('|     frequency     |\n')
        if row:
            ip.txt('|  row percentage   |\n')
        if col:
            ip.txt('| column percentage |\n')
        if cell:
            ip.txt('|  cell percentage  |\n')
        ip.txt('+-------------------+\n\n')
    width = cw * len(l2)
    ip.txt(f'{"":>{labw}} |{hdr_title:^{width}}|\n')
    ip.txt(f'{abbrev(v1, labw):>{labw}} |' + ''.join(f'{_lab(d, v2, l, nol)[:cw - 1]:>{cw}}' for l in l2) + f' |{"Total":>10}\n')
    ip.txt('-' * (labw + 1) + '+' + '-' * width + '-+' + '-' * 10 + '\n')
    for i, u in enumerate(l1):
        ip.txt(f'{_lab(d, v1, u, nol)[:labw]:>{labw}} |')
        ip.res(''.join(f'{comma_int(T[i, j]):>{cw}}' for j in range(len(l2))) + f' |{comma_int(rs[i]):>10}\n')
        if row:
            ip.txt(f'{"":>{labw}} |')
            ip.res(''.join(f'{100 * T[i, j] / rs[i] if rs[i] else 0:>{cw}.2f}' for j in range(len(l2))) + f' |{100.0:>10.2f}\n')
        if col:
            ip.txt(f'{"":>{labw}} |')
            ip.res(''.join(f'{100 * T[i, j] / cs[j] if cs[j] else 0:>{cw}.2f}' for j in range(len(l2))) + f' |{100 * rs[i] / tot:>10.2f}\n')
        if cell:
            ip.txt(f'{"":>{labw}} |')
            ip.res(''.join(f'{100 * T[i, j] / tot:>{cw}.2f}' for j in range(len(l2))) + f' |{100 * rs[i] / tot:>10.2f}\n')
        if (row or col or cell) and i < len(l1) - 1:
            ip.txt(f'{"":>{labw}} |{"":>{width}} |\n' if False else '')
    ip.txt('-' * (labw + 1) + '+' + '-' * width + '-+' + '-' * 10 + '\n')
    ip.txt(f'{"Total":>{labw}} |')
    ip.res(''.join(f'{comma_int(cs[j]):>{cw}}' for j in range(len(l2))) + f' |{comma_int(tot):>10}\n')
    if row:
        ip.txt(f'{"":>{labw}} |')
        ip.res(''.join(f'{100 * cs[j] / tot:>{cw}.2f}' for j in range(len(l2))) + f' |{100.0:>10.2f}\n')
    if col:
        ip.txt(f'{"":>{labw}} |')
        ip.res(''.join(f'{100.0:>{cw}.2f}' for j in range(len(l2))) + f' |{100.0:>10.2f}\n')
    res = {'N': tot, 'r': len(l1), 'c': len(l2)}
    if p.opts.has('chi2', 2) or True:
        E_ = np.outer(rs, cs) / tot if tot else T
        with np.errstate(all='ignore'):
            chi2 = np.nansum((T - E_) ** 2 / E_)
        df = (len(l1) - 1) * (len(l2) - 1)
        res['chi2'] = chi2
        res['p'] = sps.chi2.sf(chi2, df) if df > 0 else np.nan
        if p.opts.has('chi2', 2):
            ip.txt(f'\n{"Pearson chi2(" + str(df) + ") = ":>32}')
            ip.res(f'{chi2:8.4f}')
            ip.txt('   Pr = ')
            ip.res(f'{res["p"]:5.3f}\n')
    ip.set_r(scalars=res)


def _tab_summ(ip, d, vars_, sv, base, p):
    y = d.cols[sv]
    if len(vars_) == 1:
        v = vars_[0]
        x = d.cols[v]
        sel = base & ~np.isnan(x) if not d.isstr(v) else base
        lv = _levels(d, v, x[sel], False)
        labw = max(12, max([len(_lab(d, v, l)) for l in lv] + [len(v)]) + 1)
        ip.txt(f'\n{"":>{labw}} |{"Summary of " + (d.varlab.get(sv) or sv):^37}\n')
        ip.txt(f'{abbrev(v, labw):>{labw}} |        Mean   Std. dev.       Freq.\n')
        ip.txt('-' * (labw + 1) + '+------------------------------------\n')
        for l in lv + ['__total__']:
            s = sel & (x == l) if l != '__total__' else sel
            yy = y[s]
            yy = yy[~np.isnan(yy)]
            if l == '__total__':
                ip.txt('-' * (labw + 1) + '+------------------------------------\n')
                ip.txt(f'{"Total":>{labw}} |')
            else:
                ip.txt(f'{_lab(d, v, l)[:labw]:>{labw}} |')
            ip.res(f'{g_fmt(yy.mean() if yy.size else np.nan, 9):>12}{g_fmt(yy.std(ddof=1) if yy.size > 1 else np.nan, 9):>12}{comma_int(yy.size):>12}\n')
        return
    v1, v2 = vars_[0], vars_[1]
    x1, x2 = d.cols[v1], d.cols[v2]
    sel = base & ~np.isnan(x1) & ~np.isnan(x2)
    l1 = _levels(d, v1, x1[sel], False)
    l2 = _levels(d, v2, x2[sel], False)
    ip.txt(f'\n{"Means, Standard Deviations and Frequencies of " + (d.varlab.get(sv) or sv):^60}\n\n')
    ip.txt(f'{"":>10} |{abbrev(v2, 22):^{11 * len(l2)}}|\n')
    ip.txt(f'{abbrev(v1, 10):>10} |' + ''.join(f'{_lab(d, v2, l)[:10]:>11}' for l in l2) + f' |{"Total":>11}\n')
    ip.txt('-' * 11 + '+' + '-' * (11 * len(l2)) + '-+' + '-' * 11 + '\n')
    for u in l1 + ['__t']:
        cells = []
        for w in l2 + ['__t']:
            s = sel.copy()
            if u != '__t':
                s &= x1 == u
            if w != '__t':
                s &= x2 == w
            yy = y[s]
            yy = yy[~np.isnan(yy)]
            cells.append(yy)
        if u == '__t':
            ip.txt('-' * 11 + '+' + '-' * (11 * len(l2)) + '-+' + '-' * 11 + '\n')
        lab = 'Total' if u == '__t' else _lab(d, v1, u)[:10]
        ip.txt(f'{lab:>10} |')
        ip.res(''.join(f'{g_fmt(c.mean() if c.size else np.nan, 9):>11}' for c in cells[:-1]) + f' |{g_fmt(cells[-1].mean() if cells[-1].size else np.nan, 9):>11}\n')
        ip.txt(f'{"":>10} |')
        ip.res(''.join(f'{g_fmt(c.std(ddof=1) if c.size > 1 else np.nan, 9):>11}' for c in cells[:-1]) + f' |{g_fmt(cells[-1].std(ddof=1) if cells[-1].size > 1 else np.nan, 9):>11}\n')
        ip.txt(f'{"":>10} |')
        ip.res(''.join(f'{comma_int(c.size):>11}' for c in cells[:-1]) + f' |{comma_int(cells[-1].size):>11}\n')


# ------------------------------------------------------------------ tabstat
@command('tabstat')
def cmd_tabstat(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    vars_ = d.expand_varlist(split_top(p.body))
    stats = (p.opts.get('statistics', 2) or p.opts.get('stats') or 'mean').split()
    byv = d.resolve(p.opts.get('by')) if p.opts.has('by') else None
    base = ip.mark(p)

    def calc(x, st):
        x = x[~np.isnan(x)]
        if st in ('n', 'count', 'N'):
            return float(x.size)
        if not x.size:
            return np.nan
        return {'mean': x.mean(), 'sd': x.std(ddof=1) if x.size > 1 else np.nan, 'min': x.min(),
                'max': x.max(), 'sum': x.sum(), 'median': _stata_pctile(x, 50), 'p50': _stata_pctile(x, 50),
                'p25': _stata_pctile(x, 25), 'p75': _stata_pctile(x, 75), 'range': x.max() - x.min(),
                'var': x.var(ddof=1) if x.size > 1 else np.nan, 'iqr': _stata_pctile(x, 75) - _stata_pctile(x, 25),
                'cv': x.std(ddof=1) / x.mean() if x.size > 1 else np.nan,
                'semean': x.std(ddof=1) / np.sqrt(x.size) if x.size > 1 else np.nan}.get(st, np.nan)
    if byv:
        bx = d.cols[byv]
        lv = _levels(d, byv, bx[base], False)
        v = vars_[0]
        ip.txt(f'\nSummary for variables: {" ".join(vars_)}\nGroup variable: {byv} {("(" + d.varlab[byv] + ")") if byv in d.varlab else ""}\n\n')
        ip.txt(f'{abbrev(byv, 8):>8} |' + ''.join(f'{s:>10}' for s in stats) + '\n')
        ip.txt('-' * 9 + '+' + '-' * (10 * len(stats)) + '\n')
        for l in lv:
            s = base & (bx == l)
            ip.txt(f'{_lab(d, byv, l)[:8]:>8} |')
            ip.res(''.join(f'{g_fmt(calc(d.cols[v][s], st), 9):>10}' for st in stats) + '\n')
        ip.txt('-' * 9 + '+' + '-' * (10 * len(stats)) + '\n')
        ip.txt(f'{"Total":>8} |')
        ip.res(''.join(f'{g_fmt(calc(d.cols[v][base], st), 9):>10}' for st in stats) + '\n')
        ip.txt('-' * 9 + '-' * (10 * len(stats) + 1) + '\n')
        return
    ip.txt(f'\n{"Variable":>12} |' + ''.join(f'{s:>10}' for s in stats) + '\n')
    ip.txt('-' * 13 + '+' + '-' * (10 * len(stats)) + '\n')
    for v in vars_:
        ip.txt(f'{abbrev(v):>12} |')
        ip.res(''.join(f'{g_fmt(calc(d.cols[v][base], st), 9):>10}' for st in stats) + '\n')
    ip.txt('-' * 13 + '-' * (10 * len(stats) + 1) + '\n')


# ------------------------------------------------------------------ ttest
@command('ttest')
def cmd_ttest(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    base = ip.mark(p)
    body = p.body.strip()
    m = re.match(r'^(\S+)\s*==\s*(\S+)$', body)
    if m and not p.opts.has('by'):
        v = d.resolve(m.group(1))
        x = d.cols[v][base]
        x = x[~np.isnan(x)]
        try:
            mu0 = float(m.group(2))
            _ttest_one(ip, v, x, mu0)
        except ValueError:
            v2 = d.resolve(m.group(2))
            sel = base & ~np.isnan(d.cols[v]) & ~np.isnan(d.cols[v2])
            diff = d.cols[v][sel] - d.cols[v2][sel]
            _ttest_one(ip, 'diff', diff, 0.0)
        return
    v = d.resolve(body)
    g = d.resolve(p.opts.get('by'))
    x = d.cols[v]
    gv = d.cols[g]
    sel = base & ~np.isnan(x) & ~np.isnan(gv)
    lv = sorted(np.unique(gv[sel]))
    if len(lv) != 2:
        raise StataError('more than 2 groups found, only 2 allowed', 420)
    unequal = p.opts.has('unequal', 3)
    a = x[sel & (gv == lv[0])]
    b = x[sel & (gv == lv[1])]
    n1, n2 = a.size, b.size
    m1, m2 = a.mean(), b.mean()
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    if unequal:
        se = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
        df = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 2 / ((s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1))
    else:
        sp = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
        se = sp * math.sqrt(1 / n1 + 1 / n2)
        df = n1 + n2 - 2
    t = (m1 - m2) / se
    lab1 = _lab(d, g, lv[0])[:8]
    lab2 = _lab(d, g, lv[1])[:8]
    ip.txt(f'\nTwo-sample t test with {"unequal" if unequal else "equal"} variances\n')
    ip.txt('-' * 78 + '\n')
    ip.txt('   Group |     Obs        Mean    Std. err.   Std. dev.   [95% conf. interval]\n')
    ip.txt('---------+' + '-' * 68 + '\n')
    allx = np.concatenate([a, b])
    for lab, xx in ((lab1, a), (lab2, b)):
        _trow(ip, lab, xx)
    ip.txt('---------+' + '-' * 68 + '\n')
    _trow(ip, 'Combined', allx)
    ip.txt('---------+' + '-' * 68 + '\n')
    crit = sps.t.ppf(0.975, df)
    ip.txt(f'{"diff":>8} |')
    ip.res(f'{"":>8}{g_fmt(m1 - m2, 9):>12}{g_fmt(se, 9):>12}{"":>12}{g_fmt(m1 - m2 - crit * se, 9):>12}{g_fmt(m1 - m2 + crit * se, 9):>12}\n')
    ip.txt('-' * 78 + '\n')
    ip.txt(f'    diff = mean({lab1}) - mean({lab2})'.ljust(66) + 't = ')
    ip.res(f'{t:8.4f}\n')
    ip.txt('H0: diff = 0' + ' ' * (40 if not unequal else 28) + ("Satterthwaite's " if unequal else '') + 'degrees of freedom = ')
    ip.res(f'{g_fmt(df, 8):>8}\n\n')
    pl = sps.t.cdf(t, df)
    pu = sps.t.sf(t, df)
    p2 = 2 * sps.t.sf(abs(t), df)
    ip.txt('    Ha: diff < 0                 Ha: diff != 0                 Ha: diff > 0\n')
    ip.txt(' Pr(T < t) = ')
    ip.res(f'{pl:6.4f}')
    ip.txt('         Pr(|T| > |t|) = ')
    ip.res(f'{p2:6.4f}')
    ip.txt('          Pr(T > t) = ')
    ip.res(f'{pu:6.4f}\n')
    ip.set_r(scalars={'N_1': n1, 'N_2': n2, 'mu_1': m1, 'mu_2': m2, 'sd_1': s1, 'sd_2': s2, 'se': se,
                      't': t, 'df_t': df, 'p': p2, 'p_l': pl, 'p_u': pu, 'sd': allx.std(ddof=1), 'level': 95})


def _trow(ip, lab, xx):
    n = xx.size
    m = xx.mean()
    sd = xx.std(ddof=1)
    se = sd / math.sqrt(n)
    crit = sps.t.ppf(0.975, n - 1)
    ip.txt(f'{lab:>8} |')
    ip.res(f'{comma_int(n):>8}{g_fmt(m, 9):>12}{g_fmt(se, 9):>12}{g_fmt(sd, 9):>12}{g_fmt(m - crit * se, 9):>12}{g_fmt(m + crit * se, 9):>12}\n')


def _ttest_one(ip, v, x, mu0):
    n = x.size
    m = x.mean()
    sd = x.std(ddof=1)
    se = sd / math.sqrt(n)
    t = (m - mu0) / se
    df = n - 1
    ip.txt('\nOne-sample t test\n' + '-' * 78 + '\n')
    ip.txt('Variable |     Obs        Mean    Std. err.   Std. dev.   [95% conf. interval]\n')
    ip.txt('---------+' + '-' * 68 + '\n')
    _trow(ip, abbrev(v, 8), x)
    ip.txt('-' * 78 + '\n')
    ip.txt(f'    mean = mean({v})'.ljust(66) + 't = ')
    ip.res(f'{t:8.4f}\n')
    ip.txt(f'H0: mean = {g_fmt(mu0)}'.ljust(52) + 'Degrees of freedom = ')
    ip.res(f'{df:>8}\n')
    ip.set_r(scalars={'N_1': n, 'mu_1': m, 'sd_1': sd, 'se': se, 't': t, 'df_t': df,
                      'p': 2 * sps.t.sf(abs(t), df)})


# ------------------------------------------------------------------ mean
@command('mean')
def cmd_mean(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    vars_ = d.expand_varlist(split_top(p.body))
    base = ip.mark(p)
    for v in vars_:
        base &= ~np.isnan(d.cols[v])
    over = d.resolve(p.opts.get('over')) if p.opts.has('over') else None
    cells = []
    if over:
        base &= ~np.isnan(d.cols[over])
        lv = sorted(np.unique(d.cols[over][base]))
        for v in vars_:
            for l in lv:
                cells.append((v, l, base & (d.cols[over] == l)))
    else:
        for v in vars_:
            cells.append((v, None, base))
    N = int(base.sum())
    b, se, names = [], [], []
    for v, l, s in cells:
        x = d.cols[v][s]
        b.append(x.mean())
        se.append(x.std(ddof=1) / math.sqrt(x.size) if x.size > 1 else np.nan)
        names.append(f'c.{v}@{g_fmt(l)}.{over}' if over else v)
    df = N - 1
    crit = sps.t.ppf(0.975, df)
    ip.txt(f'\nMean estimation{"Number of obs = ":>41}')
    ip.res(f'{comma_int(N)}\n\n')
    ip.txt('-' * 63 + '\n')
    ip.txt('              |       Mean   Std. err.     [95% conf. interval]\n')
    ip.txt('--------------+' + '-' * 48 + '\n')
    prev = None
    for (v, l, s), bb, ss in zip(cells, b, se):
        if over:
            if v != prev:
                ip.txt(f'{("c." + v + "@" + over)[:13]:<13} |\n')
                prev = v
            lab = _lab(d, over, l)[:12]
            ip.txt(f'{lab:>13} |')
        else:
            ip.txt(f'{abbrev(v, 13):>13} |')
        ip.res(f'{g_fmt(bb, 9):>11}{g_fmt(ss, 9):>12}{g_fmt(bb - crit * ss, 9):>14}{g_fmt(bb + crit * ss, 9):>12}\n')
    ip.txt('-' * 63 + '\n')
    b = np.array(b)
    se = np.array(se)
    tab = np.vstack([b, se, b / se, 2 * sps.t.sf(np.abs(b / se), df), b - crit * se, b + crit * se,
                     np.full_like(b, df), np.full_like(b, crit), np.zeros_like(b)])
    T = E.Matrix(tab, ['b', 'se', 't', 'pvalue', 'll', 'ul', 'df', 'crit', 'eform'], names)
    ip.set_r(matrices={'table': T})
    ip.set_estimation({'names': names, 'b': b, 'V': np.diag(se ** 2), 'sample': base,
                       'scalars': {'N': N, 'df_r': df}, 'macros': {'cmd': 'mean'}})
    ip.r.matrices['table'] = T


# ------------------------------------------------------------------ correlate / pwcorr
@command('correlate', 'pwcorr')
def cmd_correlate(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    vars_ = d.expand_varlist(split_top(p.body))
    base = ip.mark(p)
    k = len(vars_)
    X = np.column_stack([d.cols[v] for v in vars_])
    if cmd == 'correlate':
        sel = base & ~np.isnan(X).any(axis=1)
        C = np.corrcoef(X[sel].T)
        ip.txt(f'(obs={comma_int(int(sel.sum()))})\n\n')
        _corr_table(ip, vars_, C, None, None)
        ip.set_r(scalars={'N': int(sel.sum()), 'rho': C[0, 1] if k > 1 else 1.0},
                 matrices={'C': E.Matrix(C, vars_, vars_)})
        return
    C = np.eye(k)
    P = np.zeros((k, k))
    Nn = np.zeros((k, k))
    for i in range(k):
        for j in range(i):
            s = base & ~np.isnan(X[:, i]) & ~np.isnan(X[:, j])
            r = np.corrcoef(X[s, i], X[s, j])[0, 1]
            n = s.sum()
            t = r * math.sqrt((n - 2) / (1 - r * r)) if abs(r) < 1 else np.inf
            C[i, j] = C[j, i] = r
            P[i, j] = P[j, i] = 2 * sps.t.sf(abs(t), n - 2)
            Nn[i, j] = n
    star = float(p.opts.get('star')) if p.opts.has('star') else None
    _corr_table(ip, vars_, C, P if (star or p.opts.has('sig')) else None, star, sig=p.opts.has('sig'))
    ip.set_r(matrices={'C': E.Matrix(C, vars_, vars_)})


def _corr_table(ip, vars_, C, P, star, sig=False):
    k = len(vars_)
    for j0 in range(0, k, 6):
        js = range(j0, min(k, j0 + 6))
        ip.txt(f'{"":>13}|' + ''.join(f'{abbrev(vars_[j], 8):>9}' for j in js) + '\n')
        ip.txt('-' * 13 + '+' + '-' * (9 * len(js)) + '\n')
        for i in range(j0, k):
            ip.txt(f'{abbrev(vars_[i]):>12} |')
            cells = ''
            for j in js:
                if j > i:
                    break
                s = f'{C[i, j]:.4f}'
                s = s.replace('0.', '.') if not s.startswith('-') else '-' + s[1:].replace('0.', '.', 1)
                if i == j:
                    s = '1.0000'
                st = '*' if (star and P is not None and i != j and P[i, j] < star) else ' '
                cells += f'{s:>8}{st}'
            ip.res(cells.rstrip() + '\n')
            if sig and P is not None:
                ip.txt(f'{"":>12} |')
                ip.res(''.join(f'{P[i, j]:>8.4f} ' if j < i else '         ' for j in js if j <= i) + '\n')
        ip.txt('\n')
