"""Extra data-management commands: decode, split, tsfill, fillin, contract, winsor2, extra egen helpers."""
import re

import numpy as np

from .data import Data
from .errors import StataError
from .fmt import comma_int, g_fmt
from .interp import command, alias
from .syntax import parse_command, unquote, quoted_strings
from .cmds_more import _pctiles

alias('decode', 'dec', 'deco', 'decod')


def _is_missing(d, v, x):
    return (x == '') if d.isstr(v) else np.isnan(x.astype(float))


# ------------------------------------------------------------------ decode
@command('decode')
def cmd_decode(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    v = d.resolve(p.body.strip())
    nm = p.opts.get('generate', 3)
    if not nm:
        raise StataError('option generate() required', 198)
    if d.isstr(v):
        raise StataError(f'{v} is not a numeric variable', 108)
    lab = d.vallab.get(v)
    if not lab:
        raise StataError(f'{v} not labeled', 182)
    tab = d.labels.get(lab, {})
    maxlen = p.opts.get('maxlength', 4)
    maxlen = int(maxlen) if maxlen else None
    mask = ip.mark(p, d)
    x = d.cols[v].astype(float)
    out = []
    for i in range(d.n):
        s = ''
        if mask[i] and not np.isnan(x[i]):
            s = tab.get(float(x[i]), '')
        if maxlen:
            s = s[:maxlen]
        out.append(s)
    w = max([len(s) for s in out] + [1])
    d.add_var(nm, np.array(out, dtype=object), f'str{w}')
    d.varlab[nm] = d.varlab.get(v, '')
    nmiss = sum(1 for s in out if s == '')
    if nmiss:
        ip.txt(f'({comma_int(nmiss)} missing value{"s" if nmiss != 1 else ""} generated)\n')


# ------------------------------------------------------------------ split
@command('split')
def cmd_split(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    v = d.resolve(p.body.strip())
    if not d.isstr(v):
        raise StataError(f'{v} not a string variable', 109)
    stub = p.opts.get('generate', 3) or v
    praw = p.opts.get('parse', 1)
    seps = [' ']
    if praw is not None and praw.strip():
        qs = quoted_strings(praw)
        seps = qs if qs else praw.split()
        seps = ['' if s == '' else s for s in seps] or [' ']
        seps = [' ' if s == '' else s for s in seps]
    limit = p.opts.get('limit', 1)
    limit = int(limit) if limit else None
    notrim = p.opts.has('notrim', 5)
    mask = ip.mark(p, d)
    pat = '|'.join(re.escape(s) for s in sorted(seps, key=len, reverse=True))
    parts = []
    for i in range(d.n):
        s = str(d.cols[v][i]) if mask[i] else ''
        if not notrim:
            s = s.strip()
        if s == '':
            parts.append([])
            continue
        pieces = re.split(pat, s)
        if seps == [' ']:
            pieces = [x for x in pieces if x != '']
        if not notrim:
            pieces = [x.strip() for x in pieces]
        if limit:
            pieces = pieces[:limit]
        parts.append(pieces)
    k = max([len(x) for x in parts] + [0])
    if limit:
        k = min(k, limit)
    newvars = [f'{stub}{j + 1}' for j in range(k)]
    for nv in newvars:
        if d.has(nv):
            raise StataError(f'variable {nv} already defined', 110)
    for j, nv in enumerate(newvars):
        col = np.array([x[j] if j < len(x) else '' for x in parts], dtype=object)
        w = max([len(s) for s in col] + [1])
        d.add_var(nv, col, f'str{w}')
    if newvars:
        ip.txt('variables created as string: \n')
        _print_names(ip, newvars)
    if p.opts.has('destring', 4) and newvars:
        from . import expr as E
        conv = []
        for nv in newvars:
            vals = [str(s) for s in d.cols[nv]]
            nums = []
            ok = True
            for s in vals:
                if s.strip() == '':
                    nums.append(np.nan)
                    continue
                r = E._real(s.strip())
                if np.isnan(r):
                    ok = False
                    break
                nums.append(r)
            if ok:
                arr = np.array(nums, dtype=float)
                typ = 'byte'
                fin = arr[~np.isnan(arr)]
                if fin.size and np.any(fin != np.round(fin)):
                    typ = 'double'
                elif fin.size and (fin.min() < -127 or fin.max() > 100):
                    typ = 'int' if fin.min() >= -32767 and fin.max() <= 32740 else 'long'
                d.types[nv] = typ
                d.cols[nv] = arr
                from .data import default_format
                d.formats[nv] = '%10.0g' if typ == 'double' else default_format(typ)
                conv.append((nv, typ))
                ip.txt(f'{nv}: all characters numeric; ')
                ip.res('replaced ')
                ip.txt(f'as {typ}\n')
            else:
                ip.txt(f'{nv}: contains nonnumeric characters; ')
                ip.res('no replace\n')
    ip.set_r(scalars={'nvars': len(newvars)}, macros={'varlist': ' '.join(newvars)})


def _print_names(ip, names):
    line = ''
    for n in names:
        cell = f'{n:<12}' if len(n) < 12 else n + ' '
        if len(line) + len(cell) > 78:
            ip.res(line.rstrip() + '\n')
            line = ''
        line += cell
    if line:
        ip.res(line.rstrip() + '\n')


# ------------------------------------------------------------------ tsfill
@command('tsfill')
def cmd_tsfill(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    if not d.ts or not d.ts.get('time'):
        raise StataError('time variable not set, use tsset varname ...', 111)
    panel, tvar = d.ts.get('panel'), d.ts['time']
    delta = d.ts.get('delta', 1) or 1
    t = d.cols[tvar].astype(float)
    if panel:
        g = d.cols[panel].astype(float)
    else:
        g = np.zeros(d.n)
    ok = ~np.isnan(t) & ~np.isnan(g)
    have = set(zip(g[ok].tolist(), t[ok].tolist()))
    add = []
    if p.opts.has('full'):
        tmin, tmax = t[ok].min(), t[ok].max()
        grid = np.arange(tmin, tmax + delta / 2, delta)
        for gg in sorted(set(g[ok].tolist())):
            for tt in grid:
                if (gg, float(tt)) not in have:
                    add.append((gg, float(tt)))
    else:
        for gg in sorted(set(g[ok].tolist())):
            ts_ = t[ok & (g == gg)]
            for tt in np.arange(ts_.min(), ts_.max() + delta / 2, delta):
                if (gg, float(tt)) not in have:
                    add.append((gg, float(tt)))
    if not add:
        return
    n0 = d.n
    _append_rows(d, len(add))
    for k, (gg, tt) in enumerate(add):
        if panel:
            d.cols[panel][n0 + k] = gg
        d.cols[tvar][n0 + k] = tt
    d.sort([x for x in (panel, tvar) if x])


def _append_rows(d, k):
    """Append k empty (missing) observations."""
    for v in d.vars:
        if d.isstr(v):
            d.cols[v] = np.concatenate([d.cols[v], np.array([''] * k, dtype=object)])
        else:
            d.cols[v] = np.concatenate([d.cols[v].astype(float) if d.cols[v].dtype.kind != 'f' else d.cols[v],
                                        np.full(k, np.nan)])
    d.n += k
    d.sortedby = []


# ------------------------------------------------------------------ fillin
@command('fillin')
def cmd_fillin(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    vars_ = d.expand_varlist(p.body.split())
    if len(vars_) < 2:
        raise StataError('too few variables specified', 102)
    if d.has('_fillin'):
        raise StataError('variable _fillin already defined', 110)
    keyf = lambda v, x: x if d.isstr(v) else (np.inf if np.isnan(x) else float(x))  # noqa: E731
    levels = [sorted(set(keyf(v, x) for x in d.cols[v]), key=lambda z: (isinstance(z, str), z)) for v in vars_]
    have = set(tuple(keyf(v, d.cols[v][i]) for v in vars_) for i in range(d.n))
    import itertools
    add = [c for c in itertools.product(*levels) if c not in have]
    d.add_var('_fillin', np.zeros(d.n), 'byte')
    n0 = d.n
    if add:
        _append_rows(d, len(add))
        for k, c in enumerate(add):
            for v, val in zip(vars_, c):
                d.cols[v][n0 + k] = (np.nan if val == np.inf else val) if not d.isstr(v) else val
            d.cols['_fillin'][n0 + k] = 1
    d.sort(vars_)


# ------------------------------------------------------------------ contract
@command('contract')
def cmd_contract(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    vars_ = d.expand_varlist(p.body.split())
    mask = ip.mark(p, d)
    if p.opts.has('nomiss', 5):
        for v in vars_:
            mask &= ~_is_missing(d, v, d.cols[v])
    w = np.ones(d.n)
    if p.weight:
        from . import expr as E
        w = np.asarray(E.eval_expr(p.weight[1], E.Ctx(ip, d)), dtype=float) * np.ones(d.n)
    fname = p.opts.get('freq') or '_freq'
    cfname = p.opts.get('cfreq')
    pname = p.opts.get('percent', 4)
    cpname = p.opts.get('cpercent', 5)
    for nm in [fname, cfname, pname, cpname]:
        if nm and nm in vars_:
            raise StataError(f'{nm} already defined', 110)
    keyf = lambda v, x: x if d.isstr(v) else (np.inf if np.isnan(x) else float(x))  # noqa: E731
    counts = {}
    for i in np.flatnonzero(mask):
        k = tuple(keyf(v, d.cols[v][i]) for v in vars_)
        counts[k] = counts.get(k, 0) + w[i]
    if p.opts.has('zero'):
        import itertools
        levels = [sorted(set(k[j] for k in counts), key=lambda z: (isinstance(z, str), z)) for j in range(len(vars_))]
        for c in itertools.product(*levels):
            counts.setdefault(c, 0)
    keys = sorted(counts, key=lambda k: tuple((isinstance(z, str), z) for z in k))
    nd = Data()
    nd.set_obs(len(keys))
    for j, v in enumerate(vars_):
        col = [k[j] for k in keys]
        if d.isstr(v):
            nd.add_var(v, np.array(col, dtype=object), d.types[v])
        else:
            nd.add_var(v, np.array([np.nan if z == np.inf else z for z in col], dtype=float), d.types[v])
        nd.formats[v] = d.formats.get(v, nd.formats[v])
        if v in d.varlab:
            nd.varlab[v] = d.varlab[v]
        if v in d.vallab:
            nd.vallab[v] = d.vallab[v]
    f = np.array([counts[k] for k in keys], dtype=float)
    ftype = 'float' if p.opts.has('float') else ('long' if f.size and f.max() > 32740 else ('int' if f.size and f.max() > 100 else 'byte'))
    nd.add_var(fname, f, ftype)
    nd.varlab[fname] = 'Frequency'
    tot = f.sum()
    fmt = p.opts.get('format')
    if cfname:
        nd.add_var(cfname, np.cumsum(f), ftype if ftype != 'byte' else 'int')
    if pname:
        nd.add_var(pname, 100 * f / tot if tot else f * np.nan, 'float', fmt=fmt or '%8.2f')
    if cpname:
        nd.add_var(cpname, 100 * np.cumsum(f) / tot if tot else f * np.nan, 'float', fmt=fmt or '%8.2f')
    nd.labels = dict(d.labels)
    ip.data = nd
    ip.data.sortedby = list(vars_)


# ------------------------------------------------------------------ winsor2 (SSC, Lian Yujun)
@command('winsor2')
def cmd_winsor2(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    vars_ = d.expand_varlist(p.body.split())
    trim = p.opts.has('trim')
    replace = p.opts.has('replace')
    suffix = p.opts.get('suffix')
    if suffix and replace:
        raise StataError('suffix() and replace may not be combined', 198)
    if not suffix:
        suffix = '_tr' if trim else '_w'
    cuts = p.opts.get('cuts')
    lo, hi = (1.0, 99.0)
    if cuts:
        c = [float(x) for x in cuts.replace(',', ' ').split()]
        if len(c) != 2 or not (0 <= c[0] <= c[1] <= 100):
            raise StataError('cuts() must contain two numbers between 0 and 100', 198)
        lo, hi = c
    byv = list(by or [])
    if p.opts.get('by'):
        byv += d.expand_varlist(p.opts.get('by').split())
    mask = ip.mark(p, d)
    if byv:
        keys = list(zip(*[d.cols[v] for v in byv]))
        groups = {}
        for i, k in enumerate(keys):
            groups.setdefault(k, []).append(i)
        glist = [np.array(ix) for ix in groups.values()]
    else:
        glist = [np.arange(d.n)]
    for v in vars_:
        if d.isstr(v):
            raise StataError(f'{v}: string variables not allowed', 109)
        x = d.cols[v].astype(float)
        out = x.copy()
        for ix in glist:
            sel = ix[mask[ix] & ~np.isnan(x[ix])]
            if sel.size == 0:
                continue
            xs = x[sel]
            ql, qh = _pctiles(xs, np.ones(xs.size), [lo, hi])
            if trim:
                bad = (xs < ql) | (xs > qh)
                vals = xs.copy()
                vals[bad] = np.nan
            else:
                vals = np.clip(xs, ql, qh)
            out[sel] = vals
        out[~mask] = x[~mask] if replace else np.nan
        if replace:
            d.set_col(v, out) if d.types[v] in ('float', 'double') else _promote(d, v, out)
        else:
            nv = v + suffix
            if d.has(nv):
                raise StataError(f'{nv} already defined', 110)
            d.add_var(nv, out, 'double' if d.types[v] == 'double' else 'float')
            lab = d.varlab.get(v, v)
            if p.opts.has('label'):
                d.varlab[nv] = (f'{lab}-Trimmed' if trim else f'{lab}-Winsorized') + f'(p{g_fmt(lo)},p{g_fmt(hi)})'
            else:
                d.varlab[nv] = lab if v in d.varlab else ''


def _promote(d, v, out):
    if np.all(np.isnan(out) | (out == np.round(out))):
        d.set_col(v, out)
    else:
        d.types[v] = 'float'
        d.set_col(v, out)
        d.formats[v] = '%9.0g'
