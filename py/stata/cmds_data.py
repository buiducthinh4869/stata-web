"""Data management commands."""
import io
import os
import re

import numpy as np

from . import expr as E
from .data import Data, read_dta, write_dta, store_cast, promote_type, is_str_type, default_format, check_name
from .errors import StataError
from .fmt import fmt_num, comma_int, abbrev, g_fmt
from .interp import command, alias
from .syntax import parse_command, split_top, unquote, Options, quoted_strings, parse_numlist

alias('generate', 'g', 'ge', 'gen', 'gene', 'gener', 'genera', 'generat')
alias('summarize', 'su', 'sum', 'summ', 'summa', 'summar', 'summari', 'summariz')
alias('describe', 'd', 'de', 'des', 'desc', 'descr', 'descri', 'describ')
alias('list', 'l', 'li', 'lis')
alias('use', 'u', 'us')
alias('save', 'sa', 'sav')
alias('count', 'cou', 'coun')
alias('rename', 'ren', 'rena', 'renam')
alias('format', 'form', 'forma')
alias('codebook', 'codeb', 'codebo', 'codeboo')
alias('browse', 'br', 'bro', 'brow', 'brows')
alias('tsset', 'tss')


def _require_data(ip):
    if ip.data.n == 0 and not ip.data.vars:
        raise StataError('no variables defined', 111)


def _by_ctx(ip, by):
    if not by:
        return None
    return ip.by_groups(by)


# ------------------------------------------------------------------ generate / replace
_GEN_RE = re.compile(r'^\s*(?:(byte|int|long|float|double|str\d+|strL|str)\s+)?([A-Za-z_]\w*)(?::(\w+))?\s*=(.*)$', re.S)


@command('generate')
def cmd_generate(ip, cmd, rest, by):
    p = parse_command(rest)
    m = _GEN_RE.match(p.body)
    if not m:
        raise StataError('invalid syntax', 198)
    typ, name, lblname, exp = m.group(1), m.group(2), m.group(3), m.group(4).strip()
    d = ip.data
    check_name(name)
    if name in d.cols:
        raise StataError(f'variable {name} already defined', 110)
    byc = _by_ctx(ip, by)
    val = E.eval_expr(exp, E.Ctx(ip, d, byc))
    mask = ip.mark(p, d, byc)
    is_s = isinstance(val, str) or (isinstance(val, np.ndarray) and val.dtype == object)
    if typ and typ.startswith('str') and not is_s:
        raise StataError('type mismatch', 109)
    if is_s:
        arr = np.broadcast_to(np.asarray(val, dtype=object), (d.n,)).copy() if d.n else np.array([], dtype=object)
        arr = np.where(mask, arr, '')
        arr = np.array([str(x) for x in arr], dtype=object)
        w = max([len(x) for x in arr] + [1])
        if typ and typ != 'str' and typ != 'strL':
            w2 = int(typ[3:])
            arr = np.array([x[:w2] for x in arr], dtype=object)
            w = w2
        d.add_var(name, arr, f'str{w}')
        nmiss = int(np.sum(arr == ''))
    else:
        arr = np.broadcast_to(np.asarray(val, dtype=float), (d.n,)).copy() if d.n else np.array([])
        arr[~mask] = np.nan
        t = typ or 'float'
        d.add_var(name, arr, t)
        nmiss = int(np.isnan(d.cols[name]).sum())
    if lblname:
        d.vallab[name] = lblname
    if nmiss:
        ip.txt(f'({comma_int(nmiss)} missing value{"s" if nmiss != 1 else ""} generated)\n')


@command('replace')
def cmd_replace(ip, cmd, rest, by):
    p = parse_command(rest)
    m = re.match(r'^\s*([A-Za-z_]\w*)\s*=(.*)$', p.body, re.S)
    if not m:
        raise StataError('invalid syntax', 198)
    d = ip.data
    name = d.resolve(m.group(1), allow_abbrev=False)
    byc = _by_ctx(ip, by)
    val = E.eval_expr(m.group(2).strip(), E.Ctx(ip, d, byc))
    mask = ip.mark(p, d, byc)
    old = d.cols[name]
    if d.isstr(name):
        if not (isinstance(val, str) or (isinstance(val, np.ndarray) and val.dtype == object)):
            raise StataError('type mismatch', 109)
        new = np.broadcast_to(np.asarray(val, dtype=object), (d.n,))
        res = np.where(mask, new, old)
        res = np.array([str(x) for x in res], dtype=object)
        changes = int(np.sum(res != old))
        d.cols[name] = res
        w = max([len(x) for x in res] + [1])
        cur = d.types[name]
        if cur != 'strL':
            cw = int(cur[3:]) if cur[3:].isdigit() else 1
            if w > cw:
                d.types[name] = f'str{w}'
        ip.txt(f'({comma_int(changes)} real change{"s" if changes != 1 else ""} made)\n')
        return
    if isinstance(val, str) or (isinstance(val, np.ndarray) and val.dtype == object):
        raise StataError('type mismatch', 109)
    new = np.broadcast_to(np.asarray(val, dtype=float), (d.n,))
    res = np.where(mask, new, old)
    t = d.types[name]
    nt = promote_type(t, res[mask]) if mask.any() else t
    if nt != t:
        ip.txt(f'variable {name} was {t} now {nt}\n')
        d.types[name] = nt
    res = store_cast(res, d.types[name])
    a = np.where(np.isnan(old), np.inf, old)
    b = np.where(np.isnan(res), np.inf, res)
    changed = a != b
    changes = int(changed.sum())
    tomiss = int(np.sum(changed & np.isnan(res)))
    d.cols[name] = res
    if tomiss:
        ip.txt(f'({comma_int(changes)} real change{"s" if changes != 1 else ""} made, {comma_int(tomiss)} to missing)\n')
    else:
        ip.txt(f'({comma_int(changes)} real change{"s" if changes != 1 else ""} made)\n')


# ------------------------------------------------------------------ egen
@command('egen')
def cmd_egen(ip, cmd, rest, by):
    p = parse_command(rest)
    m = re.match(r'^\s*(?:(byte|int|long|float|double|str\d+)\s+)?([A-Za-z_]\w*)\s*=\s*(\w+)\((.*)\)\s*$', p.body, re.S)
    if not m:
        raise StataError('invalid egen syntax', 198)
    typ, name, fn, arg = m.group(1), m.group(2), m.group(3), m.group(4)
    d = ip.data
    byv = list(by or [])
    if p.opts.has('by'):
        byv += d.expand_varlist(p.opts.get('by').split())
    mask = ip.mark(p, d)
    n = d.n
    if byv:
        order = np.lexsort([np.where(np.isnan(d.cols[v]), np.inf, d.cols[v]) if not d.isstr(v)
                            else np.array([str(x) for x in d.cols[v]]) for v in reversed(byv)]) \
            if not any(d.isstr(v) for v in byv) else np.array(sorted(range(n), key=lambda i: tuple(str(d.cols[v][i]) for v in byv)))
        keys = [d.cols[v][order] for v in byv]
        newg = np.zeros(n, dtype=bool)
        if n:
            newg[0] = True
        for k in keys:
            kk = np.where(np.isnan(k.astype(float)), np.inf, k.astype(float)) if k.dtype != object else k
            newg[1:] |= kk[1:] != kk[:-1]
        gsorted = np.cumsum(newg) - 1
        gid = np.empty(n, dtype=int)
        gid[order] = gsorted
    else:
        gid = np.zeros(n, dtype=int)
    ngroups = int(gid.max()) + 1 if n else 0
    res = np.full(n, np.nan)
    fn_l = fn.lower()

    def xvals(a):
        return np.asarray(E.eval_expr(a, E.Ctx(ip, d)), dtype=float) * np.ones(n)

    if fn_l in ('mean', 'sd', 'total', 'sum', 'count', 'max', 'min', 'median', 'p50', 'std', 'iqr', 'mad', 'pctile', 'skew', 'kurt', 'mode', 'first', 'last'):
        x = xvals(arg)
        valid = mask & ~np.isnan(x)
        pct = None
        if fn_l == 'pctile':
            pct = float(p.opts.get('p', 50) or 50)
        for g in range(ngroups):
            sel = (gid == g)
            vv = x[sel & valid]
            if fn_l == 'mean':
                v = vv.mean() if vv.size else np.nan
            elif fn_l == 'sd':
                v = vv.std(ddof=1) if vv.size > 1 else np.nan
            elif fn_l in ('total', 'sum'):
                v = vv.sum()
            elif fn_l == 'count':
                v = float(vv.size)
            elif fn_l == 'max':
                v = vv.max() if vv.size else np.nan
            elif fn_l == 'min':
                v = vv.min() if vv.size else np.nan
            elif fn_l in ('median', 'p50'):
                v = _stata_pctile(vv, 50) if vv.size else np.nan
            elif fn_l == 'pctile':
                v = _stata_pctile(vv, pct) if vv.size else np.nan
            elif fn_l == 'iqr':
                v = _stata_pctile(vv, 75) - _stata_pctile(vv, 25) if vv.size else np.nan
            elif fn_l == 'std':
                mu = vv.mean() if vv.size else np.nan
                sd = vv.std(ddof=1) if vv.size > 1 else np.nan
                res[sel] = (x[sel] - mu) / sd
                continue
            else:
                v = np.nan
            res[sel] = v
        if fn_l not in ('std',):
            res[~mask] = np.nan
        else:
            res[~valid] = np.nan
    elif fn_l in ('group',):
        vars_ = d.expand_varlist(arg.split())
        keys = []
        for v in vars_:
            keys.append([str(x) if d.isstr(v) else (np.inf if np.isnan(x) else x) for x in d.cols[v]])
        tuples = list(zip(*keys))
        ok = mask.copy()
        for v in vars_:
            if not d.isstr(v):
                ok &= ~np.isnan(d.cols[v])
        uniq = sorted(set(t for t, o in zip(tuples, ok) if o))
        idx = {t: i + 1 for i, t in enumerate(uniq)}
        res = np.array([idx[t] if o else np.nan for t, o in zip(tuples, ok)], dtype=float)
        if not typ:
            typ = 'float'
    elif fn_l == 'tag':
        vars_ = d.expand_varlist(arg.split())
        seen = set()
        res = np.zeros(n)
        for i in range(n):
            if not mask[i]:
                continue
            t = tuple(d.cols[v][i] if d.isstr(v) else float(d.cols[v][i]) for v in vars_)
            if t not in seen:
                seen.add(t)
                res[i] = 1
        typ = typ or 'byte'
    elif fn_l in ('rowmean', 'rowtotal', 'rowmax', 'rowmin', 'rowsd', 'rownonmiss', 'rowmiss'):
        vars_ = d.expand_varlist(arg.split())
        M = np.column_stack([d.cols[v] for v in vars_])
        with np.errstate(all='ignore'), np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            if fn_l == 'rowmean':
                res = np.nanmean(M, axis=1)
            elif fn_l == 'rowtotal':
                res = np.nansum(M, axis=1)
            elif fn_l == 'rowmax':
                res = np.nanmax(M, axis=1)
            elif fn_l == 'rowmin':
                res = np.nanmin(M, axis=1)
            elif fn_l == 'rowsd':
                res = np.nanstd(M, axis=1, ddof=1)
            elif fn_l == 'rownonmiss':
                res = (~np.isnan(M)).sum(axis=1).astype(float)
            else:
                res = np.isnan(M).sum(axis=1).astype(float)
        res[~mask] = np.nan
    elif fn_l == 'rank':
        x = xvals(arg)
        from scipy.stats import rankdata
        for g in range(ngroups):
            sel = (gid == g) & mask & ~np.isnan(x)
            if sel.any():
                res[sel] = rankdata(x[sel])
    elif fn_l == 'cut':
        x = xvals(arg)
        at = parse_numlist(p.opts.get('at'))
        res = np.full(n, np.nan)
        for k in range(len(at) - 1):
            sel = (x >= at[k]) & (x < at[k + 1])
            res[sel] = at[k]
    else:
        raise StataError(f'unknown egen function {fn}()', 133)
    d.add_var(name, res, typ or 'float')
    nmiss = int(np.isnan(d.cols[name]).sum())
    if nmiss:
        ip.txt(f'({comma_int(nmiss)} missing value{"s" if nmiss != 1 else ""} generated)\n')


def _stata_pctile(x, p):
    """Stata's default (altdef off) percentile definition."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0:
        return np.nan
    nq = n * p / 100.0
    i = int(np.floor(nq))
    if nq > i:
        return x[min(i, n - 1)]
    if i == 0:
        return x[0]
    if i >= n:
        return x[-1]
    return (x[i - 1] + x[i]) / 2.0


# ------------------------------------------------------------------ drop / keep
@command('drop')
def cmd_drop(ip, cmd, rest, by):
    _dropkeep(ip, rest, by, keep=False)


@command('keep')
def cmd_keep(ip, cmd, rest, by):
    _dropkeep(ip, rest, by, keep=True)


def _dropkeep(ip, rest, by, keep):
    p = parse_command(rest)
    d = ip.data
    if p.if_exp is not None or p.in_range is not None:
        if p.body.strip():
            raise StataError('invalid syntax', 198)
        byc = _by_ctx(ip, by)
        m = ip.mark(p, d, byc)
        if keep:
            nd = int((~m).sum())
            d.keep_obs(m)
        else:
            nd = int(m.sum())
            d.keep_obs(~m)
        ip.txt(f'({comma_int(nd)} observation{"s" if nd != 1 else ""} deleted)\n')
        return
    names = d.expand_varlist(split_top(p.body))
    if keep:
        drop = [v for v in d.vars if v not in names]
    else:
        drop = list(dict.fromkeys(names))
    d.drop_vars(drop)


@command('rename')
def cmd_rename(ip, cmd, rest, by):
    r = rest.strip()
    m = re.match(r'^\((.*?)\)\s*\((.*?)\)$', r)
    d = ip.data
    if m:
        olds = m.group(1).split()
        news = m.group(2).split()
        for o, nw in zip(olds, news):
            d.rename(d.resolve(o), nw)
        return
    parts = r.split()
    d.rename(d.resolve(parts[0]), parts[1])


@command('order')
def cmd_order(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    names = d.expand_varlist(split_top(p.body))
    others = [v for v in d.vars if v not in names]
    if p.opts.has('last'):
        d.vars = others + names
    else:
        d.vars = names + others


@command('sort')
def cmd_sort(ip, cmd, rest, by):
    p = parse_command(rest)
    names = ip.data.expand_varlist(split_top(p.body))
    ip.data.sort(names)


@command('gsort')
def cmd_gsort(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    toks = split_top(p.body)
    arrays = []
    for t in reversed(toks):
        desc = t.startswith('-')
        v = d.resolve(t.lstrip('+-'))
        a = d.cols[v]
        if d.isstr(v):
            from scipy.stats import rankdata
            r = rankdata([str(x) for x in a], method='dense').astype(float)
        else:
            r = np.where(np.isnan(a), np.inf, a)
        arrays.append(-r if desc else r)
    idx = np.lexsort(arrays)
    d.reorder(idx)
    d.sortedby = []


@command('expand')
def cmd_expand(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    k = np.asarray(E.eval_expr(p.body, E.Ctx(ip, d)), dtype=float) * np.ones(d.n)
    mask = ip.mark(p)
    k = np.where(mask & ~np.isnan(k), np.maximum(np.floor(k), 1), 1).astype(int)
    extra = np.repeat(np.arange(d.n), k - 1)
    idx = np.concatenate([np.arange(d.n), extra])
    for v in d.vars:
        d.cols[v] = d.cols[v][idx]
    added = len(extra)
    d.n = len(idx)
    d.sortedby = []
    ip.txt(f'({comma_int(added)} observation{"s" if added != 1 else ""} created)\n')


@command('duplicates')
def cmd_duplicates(ip, cmd, rest, by):
    p = parse_command(rest)
    parts = p.body.split()
    sub = parts[0]
    d = ip.data
    vars_ = d.expand_varlist(parts[1:]) if len(parts) > 1 else list(d.vars)
    tuples = list(zip(*[d.cols[v] if d.isstr(v) else np.where(np.isnan(d.cols[v]), np.inf, d.cols[v]) for v in vars_]))
    from collections import Counter
    cnt = Counter(tuples)
    if sub.startswith('r'):
        ip.txt(f'\nDuplicates in terms of {" ".join(vars_) if len(parts) > 1 else "all variables"}\n\n')
        ip.txt('--------------------------------------\n')
        ip.txt('   Copies | Observations       Surplus\n')
        ip.txt('----------+---------------------------\n')
        byc = Counter(cnt.values())
        for c in sorted(byc):
            obs = byc[c] * c
            ip.txt(f'{c:>9} | ')
            ip.res(f'{comma_int(obs):>12}{comma_int(obs - byc[c]):>14}\n')
        ip.txt('--------------------------------------\n')
    elif sub.startswith('drop'):
        seen = set()
        keep = np.ones(d.n, dtype=bool)
        for i, t in enumerate(tuples):
            if t in seen:
                keep[i] = False
            seen.add(t)
        nd = int((~keep).sum())
        d.keep_obs(keep)
        ip.txt(f'({comma_int(nd)} observations deleted)\n')
    elif sub.startswith('tag'):
        nm = p.opts.get('generate', 3)
        d.add_var(nm, np.array([cnt[t] - 1 for t in tuples], dtype=float), 'byte')
    elif sub.startswith('l'):
        pass


# ------------------------------------------------------------------ label
def do_label(ip, rest):
    r = rest.strip()
    d = ip.data
    m = re.match(r'^(?:var|vari|varia|variab|variabl|variable)\s+(\S+)\s*(.*)$', r, re.S)
    if m:
        v = d.resolve(m.group(1))
        d.varlab[v] = unquote(m.group(2).strip())[:80]
        return
    m = re.match(r'^(?:def|defi|defin|define)\s+(\w+)\s*(.*)$', r, re.S)
    if m:
        name, body = m.group(1), m.group(2)
        k = body.find(',')
        opts = ''
        # options only after last quote
        if ',' in body:
            lastq = max(body.rfind('"'), body.rfind("'"))
            k2 = body.find(',', lastq + 1)
            if k2 >= 0:
                opts = body[k2 + 1:]
                body = body[:k2]
        toks = split_top(body)
        lab = d.labels.get(name) if ('modify' in opts or 'add' in opts) else None
        if name in d.labels and not lab and 'replace' not in opts and 'modify' not in opts and 'add' not in opts:
            raise StataError(f'label {name} already defined', 110)
        lab = dict(lab or {})
        for i in range(0, len(toks) - 1, 2):
            lab[float(toks[i])] = unquote(toks[i + 1])
        d.labels[name] = lab
        return
    m = re.match(r'^(?:val|valu|value|values)\s+(.*)$', r)
    if m:
        toks = m.group(1).split()
        if len(toks) == 1:
            for v in d.expand_varlist(toks):
                d.vallab.pop(v, None)
            return
        lname = toks[-1]
        for v in d.expand_varlist(toks[:-1]):
            if lname == '.':
                d.vallab.pop(v, None)
            else:
                d.vallab[v] = lname
        return
    m = re.match(r'^(?:data)\s*(.*)$', r)
    if m:
        d.label = unquote(m.group(1))
        return
    m = re.match(r'^(?:drop)\s+(.*)$', r)
    if m:
        for nm in m.group(1).split():
            if nm == '_all':
                d.labels.clear()
            d.labels.pop(nm, None)
        return
    m = re.match(r'^(?:list|li|dir)\s*(.*)$', r)
    if m:
        names = m.group(1).split() or list(d.labels)
        for nm in names:
            ip.txt(f'{nm}:\n')
            for k, t in sorted(d.labels.get(nm, {}).items()):
                ip.res(f'{fmt_num(k, "%12.0g")} {t}\n')
        return
    raise StataError('label syntax not supported', 198)


@command('labmask')
def cmd_labmask(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    v = d.resolve(p.body.strip())
    src = d.resolve(p.opts.get('values'))
    lab = {}
    for a, b in zip(d.cols[v], d.cols[src]):
        if not np.isnan(a):
            lab[float(a)] = str(b) if d.isstr(src) else g_fmt(b)
    lname = p.opts.get('lblname') or v
    d.labels[lname] = lab
    d.vallab[v] = lname


@command('format')
def cmd_format(ip, cmd, rest, by):
    toks = rest.split()
    d = ip.data
    if toks and toks[0].startswith('%'):
        fmt, vars_ = toks[0], toks[1:]
    else:
        fmt, vars_ = toks[-1], toks[:-1]
    for v in d.expand_varlist(vars_):
        d.formats[v] = fmt


@command('recode')
def cmd_recode(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    body = p.body
    k = body.find('(')
    vars_ = d.expand_varlist(body[:k].split())
    rules = re.findall(r'\(([^()]*)\)', body[k:])
    mask = ip.mark(p)
    newnames = p.opts.get('generate', 3)
    newnames = newnames.split() if newnames else None
    for vi, v in enumerate(vars_):
        x = d.cols[v]
        out = x.copy()
        done = np.zeros(d.n, dtype=bool)
        lab = {}
        for r in rules:
            lhs, rhs = r.split('=', 1)
            rhs = rhs.strip()
            mlab = re.match(r'^(\S+)\s+"(.*)"$', rhs)
            if mlab:
                newv = float(mlab.group(1))
                lab[newv] = mlab.group(2)
            else:
                newv = float(rhs) if rhs not in ('.',) else np.nan
            sel = np.zeros(d.n, dtype=bool)
            for tok in lhs.split():
                if '/' in tok:
                    a, b = tok.split('/')
                    lo = -np.inf if a == 'min' else float(a)
                    hi = np.inf if b == 'max' else float(b)
                    xx = np.where(np.isnan(x), np.inf, x)
                    sel |= (xx >= lo) & (xx <= hi) & ~np.isnan(x)
                elif tok in ('missing', '.'):
                    sel |= np.isnan(x)
                elif tok == 'nonmissing':
                    sel |= ~np.isnan(x)
                elif tok == 'else' or tok == '*':
                    sel |= True
                else:
                    sel |= x == float(tok)
            sel &= ~done & mask
            out[sel] = newv
            done |= sel
        if newnames:
            nn = newnames[vi]
            d.add_var(nn, np.where(mask, out, np.nan), 'float')
            if lab:
                d.labels[nn] = lab
                d.vallab[nn] = nn
            ip.txt(f'({comma_int(int(mask.sum()))} differences between {v} and {nn})\n')
        else:
            changes = int(np.sum(out != x))
            d.cols[v] = store_cast(out, d.types[v])
            ip.txt(f'({v}: {changes} changes made)\n')


@command('encode')
def cmd_encode(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    v = d.resolve(p.body.strip())
    nm = p.opts.get('generate', 3)
    vals = sorted(set(x for x in d.cols[v] if x != ''))
    mp = {s: i + 1 for i, s in enumerate(vals)}
    d.add_var(nm, np.array([mp.get(x, np.nan) for x in d.cols[v]], dtype=float), 'long')
    d.labels[nm] = {float(i): s for s, i in mp.items()}
    d.vallab[nm] = nm


@command('destring')
def cmd_destring(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    for v in d.expand_varlist(p.body.split()):
        if d.isstr(v):
            vals = np.array([E._real(str(x)) if str(x) != '' else np.nan for x in d.cols[v]])
            if p.opts.has('replace'):
                d.types[v] = 'double'
                d.cols[v] = vals
                d.formats[v] = '%10.0g'


@command('tostring')
def cmd_tostring(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    for v in d.expand_varlist(p.body.split()):
        if not d.isstr(v):
            vals = np.array([g_fmt(x) if not np.isnan(x) else '' for x in d.cols[v]], dtype=object)
            d.types[v] = 'str%d' % max(1, max((len(s) for s in vals), default=1))
            d.cols[v] = vals


# ------------------------------------------------------------------ use / save / import
@command('use')
def cmd_use(ip, cmd, rest, by):
    p = parse_command(rest)
    fn = p.using if p.using is not None else unquote(p.body)
    if p.using is not None and p.body.strip():
        vars_ = p.body.split()
    else:
        vars_ = None
    if ip.data.n and not p.opts.has('clear') and False:
        raise StataError('no; dataset in memory has changed since last saved', 4)
    d = _load_dataset(ip, fn)
    if vars_:
        keep = d.expand_varlist(vars_)
        d.drop_vars([v for v in d.vars if v not in keep])
    if p.if_exp or p.in_range:
        old = ip.data
        ip.data = d
        m = ip.mark(p, d)
        d.keep_obs(m)
        ip.data = old
    ip.data = d


def _load_dataset(ip, fn):
    fn = fn.strip()
    if fn.startswith('http'):
        base = os.path.basename(fn)
        local = ip.path(base, must_exist=False)
        for cand in (local, os.path.join(ip.workdir, 'data', base), os.path.join(ip.workdir, base),
                     os.path.join('/examples/data', base)):
            if os.path.exists(cand):
                d = read_dta(cand)
                d.filename = base
                return d
        try:
            from .web import fetch_bytes
            data = fetch_bytes(fn)
            with open(local, 'wb') as f:
                f.write(data)
            d = read_dta(local)
            d.filename = base
            return d
        except Exception as e:  # noqa: BLE001
            raise StataError(f'file {fn} could not be opened ({e})', 601)
    path = ip.path(fn, must_exist=True, ext='.dta')
    d = read_dta(path)
    d.filename = os.path.basename(path)
    return d


@command('webuse', 'sysuse')
def cmd_webuse(ip, cmd, rest, by):
    p = parse_command(rest)
    nm = unquote(p.body.strip())
    if not nm.endswith('.dta'):
        nm += '.dta'
    for cand in (os.path.join(ip.workdir, 'data', nm), os.path.join(ip.workdir, nm), os.path.join('/examples/data', nm)):
        if os.path.exists(cand):
            d = read_dta(cand)
            d.filename = nm
            ip.data = d
            return
    try:
        from .web import fetch_bytes
        raw = fetch_bytes('https://www.stata-press.com/data/r18/' + nm)
        path = os.path.join(ip.workdir, 'data', nm)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(raw)
        d = read_dta(path)
        d.filename = nm
        ip.data = d
    except Exception as e:  # noqa: BLE001
        raise StataError(f'file {nm} not found\n(Trình duyệt chặn tải trực tiếp từ stata-press.com. Hãy tải '
                         f'https://www.stata-press.com/data/r18/{nm} về máy rồi đặt vào thư mục làm việc '
                         f'hoặc kéo thả vào web, sau đó chạy lại.)', 601)


@command('save')
def cmd_save(ip, cmd, rest, by):
    p = parse_command(rest)
    fn = unquote(p.body.strip()) or ip.data.filename
    path = ip.path(fn, ext='.dta')
    if os.path.exists(path) and not p.opts.has('replace'):
        raise StataError(f'file {fn} already exists', 602)
    dd = os.path.dirname(path)
    if dd:
        os.makedirs(dd, exist_ok=True)
    write_dta(ip.data, path)
    ip.data.filename = os.path.basename(path)
    ip.txt(f'file {fn if fn.endswith(".dta") else fn + ".dta"} saved\n')


@command('export')
def cmd_export(ip, cmd, rest, by):
    p = parse_command(rest)
    body = p.body.split()
    if not body or not body[0].startswith('delim'):
        raise StataError('only export delimited is supported', 198)
    d = ip.data
    vars_ = d.expand_varlist(body[1:]) if len(body) > 1 else d.vars
    path = ip.path(p.using)
    import csv
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(vars_)
        for i in range(d.n):
            row = []
            for v in vars_:
                x = d.cols[v][i]
                if d.isstr(v):
                    row.append(x)
                else:
                    if np.isnan(x):
                        row.append('')
                    else:
                        txt = d.value_text(v, x) if not p.opts.has('nolabel') else None
                        row.append(txt if txt is not None else (repr(float(x)) if x != int(x) else str(int(x))))
            w.writerow(row)
    ip.txt(f'file {p.using} saved\n')


@command('import')
def cmd_import(ip, cmd, rest, by):
    p = parse_command(rest)
    body = p.body.split(None, 1)
    kind = body[0]
    fn = p.using if p.using is not None else unquote(body[1] if len(body) > 1 else '')
    if not kind.startswith('delim') and kind not in ('excel',):
        raise StataError(f'import {kind} not supported', 198)
    import pandas as pd
    path = ip.path(fn, must_exist=True)
    if kind == 'excel':
        df = pd.read_excel(path, header=0 if p.opts.has('firstrow') else None)
    else:
        delim = unquote(p.opts.get('delimiters', 5) or ',')
        df = pd.read_csv(path, sep=delim)
    d = Data()
    d.n = len(df)
    for c in df.columns:
        nm = re.sub(r'\W', '_', str(c).strip()).lower() if not p.opts.has('case') else str(c)
        col = df[c]
        if not pd.api.types.is_numeric_dtype(col):
            vals = np.array(['' if pd.isna(x) else str(x) for x in col], dtype=object)
            d.add_var(nm, vals, 'str%d' % max(1, max((len(s) for s in vals), default=1)))
        else:
            a = col.to_numpy(dtype=float)
            ok = a[~np.isnan(a)]
            if ok.size and np.all(ok == np.trunc(ok)):
                t = 'byte' if ok.max() <= 100 and ok.min() >= -127 else ('int' if ok.max() < 32740 and ok.min() > -32767 else 'long')
            else:
                t = 'double' if kind.startswith('delim') else 'double'
            d.add_var(nm, a, t)
    ip.data = d
    ip.txt(f'({len(df.columns)} vars, {comma_int(len(df))} obs)\n')


# ------------------------------------------------------------------ merge / append
@command('merge')
def cmd_merge(ip, cmd, rest, by):
    p = parse_command(rest)
    m = re.match(r'^\s*(1:1|m:1|1:m|m:m)\s+(.*)$', p.body)
    if not m:
        raise StataError('merge syntax: merge 1:1 varlist using filename', 198)
    kind = m.group(1)
    d = ip.data
    keys = d.expand_varlist(m.group(2).split())
    u = _load_dataset(ip, p.using)
    for k in keys:
        if k not in u.cols:
            raise StataError(f'variable {k} not found in using data', 111)

    def keytuples(dd):
        return [tuple((dd.cols[k][i] if dd.isstr(k) else float(dd.cols[k][i])) for k in keys) for i in range(dd.n)]
    km = keytuples(d)
    ku = keytuples(u)
    if kind[2] == '1':
        if len(set(ku)) != len(ku):
            raise StataError('variable(s) ' + ' '.join(keys) + ' do not uniquely identify observations in the using data', 459)
    if kind[0] == '1':
        if len(set(km)) != len(km):
            raise StataError('variable(s) ' + ' '.join(keys) + ' do not uniquely identify observations in the master data', 459)
    umap = {}
    for i, t in enumerate(ku):
        umap.setdefault(t, []).append(i)
    rows_m, rows_u, merge = [], [], []
    used = set()
    for i, t in enumerate(km):
        js = umap.get(t)
        if js:
            for j in (js if kind == '1:m' else js[:1]):
                rows_m.append(i)
                rows_u.append(j)
                merge.append(3)
                used.add(j)
        else:
            rows_m.append(i)
            rows_u.append(-1)
            merge.append(1)
    for j in range(u.n):
        if j not in used:
            rows_m.append(-1)
            rows_u.append(j)
            merge.append(2)
    rows_m = np.array(rows_m, dtype=int)
    rows_u = np.array(rows_u, dtype=int)
    merge = np.array(merge, dtype=float)
    upd = p.opts.has('update')
    keep_opt = p.opts.get('keep')
    sel = np.ones(len(merge), dtype=bool)
    if keep_opt:
        allowed = set()
        for t in keep_opt.split():
            if t.startswith('mat') or t == '3':
                allowed.add(3)
            if t.startswith('mas') or t == '1':
                allowed.add(1)
            if t.startswith('us') or t == '2':
                allowed.add(2)
        sel = np.isin(merge, list(allowed))
    nd = Data()
    n = int(sel.sum())
    nd.n = n
    rm, ru, mg = rows_m[sel], rows_u[sel], merge[sel]
    for v in d.vars:
        col = d.cols[v]
        if d.isstr(v):
            vals = np.array([col[i] if i >= 0 else '' for i in rm], dtype=object)
        else:
            vals = np.array([col[i] if i >= 0 else np.nan for i in rm], dtype=float)
        if v in keys:
            ucol = u.cols[v]
            for k in range(n):
                if rm[k] < 0:
                    vals[k] = ucol[ru[k]]
        nd.vars.append(v)
        nd.cols[v] = vals
        nd.types[v] = d.types[v]
        nd.formats[v] = d.formats[v]
        if v in d.varlab:
            nd.varlab[v] = d.varlab[v]
        if v in d.vallab:
            nd.vallab[v] = d.vallab[v]
    for v in u.vars:
        if v in keys:
            continue
        col = u.cols[v]
        if v in nd.cols:
            if upd:
                cur = nd.cols[v]
                for k in range(n):
                    if ru[k] >= 0 and (cur[k] == '' if u.isstr(v) else np.isnan(cur[k])):
                        cur[k] = col[ru[k]]
            continue
        if u.isstr(v):
            vals = np.array([col[i] if i >= 0 else '' for i in ru], dtype=object)
        else:
            vals = np.array([col[i] if i >= 0 else np.nan for i in ru], dtype=float)
        nd.vars.append(v)
        nd.cols[v] = vals
        nd.types[v] = u.types[v]
        nd.formats[v] = u.formats[v]
        if v in u.varlab:
            nd.varlab[v] = u.varlab[v]
        if v in u.vallab:
            nd.vallab[v] = u.vallab[v]
    nd.labels = dict(d.labels)
    for k, v in u.labels.items():
        nd.labels.setdefault(k, v)
    nd.label = d.label
    nd.ts = d.ts
    nd.filename = d.filename
    if not p.opts.has('nogenerate') and not p.opts.has('nogen'):
        gname = p.opts.get('generate', 3) or '_merge'
        nd.add_var(gname, mg, 'byte')
        nd.labels['_merge'] = {1.0: 'Master only (1)', 2.0: 'Using only (2)', 3.0: 'Matched (3)'}
        nd.vallab[gname] = '_merge'
    ip.data = nd
    n1, n2, n3 = int((merge == 1).sum()), int((merge == 2).sum()), int((merge == 3).sum())
    if not p.opts.has('noreport'):
        ip.txt('\n    Result                      Number of obs\n')
        ip.txt('    -----------------------------------------\n')
        ip.txt(f'    Not matched              {comma_int(n1 + n2):>16}\n')
        if n1 + n2:
            ip.txt(f'        from master          {comma_int(n1):>16}  (_merge==1)\n')
            ip.txt(f'        from using           {comma_int(n2):>16}  (_merge==2)\n')
        ip.txt('\n')
        ip.txt(f'    Matched                  {comma_int(n3):>16}  (_merge==3)\n')
        ip.txt('    -----------------------------------------\n')


@command('append')
def cmd_append(ip, cmd, rest, by):
    p = parse_command(rest)
    files = split_top(p.using) if p.using else []
    d = ip.data
    for fn in files:
        u = _load_dataset(ip, unquote(fn))
        n0, n1 = d.n, u.n
        for v in d.vars:
            if v in u.cols:
                if d.isstr(v) != u.isstr(v):
                    raise StataError(f'variable {v} is str in one dataset and numeric in other', 106)
                d.cols[v] = np.concatenate([d.cols[v], u.cols[v]])
            else:
                fill = np.array([''] * n1, dtype=object) if d.isstr(v) else np.full(n1, np.nan)
                d.cols[v] = np.concatenate([d.cols[v], fill])
        for v in u.vars:
            if v not in d.cols:
                fill = np.array([''] * n0, dtype=object) if u.isstr(v) else np.full(n0, np.nan)
                d.vars.append(v)
                d.cols[v] = np.concatenate([fill, u.cols[v]])
                d.types[v] = u.types[v]
                d.formats[v] = u.formats[v]
                if v in u.varlab:
                    d.varlab[v] = u.varlab[v]
        for k, v in u.labels.items():
            d.labels.setdefault(k, v)
        d.n = n0 + n1
        d.sortedby = []


# ------------------------------------------------------------------ collapse
_STATS = ('mean', 'sd', 'sum', 'rawsum', 'count', 'min', 'max', 'median', 'first', 'last', 'firstnm',
          'lastnm', 'semean', 'sebinomial', 'iqr', 'p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99')


@command('collapse')
def cmd_collapse(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    byv = d.expand_varlist(p.opts.get('by').split()) if p.opts.has('by') else []
    mask = ip.mark(p)
    w = None
    if p.weight:
        w = np.asarray(E.eval_expr(p.weight[1], E.Ctx(ip, d)), dtype=float) * np.ones(d.n)
    specs = []
    stat = 'mean'
    for tok in split_top(p.body):
        m = re.match(r'^\((\w+)\)$', tok)
        if m:
            stat = m.group(1)
            continue
        if '=' in tok:
            nn, src = tok.split('=', 1)
            specs.append((stat, nn, d.resolve(src)))
        else:
            for v in d.expand_varlist([tok]):
                specs.append((stat, v, v))
    if byv:
        keys = list(zip(*[d.cols[v] if d.isstr(v) else d.cols[v].astype(float) for v in byv]))
    else:
        keys = [()] * d.n
    groups = {}
    for i, k in enumerate(keys):
        if mask[i]:
            kk = tuple((np.inf if (not isinstance(x, str) and np.isnan(x)) else x) for x in k)
            groups.setdefault(kk, []).append(i)
    gkeys = sorted(groups.keys(), key=lambda t: tuple((0, x) if isinstance(x, str) else (1, x) for x in t))
    nd = Data()
    nd.n = len(gkeys)
    for j, v in enumerate(byv):
        vals = [k[j] for k in gkeys]
        if d.isstr(v):
            nd.add_var(v, np.array(vals, dtype=object), d.types[v])
        else:
            nd.add_var(v, np.array([np.nan if x == np.inf else x for x in vals], dtype=float), d.types[v])
        if v in d.varlab:
            nd.varlab[v] = d.varlab[v]
        if v in d.vallab:
            nd.vallab[v] = d.vallab[v]
    for stat, nn, src in specs:
        x = d.cols[src].astype(float)
        out = np.full(nd.n, np.nan)
        for gi, k in enumerate(gkeys):
            idx = np.array(groups[k])
            xv = x[idx]
            ok = ~np.isnan(xv)
            ww = w[idx] if w is not None else np.ones(len(idx))
            ok &= ~np.isnan(ww)
            xs, wsel = xv[ok], ww[ok]
            if stat == 'mean':
                out[gi] = np.sum(xs * wsel) / np.sum(wsel) if xs.size and np.sum(wsel) > 0 else np.nan
            elif stat in ('sum', 'rawsum'):
                out[gi] = np.sum(xs * (wsel if stat == 'sum' and p.weight and p.weight[0] != 'aweight' else 1))
            elif stat == 'count':
                out[gi] = xs.size
            elif stat == 'sd':
                if xs.size > 1:
                    mu = np.sum(xs * wsel) / np.sum(wsel)
                    V = np.sum(wsel * (xs - mu) ** 2) / np.sum(wsel) * xs.size / (xs.size - 1)
                    out[gi] = np.sqrt(V)
            elif stat == 'semean':
                if xs.size > 1:
                    out[gi] = xs.std(ddof=1) / np.sqrt(xs.size)
            elif stat == 'min':
                out[gi] = xs.min() if xs.size else np.nan
            elif stat == 'max':
                out[gi] = xs.max() if xs.size else np.nan
            elif stat in ('median', 'p50'):
                out[gi] = _stata_pctile(xs, 50) if xs.size else np.nan
            elif re.match(r'^p\d+$', stat):
                out[gi] = _stata_pctile(xs, float(stat[1:])) if xs.size else np.nan
            elif stat == 'iqr':
                out[gi] = _stata_pctile(xs, 75) - _stata_pctile(xs, 25) if xs.size else np.nan
            elif stat in ('first', 'firstnm'):
                out[gi] = x[idx][0] if stat == 'first' else (xs[0] if xs.size else np.nan)
            elif stat in ('last', 'lastnm'):
                out[gi] = x[idx][-1] if stat == 'last' else (xs[-1] if xs.size else np.nan)
        t = 'double' if stat not in ('count',) else 'long'
        if stat in ('min', 'max', 'first', 'last', 'firstnm', 'lastnm', 'median', 'p50') and d.types.get(src) in ('byte', 'int', 'long', 'float'):
            t = d.types[src]
        if stat == 'mean' and d.types.get(src) != 'double':
            t = 'float'
        nd.add_var(nn, out, t)
        lab = d.varlab.get(src, src)
        nd.varlab[nn] = f'({stat}) {lab}' if nn != src or stat != 'mean' else f'(mean) {lab}'
    nd.labels = dict(d.labels)
    ip.data = nd
    ip.data.sortedby = list(byv)


# ------------------------------------------------------------------ describe
@command('describe')
def cmd_describe(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    vars_ = d.expand_varlist(split_top(p.body)) if p.body.strip() else d.vars
    if not p.body.strip() or p.opts.has('short'):
        fn = d.filename or ''
        if fn:
            ip.txt(f'\nContains data from ')
            ip.res(fn + '\n')
        else:
            ip.txt('\nContains data\n')
        ip.txt(f' Observations: {comma_int(d.n):>13}{"  " + d.label if d.label else ""}\n')
        ip.txt(f'    Variables: {len(d.vars):>13}\n')
    if p.opts.has('short'):
        return
    ip.txt('-' * 79 + '\n')
    ip.txt('Variable      Storage   Display    Value\n')
    ip.txt('    name         type    format    label      Variable label\n')
    ip.txt('-' * 79 + '\n')
    for v in vars_:
        name = abbrev(v, 15)
        ip.res(f'{name:<16}')
        ip.txt(f'{d.types[v]:<8}{d.formats.get(v, ""):<11}{(d.vallab.get(v, "") or "")[:10]:<11}{d.varlab.get(v, "")}\n')
    ip.txt('-' * 79 + '\n')
    sb = ' '.join(d.sortedby)
    ip.txt(f'Sorted by: {sb}\n')
    ip.set_r(scalars={'N': d.n, 'k': len(d.vars)})


@command('count')
def cmd_count(ip, cmd, rest, by):
    p = parse_command(rest)
    byc = _by_ctx(ip, by)
    m = ip.mark(p, ip.data, byc)
    if by:
        gid = byc[0]
        for g in range(int(gid.max()) + 1 if len(gid) else 0):
            sel = gid == g
            first = np.flatnonzero(sel)[0]
            hdr = ', '.join(f'{v} = {_valtxt(ip.data, v, ip.data.cols[v][first])}' for v in by)
            ip.txt('-' * 79 + f'\n-> {hdr}\n')
            ip.res(f'  {comma_int(int(m[sel].sum()))}\n')
    n = int(m.sum())
    if not by:
        ip.res(f'  {comma_int(n)}\n')
    ip.set_r(scalars={'N': n})


def _valtxt(d, v, x):
    if d.isstr(v):
        return x
    t = d.value_text(v, x)
    return t if t is not None else g_fmt(x)


@command('list')
def cmd_list(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    vars_ = d.expand_varlist(split_top(p.body)) if p.body.strip() else list(d.vars)
    m = ip.mark(p)
    idx = np.flatnonzero(m)
    nolab = p.opts.has('nolabel')
    sepby = p.opts.get('sepby')
    sepby = d.expand_varlist(sepby.split()) if sepby else None
    cols = []
    for v in vars_:
        vals = []
        for i in idx:
            x = d.cols[v][i]
            if d.isstr(v):
                vals.append(str(x))
            else:
                t = None if nolab else d.value_text(v, x)
                if t is not None:
                    vals.append(t)
                else:
                    f = d.formats.get(v, '%9.0g')
                    vals.append(fmt_num(x, f).strip())
        w = max([len(abbrev(v, 8))] + [len(s) for s in vals]) if vals else len(v)
        w = min(w, 32)
        cols.append((abbrev(v, max(8, min(w, 12))), vals, w, d.isstr(v)))
    if not idx.size:
        return
    total = sum(c[2] + 3 for c in cols) - 1
    width = len(str(idx[-1] + 1)) + 2
    pad = ' ' * (width + 1)
    ip.txt('\n' + pad + '+' + '-' * total + '+\n')
    ip.txt(pad + '|' + ''.join((' ' + c[0][:c[2]].rjust(c[2]) + ' ' + ('' if k == len(cols) - 1 else ' ')) for k, c in enumerate(cols)).rstrip().ljust(total) + '|\n')
    ip.txt(pad + '|' + '-' * total + '|\n')
    prevkey = None
    for r, i in enumerate(idx):
        if sepby:
            key = tuple(d.cols[v][i] for v in sepby)
            if prevkey is not None and key != prevkey:
                ip.txt(pad + '|' + '-' * total + '|\n')
            prevkey = key
        elif r and r % 5 == 0 and not p.opts.has('separator') and not p.opts.has('sep'):
            ip.txt(pad + '|' + '-' * total + '|\n')
        line = ''.join((' ' + (c[1][r][:c[2]].ljust(c[2]) if c[3] else c[1][r][:c[2]].rjust(c[2])) + ' ' + ('' if k == len(cols) - 1 else ' ')) for k, c in enumerate(cols))
        ip.txt(f'{str(i + 1) + ".":>{width}} |')
        ip.res(line.rstrip().ljust(total))
        ip.txt('|\n')
    ip.txt(pad + '+' + '-' * total + '+\n')


@command('browse', 'edit')
def cmd_browse(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    vars_ = d.expand_varlist(split_top(p.body)) if p.body.strip() else list(d.vars)
    m = ip.mark(p)
    ip.browse_request = (vars_, np.flatnonzero(m))
    ip.txt('(Data Browser opened in the "Data" tab)\n')


@command('codebook')
def cmd_codebook(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    vars_ = d.expand_varlist(split_top(p.body)) if p.body.strip() else d.vars
    for v in vars_:
        ip.txt('-' * 79 + '\n')
        ip.res(f'{v}')
        ip.txt(f'{d.varlab.get(v, "(unlabeled)"):>{79 - len(v)}}\n')
        ip.txt('-' * 79 + '\n\n')
        x = d.cols[v]
        if d.isstr(v):
            uniq = len(set(x))
            miss = int(np.sum(x == ''))
            ip.txt(f'{"Type:":>20} String ({d.types[v]})\n\n')
            ip.txt(f'{"Unique values:":>20} {uniq:<12}Missing "": {miss}/{comma_int(d.n)}\n\n')
            continue
        ok = x[~np.isnan(x)]
        uniq = np.unique(ok)
        miss = int(np.isnan(x).sum())
        ip.txt(f'{"Type:":>20} Numeric ({d.types[v]})\n')
        if v in d.vallab:
            ip.txt(f'{"Label:":>20} {d.vallab[v]}\n')
        ip.txt('\n')
        if ok.size:
            ip.txt(f'{"Range:":>20} [{g_fmt(ok.min())},{g_fmt(ok.max())}]{"Units:":>20} {_units(ok)}\n')
        ip.txt(f'{"Unique values:":>20} {comma_int(len(uniq)):<16}Missing .: {comma_int(miss)}/{comma_int(d.n)}\n\n')
        if len(uniq) <= 9 or (p.opts.has('tab') and len(uniq) <= int(p.opts.get('tab') or 9)):
            ip.txt(f'{"Tabulation:":>20} Freq.   Value\n')
            for u in uniq:
                c = int(np.sum(ok == u))
                lab = d.value_text(v, u)
                ip.res(f'{comma_int(c):>27}  ')
                ip.txt(f'{g_fmt(u):>8}' + (f'  {lab}' if lab else '') + '\n')
            if miss:
                ip.res(f'{comma_int(miss):>27}  ')
                ip.txt(f'{".":>8}\n')
        elif ok.size:
            from .cmds_data import _stata_pctile as pc
            ip.txt(f'{"Mean:":>20} {g_fmt(ok.mean())}\n{"Std. dev.:":>20} {g_fmt(ok.std(ddof=1) if ok.size > 1 else np.nan)}\n\n')
            ip.txt(f'{"Percentiles:":>20}{"10%":>10}{"25%":>10}{"50%":>10}{"75%":>10}{"90%":>10}\n')
            ip.res(' ' * 20 + ''.join(f'{g_fmt(pc(ok, q)):>10}' for q in (10, 25, 50, 75, 90)) + '\n')
        ip.txt('\n')


def _units(ok):
    for u in (1, 0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7):
        if np.allclose(ok / u, np.round(ok / u), atol=1e-6):
            return g_fmt(u)
    return '1.000e-07'


@command('misstable')
def cmd_misstable(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    parts = p.body.split()
    sub = parts[0] if parts else 'summarize'
    vars_ = d.expand_varlist(parts[1:]) if len(parts) > 1 else [v for v in d.vars if not d.isstr(v)]
    if sub.startswith('sum'):
        ip.txt(f'{"Obs<.":>51}\n')
        ip.txt(f'{"+------------------------------":>63}\n')
        ip.txt(f'{"|":>33}{"| Unique":>30}\n')
        ip.txt(f'{"Variable |     Obs=.     Obs>.     Obs<.  |   values        Min         Max":>79}\n')
        ip.txt('-------------+--------------------------------+------------------------------\n')
        for v in vars_:
            x = d.cols[v]
            miss = int(np.isnan(x).sum())
            if not miss:
                continue
            ok = x[~np.isnan(x)]
            ip.txt(f'{abbrev(v):>12} |')
            ip.res(f'{comma_int(miss):>10}{"":>10}{comma_int(ok.size):>10}')
            ip.txt('  |')
            ip.res(f'{len(np.unique(ok)):>9}{g_fmt(ok.min()):>12}{g_fmt(ok.max()):>12}\n')
        ip.txt('-' * 78 + '\n')
    elif sub.startswith('pat'):
        M = np.column_stack([np.isnan(d.cols[v]) for v in vars_])
        anym = [v for k, v in enumerate(vars_) if M[:, k].any()]
        cols = [vars_.index(v) for v in anym]
        from collections import Counter
        pats = Counter(tuple(int(not M[i, c]) for c in cols) for i in range(d.n))
        ip.txt('\n    Missing-value patterns\n      (1 means complete)\n\n')
        ip.txt('              |   Pattern\n    Percent   |  ' + '  '.join(str(k + 1) for k in range(len(cols))) + '\n')
        ip.txt('  ------------+-------------\n')
        for pat, c in sorted(pats.items(), key=lambda t: -t[1]):
            ip.res(f'  {100 * c / d.n:>8.0f}%   |  ' + '  '.join(str(x) for x in pat) + '\n')
        ip.txt('  ------------+-------------\n       100%\n\n  Variables are  ' +
               ' '.join(f'({k + 1}) {v}' for k, v in enumerate(anym)) + '\n')


# ------------------------------------------------------------------ xtset / tsset
@command('xtset', 'tsset')
def cmd_xtset(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    toks = p.body.split()
    if not toks:
        if not d.ts:
            raise StataError('panel variable not set; use xtset varname ...', 459)
        toks = [x for x in (d.ts.get('panel'), d.ts.get('time')) if x]
    if p.body.strip() == 'clear':
        d.ts = None
        return
    toks = d.expand_varlist(toks)
    if cmd == 'xtset' or len(toks) == 2:
        panel = toks[0]
        time = toks[1] if len(toks) > 1 else None
    else:
        panel, time = None, toks[0]
    d.ts = {'panel': panel, 'time': time, 'delta': float(p.opts.get('delta') or 1)}
    keys = ([panel] if panel else []) + ([time] if time else [])
    d.sort(keys)
    if panel:
        pv = d.cols[panel]
        groups = np.unique(pv[~np.isnan(pv)])
        bal = 'strongly balanced'
        if time:
            tv = d.cols[time]
            cnt = {g: np.sum(pv == g) for g in groups}
            ts = set(tv[~np.isnan(tv)])
            if len(set(cnt.values())) > 1:
                bal = 'unbalanced'
            elif len(ts) * len(groups) != d.n:
                bal = 'weakly balanced'
        ip.txt(f'\nPanel variable: ')
        ip.res(f'{panel} ({bal})\n')
    if time:
        tv = d.cols[time]
        ok = tv[~np.isnan(tv)]
        ip.txt(f' Time variable: ' if panel else 'Time variable: ')
        ip.res(f'{time}, {g_fmt(ok.min())} to {g_fmt(ok.max())}')
        gaps = False
        if panel:
            gaps = False
        ip.res(('\n' if not gaps else ', but with gaps\n'))
        ip.txt(f'         Delta: ' if panel else '        Delta: ')
        ip.res('1 unit\n')
    ip.set_r(macros={'panelvar': panel or '', 'timevar': time or ''})


@command('svmat')
def cmd_svmat(ip, cmd, rest, by):
    p = parse_command(rest)
    M = ip.get_matrix(p.body.split()[-1].strip())
    d = ip.data
    names_opt = p.opts.get('names') or 'col'
    r, c = M.a.shape
    if d.n < r:
        d.set_obs(r)
    for j in range(c):
        if names_opt == 'col':
            nm = M.colnames[j] if re.match(r'^[A-Za-z_]\w*$', M.colnames[j]) else f'c{j+1}'
        elif names_opt == 'eqcol':
            nm = M.colnames[j]
        else:
            nm = f'{names_opt}{j+1}'
        vals = np.full(d.n, np.nan)
        vals[:r] = M.a[:, j]
        d.add_var(nm, vals, 'float')


@command('fillin', 'isid')
def cmd_isid(ip, cmd, rest, by):
    d = ip.data
    vars_ = d.expand_varlist(rest.split(',')[0].split())
    t = list(zip(*[d.cols[v] for v in vars_]))
    if len(set(t)) != len(t):
        raise StataError('variables ' + ' '.join(vars_) + ' do not uniquely identify the observations', 459)


@command('xtdescribe', 'xtdes')
def cmd_xtdescribe(ip, cmd, rest, by):
    d = ip.data
    if not d.ts or not d.ts.get('panel'):
        raise StataError('panel variable not set', 459)
    pv, tv = d.cols[d.ts['panel']], d.cols[d.ts['time']]
    ug = np.unique(pv[~np.isnan(pv)])
    ut = np.unique(tv[~np.isnan(tv)])
    from collections import Counter
    Ti = Counter(pv[~np.isnan(pv)])
    t = np.array(list(Ti.values()))
    ip.txt(f'\n{d.ts["panel"]:>8}:  {g_fmt(ug.min())}, ..., {g_fmt(ug.max())}{"n =":>40}{len(ug):>12}\n')
    ip.txt(f'{d.ts["time"]:>8}:  {g_fmt(ut.min())}, ..., {g_fmt(ut.max())}{"T =":>40}{len(ut):>12}\n')
    ip.txt(f'{"Delta(" + d.ts["time"] + ") = 1 unit":>30}\n\n')
    ip.txt('Distribution of T_i:   min      5%     25%       50%       75%     95%     max\n')
    q = [np.min(t)] + [np.quantile(t, x) for x in (0.05, 0.25, 0.5, 0.75, 0.95)] + [np.max(t)]
    ip.res(' ' * 21 + ''.join(f'{g_fmt(v):>8}' for v in q) + '\n')


@command('xtsum')
def cmd_xtsum(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    pv = d.cols[d.ts['panel']]
    ip.txt('\nVariable         |      Mean   Std. dev.       Min        Max |    Observations\n')
    ip.txt('-----------------+--------------------------------------------+----------------\n')
    for v in d.expand_varlist(split_top(p.body)):
        x = d.cols[v]
        ok = ~np.isnan(x) & ~np.isnan(pv)
        xs, gs = x[ok], pv[ok]
        ug, inv = np.unique(gs, return_inverse=True)
        gm = np.bincount(inv, weights=xs) / np.bincount(inv)
        within = xs - gm[inv] + xs.mean()
        Tbar = len(xs) / len(ug)
        rows = [('overall', xs.mean(), xs.std(ddof=1), xs.min(), xs.max(), f'N = {comma_int(len(xs))}'),
                ('between', None, gm.std(ddof=1), gm.min(), gm.max(), f'n = {comma_int(len(ug))}'),
                ('within', None, within.std(ddof=1), within.min(), within.max(), f'T-bar = {g_fmt(Tbar)}')]
        for k, (lab, mu, sd, mn, mx, obs) in enumerate(rows):
            ip.txt(f'{abbrev(v, 8) if k == 0 else "":<8} {lab:<8}|')
            ip.res(f'{g_fmt(mu, 9) if mu is not None else "":>10}{g_fmt(sd, 9):>12}{g_fmt(mn, 9):>11}{g_fmt(mx, 9):>11}')
            ip.txt(f' |{obs:>16}\n')
        ip.txt('                 |                                            |\n')
