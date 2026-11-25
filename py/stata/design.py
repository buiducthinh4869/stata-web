"""Design matrix construction with Stata factor-variable notation and coefficient tables."""
import math
import re

import numpy as np
from scipy import stats as sps

from . import expr as E
from .errors import StataError
from .fmt import g_fmt, fmt_num, abbrev, num_to_macro, comma_int
from .syntax import parse_terms, split_top


class Col:
    __slots__ = ('name', 'values', 'base', 'omitted', 'term', 'group', 'level', 'kind', 'vars', 'compact')

    def __init__(self, **kw):
        self.base = False
        self.omitted = False
        self.compact = False
        for k, v in kw.items():
            setattr(self, k, v)


def _comp_values(ip, d, comp, override=None):
    var = d.resolve(comp.var)
    if comp.tsop:
        op, k = comp.tsop
        return ip.ts_op(E.Ctx(ip, d), op, k, var), var
    if override and var in override:
        return np.asarray(override[var], dtype=float), var
    if d.isstr(var):
        raise StataError(f'{var}: string variables may not be used as factor variables or regressors', 109)
    return d.cols[var].astype(float), var


def _lvl(v):
    return num_to_macro(v)


def term_vars(ip, d, tokens):
    """All variables referenced by a varlist (for sample marking)."""
    terms = parse_terms(tokens)
    out = []
    for t in terms:
        for c in t:
            out.append(c)
    return terms, out


def build_design(ip, d, tokens, mask, override=None, levels_from=None, base_override=None):
    """Return (list[Col], mask) ; mask is updated for missing values."""
    terms = parse_terms(tokens)
    mask = mask.copy()
    compvals = {}
    for t in terms:
        for c in t:
            vals, var = _comp_values(ip, d, c, override)
            key = (c.kind, var, c.tsop)
            compvals[key] = vals
            mask &= ~np.isnan(vals)
            if c.kind == 'i':
                ok = vals[~np.isnan(vals)]
                if ok.size and (np.any(ok < 0) or np.any(ok != np.trunc(ok))):
                    raise StataError(f'{var}: factor variables may not contain noninteger or negative values', 452)
    # lower-order term keys present
    term_keys = set()
    for t in terms:
        term_keys.add(tuple(sorted((c.kind, c.var) for c in t)))
    cols = []
    for ti, t in enumerate(terms):
        parts = []    # list of list of (label, values, isbase, namepart)
        for c in t:
            var = d.resolve(c.var)
            vals = compvals[(c.kind, var, c.tsop)]
            if c.kind == 'c':
                nm = var if not c.tsop else f'{c.tsop[0]}{"" if c.tsop[1] == 1 else c.tsop[1]}.{var}'
                parts.append([('', vals, False, ('c.' + nm) if len(t) > 1 else nm, None)])
            else:
                if levels_from and var in levels_from:
                    lv = levels_from[var]
                else:
                    lv = sorted(np.unique(vals[mask]).tolist())
                base = None
                if c.base == 'none':
                    base = None
                elif c.base is None:
                    base = lv[0] if lv else None
                elif c.base == 'last':
                    base = lv[-1]
                elif c.base == 'first':
                    base = lv[0]
                elif c.base == 'freq':
                    cnt = [np.sum(vals[mask] == l) for l in lv]
                    base = lv[int(np.argmax(cnt))]
                else:
                    base = float(c.base)
                if base_override and var in base_override:
                    base = base_override[var]
                # does base apply? (lower-order term present)
                rest = tuple(sorted((cc.kind, cc.var) for cc in t if cc is not c))
                applies = (len(t) == 1) or (rest in term_keys)
                if c.base == 'none':
                    applies = False
                elif c.base is not None and len(t) > 1:
                    applies = True
                lst = []
                for l in lv:
                    isb = applies and base is not None and l == base
                    lst.append((l, (vals == l).astype(float), isb, None, var))
                parts.append(lst)
        # cartesian product
        combos = [[]]
        for plist in parts:
            combos = [cmb + [p] for cmb in combos for p in plist]
        factor_vars = [d.resolve(c.var) for c in t if c.kind == 'i']
        for cmb in combos:
            v = np.ones(d.n)
            namebits = []
            labbits = []
            isbase = False
            for (lab, vals, isb, nmpart, var), c in zip(cmb, t):
                v = v * np.nan_to_num(vals)
                if c.kind == 'c':
                    namebits.append(nmpart)
                else:
                    namebits.append(f'{_lvl(lab)}{"b" if isb else ""}.{var}')
                    labbits.append((var, lab))
                    isbase = isbase or isb
            name = '#'.join(namebits)
            if isbase and any(c.kind == 'c' for c in t):
                # continuous part of a base cell is marked "co." in Stata names
                name = '#'.join(('co.' + nb[2:]) if nb.startswith('c.') else nb for nb in namebits)
            group_parts = []
            for c in t:
                vv = d.resolve(c.var)
                group_parts.append(('c.' + vv) if (c.kind == 'c' and len(t) > 1 and factor_vars) else vv)
            group = '#'.join(group_parts)
            if factor_vars:
                if len(t) > 1 and any(c.kind == 'c' for c in t) and len(factor_vars) >= 1:
                    group = '#'.join((('c.' + d.resolve(c.var)) if c.kind == 'c' else d.resolve(c.var)) for c in t)
                lab = ' '.join(_level_label(d, var, l) for var, l in labbits)
                kind = 'factor'
            else:
                lab = ''
                kind = 'cont' if len(t) == 1 else 'cinter'
                if len(t) > 1:
                    group = '#'.join('c.' + d.resolve(c.var) for c in t)
            cols.append(Col(name=name, values=v, base=isbase, term=ti, group=group, level=lab,
                            kind=kind, vars=factor_vars))
    # compact display for single-factor terms with 1 non-base level and no value label
    byterm = {}
    for c in cols:
        byterm.setdefault(c.term, []).append(c)
    for ti, cl in byterm.items():
        t = terms[ti]
        if len(t) == 1 and t[0].kind == 'i':
            nb = [c for c in cl if not c.base]
            var = d.resolve(t[0].var)
            if len(nb) == 1 and var not in d.vallab:
                nb[0].compact = True
    return cols, mask


def _level_label(d, var, l):
    t = d.value_text(var, l)
    return t if t is not None else _lvl(l)


def mark_collinear(X, names, keep_first=None, tol=1e-9):
    """Sequential collinearity detection. Returns boolean array of omitted columns."""
    n, k = X.shape
    omitted = np.zeros(k, dtype=bool)
    Q = np.zeros((n, 0))
    for j in range(k):
        x = X[:, j]
        nx = float(x @ x)
        if nx <= 1e-300:
            omitted[j] = True
            continue
        r = x - Q @ (Q.T @ x) if Q.shape[1] else x.copy()
        if Q.shape[1]:
            r = r - Q @ (Q.T @ r)
        nr = float(r @ r)
        if nr <= tol * nx:
            omitted[j] = True
            continue
        Q = np.column_stack([Q, r / math.sqrt(nr)])
    return omitted


def finalize_cols(cols, X_active_fn=None):
    pass


# ---------------------------------------------------------------- table printing
def _split_name(name, w=12):
    if len(name) <= w:
        return [name]
    if '#' in name:
        k = name.rfind('#', 0, w)
        if k > 0:
            first = name[:k + 1]
            second = name[k + 1:]
            return [abbrev(first, w), abbrev(second, w)]
    return [abbrev(name, w)]


def coef_table(ip, est, stat='t', robust=None, level=95, depvar=None, beta=None, show_base=False,
               extra_rows=None, eq_title=None, colhead=None, width=78):
    """Print a standard Stata coefficient table."""
    names = est['names']
    b = np.asarray(est['b'], float)
    V = np.asarray(est['V'], float)
    info = est.get('colinfo')
    df = est.get('df_r')
    depvar = depvar or est.get('depvar', '')
    se = np.sqrt(np.maximum(np.diag(V), 0))
    alpha = (100 - level) / 200
    if stat == 't' and df is not None and not (isinstance(df, float) and math.isnan(df)):
        crit = sps.t.ppf(1 - alpha, df)
    else:
        crit = sps.norm.ppf(1 - alpha)
    hdr2 = 'Coefficient  Std. err.' if not robust else 'Coefficient  std. err.'
    statlab = '      t    P>|t|' if stat == 't' else '      z    P>|z|'
    ip.txt('-' * 78 + '\n')
    if robust:
        ip.txt(f'{"":>12} |{robust:>21}\n' if len(robust) < 12 else f'{"":>12} | {robust}\n')
    dv = _split_name(depvar)
    if len(dv) > 1:
        ip.txt(f'{dv[0]:>12} |\n')
    tail = '     [95% conf. interval]' if beta is None else '                  Beta'
    if level != 95 and beta is None:
        tail = f'     [{level}% conf. interval]'
    lab = colhead or hdr2
    ip.txt(f'{dv[-1]:>12} | {lab}{statlab}{tail}\n')
    ip.txt('-' * 13 + '+' + '-' * 64 + '\n')
    if eq_title:
        ip.res(f'{eq_title:<12}')
        ip.txt(' |\n')
    lastgroup = None
    idx = list(range(len(names)))
    # _cons last
    order = [i for i in idx if names[i] != '_cons'] + [i for i in idx if names[i] == '_cons']
    for i in order:
        ci = info[i] if info else None
        if ci is not None and ci.base and not show_base:
            continue
        nm = names[i]
        if ci is not None and ci.kind == 'factor' and not ci.compact:
            if ci.group != lastgroup:
                ip.txt(f'{"":>12} |\n')
                gl = _split_name(ci.group)
                for g in gl[:-1]:
                    ip.txt(f'{g:>12} |\n')
                ip.txt(f'{gl[-1]:>12} |\n')
                lastgroup = ci.group
            label = ci.level
            if len(label) > 10:
                label = label[:10]
            rowname = f'{label:>11} '
        else:
            if ci is not None and ci.kind == 'factor' and ci.compact:
                rowname = abbrev(nm.replace('b.', '.'), 12)
                if lastgroup is not None:
                    pass
                lastgroup = None
            elif ci is not None and ci.kind == 'cinter':
                parts = _split_name(nm)
                for pp in parts[:-1]:
                    ip.txt(f'{pp:>12} |\n')
                rowname = parts[-1]
                lastgroup = None
            else:
                if lastgroup is not None and nm != '_cons':
                    pass
                if lastgroup is not None and nm == '_cons':
                    ip.txt(f'{"":>12} |\n')
                rowname = abbrev(nm.replace('o.', ''), 12)
                lastgroup = None
        ip.txt(f'{rowname:>12} |')
        if ci is not None and ci.base:
            ip.res(f'{"0":>11}')
            ip.txt('  (base)\n')
            continue
        if (ci is not None and ci.omitted) or (se[i] == 0 and b[i] == 0 and nm != '_cons'):
            ip.res(f'{"0":>11}')
            ip.txt('  (omitted)\n')
            continue
        t = b[i] / se[i] if se[i] > 0 else np.nan
        if stat == 't' and df is not None:
            p = 2 * sps.t.sf(abs(t), df) if not np.isnan(t) else np.nan
        else:
            p = 2 * sps.norm.sf(abs(t)) if not np.isnan(t) else np.nan
        tstr = f'{t:9.2f}' if not np.isnan(t) else f'{".":>9}'
        if not np.isnan(t) and abs(t) >= 1e5:
            tstr = f'{g_fmt(t, 8):>9}'
        pstr = f'{p:8.3f}' if not np.isnan(p) else f'{".":>8}'
        if beta is None:
            lo, hi = b[i] - crit * se[i], b[i] + crit * se[i]
            tail = f'{g_fmt(lo, 9):>13}{g_fmt(hi, 9):>12}'
        else:
            bv = beta[i] if nm != '_cons' else np.nan
            tail = f'{g_fmt(bv, 9) if not np.isnan(bv) else ".":>25}' if nm != '_cons' else ''
        ip.res(f'{g_fmt(b[i], 9):>11}{g_fmt(se[i], 9):>11}{tstr}{pstr}{tail}\n')
    if extra_rows:
        ip.txt('-' * 13 + '+' + '-' * 64 + '\n')
        for nm, val, note in extra_rows:
            ip.txt(f'{nm:>12} |')
            ip.res(f'{g_fmt(val, 9):>11}')
            ip.txt((f'   {note}' if note else '') + '\n')
    ip.txt('-' * 78 + '\n')


def header_line(ip, left, key, val, valfmt=None, left_style='txt'):
    """Right-hand header 'key = value' aligned like Stata (col 49)."""
    if isinstance(val, str):
        v = val
    elif valfmt:
        v = fmt_num(val, valfmt).strip()
    else:
        v = g_fmt(val)
    ip.txt(f'{left:<48}')
    ip.txt(f'{key:<18}= ')
    ip.res(f'{v:>10}\n')


def right_lines(ip, left_lines, right):
    """Print a header block: left_lines (list of str) and right (list of (key,val_str))."""
    n = max(len(left_lines), len(right))
    for i in range(n):
        lft = left_lines[i] if i < len(left_lines) else ''
        if i < len(right) and right[i] is not None:
            k, v = right[i]
            ip.txt(f'{lft:<48}{k:<17} = ')
            ip.res(f'{v:>10}\n')
        else:
            ip.txt(f'{lft}\n')
