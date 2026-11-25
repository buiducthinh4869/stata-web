"""Regression tables: estimates table, esttab/estout, eststo, estpost, outreg2."""
import math
import os
import re

import numpy as np
from scipy import stats as sps

from .errors import StataError
from .fmt import fmt_num, g_fmt, comma_int, abbrev
from .interp import command, canon_coef
from .syntax import parse_command, split_top, unquote, Options, quoted_strings, find_top


def _pvals(est):
    b = est['b']
    se = np.sqrt(np.maximum(np.diag(est['V']), 0))
    with np.errstate(all='ignore'):
        t = b / se
    df = est.get('df_r')
    if est.get('stat') == 't' and df:
        p = 2 * sps.t.sf(np.abs(t), df)
    else:
        p = 2 * sps.norm.sf(np.abs(t))
    return se, t, p


def _stars(p, levels):
    s = ''
    for sym, lv in levels:
        if p < lv:
            s = sym
    return s


def _parse_star_levels(raw, default):
    if raw is None:
        return default
    toks = split_top(raw)
    out = []
    for i in range(0, len(toks) - 1, 2):
        try:
            out.append((toks[i], float(toks[i + 1])))
        except ValueError:
            pass
    out.sort(key=lambda t: -t[1])
    return out or default


def _models(ip, names):
    if not names:
        names = list(ip.eststo_order) or ['.']
    out = []
    for nm in names:
        if nm in ('.', ''):
            if ip.est is None:
                raise StataError('no estimates', 301)
            out.append(('.', ip.est))
        elif '*' in nm:
            import fnmatch
            for k in ip.stored:
                if fnmatch.fnmatchcase(k, nm):
                    out.append((k, ip.stored[k]))
        else:
            if nm not in ip.stored:
                raise StataError(f'estimation result {nm} not found', 111)
            out.append((nm, ip.stored[nm]))
    return out


def _coef_rows(models, keep, drop, order):
    import fnmatch
    rows = []
    for _, est in models:
        info = est.get('colinfo')
        for j, n in enumerate(est['names']):
            if info and (info[j].base or info[j].omitted):
                continue
            if n not in rows:
                rows.append(n)
    if '_cons' in rows:
        rows.remove('_cons')
        rows.append('_cons')

    def m(n, pats):
        c = canon_coef(n)
        return any(fnmatch.fnmatchcase(c, canon_coef(pt)) or fnmatch.fnmatchcase(n, pt) for pt in pats)
    if keep:
        rows = [r for r in rows if m(r, keep)]
        rows.sort(key=lambda r: next(i for i, pt in enumerate(keep) if m(r, [pt])))
    if drop:
        rows = [r for r in rows if not m(r, drop)]
    if order:
        first = [r for pt in order for r in rows if m(r, [pt])]
        first = list(dict.fromkeys(first))
        rows = first + [r for r in rows if r not in first]
    return rows


def _lookup(est, name):
    c = canon_coef(name)
    for j, n in enumerate(est['names']):
        if canon_coef(n) == c:
            info = est.get('colinfo')
            if info and (info[j].base or info[j].omitted):
                return None
            return j
    return None


def _stat_value(est, key):
    sc = dict(est.get('scalars', {}))
    sc.update(est.get('e_scalars', {}))
    return sc.get(key, np.nan)


def _fmt_dec(x, d):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ''
    return f'{x:.{d}f}'


def build_table(ip, models, o, varlabels=None, uselabels=False):
    """Return dict with header, rows (list of (label, cells)), stats rows, notes."""
    keep = split_top(o.get('keep')) if o.get('keep') else None
    drop = split_top(o.get('drop')) if o.get('drop') else None
    order = split_top(o.get('order')) if o.get('order') else None
    rows = _coef_rows(models, keep, drop, order)
    cells_raw = o.get('cells')
    bdec, sedec = 3, 3
    show = 't'
    if o.get('b') is not None:
        bdec = _dec_of(o.get('b'), 3)
    if o.has('se'):
        show = 'se'
        sedec = _dec_of(o.get('se'), bdec)
    if o.has('t') and o.get('t'):
        sedec = _dec_of(o.get('t'), 2)
    if o.has('p'):
        show = 'p'
    if o.has('ci'):
        show = 'ci'
    if cells_raw:
        mm = re.search(r'\bb\(([^)]*)\)', cells_raw)
        if mm:
            f = re.search(r'fmt\((\d+)', mm.group(1))
            bdec = int(f.group(1)) if f else 3
        mm = re.search(r'\bse\(([^)]*)\)', cells_raw)
        if mm:
            show = 'se'
            f = re.search(r'fmt\((\d+)', mm.group(1))
            sedec = int(f.group(1)) if f else 3
    default_star = [('*', 0.05), ('**', 0.01), ('***', 0.001)]
    if o.has('star'):
        levels = _parse_star_levels(o.get('star'), default_star) if o.get('star') else default_star
    elif o.has('starlevels'):
        levels = _parse_star_levels(o.get('starlevels'), default_star)
    elif cells_raw and 'star' in cells_raw:
        levels = default_star
    else:
        levels = default_star
    nostar = o.has('nostar')
    vl = dict(varlabels or {})
    if o.get('varlabels'):
        toks = split_top(o.get('varlabels'))
        for i in range(0, len(toks) - 1, 2):
            vl[canon_coef(toks[i])] = unquote(toks[i + 1])
            vl[toks[i]] = unquote(toks[i + 1])
    out_rows = []
    for r in rows:
        lab = vl.get(r) or vl.get(canon_coef(r))
        if not lab and uselabels and r in ip.data.cols:
            lab = ip.data.varlab.get(r)
        lab = lab or r
        bc, sc = [], []
        for _, est in models:
            j = _lookup(est, r)
            if j is None:
                bc.append('')
                sc.append('')
                continue
            se, t, p = _pvals(est)
            st = '' if nostar else _stars(p[j], levels)
            bc.append(_fmt_dec(est['b'][j], bdec) + st)
            if show == 'se':
                sc.append(f'({_fmt_dec(se[j], sedec)})')
            elif show == 'p':
                sc.append(f'({p[j]:.3f})')
            elif show == 'ci':
                cr = 1.96
                sc.append(f'[{_fmt_dec(est["b"][j] - cr * se[j], bdec)},{_fmt_dec(est["b"][j] + cr * se[j], bdec)}]')
            else:
                sc.append(f'({_fmt_dec(t[j], 2)})')
        out_rows.append((lab, bc, sc))
    # stats
    stats = []
    st_raw = o.get('stats')
    keys = []
    labels = {}
    fmts = {}
    if st_raw:
        k = find_top(st_raw, ',')
        keys = split_top(st_raw if k < 0 else st_raw[:k])
        so = Options(st_raw[k + 1:] if k >= 0 else '')
        if so.get('labels'):
            ls = [unquote(x) for x in split_top(so.get('labels'))]
            labels = dict(zip(keys, ls))
        if so.get('fmt'):
            fs = split_top(so.get('fmt'))
            for kk, f in zip(keys, fs + [fs[-1]] * len(keys)):
                fmts[kk] = f
    else:
        keys = ['N']
        if o.has('r2'):
            keys.append('r2')
        if o.has('ar2'):
            keys.append('r2_a')
        sc_raw = o.get('scalars')
        if sc_raw:
            for tok in split_top(sc_raw):
                q = unquote(tok).split(None, 1)
                if q[0] not in keys:
                    keys.append(q[0])
                if len(q) > 1:
                    labels[q[0]] = q[1]
    for kk in keys:
        vals = []
        for _, est in models:
            v = _stat_value(est, kk)
            if kk == 'r2' and (isinstance(v, float) and math.isnan(v)):
                v = _stat_value(est, 'r2_w')
            f = fmts.get(kk)
            if isinstance(v, float) and math.isnan(v):
                vals.append('')
            elif kk in ('N', 'N_clust', 'N_g') or (f is not None and str(f) == '0'):
                vals.append(f'{int(v)}')
            else:
                d = int(f) if f and str(f).isdigit() else 3
                vals.append(f'{v:.{d}f}')
        default_lab = {'N': 'N', 'r2': 'R-sq', 'r2_a': 'adj. R-sq', 'F': 'F', 'N_clust': 'N_clust'}.get(kk, kk)
        stats.append((labels.get(kk, default_lab), vals))
    if show == 't':
        note1 = 't statistics in parentheses'
    elif show == 'se':
        note1 = 'Standard errors in parentheses'
    elif show == 'p':
        note1 = 'p-values in parentheses'
    else:
        note1 = '95% confidence intervals in brackets'
    note2 = ', '.join(f'{s} p<{g_fmt(l)}' for s, l in sorted(levels, key=lambda t: -t[1]))
    notes = [] if o.has('nonotes') else [note1, note2]
    for raw in (o.get('addnotes'), o.get('note')):
        if raw:
            notes += quoted_strings(raw) or [raw]
    return {'rows': out_rows, 'stats': stats, 'notes': notes}


def _dec_of(s, default):
    if s is None or s == '':
        return default
    m = re.search(r'(\d+)', s)
    if m:
        return int(m.group(1)) if '.' not in s else int(s.split('.')[-1][0]) if s.split('.')[-1][0].isdigit() else default
    return default


def _titles(ip, models, o):
    mt = o.get('mtitles')
    if mt is not None and mt != '':
        q = quoted_strings(mt)
        return q + [''] * (len(models) - len(q))
    if o.has('mtitles') or o.has('mtitle'):
        return [nm for nm, _ in models]
    if o.has('nomtitles'):
        return [''] * len(models)
    return [est.get('depvar', '') for _, est in models]


def render_text(T, models, titles, title=None, width=None):
    ncol = len(models)
    labw = max([len(r[0]) for r in T['rows']] + [len(s[0]) for s in T['stats']] + [12])
    labw = min(labw, 30)
    cw = max([len(c) for r in T['rows'] for c in r[1] + r[2]] + [len(t) for t in titles] + [10]) + 2
    line = '-' * (labw + cw * ncol)
    out = []
    if title:
        out.append(title)
    out.append(line)
    out.append(' ' * labw + ''.join(f'({k + 1})'.center(cw) for k in range(ncol)).rstrip())
    if any(titles):
        out.append(' ' * labw + ''.join(t[:cw - 1].center(cw) for t in titles).rstrip())
    out.append(line)
    for lab, bc, sc in T['rows']:
        out.append(f'{lab[:labw]:<{labw}}' + ''.join(c.center(cw) for c in bc).rstrip())
        out.append(' ' * labw + ''.join(c.center(cw) for c in sc).rstrip())
    out.append(line)
    for lab, vals in T['stats']:
        out.append(f'{lab[:labw]:<{labw}}' + ''.join(v.center(cw) for v in vals).rstrip())
    out.append(line)
    out.extend(T['notes'])
    return '\n'.join(out)


def render_rtf(T, models, titles, title=None):
    def esc(s):
        s = s.replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}')
        return ''.join(c if ord(c) < 128 else f'\\u{ord(c) if ord(c) < 32768 else ord(c) - 65536}?' for c in s)
    ncol = len(models)
    cw = 1600
    lw = 3200

    def row(cells, bold=False, top=False, bottom=False):
        b = ''
        x = lw
        defs = ''
        for k in range(ncol + 1):
            bd = ('\\clbrdrt\\brdrs' if top else '') + ('\\clbrdrb\\brdrs' if bottom else '')
            defs += f'{bd}\\cellx{lw + k * cw if k else lw}'
        s = '\\trowd\\trgaph60' + defs
        for k, c in enumerate(cells):
            al = '\\ql' if k == 0 else '\\qc'
            s += f'\\pard\\intbl{al} {"\\b " if bold else ""}{esc(c)}\\cell'
        return s + '\\row\n'
    out = ['{\\rtf1\\ansi\\deff0 {\\fonttbl{\\f0 Times New Roman;}}\\fs20\n']
    if title:
        out.append(f'\\pard\\qc\\b {esc(title)}\\b0\\par\n')
    out.append(row([''] + [f'({k + 1})' for k in range(ncol)], top=True))
    if any(titles):
        out.append(row([''] + titles, bottom=True))
    for lab, bc, sc in T['rows']:
        out.append(row([lab] + bc))
        out.append(row([''] + sc))
    for i, (lab, vals) in enumerate(T['stats']):
        out.append(row([lab] + vals, top=(i == 0), bottom=(i == len(T['stats']) - 1)))
    for n in T['notes']:
        out.append(f'\\pard\\ql\\fs16 {esc(n)}\\par\n')
    out.append('}')
    return ''.join(out)


def render_tex(T, models, titles, title=None, booktabs=False):
    def esc(s):
        return s.replace('&', '\\&').replace('%', '\\%').replace('_', '\\_').replace('#', '\\#')
    ncol = len(models)
    top, mid, bot = ('\\toprule', '\\midrule', '\\bottomrule') if booktabs else ('\\hline\\hline', '\\hline', '\\hline\\hline')
    out = ['{', '\\def\\sym#1{\\ifmmode^{#1}\\else\\(^{#1}\\)\\fi}']
    out.append('\\begin{tabular}{l*{' + str(ncol) + '}{c}}')
    out.append(top)
    out.append(' & ' + ' & '.join(f'\\multicolumn{{1}}{{c}}{{({k + 1})}}' for k in range(ncol)) + '\\\\')
    if any(titles):
        out.append(' & ' + ' & '.join(f'\\multicolumn{{1}}{{c}}{{{esc(t)}}}' for t in titles) + '\\\\')
    out.append(mid)
    for lab, bc, sc in T['rows']:
        bcx = [re.sub(r'(\*+)$', r'\\sym{\1}', c) for c in bc]
        out.append(esc(lab) + ' & ' + ' & '.join(bcx) + '\\\\')
        out.append(' & ' + ' & '.join(sc) + '\\\\[1em]')
    out.append(mid)
    for lab, vals in T['stats']:
        out.append(esc(lab) + ' & ' + ' & '.join(vals) + '\\\\')
    out.append(bot)
    out.append('\\multicolumn{' + str(ncol + 1) + '}{l}{\\footnotesize ' + ' \\\\ '.join(esc(n) for n in T['notes']) + '}\\\\')
    out.append('\\end{tabular}')
    out.append('}')
    s = '\n'.join(out)
    if title:
        s = f'\\begin{{table}}[htbp]\\centering\n\\caption{{{esc(title)}}}\n{s}\n\\end{{table}}'
    return s


def _write_out(ip, p, T, models, titles, title):
    fn = p.using
    path = ip.path(fn)
    ext = os.path.splitext(path)[1].lower()
    if os.path.exists(path) and not (p.opts.has('replace') or p.opts.has('append')):
        raise StataError(f'file {fn} already exists', 602)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if ext == '.rtf' or p.opts.has('rtf'):
        txt = render_rtf(T, models, titles, title)
    elif ext == '.tex' or p.opts.has('tex') or p.opts.has('booktabs'):
        txt = render_tex(T, models, titles, title, booktabs=p.opts.has('booktabs'))
    elif ext == '.csv':
        txt = render_text(T, models, titles, title).replace('  ', ',')
    else:
        txt = render_text(T, models, titles, title)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(txt)
    ip.txt(f'(output written to {fn})\n')


# ---------------------------------------------------------------- commands
@command('esttab', 'estout')
def cmd_esttab(ip, cmd, rest, by):
    p = parse_command(rest)
    if getattr(ip, 'estpost', None) and ('cells' in (p.opts.raw or '') and 'mean' in (p.opts.get('cells') or '')):
        return _esttab_summary(ip, p)
    models = _models(ip, split_top(p.body))
    titles = _titles(ip, models, p.opts)
    T = build_table(ip, models, p.opts, uselabels=p.opts.has('label'))
    tl = quoted_strings(p.opts.get('title') or '')
    title = ' '.join(tl) if tl else None
    if p.using:
        _write_out(ip, p, T, models, titles, title)
        return
    ip.txt('\n' + render_text(T, models, titles, title) + '\n')


@command('eststo')
def cmd_eststo(ip, cmd, rest, by):
    r = rest.strip()
    if r.startswith('clear'):
        for nm in ip.eststo_order:
            ip.stored.pop(nm, None)
        ip.eststo_order = []
        ip.estpost = None
        return
    nm = r.split(',')[0].strip() or f'est{len(ip.eststo_order) + 1}'
    ip.store_estimates(nm, eststo=True)


@command('estpost')
def cmd_estpost(ip, cmd, rest, by):
    sub = rest.strip().split()[0]
    p = parse_command(rest.strip()[len(sub):])
    d = ip.data
    vars_ = d.expand_varlist(split_top(p.body))
    stats = {}
    from .cmds_data import _stata_pctile
    for v in vars_:
        x = d.cols[v][ip.mark(p)]
        x = x[~np.isnan(x)]
        stats[v] = {'mean': x.mean(), 'sd': x.std(ddof=1), 'min': x.min(), 'max': x.max(), 'count': x.size,
                    'p50': _stata_pctile(x, 50)}
    ip.estpost = stats
    ip.txt(f'(estpost {sub}: results stored for {len(vars_)} variables)\n')


def _esttab_summary(ip, p):
    stats = ip.estpost
    cells = unquote(p.opts.get('cells'))
    spec = re.findall(r'(\w+)(?:\(fmt\((\d+)\)\))?', cells)
    heads = quoted_strings(p.opts.get('collabels') or '') or [s for s, _ in spec]
    lines = ['-' * (14 + 12 * len(spec)), f'{"":<14}' + ''.join(f'{h:>12}' for h in heads),
             '-' * (14 + 12 * len(spec))]
    for v, st in stats.items():
        lines.append(f'{v:<14}' + ''.join(f'{st.get(s, np.nan):>12.{int(f) if f else 3}f}' for s, f in spec))
    lines.append('-' * (14 + 12 * len(spec)))
    txt = '\n'.join(lines)
    if p.using:
        path = ip.path(p.using)
        if path.lower().endswith('.rtf'):
            body = txt.replace('\n', '\\par\n')
            txt = '{\\rtf1\\ansi{\\fonttbl{\\f0 Courier New;}}\\f0\\fs18 ' + body + '}'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(txt)
        ip.txt(f'(output written to {p.using})\n')
    else:
        ip.txt('\n' + txt + '\n')


def estimates_table(ip, arg):
    p = parse_command(arg)
    models = _models(ip, split_top(p.body))
    o = p.opts
    keep = split_top(o.get('keep')) if o.get('keep') else None
    drop = split_top(o.get('drop')) if o.get('drop') else None
    rows = _coef_rows(models, keep, drop, None)
    bfmt = o.get('b') or '%9.0g'
    sefmt = o.get('se')
    star = o.has('star')
    cw = 13
    ttl = quoted_strings(o.get('title') or '')
    ip.txt('\n')
    if ttl:
        ip.txt(ttl[0] + '\n')
    w = 13 + cw * len(models)
    ip.txt('-' * (w + 1) + '\n')
    ip.txt(f'{"Variable":>12} |' + ''.join(f'{(nm if nm != "." else "Active")[:12]:^{cw}}' for nm, _ in models) + '\n')
    ip.txt('-' * 13 + '+' + '-' * (cw * len(models)) + '\n')
    for r in rows:
        ip.txt(f'{abbrev(r):>12} |')
        line = ''
        line2 = ''
        for _, est in models:
            j = _lookup(est, r)
            if j is None:
                line += ' ' * cw
                line2 += ' ' * cw
                continue
            se, t, pv = _pvals(est)
            s = fmt_num(est['b'][j], bfmt).strip()
            if star:
                s += _stars(pv[j], [('*', 0.05), ('**', 0.01), ('***', 0.001)])
            line += f'{s:>{cw - 3}}   '
            if sefmt is not None:
                line2 += f'{fmt_num(se[j], sefmt or "%9.0g").strip():>{cw - 3}}   '
        ip.res(line + '\n')
        if sefmt is not None:
            ip.txt(f'{"":>12} |')
            ip.res(line2 + '\n')
    st = o.get('stats')
    if st:
        ip.txt('-' * 13 + '+' + '-' * (cw * len(models)) + '\n')
        for kk in split_top(st):
            ip.txt(f'{kk:>12} |')
            vals = ''
            for _, est in models:
                v = _stat_value(est, kk)
                vals += f'{(fmt_num(v, "%9.0g").strip() if v == v else "."):>{cw - 3}}   '
            ip.res(vals + '\n')
    ip.txt('-' * (w + 1) + '\n')
    if star:
        ip.txt(f'{"Legend: * p<0.05; ** p<0.01; *** p<0.001":>{w}}\n')


# ---------------------------------------------------------------- outreg2
@command('outreg2')
def cmd_outreg2(ip, cmd, rest, by):
    p = parse_command(rest)
    if ip.est is None:
        raise StataError('no estimates', 301)
    fn = p.using
    path = ip.path(fn)
    if not hasattr(ip, 'outreg'):
        ip.outreg = {}
    est = ip.est
    if p.opts.has('replace') or path not in ip.outreg:
        ip.outreg[path] = []
    ip.outreg[path].append((unquote(p.opts.get('ctitle') or est.get('depvar', '')), dict(est)))
    cols = ip.outreg[path]
    models = [(f'({k + 1})', e) for k, (_, e) in enumerate(cols)]
    titles = [t for t, _ in cols]
    dec = int(p.opts.get('dec') or 3)
    o = Options(f'b({dec}) se({dec}) star(*** 0.01 ** 0.05 * 0.1) r2')
    T = build_table(ip, models, o, uselabels=p.opts.has('label'))
    T['notes'] = ['Standard errors in parentheses', '*** p<0.01, ** p<0.05, * p<0.1']
    if p.opts.get('addnote'):
        T['notes'] += quoted_strings(p.opts.get('addnote'))
    tl = quoted_strings(p.opts.get('title') or '')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    rtf = render_rtf(T, models, titles, tl[0] if tl else None)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(rtf)
    if p.opts.has('word') and not path.lower().endswith(('.doc', '.rtf')):
        with open(os.path.splitext(path)[0] + '.rtf', 'w', encoding='utf-8') as f:
            f.write(rtf)
    ip.txt(f'{fn}\ndir : seeout\n')
