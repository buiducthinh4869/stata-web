"""irf set/create/table/ctable/graph/describe/drop after var and svar (ported from Stata's varirf_*.ado,
_varirf.ado and varirf_ctable.ado).  Standard errors are the asymptotic (delta-method) ones of _varirf.ado:
Var(vec A) = (X'X)^-1 (x) Sigma for the lag coefficients, Var(vech Sigma) = (2/T) D+ (Sigma (x) Sigma) D+',
and, after svar, e(V) of the structural parameters."""
import math
import re

import numpy as np
from scipy import stats as sps

from .errors import StataError
from .fmt import fmt_num
from .interp import command
from .syntax import parse_command, split_top
from .tsvar import lag_matrices, _lagname

STATS = ['irf', 'cirf', 'oirf', 'coirf', 'sirf', 'csirf', 'fevd', 'sfevd', 'mse']
LABELS = {'irf': 'impulse-response function (irf)', 'cirf': 'cumulative irf', 'oirf': 'orthogonalized irf',
          'coirf': 'cumulative orthogonalized irf', 'sirf': 'structural irf', 'fevd': 'fraction of mse due to impulse',
          'sfevd': '(structural) fraction of mse due to impulse', 'mse': 'SE of forecast of response variable'}
GLABEL = {'irf': 'impulse-response function (irf)', 'cirf': 'cumulative irf', 'oirf': 'orthogonalized irf',
          'coirf': 'cumulative orthogonalized irf', 'sirf': 'structural irf', 'fevd': 'fraction of mse due to impulse',
          'sfevd': '(structural) fraction of mse due to impulse'}


def _state(ip):
    st = getattr(ip, '_irf', None)
    if st is None:
        st = {'active': None, 'files': {}}
        ip._irf = st
    return st


def _fname(raw):
    f = raw.strip().strip('"')
    if not re.search(r'\.(irf|vrf)$', f, re.I):
        f += '.irf'
    return f


def _set_file(ip, raw, replace=False, loud=True):
    st = _state(ip)
    f = _fname(raw)
    if replace or f not in st['files']:
        st['files'][f] = {'irfnames': [], 'res': {}}
        if loud:
            ip.txt(f'(file {f} created)\n')
    st['active'] = f
    if loud:
        ip.txt(f'(file {f} now active)\n')
    return f


def _active(ip):
    st = _state(ip)
    f = st['active']
    if not f:
        raise StataError('no irf file active', 198)
    return st['files'][f], f


@command('irf')
def cmd_irf(ip, cmd, rest, by):
    m = re.match(r'\s*(\w*)(.*)$', rest, re.S)
    sub, tail = m.group(1), m.group(2)
    s = sub.lower()
    if s == 'set':
        return _irf_set(ip, tail)
    if s in ('cr', 'cre', 'crea', 'creat', 'create'):
        return _irf_create(ip, tail)
    if s in ('t', 'ta', 'tab', 'tabl', 'table'):
        return _irf_table(ip, tail)
    if s in ('ct', 'cta', 'ctab', 'ctabl', 'ctable'):
        return _irf_ctable(ip, tail)
    if s in ('g', 'gr', 'gra', 'grap', 'graph'):
        return _irf_graph(ip, tail)
    if s in ('cg', 'cgr', 'cgraph'):
        return _irf_graph(ip, tail, combined=True)
    if s in ('og', 'ograph'):
        return _irf_ograph(ip, tail)
    if s in ('d', 'de', 'des', 'desc', 'descr', 'describe'):
        return _irf_describe(ip, tail)
    if s == 'drop':
        return _irf_drop(ip, tail)
    if s in ('ren', 'rename'):
        return _irf_rename(ip, tail)
    raise StataError(f'irf {sub}: unknown subcommand', 198)


def _irf_set(ip, tail):
    p = parse_command(tail)
    st = _state(ip)
    if p.opts.has('clear'):
        st['active'] = None
        ip.txt('irf file has been cleared\n')
        return
    if not p.body.strip():
        if not st['active']:
            ip.txt('no irf file active\n')
        else:
            ip.txt(f"(file {st['active']} now active)\n")
            ip.set_r(macros={'irffile': st['active']})
        return
    _set_file(ip, p.body, replace=p.opts.has('replace'))


# ---------------------------------------------------------------- computation
def _dupl_plus(K):
    D = np.zeros((K * K, K * (K + 1) // 2))
    c = 0
    for j in range(K):
        for i in range(j, K):
            D[i + j * K, c] = 1
            D[j + i * K, c] = 1
            c += 1
    return np.linalg.pinv(D)


def _vech(S):
    K = S.shape[0]
    return np.array([S[i, j] for j in range(K) for i in range(j, K)])


def _unvech(v, K):
    S = np.zeros((K, K))
    c = 0
    for j in range(K):
        for i in range(j, K):
            S[i, j] = S[j, i] = v[c]
            c += 1
    return S


def irf_stats(A, Sig, step, perm, Ainv_B=None):
    """All irf statistics for lag matrices A (list), residual covariance Sig; arrays [step, response, impulse]
    in the permuted order."""
    K = Sig.shape[0]
    P_ = np.eye(K)[perm]
    A = [P_ @ a @ P_.T for a in A]
    Sig = P_ @ Sig @ P_.T
    p = len(A)
    Phi = [np.eye(K)]
    for i in range(1, step + 1):
        M = np.zeros((K, K))
        for j in range(1, min(i, p) + 1):
            M = M + Phi[i - j] @ A[j - 1]
        Phi.append(M)
    Phi = np.array(Phi)
    try:
        Pc = np.linalg.cholesky(Sig)
    except np.linalg.LinAlgError:
        Pc = np.full((K, K), np.nan)
    Th = Phi @ Pc
    out = {'irf': Phi, 'cirf': np.cumsum(Phi, axis=0), 'oirf': Th, 'coirf': np.cumsum(Th, axis=0)}
    fe, ms = _fevd(Th, step, K)
    out['fevd'] = fe
    out['mse'] = ms
    if Ainv_B is not None:
        Sm = Phi @ (P_ @ Ainv_B @ P_.T)
        out['sirf'] = Sm
        out['csirf'] = np.cumsum(Sm, axis=0)
        sf, _ = _fevd(Sm, step, K)
        out['sfevd'] = sf
    return out


def _fevd(Th, step, K):
    fe = np.zeros((step + 1, K, K))
    ms = np.zeros((step + 1, K, K))
    om = np.zeros((K, K))
    for h in range(1, step + 1):
        om = om + Th[h - 1] ** 2
        mse = om.sum(axis=1)
        fe[h] = om / mse[:, None]
        ms[h] = np.sqrt(mse)[:, None]
    return fe, ms


def irf_compute(ip, e, step, order=None, se=True):
    """Returns dict: order names, stats, std."""
    ve = e.get('varest', e)
    endog = ve['endog']
    K = len(endog)
    if order:
        if sorted(order) != sorted(endog):
            raise StataError(f"order({' '.join(order)}) must contain all the endogenous variables", 198)
        perm = [endog.index(v) for v in order]
    else:
        perm = list(range(K))
    names = [endog[i] for i in perm]
    A0 = lag_matrices(ve)
    Sig0 = ve['Sigma']
    T = ve['T']
    lags = ve['lags']
    svar = e.get('cmd') == 'svar'
    AB0 = None
    if svar:
        AB0 = np.linalg.inv(e['As']) @ e['Bs']
    base = irf_stats(A0, Sig0, step, perm, AB0)
    std = {}
    if se:
        # parameter vector: included lag coefficients A_l[i,j], vech(Sigma), svar free parameters
        X = ve['xnames']
        XX = ve['XtXi']
        idx = [(l, i, j) for l in lags for i in range(K) for j in range(K)]
        rr = [X.index(_lagname(l, endog[j])[0]) for (l, i, j) in idx]
        Ca = np.array([[XX[rr[a], rr[b]] * Sig0[idx[a][1], idx[b][1]] for b in range(len(idx))]
                       for a in range(len(idx))]) if idx else np.zeros((0, 0))
        Dp = _dupl_plus(K)
        Cs = (2.0 / T) * Dp @ np.kron(Sig0, Sig0) @ Dp.T
        th0 = [A0[l - 1][i, j] for (l, i, j) in idx] + list(_vech(Sig0))
        blocks = [Ca, Cs]
        nsv = 0
        if svar:
            fr = e['svar_free']
            nsv = len(fr)
            th0 += [e['svar_theta'][k] for k in range(nsv)]
            blocks.append(e['svar_Vfree'])
        th0 = np.array(th0, float)
        n = th0.size
        C = np.zeros((n, n))
        o = 0
        for bl in blocks:
            k = bl.shape[0]
            C[o:o + k, o:o + k] = bl
            o += k
        na = len(idx)
        ns = K * (K + 1) // 2

        def f(th):
            A = [a.copy() for a in A0]
            for (l, i, j), v in zip(idx, th[:na]):
                A[l - 1][i, j] = v
            Sg = _unvech(th[na:na + ns], K)
            ab = None
            if svar:
                Am, Bm = e['svar_build'](th[na + ns:])
                ab = np.linalg.inv(Am) @ Bm
            r = irf_stats(A, Sg, step, perm, ab)
            return np.concatenate([r[k].ravel() for k in sorted(r)])
        keys = sorted(base)
        sizes = [base[k].size for k in keys]
        J = np.zeros((sum(sizes), n))
        sd = np.sqrt(np.maximum(np.diag(C), 0))
        for k in range(n):
            h = 1e-4 * sd[k] if sd[k] > 0 else 1e-7
            tp, tm = th0.copy(), th0.copy()
            tp[k] += h
            tm[k] -= h
            J[:, k] = (f(tp) - f(tm)) / (2 * h)
        var = np.einsum('ij,jk,ik->i', J, C, J)
        o = 0
        for k, sz in zip(keys, sizes):
            std[k] = np.sqrt(np.maximum(var[o:o + sz], 0)).reshape(base[k].shape)
            o += sz
        std.pop('mse', None)
    return {'order': names, 'step': step, 'stats': base, 'std': std, 'model': 'svar' if svar else 'var',
            'endog': endog, 'tmins': ve['tmins'], 'tmaxs': ve['tmaxs'], 'lags': ' '.join(map(str, lags)),
            'stderr': 'asymptotic' if se else 'none'}


def _irf_create(ip, tail):
    p = parse_command(tail)
    name = p.body.strip()
    if not name or not re.match(r'^[A-Za-z_]\w*$', name):
        raise StataError('irf name required', 198)
    if len(name) > 15:
        raise StataError('irfname must have 15 or fewer characters', 198)
    e = ip.est
    if not e or e.get('cmd') not in ('var', 'svar', 'vec'):
        raise StataError('irf create can only estimate IRFs after var, svar, and vec', 198)
    if e.get('cmd') == 'vec':
        return __import__('stata.tsvec', fromlist=['x']).vec_irf_create(ip, p, name)
    if p.opts.has('bs') or p.opts.has('bsp'):
        raise StataError('bootstrap standard errors (bs, bsp) are not available in Stata Web; '
                         'asymptotic standard errors are computed by default', 198)
    order = p.opts.get('order', 3)
    if order and e.get('cmd') == 'svar':
        raise StataError('order cannot be specified after svar', 198)
    order = order.split() if order else None
    stp = p.opts.get('step', 2)
    step = int(stp) if stp not in (None, '') else 8
    if step < 1:
        raise StataError('step() must specify an integer greater than or equal to 1', 198)
    sv = p.opts.get('set', 3)
    if sv:
        _set_file(ip, sv, replace=False)
    fobj, f = _active(ip)
    if name in fobj['res']:
        if p.opts.has('replace'):
            fobj['irfnames'].remove(name)
            del fobj['res'][name]
        else:
            raise StataError(f'irfname {name} is already in {f}', 498)
    R = irf_compute(ip, e, step, order, se=not p.opts.has('nose'))
    fobj['res'][name] = R
    fobj['irfnames'].append(name)
    ip.txt(f'(file {f} updated)\n')


def _get_irfs(ip, p):
    fobj, f = _active(ip)
    want = (p.opts.get('irf', 2) or '').split()
    if not want:
        want = list(fobj['irfnames'])
    for w in want:
        if w not in fobj['res']:
            raise StataError(f'{w} is not a valid irfname', 498)
    return fobj, want


def _value(R, stat, resp, imp, kind):
    """kind: '' (value) / 'se' / ('lo', z) / ('hi', z)."""
    arr = R['stats'].get(stat)
    o = R['order']
    if arr is None:
        return np.full(R['step'] + 1, np.nan)
    v = arr[:, o.index(resp), o.index(imp)]
    if kind == '':
        return v
    sd = R['std'].get(stat)
    if sd is None:
        return np.full(v.shape, np.nan)
    s = sd[:, o.index(resp), o.index(imp)]
    if kind == 'se':
        return s
    z = kind[1]
    return v - z * s if kind[0] == 'lo' else v + z * s


def _irf_table(ip, tail):
    p = parse_command(tail)
    stats = p.body.split()
    if not stats:
        stats = ['irf', 'oirf', 'fevd'] if p.opts.has('nostructural', 5) else ['irf', 'oirf', 'sirf', 'fevd', 'sfevd']
    for s in stats:
        if s not in ('irf', 'cirf', 'oirf', 'coirf', 'sirf', 'csirf', 'fevd', 'sfevd', 'dm', 'cdm'):
            raise StataError(f'{s} not valid', 198)
    fobj, irfs = _get_irfs(ip, p)
    imp = (p.opts.get('impulse', 1) or '').split()
    resp = (p.opts.get('response', 1) or '').split()
    specs = []
    for nm in irfs:
        R = fobj['res'][nm]
        for i in (imp or R['order']):
            if i not in R['order']:
                raise StataError(f'{i} is not a valid impulse', 498)
            for r in (resp or R['order']):
                if r not in R['order']:
                    raise StataError(f'{r} is not a valid response', 498)
                specs.append((nm, i, r, stats))
    title = p.opts.get('title', 2)
    title = title.strip().strip('"') if title else f"Results from {' '.join(irfs)}"
    _ctable(ip, fobj, specs, noci=p.opts.has('noci', 4), stderror=p.opts.has('stderror', 3),
            level=p.opts.get('level', 1), stepmax=p.opts.get('step', 2), title=title)


def _irf_ctable(ip, tail):
    p = parse_command(tail)
    fobj, _ = _active(ip)
    specs = []
    for grp in re.findall(r'\(([^()]*)\)', p.body):
        g = grp.split(',')[0].split()
        if len(g) < 4:
            raise StataError(f'tables not properly specified in {p.body}', 198)
        if g[0] not in fobj['res']:
            raise StataError(f'{g[0]} is not a valid irfname', 498)
        specs.append((g[0], g[1], g[2], g[3:]))
    title = p.opts.get('title', 2)
    _ctable(ip, fobj, specs, noci=p.opts.has('noci', 4), stderror=p.opts.has('stderror', 3),
            level=p.opts.get('level', 1), stepmax=p.opts.get('step', 2),
            title=title.strip().strip('"') if title else None)


def _ctable(ip, fobj, specs, noci, stderror, level, stepmax, title):
    lev = float(level) if level not in (None, '') else 95.0
    z = sps.norm.ppf(1 - (100 - lev) / 200)
    triples, cols = [], []     # cols: (k_u, header, values, blocklen)
    anyci = False
    for nm, i, r, stats in specs:
        R = fobj['res'][nm]
        key = (nm, i, r)
        if key not in triples:
            triples.append(key)
        ku = triples.index(key) + 1
        if R['model'] == 'vec':
            noci_ = True
        else:
            noci_ = noci
        for s in stats:
            if s in ('sirf', 'sfevd', 'csirf') and R['model'] == 'var':
                pass
            blk = [(s, _value(R, s, r, i, ''))]
            if not noci_:
                anyci = True
                blk += [('Lower', _value(R, s, r, i, ('lo', z))), ('Upper', _value(R, s, r, i, ('hi', z)))]
            if stderror:
                blk.append(('S.E.', _value(R, s, r, i, 'se')))
            for hd, v in blk:
                cols.append((ku, hd, v, len(blk)))
    smax = max(fobj['res'][s[0]]['step'] for s in specs) + 1
    if stepmax not in (None, ''):
        smax = min(smax, int(stepmax) + 1)
    lstepc, lcol, linesize = 10, 12, 80
    cpline = int((linesize - lstepc) / lcol)
    if title:
        ip.txt('\n')
        if len(cols) >= cpline:
            ip.txt(title.center(linesize).rstrip() + '\n')
        else:
            ind = int((lstepc + len(cols) * lcol - len(title)) / 2)
            ip.txt(' ' * max(ind - 1, 0) + title + '\n')
    # split into tables of whole blocks
    tables, cur, cpos, k = [], [], lstepc, 0
    while k < len(cols):
        bl = cols[k][3]
        blk = cols[k:k + bl]
        if cur and cpos + bl * lcol > linesize:
            tables.append(cur)
            cur, cpos = [], lstepc
        cur.append(blk)
        cpos += bl * lcol
        k += bl
    if cur:
        tables.append(cur)
    for tb in tables:
        top, mid, bot = '+' + '-' * (lstepc - 2) + '+', '|' + '-' * (lstepc - 2) + '+', '+' + '-' * (lstepc - 2) + '+'
        h1 = '|' + ' ' * (lstepc - 2) + '|'
        h2 = _at('|', 4, 'step')
        h2 = h2 + ' ' * (lstepc - 1 - len(h2)) + '|'
        rows = []
        for s in range(smax):
            v = fmt_num(float(s), '%-6.0f')
            line = '|' + v
            rows.append([(line + ' ' * (lstepc - 1 - len(line)) + '|', 'txt')])
        cpos = lstepc
        nb = len(tb)
        for bi, blk in enumerate(tb):
            for ci, (ku, hd, vals, bl) in enumerate(blk):
                end = ci == len(blk) - 1
                last = bi == nb - 1
                ch_top, ch_mid, ch_bot = (('+', '|' if last else '+', '+') if end else ('-', '-', '-'))
                top += '-' * (lcol - 1) + ch_top
                mid += '-' * (lcol - 1) + ch_mid
                bot += '-' * (lcol - 1) + ch_bot
                ktxt = f'({ku})'
                h1 = _pad_to(h1, cpos + int((lcol - len(str(ku))) / 2)) + ktxt
                h2 = _pad_to(h2, cpos + int((lcol - len(hd)) / 2)) + hd
                if end:
                    h1 = _pad_to(h1, cpos + lcol) + '|'
                    h2 = _pad_to(h2, cpos + lcol) + '|'
                for s in range(smax):
                    v = vals[s] if s < len(vals) else np.nan
                    txt = fmt_num(v, '%-8.6g')
                    seg = rows[s]
                    ln = sum(len(t) for t, _ in seg)
                    target = 2 + cpos
                    seg.append((' ' * max(target - 1 - ln, 0) + txt, 'res'))
                    if end:
                        ln = sum(len(t) for t, _ in seg)
                        seg.append((' ' * max(cpos + lcol - 1 - ln, 0) + '|', 'txt'))
                cpos += lcol
        ip.txt('\n' + top + '\n' + h1 + '\n' + h2 + '\n' + mid + '\n')
        for seg in rows:
            for t, sty in seg:
                (ip.res if sty == 'res' else ip.txt)(t)
            ip.txt('\n')
        ip.txt(bot + '\n')
    if anyci:
        ip.txt(f'{fmt_num(lev, "%9.0g").strip()}% lower and upper bounds reported\n')
    for k, (nm, i, r) in enumerate(triples, start=1):
        ip.txt(f'({k}) irfname = {nm}, impulse = {i}, and response = {r}\n')
    ip.set_r(scalars={'ncols': len(cols), 'k_umax': len(triples), 'k': len(specs)},
             macros={f'key{k}': f'{nm} {i} {r}' for k, (nm, i, r) in enumerate(triples, start=1)})


def _pad_to(s, col):
    """pad string so the next char is written at 1-based column `col`."""
    return s + ' ' * max(col - 1 - len(s), 0)


def _at(s, col, txt):
    return _pad_to(s, col) + txt


# ---------------------------------------------------------------- graphs
def _irf_graph(ip, tail, combined=False):
    from .graphs import Figure, finish_graph, stata_color, apply_global_options
    p = parse_command(tail)
    stats = p.body.split()
    if not stats:
        raise StataError('statlist required', 198)
    if len(stats) > 4:
        raise StataError('at most 4 statistics may be graphed', 198)
    fobj, irfs = _get_irfs(ip, p)
    imp = (p.opts.get('impulse', 1) or '').split()
    resp = (p.opts.get('response', 1) or '').split()
    lev = p.opts.get('level', 1)
    lev = float(lev) if lev not in (None, '') else 95.0
    z = sps.norm.ppf(1 - (100 - lev) / 200)
    noci = p.opts.has('noci', 4)
    panels = []
    for nm in irfs:
        R = fobj['res'][nm]
        for i in (imp or R['order']):
            for r in (resp or R['order']):
                if i not in R['order'] or r not in R['order']:
                    raise StataError(f'{i if i not in R["order"] else r} is not valid', 498)
                panels.append((nm, i, r, R))
    n = len(panels)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig = Figure(ip, rows, cols, figsize=(min(3.6 * cols, 13), 2.9 * rows + 1.0), share=True)
    colors = [stata_color(c) for c in ('navy', 'maroon', 'forest_green', 'dkorange')]
    handles = []
    stmax = p.opts.get('ustep', 5)
    for k, ax in enumerate(fig.axes):
        if k >= n:
            ax.set_visible(False)
            continue
        nm, i, r, R = panels[k]
        steps = np.arange(R['step'] + 1)
        if stmax not in (None, ''):
            steps = steps[steps <= int(stmax)]
        hs = []
        for si, s in enumerate(stats):
            v = _value(R, s, r, i, '')[:steps.size]
            if not noci and R['std'].get(s) is not None:
                lo = _value(R, s, r, i, ('lo', z))[:steps.size]
                hi = _value(R, s, r, i, ('hi', z))[:steps.size]
                cc = colors[si][:3]
                fc = (0.75, 0.77, 0.80) if si == 0 else tuple(0.55 + 0.45 * c for c in cc)
                h1 = ax.fill_between(steps, lo, hi, color=fc, alpha=1.0 if si == 0 else 0.6, lw=0)
                hs.append((h1, f'{lev:g}% CI' + ('' if len(stats) == 1 else f' ({s})')))
            h2, = ax.plot(steps, v, color=colors[si], lw=1.4)
            hs.append((h2, GLABEL.get(s, s)))
        ax.set_title(f'{nm}, {i}, {r}', fontsize=9, color='#22272b')
        if k == 0:
            handles = hs
    fig.legend_items = []
    gopts = p.opts
    apply_global_options(fig, gopts)
    H = fig.fig.get_figheight()
    showleg = handles and not str(p.opts.get('legend', 3) or '').startswith('off')
    yleg = 0.32 / H
    fig.fig.supxlabel(fig.titles.pop('xtitle', None) or 'step', fontsize=9.5,
                      y=(yleg + 0.30 / H) if showleg else 0.30 / H)
    if showleg:
        fig.fig.legend([h for h, _ in handles], [l for _, l in handles], loc='lower center',
                       bbox_to_anchor=(0.5, yleg), ncol=min(4, len(handles)), fontsize=8.5, frameon=False)
        fig.bottom_extra = 0.62 / H
    else:
        fig.bottom_extra = 0.25 / H
    fig.legend_opts = 'off'
    fig.titles.setdefault('note', 'Graphs by irfname, impulse variable, and response variable')
    finish_graph(ip, fig, gopts)


def _irf_ograph(ip, tail):
    from .graphs import Figure, finish_graph, stata_color, apply_global_options
    p = parse_command(tail)
    fobj, _ = _active(ip)
    fig = Figure(ip)
    ax = fig.ax
    cl = ['navy', 'maroon', 'forest_green', 'dkorange', 'teal', 'cranberry']
    for k, grp in enumerate(re.findall(r'\(([^()]*)\)', p.body)):
        g = grp.split(',')[0].split()
        if len(g) < 4:
            raise StataError('ograph: (irfname impulse response stat) required', 198)
        nm, i, r, s = g[:4]
        if nm not in fobj['res']:
            raise StataError(f'{nm} is not a valid irfname', 498)
        R = fobj['res'][nm]
        v = _value(R, s, r, i, '')
        ax.plot(np.arange(v.size), v, color=stata_color(cl[k % len(cl)]), lw=1.4, label=f'{nm}: {s} of {i} -> {r}')
        fig.legend_items.append((ax.lines[-1], f'{nm}: {s} of {i} -> {r}'))
    fig.set_default_titles(xtitle='step')
    apply_global_options(fig, p.opts)
    finish_graph(ip, fig, p.opts)


def _irf_describe(ip, tail):
    fobj, f = _active(ip)
    ip.txt(f'\nContains irf results from {f}\n\n')
    ip.txt(f'{"irfname":<15} {"model":<8} {"endogenous variables and order (*)":<40}\n')
    ip.txt('-' * 78 + '\n')
    for nm in fobj['irfnames']:
        R = fobj['res'][nm]
        ip.txt(f"{nm:<15} {R['model']:<8} {' '.join(R['order'])}\n")
    ip.txt('-' * 78 + '\n')
    ip.set_r(macros={'irfnames': ' '.join(fobj['irfnames'])})


def _irf_drop(ip, tail):
    p = parse_command(tail)
    fobj, f = _active(ip)
    for nm in p.body.split():
        if nm not in fobj['res']:
            raise StataError(f'{nm} not found in {f}', 498)
        del fobj['res'][nm]
        fobj['irfnames'].remove(nm)
        ip.txt(f'({nm} dropped)\n')
    ip.txt(f'file {f} updated\n')


def _irf_rename(ip, tail):
    toks = split_top(parse_command(tail).body)
    fobj, f = _active(ip)
    if len(toks) != 2 or toks[0] not in fobj['res']:
        raise StataError('irf rename oldname newname', 198)
    fobj['res'][toks[1]] = fobj['res'].pop(toks[0])
    fobj['irfnames'][fobj['irfnames'].index(toks[0])] = toks[1]
    ip.txt(f'({toks[0]} renamed to {toks[1]})\n')


