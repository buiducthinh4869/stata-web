"""SCM (synth2), staggered DID (csdid), SDID, didregress, ritest."""
import math
import re

import numpy as np
from scipy import optimize as spo
from scipy import stats as sps

from . import expr as E
from .design import coef_table, right_lines, build_design, mark_collinear
from .errors import StataError
from .estim import parse_vce, cluster_ids, sandwich, wald_test
from .fmt import g_fmt, fmt_num, comma_int, abbrev, num_to_macro
from .interp import command, Results
from .syntax import parse_command, split_top, unquote, Options, parse_numlist


def _num(s, default=None):
    try:
        return float(str(s).replace('`', '').replace("'", '').strip())
    except (TypeError, ValueError):
        return default


# ================================================================ synth (core)
def _simplex_ls(A, b, Vd=None):
    """min ||A w - b||^2_V  s.t. w >= 0, sum w = 1 (via NNLS with heavy penalty)."""
    from scipy.optimize import nnls
    if Vd is not None:
        s = np.sqrt(Vd)
        A = A * s[:, None]
        b = b * s
    scale = max(1.0, float(np.abs(A).max()))
    lam = 1e4 * scale
    A2 = np.vstack([A, lam * np.ones((1, A.shape[1]))])
    b2 = np.concatenate([b, [lam]])
    w, _ = nnls(A2, b2, maxiter=5000)
    s_ = w.sum()
    if s_ > 0:
        w = w / s_
    return w


class SynthResult:
    pass


def synth_fit(Y, X1, X0, Z1, Z0, nested=True, allopt=False, fast=False):
    """Y unused; X1 (k,), X0 (k,J); Z1 (T0,), Z0 (T0,J). Returns (w, v)."""
    k = X1.shape[0]
    sd = np.std(np.column_stack([X1[:, None], X0]), axis=1, ddof=1)
    sd[sd == 0] = 1
    X1n, X0n = X1 / sd, X0 / sd[:, None]

    def W_of(v):
        return _simplex_ls(X0n, X1n, v)

    def loss_v(theta):
        v = np.exp(theta - theta.max())
        v = v / v.sum()
        w = W_of(v)
        r = Z1 - Z0 @ w
        return float(r @ r) / len(r)
    starts = [np.zeros(k)]
    # regression-based start: weights from regressing Z on X
    try:
        XX = np.column_stack([X1n, X0n]).T
        ZZ = np.column_stack([Z1, Z0]).T
        B = np.linalg.lstsq(np.column_stack([np.ones(XX.shape[0]), XX]), ZZ, rcond=None)[0][1:]
        vr = np.sum(B ** 2, axis=1)
        vr = vr / vr.sum() if vr.sum() > 0 else np.ones(k) / k
        starts.append(np.log(np.maximum(vr, 1e-8)))
    except np.linalg.LinAlgError:
        pass
    if allopt:
        rng = np.random.default_rng(1)
        for _ in range(3):
            starts.append(rng.normal(0, 1.5, k))
    best = None
    if not nested:
        theta = starts[-1]
        v = np.exp(theta - theta.max())
        v /= v.sum()
        return W_of(v), v, loss_v(theta)
    if fast:
        starts = starts[-1:]
    for st in starts:
        r = spo.minimize(loss_v, st, method='Nelder-Mead',
                         options={'maxiter': (300 if fast else 1500) * k, 'xatol': 1e-5 if fast else 1e-6,
                                  'fatol': 1e-8 if fast else 1e-10, 'adaptive': True})
        if fast:
            cand = r
            if best is None or cand.fun < best.fun:
                best = cand
            continue
        r2 = spo.minimize(loss_v, r.x, method='Powell', options={'maxiter': 3000, 'xtol': 1e-6, 'ftol': 1e-12})
        cand = r2 if r2.fun < r.fun else r
        if best is None or cand.fun < best.fun:
            best = cand
    theta = best.x
    v = np.exp(theta - theta.max())
    v /= v.sum()
    return W_of(v), v, float(best.fun)


def _synth_spec(ip, d, p, unitvar, timevar, trperiod, units, times):
    toks = split_top(p.body)
    yv = d.resolve(toks[0])
    preds = toks[1:]
    xperiod = p.opts.get('xperiod')
    pre_times = [t for t in times if t < trperiod]
    xp = parse_numlist(xperiod) if xperiod else pre_times
    mspe = p.opts.get('mspeperiod')
    mp = parse_numlist(mspe) if mspe else pre_times
    rows = []
    for tk in preds:
        m = re.match(r'^(\w+)\((.*)\)$', tk)
        if m:
            var = d.resolve(m.group(1))
            per = m.group(2)
            if '&' in per:
                pl = [float(x) for x in per.split('&')]
            else:
                pl = parse_numlist(per)
            rows.append((f'{var}({per})', var, pl))
        else:
            var = d.resolve(tk)
            rows.append((var, var, xp))
    return yv, rows, mp


def _panel_matrix(d, var, unitvar, timevar, units, times):
    u = d.cols[unitvar]
    t = d.cols[timevar]
    x = d.cols[var]
    M = np.full((len(times), len(units)), np.nan)
    ui = {v: i for i, v in enumerate(units)}
    ti = {v: i for i, v in enumerate(times)}
    for a, b, c in zip(u, t, x):
        if a in ui and b in ti:
            M[ti[b], ui[a]] = c
    return M


def run_synth(ip, d, p, trunit, trperiod, donors_excl=(), quiet=False, mspe_override=None, fast=False):
    unitvar, timevar = d.ts['panel'], d.ts['time']
    units = sorted(np.unique(d.cols[unitvar][~np.isnan(d.cols[unitvar])]).tolist())
    times = sorted(np.unique(d.cols[timevar][~np.isnan(d.cols[timevar])]).tolist())
    yv, rows, mp = _synth_spec(ip, d, p, unitvar, timevar, trperiod, units, times)
    Yp = _panel_matrix(d, yv, unitvar, timevar, units, times)
    tr_i = units.index(trunit)
    donors = [i for i, u in enumerate(units) if i != tr_i and u not in donors_excl]
    Xall = []
    for lab, var, per in rows:
        M = _panel_matrix(d, var, unitvar, timevar, units, times)
        ti = [times.index(t) for t in per if t in times]
        with np.errstate(all='ignore'):
            Xall.append(np.nanmean(M[ti, :], axis=0))
    X = np.array(Xall)
    mpi = [times.index(t) for t in (mspe_override or mp) if t in times]
    w, v, loss = synth_fit(None, X[:, tr_i], X[:, donors], Yp[mpi, tr_i], Yp[np.ix_(mpi, donors)],
                           nested=p.opts.has('nested'), allopt=p.opts.has('allopt') and not fast, fast=fast)
    synth_y = Yp[:, donors] @ w
    res = SynthResult()
    res.units, res.times, res.tr_i, res.donors = units, times, tr_i, donors
    res.w, res.v, res.X, res.rows, res.Yp = w, v, X, rows, Yp
    res.y_tr = Yp[:, tr_i]
    res.y_syn = synth_y
    res.yv = yv
    pre = [k for k, t in enumerate(times) if t < trperiod]
    post = [k for k, t in enumerate(times) if t >= trperiod]
    res.pre, res.post = pre, post
    gap = res.y_tr - synth_y
    res.gap = gap
    res.pre_mspe = float(np.mean(gap[pre] ** 2))
    res.post_mspe = float(np.mean(gap[post] ** 2)) if post else np.nan
    res.rmspe = math.sqrt(res.pre_mspe)
    return res


@command('synth', 'synth2')
def cmd_synth2(ip, cmd, rest, by):
    from .graphs import Figure, finish_graph, stata_color
    p = parse_command(rest)
    d = ip.data
    if not d.ts or not d.ts.get('panel'):
        raise StataError('panel variable not set; use tsset panelvar timevar', 459)
    unitvar, timevar = d.ts['panel'], d.ts['time']
    trunit = _num(p.opts.get('trunit'))
    trperiod = _num(p.opts.get('trperiod'))
    if trunit is None or trperiod is None:
        raise StataError('options trunit() and trperiod() required', 198)
    placebo = p.opts.get('placebo')
    show_fig = p.opts.has('fig') or cmd == 'synth2'
    res = run_synth(ip, d, p, trunit, trperiod)
    units, times = res.units, res.times
    trlab = d.value_text(unitvar, trunit) or g_fmt(trunit)
    pre_y = res.y_tr[res.pre]
    r2 = 1 - np.sum((pre_y - res.y_syn[res.pre]) ** 2) / np.sum((pre_y - pre_y.mean()) ** 2)
    D80 = '-' * 80 + '\n'
    ip.txt('Fitting results in the pretreatment periods:\n' + D80)
    ip.txt(' Number of Control Units  = '); ip.res(f'{len(res.donors):>10}')
    ip.txt('     Root Mean Squared Error  = '); ip.res(f'{res.rmspe:>10.5f}\n')
    ip.txt(' Number of Covariates     = '); ip.res(f'{len(res.rows):>10}')
    ip.txt('     R-squared                = '); ip.res(f'{r2:>10.5f}\n')
    ip.txt(D80)
    ip.txt('\nCovariate balance in the pretreatment periods:\n' + D80)
    ip.txt('   Covariate   |  V.weight    Treated    Synthetic Control     Average Control  \n')
    ip.txt('               |                        Value          Bias   Value        Bias \n')
    ip.txt('-' * 15 + '+' + '-' * 64 + '\n')
    bal = []
    for k, (lab, var, per) in enumerate(res.rows):
        xt = res.X[k, res.tr_i]
        xs = float(res.X[k, res.donors] @ res.w)
        avg = float(np.mean(res.X[k, res.donors]))
        bal.append([xt, xs, avg])
        b1 = (xs - xt) / xt * 100 if xt else 0
        b2 = (avg - xt) / xt * 100 if xt else 0
        ip.txt(f'{lab[-14:]:>14} |')
        ip.res(f'{res.v[k]:>9.4f}{xt:>12.4f}{xs:>12.4f}{b1:>9.2f}%{avg:>11.4f}{b2:>9.2f}%\n')
    ip.txt(D80)
    ip.txt('Note: "V.weight" is the optimal covariate weight in the diagonal of V matrix.\n'
           '      "Synthetic Control" is the weighted average of donor units with optimal\n'
           '      weights.\n'
           '      "Average Control" is the simple average of all control units with equal\n'
           '      weights.\n')
    ip.txt('\nOptimal Unit Weights:\n' + '-' * 26 + '\n     Unit    |    U.weight\n' + '-' * 13 + '+' + '-' * 12 + '\n')
    zero = []
    for j, w in sorted(zip(res.donors, res.w), key=lambda t: -t[1]):
        lab = d.value_text(unitvar, units[j]) or g_fmt(units[j])
        if round(w, 3) > 0:
            ip.txt(f'{lab[-12:]:>12} |'); ip.res(f'{round(w, 3):>11.4f} \n')
        else:
            zero.append((j, lab))
    ip.txt('-' * 26 + '\n')
    if zero:
        words = ['The', 'unit'] + [lb for _, lb in sorted(zero)] + ['in', 'the', 'donor', 'pool', 'get', 'a', 'weight', 'of', '0.']
        line, first = 'Note:', True
        for w_ in words:
            if len(line) + 1 + len(w_) > 78:
                ip.txt(line + '\n'); line = '     '
            line += ' ' + w_
        ip.txt(line + '\n')
    ip.txt('\nPrediction results in the posttreatment periods:\n' + '-' * 59 + '\n')
    ip.txt(' Time | Actual Outcome  Synthetic Outcome  Treatment Effect\n' + '-' * 6 + '+' + '-' * 52 + '\n')
    for k in res.post:
        ip.txt(f'{g_fmt(times[k]).strip():>5} |')
        ip.res(f'{res.y_tr[k]:>14.4f}{res.y_syn[k]:>19.4f}{res.gap[k]:>18.4f} \n')
    ip.txt('-' * 6 + '+' + '-' * 52 + '\n')
    att = float(np.mean(res.gap[res.post]))
    ip.txt(' Mean |')
    ip.res(f'{np.mean(res.y_tr[res.post]):>14.4f}{np.mean(res.y_syn[res.post]):>19.4f}{att:>18.4f} \n')
    ip.txt('-' * 59 + '\n')
    ip.txt(f'Note: The average treatment effect over the posttreatment period is {att:.4f}.\n')
    e_sc = {'N': len(units) * len(times), 'T': len(times), 'T0': len(res.pre), 'T1': len(res.post),
            'K': len(res.rows), 'J': len(units), 'rmse': res.rmspe, 'r2': r2, 'att': att, 'mspe': res.pre_mspe}
    U = E.Matrix(np.column_stack([[units[j] for j in res.donors], res.w]),
                 [str(int(units[j])) if float(units[j]).is_integer() else str(units[j]) for j in res.donors],
                 ['unit', 'U.weight'])
    Vm = E.Matrix(res.v.reshape(-1, 1), [r[0] for r in res.rows], ['V.weight'])
    B = E.Matrix(np.array(bal), [r[0] for r in res.rows], ['Treated', 'Synthetic', 'Average'])
    P = E.Matrix(np.column_stack([times, res.y_tr, res.y_syn, res.gap]), [g_fmt(t) for t in times],
                 ['time', 'actual', 'synthetic', 'effect'])
    mats = {'U_wt': U, 'V_wt': Vm, 'bal': B, 'pred': P}
    # ----- placebo / loo
    if placebo:
        _synth_placebo(ip, d, p, res, trunit, trperiod, placebo, e_sc, mats)
    if p.opts.has('loo'):
        _synth_loo(ip, d, p, res, trunit, trperiod)
    ip.set_estimation({'names': [], 'b': np.zeros(0), 'V': np.zeros((0, 0)), 'cmd': 'synth2',
                       'scalars': e_sc, 'macros': {'cmd': 'synth2', 'depvar': res.yv, 'panelvar': unitvar,
                                                   'timevar': timevar, 'trunit': g_fmt(trunit),
                                                   'trperiod': g_fmt(trperiod)},
                       'matrices': mats})
    if show_fig:
        _synth_figs(ip, d, p, res, trperiod, trlab)


def _synth_figs(ip, d, p, res, trperiod, trlab):
    from .graphs import Figure, finish_graph, stata_color
    times = res.times
    fig = Figure(ip)
    ax = fig.ax
    ax.plot(times, res.y_tr, color=stata_color('navy'), lw=2, label=f'{trlab}')
    ax.plot(times, res.y_syn, color=stata_color('maroon'), lw=2, ls='--', label=f'Synthetic {trlab}')
    ax.axvline(trperiod - 0.5 if all(float(t).is_integer() for t in times) else trperiod, color='gray', ls=':')
    ax.set_xlabel(d.varlab.get(d.ts['time'], d.ts['time']))
    ax.set_ylabel(d.varlab.get(res.yv, res.yv))
    ax.set_title('Actual and Synthetic Outcomes', fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    finish_graph(ip, fig, Options(''), name='pred')
    fig = Figure(ip)
    ax = fig.ax
    ax.plot(times, res.gap, color=stata_color('navy'), lw=2)
    ax.axhline(0, color='black', lw=0.7)
    ax.axvline(trperiod - 0.5, color='gray', ls=':')
    ax.set_title('Treatment Effects', fontsize=11)
    ax.set_xlabel(d.ts['time'])
    ax.set_ylabel('Treatment effect')
    finish_graph(ip, fig, Options(''), name='eff')


def _synth_placebo(ip, d, p, res, trunit, trperiod, spec, e_sc, mats):
    from .graphs import Figure, finish_graph, stata_color
    unitvar = d.ts['panel']
    units, times = res.units, res.times
    m = re.search(r'unit(?:\s+cut\(([^)]*)\))?', spec)
    mp = re.search(r'period\(([^)]*)\)', spec)
    sigf = int(_num(p.opts.get('sigf'), 3))
    if mp:
        per = parse_numlist(mp.group(1))
        for fake in per:
            ip.txt(f'\nIn-time placebo test: fake treatment time = {g_fmt(fake)}\n')
            pre_t = [t for t in times if t < fake]
            r2 = run_synth(ip, d, p, trunit, fake, mspe_override=pre_t)
            ip.txt('-' * 60 + '\n')
            ip.txt(f'{"Time":>10} |{"Actual":>12}{"Synthetic":>12}{"Effect":>12}\n' + '-' * 11 + '+' + '-' * 36 + '\n')
            for k, t in enumerate(times):
                if t >= fake:
                    ip.txt(f'{g_fmt(t):>10} |')
                    ip.res(f'{g_fmt(r2.y_tr[k], 9):>12}{g_fmt(r2.y_syn[k], 9):>12}{g_fmt(r2.gap[k], 9):>12}\n')
            ip.txt('-' * 60 + '\n')
            fig = Figure(ip)
            ax = fig.ax
            ax.plot(times, r2.y_tr, color=stata_color('navy'), lw=2, label='Actual')
            ax.plot(times, r2.y_syn, color=stata_color('maroon'), lw=2, ls='--', label='Synthetic (fake time)')
            ax.axvline(fake - 0.5, color='gray', ls=':')
            ax.axvline(trperiod - 0.5, color='black', ls='--', lw=0.8)
            ax.set_title(f'In-time placebo test (fake treatment time {g_fmt(fake)})', fontsize=11)
            ax.legend(frameon=False, fontsize=9)
            finish_graph(ip, fig, Options(''), name='placebo_period')
        return
    cut = _num(m.group(1)) if (m and m.group(1)) else None
    gaps = {}
    ratios = {}
    pre_mspe = {}
    ip.txt(f'\nIn-space placebo test: fitting {len(res.donors)} placebo units ...\n')
    for j in res.donors:
        u = units[j]
        r2 = run_synth(ip, d, p, u, trperiod, donors_excl=(trunit,), fast=True)
        gaps[u] = r2.gap
        pre_mspe[u] = r2.pre_mspe
        ratios[u] = r2.post_mspe / r2.pre_mspe if r2.pre_mspe > 0 else np.nan
    keep = [u for u in gaps if cut is None or pre_mspe[u] <= cut * res.pre_mspe]
    tr_gap = res.gap
    post = res.post
    pvals = []
    for k in post:
        cnt = sum(abs(gaps[u][k]) >= abs(tr_gap[k]) for u in keep)
        pvals.append((cnt + 1) / (len(keep) + 1))
    tr_ratio = res.post_mspe / res.pre_mspe
    rank = 1 + sum(ratios[u] >= tr_ratio for u in keep)
    p_ratio = rank / (len(keep) + 1)
    ip.txt(f'(placebo units kept: {len(keep)} of {len(gaps)}' + (f'; dropped units with pre-MSPE > {g_fmt(cut)} times the treated unit)\n' if cut else ')\n'))
    ip.txt('\nIn-space placebo test results using fake treatment units\n' + '-' * 66 + '\n')
    ip.txt(f'{"Time":>10} |{"Treatment effect":>18}{"p-value (two-sided)":>22}\n' + '-' * 11 + '+' + '-' * 42 + '\n')
    for k, pv in zip(post, pvals):
        ip.txt(f'{g_fmt(res.times[k]):>10} |')
        ip.res(f'{g_fmt(tr_gap[k], 9):>18}{pv:>22.{sigf}f}\n')
    ip.txt('-' * 66 + '\n')
    ip.txt(f'Note: post/pre-treatment MSPE ratio of treated unit ranks {rank} of {len(keep) + 1} (p = {p_ratio:.{sigf}f})\n')
    e_sc['p_ratio'] = p_ratio
    mats['pval'] = E.Matrix(np.column_stack([[res.times[k] for k in post], pvals]),
                            [g_fmt(res.times[k]) for k in post], ['time', 'pvalue'])
    mats['mspe'] = E.Matrix(np.array([[tr_ratio]] + [[ratios[u]] for u in keep]),
                            ['treated'] + [g_fmt(u) for u in keep], ['ratio'])
    fig = Figure(ip)
    ax = fig.ax
    for u in keep:
        ax.plot(res.times, gaps[u], color='0.75', lw=0.8)
    ax.plot(res.times, tr_gap, color=stata_color('navy'), lw=2.2, label='Treated unit')
    ax.axhline(0, color='black', lw=0.6)
    ax.axvline(trperiod - 0.5, color='gray', ls=':')
    ax.set_title('In-space placebo test: treatment effects', fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    finish_graph(ip, fig, Options(''), name='placebo_unit')


def _synth_loo(ip, d, p, res, trunit, trperiod):
    from .graphs import Figure, finish_graph, stata_color
    units = res.units
    pos = [j for j, w in zip(res.donors, res.w) if w > 1e-4]
    fig = Figure(ip)
    ax = fig.ax
    ip.txt('\nLeave-one-out robustness test\n' + '-' * 50 + '\n')
    for j in pos:
        r2 = run_synth(ip, d, p, trunit, trperiod, donors_excl=(units[j],), fast=True)
        ax.plot(res.times, r2.y_syn, color='0.7', lw=0.9)
        ip.txt(f'  excluding unit {g_fmt(units[j]):>8}: mean effect = ')
        ip.res(f'{np.mean(r2.gap[res.post]):.4f}\n')
    ax.plot(res.times, res.y_tr, color=stata_color('navy'), lw=2, label='Treated')
    ax.plot(res.times, res.y_syn, color=stata_color('maroon'), lw=2, ls='--', label='Synthetic')
    ax.axvline(trperiod - 0.5, color='gray', ls=':')
    ax.set_title('Leave-one-out robustness test', fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    finish_graph(ip, fig, Options(''), name='loo')


# ================================================================ csdid
@command('csdid')
def cmd_csdid(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    yv = d.resolve(split_top(p.body)[0])
    iv = d.resolve(p.opts.get('ivar'))
    tv = d.resolve(p.opts.get('time'))
    gv = d.resolve(p.opts.get('gvar'))
    notyet = p.opts.has('notyet')
    base = ip.mark(p)
    for v in (yv, iv, tv, gv):
        base &= ~np.isnan(d.cols[v])
    ids = d.cols[iv][base]
    tt = d.cols[tv][base]
    gg = d.cols[gv][base]
    yy = d.cols[yv][base]
    uid, uinv = np.unique(ids, return_inverse=True)
    times = np.unique(tt)
    n = len(uid)
    Y = np.full((n, len(times)), np.nan)
    tix = {t: k for k, t in enumerate(times)}
    for i, t, y in zip(uinv, tt, yy):
        Y[i, tix[t]] = y
    G = np.zeros(n)
    G[uinv] = gg
    groups = sorted(g for g in np.unique(G) if g > 0)
    cells = []
    for g in groups:
        for k, t in enumerate(times):
            if t == times[0]:
                continue
            if t < g:
                base_t = times[k - 1]
            else:
                cand = [s for s in times if s < g]
                if not cand:
                    continue
                base_t = cand[-1]
            kb = tix[base_t]
            dy = Y[:, k] - Y[:, kb]
            treat = G == g
            if notyet:
                ctrl = (G == 0) | (G > max(t, base_t))
            else:
                ctrl = G == 0
            ok = ~np.isnan(dy)
            tr, ct = treat & ok, ctrl & ok
            if tr.sum() == 0 or ct.sum() == 0:
                continue
            pg = tr.mean()
            pc = ct.mean()
            mt, mc = dy[tr].mean(), dy[ct].mean()
            att = mt - mc
            IF = np.zeros(n)
            IF[tr] = (dy[tr] - mt) / pg
            IF[ct] -= (dy[ct] - mc) / pc
            cells.append({'g': g, 't': t, 'att': att, 'IF': IF, 'e': t - g})
    names = [f'g{g_fmt(c["g"])}:t_{g_fmt(times[tix[c["t"]] - 1] if c["t"] < c["g"] else [s for s in times if s < c["g"]][-1])}_{g_fmt(c["t"])}' for c in cells]
    b = np.array([c['att'] for c in cells])
    IFm = np.column_stack([c['IF'] for c in cells])
    V = IFm.T @ IFm / n ** 2
    est = {'names': names, 'b': b, 'V': V, 'colinfo': None, 'df_r': None, 'depvar': yv, 'cmd': 'csdid',
           'stat': 'z', 'sample': base,
           'scalars': {'N': int(base.sum())}, 'macros': {'cmd': 'csdid', 'depvar': yv, 'ivar': iv, 'tvar': tv,
                                                         'gvar': gv, 'control': 'notyet' if notyet else 'never'}}
    est['_cs'] = {'cells': cells, 'G': G, 'n': n, 'groups': groups}
    est['estat'] = _csdid_estat
    ip.set_estimation(est)
    ip.txt('\nDifference-in-difference with Multiple Time Periods\n\n')
    ip.txt(f'{"Number of obs":>70} = ')
    ip.res(f'{comma_int(int(base.sum()))}\n')
    ip.txt('Outcome model  : regression adjustment\nTreatment model: inverse probability\n')
    _cs_table(ip, names, b, V, groups=True)
    ip.txt(f'Control: {"Not yet Treated" if notyet else "Never Treated"}\n\n'
           "See Callaway and Sant'Anna (2021) for details\n")


def _cs_table(ip, names, b, V, groups=False, title=None):
    ip.txt('-' * 78 + '\n')
    ip.txt(f'{"":>12} | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]\n')
    ip.txt('-' * 13 + '+' + '-' * 64 + '\n')
    lastg = None
    se = np.sqrt(np.maximum(np.diag(V), 0))
    for nm, bb, s in zip(names, b, se):
        if groups and ':' in nm:
            g, rest = nm.split(':', 1)
            if g != lastg:
                ip.txt(f'{g:<12} |\n')
                lastg = g
            label = rest
        else:
            label = nm
        z = bb / s if s > 0 else np.nan
        pv = 2 * sps.norm.sf(abs(z))
        ip.txt(f'{abbrev(label, 12):>12} |')
        ip.res(f'{g_fmt(bb, 9):>11}{g_fmt(s, 9):>11}{z:9.2f}{pv:8.3f}{g_fmt(bb - 1.96 * s, 9):>13}{g_fmt(bb + 1.96 * s, 9):>12}\n')
    ip.txt('-' * 78 + '\n')


def _agg(cells, sel, G, n, w_by_group=True):
    """Weighted average of ATT(g,t) cells with influence function incl. weight estimation."""
    cs = [c for c in cells if sel(c)]
    if not cs:
        return np.nan, np.zeros(n)
    pg = np.array([np.mean(G == c['g']) for c in cs])
    w = pg / pg.sum()
    att = float(np.sum(w * np.array([c['att'] for c in cs])))
    IF = sum(wk * c['IF'] for wk, c in zip(w, cs))
    # weight influence function
    S = pg.sum()
    ind = np.column_stack([(G == c['g']).astype(float) - pk for c, pk in zip(cs, pg)])
    wif = ind / S - np.outer(ind.sum(axis=1), pg) / S ** 2
    IF = IF + wif @ np.array([c['att'] for c in cs])
    return att, IF


def _csdid_estat(ip, sub, p):
    est = ip.est
    cs = est['_cs']
    cells, G, n, groups = cs['cells'], cs['G'], cs['n'], cs['groups']
    if sub.startswith('sim'):
        att, IF = _agg(cells, lambda c: c['t'] >= c['g'], G, n)
        names, b, IFs = ['ATT'], [att], [IF]
        title = 'Average Treatment Effect on Treated'
    elif sub.startswith('gr'):
        names, b, IFs = [], [], []
        thetas = []
        for g in groups:
            sel = [c for c in cells if c['g'] == g and c['t'] >= g]
            th = float(np.mean([c['att'] for c in sel]))
            IF = sum(c['IF'] for c in sel) / len(sel)
            thetas.append((g, th, IF))
        pg = np.array([np.mean(G == g) for g in groups])
        w = pg / pg.sum()
        avg = float(np.sum(w * np.array([t[1] for t in thetas])))
        IFa = sum(wk * t[2] for wk, t in zip(w, thetas))
        ind = np.column_stack([(G == g).astype(float) - pk for g, pk in zip(groups, pg)])
        wif = ind / pg.sum() - np.outer(ind.sum(axis=1), pg) / pg.sum() ** 2
        IFa = IFa + wif @ np.array([t[1] for t in thetas])
        names = ['GAverage'] + [f'G{g_fmt(g)}' for g in groups]
        b = [avg] + [t[1] for t in thetas]
        IFs = [IFa] + [t[2] for t in thetas]
        title = 'ATT by group'
    elif sub.startswith('ev'):
        es = sorted(set(c['e'] for c in cells))
        rows = []
        for e in es:
            att, IF = _agg(cells, lambda c, e=e: c['e'] == e, G, n)
            rows.append((e, att, IF))
        pre = [r for r in rows if r[0] < 0]
        post = [r for r in rows if r[0] >= 0]
        names, b, IFs = [], [], []
        if pre:
            names.append('Pre_avg')
            b.append(float(np.mean([r[1] for r in pre])))
            IFs.append(sum(r[2] for r in pre) / len(pre))
        if post:
            names.append('Post_avg')
            b.append(float(np.mean([r[1] for r in post])))
            IFs.append(sum(r[2] for r in post) / len(post))
        for e, att, IF in rows:
            names.append(f'Tm{-int(e)}' if e < 0 else f'Tp{int(e)}')
            b.append(att)
            IFs.append(IF)
        title = 'ATT by Periods Before and After treatment\nEvent Study:Dynamic effects'
        ip.csdid_event = [(e, a, math.sqrt(float(I @ I) / n ** 2)) for e, a, I in rows]
    elif sub.startswith('cal'):
        ts = sorted(set(c['t'] for c in cells if c['t'] >= c['g']))
        names, b, IFs = [], [], []
        for t in ts:
            att, IF = _agg(cells, lambda c, t=t: c['t'] == t and c['t'] >= c['g'], G, n)
            names.append(f'T{g_fmt(t)}')
            b.append(att)
            IFs.append(IF)
        title = 'ATT by Calendar Period'
    else:
        raise StataError(f'estat {sub} not valid after csdid', 321)
    b = np.array(b)
    IFm = np.column_stack(IFs)
    V = IFm.T @ IFm / n ** 2
    ip.txt(f'\n{title}\n')
    _cs_table(ip, names, b, V)
    ip.set_r(matrices={'b': E.Matrix(b.reshape(1, -1), ['y1'], names), 'V': E.Matrix(V, names, names),
                       'table': E.Matrix(np.vstack([b, np.sqrt(np.diag(V))]), ['b', 'se'], names)})


@command('csdid_plot')
def cmd_csdid_plot(ip, cmd, rest, by):
    from .graphs import Figure, apply_global_options, finish_graph, stata_color
    p = parse_command(rest)
    ev = getattr(ip, 'csdid_event', None)
    if ev is None:
        ip.dispatch('quietly estat event')
        ev = ip.csdid_event
    fig = Figure(ip)
    ax = fig.ax
    for e, a, s in ev:
        col = stata_color('navy') if e < 0 else stata_color('maroon')
        ax.errorbar([e], [a], yerr=[[1.96 * s], [1.96 * s]], fmt='o', color=col, capsize=3, ms=5)
    ax.axhline(0, color='black', lw=0.7)
    ax.plot([], [], 'o', color=stata_color('navy'), label='Pre-treatment')
    ax.plot([], [], 'o', color=stata_color('maroon'), label='Post-treatment')
    fig.set_default_titles(xtitle='Periods to Treatment', ytitle='ATT')
    fig.legend_default = True
    apply_global_options(fig, p.opts)
    finish_graph(ip, fig, p.opts)


# ================================================================ sdid
def _sdid_weights(Y, N0, T0, N1, T1):
    """Y: (N, T) with controls first then treated; returns omega (N0), lambda (T0), tau."""
    Yc_pre = Y[:N0, :T0]
    Yc_post = Y[:N0, T0:]
    Yt_pre = Y[N0:, :T0]
    Yt_post = Y[N0:, T0:]
    diffs = np.diff(Yc_pre, axis=1)
    sigma = float(np.std(diffs, ddof=1))
    zeta_o = (N1 * T1) ** 0.25 * sigma
    zeta_l = 1e-6 * sigma
    # unit weights: rows = pre periods; columns = controls; target = treated mean; with intercept
    A = Yc_pre.T                      # (T0, N0)
    bt = Yt_pre.mean(axis=0)          # (T0,)
    Ac = A - A.mean(axis=0)
    btc = bt - bt.mean()
    A_aug = np.vstack([Ac, zeta_o * math.sqrt(T0) * np.eye(N0)])
    b_aug = np.concatenate([btc, np.zeros(N0)])
    from .causal2 import _simplex_ls as sls
    omega = sls(A_aug, b_aug)
    # time weights: rows = controls; columns = pre periods; target = post mean
    B = Yc_pre                        # (N0, T0)
    bl = Yc_post.mean(axis=1)         # (N0,)
    Bc = B - B.mean(axis=0)
    blc = bl - bl.mean()
    B_aug = np.vstack([Bc, zeta_l * math.sqrt(N0) * np.eye(T0)])
    l_aug = np.concatenate([blc, np.zeros(T0)])
    lam = sls(B_aug, l_aug)
    tr = Yt_post.mean() - float(Yt_pre.mean(axis=0) @ lam)
    co = float(omega @ (Yc_post.mean(axis=1) - Yc_pre @ lam))
    tau = tr - co
    return omega, lam, tau


@command('sdid')
def cmd_sdid(ip, cmd, rest, by):
    from .graphs import Figure, finish_graph, stata_color
    p = parse_command(rest)
    d = ip.data
    toks = split_top(p.body)
    yv, iv, tv, dv = (d.resolve(t) for t in toks[:4])
    ids = d.cols[iv]
    tt = d.cols[tv]
    yy = d.cols[yv]
    DD = d.cols[dv]
    uid = np.unique(ids)
    times = np.unique(tt)
    N, T = len(uid), len(times)
    ui = {u: k for k, u in enumerate(uid)}
    ti = {t: k for k, t in enumerate(times)}
    Y = np.full((N, T), np.nan)
    Dm = np.zeros((N, T))
    for a, b_, y, dd in zip(ids, tt, yy, DD):
        Y[ui[a], ti[b_]] = y
        Dm[ui[a], ti[b_]] = dd
    treated_units = Dm.max(axis=1) > 0
    first = [np.flatnonzero(Dm[i] > 0)[0] for i in range(N) if treated_units[i]]
    T0 = int(min(first))
    ctrl = np.flatnonzero(~treated_units)
    trt = np.flatnonzero(treated_units)
    Yo = np.vstack([Y[ctrl], Y[trt]])
    N0, N1, T1 = len(ctrl), len(trt), T - T0
    omega, lam, tau = _sdid_weights(Yo, N0, T0, N1, T1)
    reps = int(_num(p.opts.get('reps'), 50))
    seed = int(_num(p.opts.get('seed'), 0) or 0)
    vce = (p.opts.get('vce') or 'bootstrap').lower()
    rng = np.random.default_rng(seed)
    boots = []
    if vce.startswith('boot'):
        while len(boots) < reps:
            draw = rng.integers(0, N, N)
            c_ = [i for i in draw if not treated_units[i]]
            t_ = [i for i in draw if treated_units[i]]
            if not c_ or not t_:
                continue
            Yb = np.vstack([Y[c_], Y[t_]])
            _, _, tb = _sdid_weights(Yb, len(c_), T0, len(t_), T1)
            boots.append(tb)
        se = float(np.std(boots, ddof=1))
    elif vce.startswith('placebo'):
        while len(boots) < reps:
            perm = rng.permutation(ctrl)
            pc, pt = perm[:N0 - N1], perm[N0 - N1:]
            Yb = np.vstack([Y[pc], Y[pt]])
            _, _, tb = _sdid_weights(Yb, len(pc), T0, len(pt), T1)
            boots.append(tb)
        se = float(np.std(boots, ddof=1))
    else:
        se = np.nan
    z = tau / se if se > 0 else np.nan
    pv = 2 * sps.norm.sf(abs(z))
    ip.txt(f'\nBootstrap replications ({reps}). This may take some time.\n')
    ip.txt('----+--- 1 ---+--- 2 ---+--- 3 ---+--- 4 ---+--- 5\n' + '.' * min(reps, 50) + '\n\n')
    ip.txt('Synthetic Difference-in-Differences Estimator\n\n')
    ip.txt('-' * 77 + '\n')
    ip.txt(f'{abbrev(yv):>12} |     ATT     Std. Err.     t      P>|t|    [95% Conf. Interval]\n')
    ip.txt('-' * 13 + '+' + '-' * 63 + '\n')
    ip.txt(f'{abbrev(dv):>12} |')
    ip.res(f'{tau:>10.5f}{se:>11.5f}{z:>9.2f}{pv:>9.3f}{tau - 1.96 * se:>12.5f}{tau + 1.96 * se:>11.5f}\n')
    ip.txt('-' * 77 + '\n')
    ip.txt('95% CIs and p-values are based on Large-Sample approximations.\n'
           'Refer to Arkhangelsky et al., (2020) for theoretical derivations.\n')
    ip.set_estimation({'names': [dv], 'b': np.array([tau]), 'V': np.array([[se ** 2]]), 'colinfo': None,
                       'cmd': 'sdid', 'stat': 'z', 'depvar': yv,
                       'scalars': {'ATT': tau, 'se': se, 'reps': reps, 'N': int(np.sum(~np.isnan(yy)))},
                       'macros': {'cmd': 'sdid', 'depvar': yv},
                       'matrices': {'omega': E.Matrix(np.column_stack([omega, uid[ctrl]])),
                                    'lambda': E.Matrix(np.column_stack([lam, times[:T0]]))}})
    if p.opts.has('graph'):
        fig = Figure(ip)
        ax = fig.ax
        tr_path = Y[trt].mean(axis=0)
        sc_path = omega @ Y[ctrl]
        off = float(tr_path[:T0] @ lam - sc_path[:T0] @ lam)
        ax.plot(times, tr_path, color=stata_color('navy'), lw=2, label='Treated')
        ax.plot(times, sc_path + off, color=stata_color('maroon'), lw=2, ls='--', label='Synthetic control')
        ax.axvline(times[T0] - 0.5, color='gray', ls=':')
        axb = ax.twinx()
        axb.bar(times[:T0], lam, color=stata_color('forest_green'), alpha=0.35, width=0.8)
        axb.set_ylim(0, max(lam.max() * 4, 1e-6))
        axb.set_yticks([])
        ax.set_title('Synthetic DID: treated vs synthetic control (bars = time weights)', fontsize=10)
        ax.legend(frameon=False, fontsize=9, loc='upper left')
        finish_graph(ip, fig, Options(''), name='sdid')


# ================================================================ didregress / xtdidregress
@command('didregress', 'xtdidregress')
def cmd_didregress(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    groups = re.findall(r'\(([^()]*)\)', p.body)
    if len(groups) < 2:
        raise StataError('syntax: didregress (ovar [omvarlist]) (tvar), group(gvar) time(tvar)', 198)
    otoks = split_top(groups[0])
    yv = d.resolve(otoks[0])
    covs = otoks[1:]
    trv = d.resolve(groups[1].split()[0])
    if cmd == 'xtdidregress':
        gvar = d.ts['panel'] if d.ts else d.resolve(p.opts.get('group'))
        timev = d.ts['time'] if d.ts else d.resolve(p.opts.get('time'))
        if p.opts.has('group'):
            gvar = d.resolve(p.opts.get('group'))
        if p.opts.has('time'):
            timev = d.resolve(p.opts.get('time'))
    else:
        gvar = d.resolve(p.opts.get('group'))
        timev = d.resolve(p.opts.get('time'))
    mask = ip.mark(p)
    for v in (yv, trv, gvar, timev):
        mask &= ~np.isnan(d.cols[v])
    vce, clv = parse_vce(p.opts)
    clv = clv or gvar
    rhs = [trv] + covs
    cols, mask = build_design(ip, d, rhs, mask)
    y = d.cols[yv][mask]
    n = len(y)
    X = np.column_stack([c.values[mask] for c in cols])
    gid = np.unique(d.cols[gvar][mask], return_inverse=True)[1]
    tid = np.unique(d.cols[timev][mask], return_inverse=True)[1]

    def demean(z):
        z = z.astype(float).copy()
        for _ in range(5000):
            old = z.copy()
            for g in (gid, tid):
                m = np.bincount(g, weights=z) / np.bincount(g)
                z = z - m[g]
            if np.max(np.abs(z - old)) < 1e-12 * (1 + np.abs(z).max()):
                break
        return z
    yt = demean(y)
    Xt = np.column_stack([demean(X[:, j]) for j in range(X.shape[1])])
    keep = [j for j in range(X.shape[1]) if not cols[j].base]
    om = mark_collinear(Xt[:, keep], None)
    act = [j for j, o in zip(keep, om) if not o]
    Xa = Xt[:, act]
    XtXi = np.linalg.pinv(Xa.T @ Xa)
    bb = XtXi @ Xa.T @ yt
    e = yt - Xa @ bb
    cgid, cname = cluster_ids(d, clv, mask)
    G = int(cgid.max()) + 1
    k = len(act)
    T_ = int(tid.max()) + 1
    ng = int(gid.max()) + 1
    qc = G / (G - 1) * (n - 1) / (n - k - (T_ - 1) - 1 - (0 if cmd == 'xtdidregress' else ng - 1))
    Vb = sandwich(Xa, Xa * e[:, None], XtXi, cgid) * qc
    j0 = 0
    att, se = bb[j0], math.sqrt(Vb[j0, j0])
    df = G - 1
    tstat = att / se
    pv = 2 * sps.t.sf(abs(tstat), df)
    crit = sps.t.ppf(0.975, df)
    trt_groups = np.unique(d.cols[gvar][mask & (d.cols[trv] != 0)])
    all_groups = np.unique(d.cols[gvar][mask])
    ctrl_groups = np.setdiff1d(all_groups, trt_groups)
    tvals = d.cols[timev][mask]
    ever = np.isin(d.cols[gvar][mask], trt_groups)
    _tm = mask & (d.cols[trv] != 0)
    _gf = {}
    for _g, _t in zip(d.cols[gvar][_tm], d.cols[timev][_tm]):
        _gf[_g] = min(_t, _gf.get(_g, np.inf))
    first_t = np.array(list(_gf.values()))
    ip.txt('\nTreatment and time information\n\n')
    ip.txt(f'Time variable: {timev}\nControl:       {trv} = 0\nTreatment:     {trv} = 1\n')
    ip.txt('-' * 35 + '\n')
    ip.txt(f'{"":>12} |   Control  Treatment\n')
    ip.txt('-' * 13 + '+' + '-' * 21 + '\n')
    ip.txt(f'{"Group":<12} |\n{abbrev(gvar):>12} |')
    ip.res(f'{len(ctrl_groups):>10}{len(trt_groups):>11}\n')
    ip.txt('-' * 13 + '+' + '-' * 21 + '\n')
    ip.txt(f'{"Time":<12} |\n{"Minimum":>12} |')
    ip.res(f'{g_fmt(tvals.min()):>10}{g_fmt(first_t.min()):>11}\n')
    ip.txt(f'{"Maximum":>12} |')
    ip.res(f'{g_fmt(tvals.min()):>10}{g_fmt(first_t.max()):>11}\n')
    ip.txt('-' * 35 + '\n\n')
    ip.txt(f'Difference-in-differences regression{"Number of obs = ":>32}')
    ip.res(f'{comma_int(n)}\n')
    ip.txt(f'Data type: {"Longitudinal" if cmd == "xtdidregress" else "Repeated cross-sectional"}\n\n')
    msg = f'(Std. err. adjusted for {comma_int(G)} clusters in {cname})'
    ip.txt(f'{msg:>78}\n')
    ip.txt('-' * 78 + '\n')
    ip.txt(f'{"":>12} |{"Robust":>21}\n')
    ip.txt(f'{abbrev(yv):>12} | Coefficient  std. err.      t    P>|t|     [95% conf. interval]\n')
    ip.txt('-' * 13 + '+' + '-' * 64 + '\n')
    ip.txt(f'{"ATET":<12} |\n{abbrev(trv):>12} |\n')
    ip.txt(f'{"(1 vs 0)":>11}  |')
    ip.res(f'{g_fmt(att, 9):>11}{g_fmt(se, 9):>11}{tstat:9.2f}{pv:8.3f}{g_fmt(att - crit * se, 9):>13}{g_fmt(att + crit * se, 9):>12}\n')
    ip.txt('-' * 78 + '\n')
    ip.txt(f'Note: ATET estimate adjusted for {"covariates, " if covs else ""}panel effects, and time effects.\n'
           if cmd == 'xtdidregress' else
           f'Note: ATET estimate adjusted for {"covariates, " if covs else ""}group effects and time effects.\n')
    names = [f'r1vs0.{trv}']
    est = {'names': names, 'b': np.array([att]), 'V': np.array([[se * se]]), 'colinfo': None, 'df_r': df,
           'depvar': yv, 'sample': mask, 'cmd': cmd, 'stat': 't',
           'scalars': {'N': n, 'N_clust': G, 'df_r': df},
           'macros': {'cmd': cmd, 'depvar': yv, 'treatment': trv, 'groupvars': gvar, 'timevar': timev,
                      'clustvar': cname}}
    est['_did'] = {'y': yv, 'trv': trv, 'gvar': gvar, 'timev': timev, 'mask': mask, 'covs': covs,
                   'trt_groups': trt_groups, 'clv': clv}
    est['estat'] = _did_estat
    ip.set_estimation(est)


def _did_estat(ip, sub, p):
    from .graphs import Figure, finish_graph, stata_color
    est = ip.est
    dd = est['_did']
    d = ip.data
    mask = dd['mask']
    y = d.cols[dd['y']][mask]
    g = d.cols[dd['gvar']][mask]
    t = d.cols[dd['timev']][mask]
    D = d.cols[dd['trv']][mask]
    ever = np.isin(g, dd['trt_groups'])
    times = np.unique(t)
    t0 = d.cols[dd['timev']][mask & (d.cols[dd['trv']] != 0)].min()
    if sub.startswith('trend'):
        fig = Figure(ip, nrows=1, ncols=2)
        ax1, ax2 = fig.axes
        mt = [y[ever & (t == s)].mean() for s in times]
        mc = [y[~ever & (t == s)].mean() for s in times]
        ax1.plot(times, mc, '-o', color=stata_color('navy'), ms=3, label='Control')
        ax1.plot(times, mt, '-o', color=stata_color('maroon'), ms=3, label='Treatment')
        ax1.axvline(t0 - 0.5, color='gray', ls=':')
        ax1.set_title('Observed means', fontsize=10)
        ax1.legend(fontsize=8, frameon=False)
        pre = t < t0
        Xc = np.column_stack([np.ones(pre.sum()), t[pre], ever[pre], t[pre] * ever[pre]])
        bc = np.linalg.lstsq(Xc, y[pre], rcond=None)[0]
        ax2.plot(times, bc[0] + bc[1] * times, color=stata_color('navy'), label='Control')
        ax2.plot(times, bc[0] + bc[2] + (bc[1] + bc[3]) * times, color=stata_color('maroon'), label='Treatment')
        ax2.axvline(t0 - 0.5, color='gray', ls=':')
        ax2.set_title('Linear-trends model', fontsize=10)
        ax2.legend(fontsize=8, frameon=False)
        fig.fig.suptitle(d.varlab.get(dd['y'], dd['y']), fontsize=11)
        finish_graph(ip, fig, Options(''))
        return
    gid = np.unique(g, return_inverse=True)[1]
    tid = np.unique(t, return_inverse=True)[1]
    cgid = np.unique(d.cols[dd['clv']][mask], return_inverse=True)[1]
    G = int(cgid.max()) + 1

    def fe_reg(extra):
        X = np.column_stack([D] + extra)

        def dm(z):
            z = z.astype(float).copy()
            for _ in range(3000):
                old = z.copy()
                for gg in (gid, tid):
                    z = z - (np.bincount(gg, weights=z) / np.bincount(gg))[gg]
                if np.max(np.abs(z - old)) < 1e-12 * (1 + np.abs(z).max()):
                    break
            return z
        yt = dm(y)
        Xt = np.column_stack([dm(X[:, j]) for j in range(X.shape[1])])
        XtXi = np.linalg.pinv(Xt.T @ Xt)
        b = XtXi @ Xt.T @ yt
        e = yt - Xt @ b
        n = len(y)
        V = sandwich(Xt, Xt * e[:, None], XtXi, cgid) * G / (G - 1) * (n - 1) / (n - Xt.shape[1] - len(times))
        return b, V
    if sub.startswith('ptr'):
        pre = (t < t0).astype(float)
        b, V = fe_reg([ever * t * pre])
        F = b[1] ** 2 / V[1, 1]
        pv = sps.f.sf(F, 1, G - 1)
        ip.txt('\nParallel-trends test (pretreatment time period)\nH0: Linear trends are parallel\n\n')
        ip.txt(f'F(1, {G - 1}) = ')
        ip.res(f'{F:8.2f}\n')
        ip.txt(f'{"Prob > F =":>{len(str(G - 1)) + 9}} ')
        ip.res(f'{pv:8.4f}\n')
        ip.set_r(scalars={'F': F, 'p': pv, 'df': 1, 'df_r': G - 1})
        return
    if sub.startswith('gra'):
        leads = [s for s in times if s < t0]
        extra = [(ever & (t == s)).astype(float) for s in leads[1:]]
        if not extra:
            ip.txt('(no pre-treatment periods to test)\n')
            return
        b, V = fe_reg(extra)
        idx = list(range(1, len(extra) + 1))
        bb = b[idx]
        F = float(bb @ np.linalg.pinv(V[np.ix_(idx, idx)]) @ bb) / len(idx)
        pv = sps.f.sf(F, len(idx), G - 1)
        ip.txt('\nGranger causality test\nH0: No effect in anticipation of treatment\n\n')
        ip.txt(f'F({len(idx)}, {G - 1}) = ')
        ip.res(f'{F:8.2f}\n')
        ip.txt(f'{"Prob > F =":>{len(str(len(idx))) + len(str(G - 1)) + 8}} ')
        ip.res(f'{pv:8.4f}\n')
        ip.set_r(scalars={'F': F, 'p': pv, 'df': len(idx), 'df_r': G - 1})
        return
    if sub.startswith('sum'):
        ip.txt(f'\n  Estimation sample {est["cmd"]}{"Number of obs = ":>36}')
        ip.res(f'{comma_int(len(y))}\n\n')
        ip.txt('  -------------------------------------------------------------\n')
        ip.txt('      Variable |         Mean      Std. dev.         Min          Max\n')
        ip.txt('  -------------+-----------------------------------------------\n')
        for v in [dd['y'], dd['trv']] + [c for c in dd['covs'] if '.' not in c and '#' not in c]:
            x = d.cols[d.resolve(v)][mask]
            ip.txt(f'  {abbrev(v):>12} |')
            ip.res(f'{g_fmt(x.mean(), 9):>13}{g_fmt(x.std(ddof=1), 9):>15}{g_fmt(x.min(), 9):>12}{g_fmt(x.max(), 9):>13}\n')
        ip.txt('  -------------------------------------------------------------\n')
        return
    raise StataError(f'estat {sub} not valid after {est["cmd"]}', 321)


# ================================================================ ritest
@command('ritest')
def cmd_ritest(ip, cmd, rest, by, inner=None):
    p = parse_command(rest)
    d = ip.data
    toks = split_top(p.body)
    rv = d.resolve(toks[0])
    exprs = toks[1:]
    reps = int(_num(p.opts.get('reps'), 100))
    seed = p.opts.get('seed')
    rng = np.random.default_rng(int(_num(seed, 0)) if seed else None)
    clv = p.opts.get('cluster')
    strata = p.opts.get('strata')
    q0 = ip.quiet
    ip.quiet += 1
    try:
        ip.dispatch(inner)
        obs = [float(ip.eval_scalar(x)) for x in exprs]
        sample = ip.est['sample'].copy() if ip.est and ip.est.get('sample') is not None else np.ones(d.n, bool)
    finally:
        ip.quiet = q0
    orig = d.cols[rv].copy()
    if clv:
        cvar = d.resolve(clv)
        cl = d.cols[cvar]
        ucl, cinv = np.unique(np.where(np.isnan(cl), -1e18, cl), return_inverse=True)
        tr_cl = np.zeros(len(ucl))
        for k in range(len(ucl)):
            vals = orig[cinv == k]
            vals = vals[~np.isnan(vals)]
            tr_cl[k] = vals[0] if len(vals) else np.nan
    cnt = np.zeros(len(exprs))
    nvalid = np.zeros(len(exprs))
    ip.txt(f'\nResampling replications ({reps})\n')
    try:
        for r in range(reps):
            if clv:
                ok = ~np.isnan(tr_cl)
                perm = tr_cl.copy()
                perm[ok] = rng.permutation(tr_cl[ok])
                new = perm[cinv]
                new = np.where(np.isnan(orig), np.nan, new)
            else:
                new = orig.copy()
                idx = np.flatnonzero(~np.isnan(orig))
                new[idx] = rng.permutation(orig[idx])
            d.cols[rv] = new
            ip.quiet += 1
            try:
                ip.dispatch(inner)
                for k, x in enumerate(exprs):
                    v = float(ip.eval_scalar(x))
                    if not math.isnan(v):
                        nvalid[k] += 1
                        if abs(v) >= abs(obs[k]):
                            cnt[k] += 1
            except StataError:
                pass
            finally:
                ip.quiet -= 1
    finally:
        d.cols[rv] = orig
        ip.quiet += 1
        try:
            ip.dispatch(inner)
        finally:
            ip.quiet -= 1
    ip.txt('\n')
    ip.txt(f'      command:  {inner}\n')
    for k, x in enumerate(exprs):
        ip.txt(f'{"_pm_" + str(k + 1):>13}:  {x}\n')
    ip.txt(f'  res. var(s):  {rv}\n   Resampling:  Permuting {rv}\n')
    ip.txt(f'Clust. var(s):  {clv or "__000001"}\n     Clusters:  {len(ucl) if clv else int((~np.isnan(orig)).sum())}\n')
    ip.txt('Strata var(s):  none\n       Strata:  1\n\n')
    ip.txt('-' * 78 + '\n')
    ip.txt(f'{"T":<12} |     T(obs)       c       n   p=c/n   SE(p) [95% Conf. Interval]\n')
    ip.txt('-' * 13 + '+' + '-' * 64 + '\n')
    pmat = []
    for k in range(len(exprs)):
        n_ = nvalid[k]
        pp = cnt[k] / n_ if n_ else np.nan
        sep = math.sqrt(pp * (1 - pp) / n_) if n_ else np.nan
        lo, hi = sps.beta.ppf(0.025, cnt[k], n_ - cnt[k] + 1) if cnt[k] > 0 else 0.0, \
            sps.beta.ppf(0.975, cnt[k] + 1, n_ - cnt[k]) if cnt[k] < n_ else 1.0
        ip.txt(f'{"_pm_" + str(k + 1):>12} |')
        ip.res(f'{g_fmt(obs[k], 9):>11}{int(cnt[k]):>8}{int(n_):>8}{pp:>8.4f}{sep:>8.4f}{lo:>8.4f}{hi:>9.4f}\n')
        pmat.append(pp)
    ip.txt('-' * 78 + '\n')
    ip.txt('Note: Confidence interval is with respect to p=c/n.\nNote: c = #{|T| >= |T(obs)|}\n')
    ip.r = Results()
    ip.r.matrices['p'] = E.Matrix(np.array([pmat]), ['c1'], [f'_pm_{k + 1}' for k in range(len(exprs))])
    ip.r.matrices['b'] = E.Matrix(np.array([obs]), ['c1'], [f'_pm_{k + 1}' for k in range(len(exprs))])
    ip.r.matrices['c'] = E.Matrix(np.array([cnt]), ['c1'], [f'_pm_{k + 1}' for k in range(len(exprs))])
    ip.r.matrices['reps'] = E.Matrix(np.array([nvalid]), ['c1'], [f'_pm_{k + 1}' for k in range(len(exprs))])
