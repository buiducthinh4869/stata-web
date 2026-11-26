"""Causal-inference commands: RDD, PSM, matching, SCM, DID variants, randomization inference."""
import math
import re

import numpy as np
from scipy import stats as sps

from . import expr as E
from .design import build_design, coef_table, right_lines
from .errors import StataError
from .estim import (parse_vce, cluster_ids, sandwich, full_bV, wald_test, assemble, detect_omitted,
                    _predict_linear, _store_rhs)
from .fmt import g_fmt, fmt_num, comma_int, abbrev, num_to_macro
from .interp import command, Results
from .syntax import parse_command, split_top, unquote, Options, parse_numlist, quoted_strings


def _num_opt(s, default=None):
    if s is None or s == '':
        return default
    s = s.replace('`', '').replace("'", '').replace('=', '').strip()
    try:
        return float(s)
    except ValueError:
        return default


def _graph_opts(p, name='graph_options'):
    raw = p.opts.get(name) or p.opts.get('graph_opt') or ''
    return Options(raw)


# ================================================================ rdrobust
@command('rdrobust')
def cmd_rdrobust(ip, cmd, rest, by):
    from rdpkg.rdrobust import rdrobust as _rd
    p = parse_command(rest)
    d = ip.data
    toks = split_top(p.body)
    yv, xv = d.resolve(toks[0]), d.resolve(toks[1])
    mask = ip.mark(p) & ~np.isnan(d.cols[yv]) & ~np.isnan(d.cols[xv])
    c = _num_opt(p.opts.get('c'), 0.0)
    pp = int(_num_opt(p.opts.get('p'), 1))
    q = p.opts.get('q')
    q = int(_num_opt(q)) if q else None
    kernel = (p.opts.get('kernel') or 'triangular').lower()
    kernel = {'tri': 'triangular', 'uni': 'uniform', 'epa': 'epanechnikov'}.get(kernel[:3], kernel)
    bws = (p.opts.get('bwselect') or 'mserd').lower()
    h = p.opts.get('h')
    b = p.opts.get('b')
    kw = {}
    if h:
        hv = [float(x) for x in h.replace('`', '').split()]
        kw['h'] = hv[0] if len(hv) == 1 else hv
    if b:
        bv = [float(x) for x in b.replace('`', '').split()]
        kw['b'] = bv[0] if len(bv) == 1 else bv
    covs = None
    if p.opts.has('covs'):
        cv = d.expand_varlist(p.opts.get('covs').split())
        for v in cv:
            mask &= ~np.isnan(d.cols[v])
        covs = np.column_stack([d.cols[v][mask] for v in cv])
    vce = (p.opts.get('vce') or 'nn').split()
    vce_kind = vce[0].lower()
    cluster = None
    if vce_kind in ('cluster', 'nncluster', 'hc0', 'hc1', 'hc2', 'hc3', 'nn'):
        if vce_kind in ('cluster', 'nncluster') and len(vce) > 1:
            cluster = d.cols[d.resolve(vce[1])][mask]
    y = d.cols[yv][mask]
    x = d.cols[xv][mask]
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = _rd(y, x, c=c, p=pp, q=q, kernel=kernel, bwselect=bws, covs=covs, vce=vce_kind,
                cluster=cluster, **kw)
    tau_cl, tau_bc = float(r.coef.iloc[0, 0]), float(r.coef.iloc[1, 0])
    se_cl, se_rb = float(r.se.iloc[0, 0]), float(r.se.iloc[2, 0])
    h_l, h_r = float(r.bws.iloc[0, 0]), float(r.bws.iloc[0, 1])
    b_l, b_r = float(r.bws.iloc[1, 0]), float(r.bws.iloc[1, 1])
    N_l, N_r = int(r.N[0]), int(r.N[1])
    Nh_l, Nh_r = int(r.N_h[0]), int(r.N_h[1])
    z_cl = tau_cl / se_cl
    z_rb = tau_bc / se_rb
    pv_cl = 2 * sps.norm.sf(abs(z_cl))
    pv_rb = 2 * sps.norm.sf(abs(z_rb))
    zc = sps.norm.ppf(0.975)
    q_used = pp + 1 if q is None else q
    allopt = p.opts.has('all')
    ip.txt('\nSharp RD estimates using local polynomial regression.\n\n')
    ip.txt(f'{"Cutoff c = " + g_fmt(c):>19} | Left of c  Right of c{"Number of obs":>27} = ')
    ip.res(f'{N_l + N_r:>10}\n')
    ip.txt('-' * 19 + '+' + '-' * 22 + f'{"BW type":>21}       = ')
    ip.res(f'{bws if not h else "Manual":>10}\n')
    ip.txt(f'{"Number of obs":>19} |')
    ip.res(f'{N_l:>10}{N_r:>12}')
    ip.txt(f'{"Kernel":>20}        = ')
    ip.res(f'{kernel.capitalize():>10}\n')
    ip.txt(f'{"Eff. Number of obs":>19} |')
    ip.res(f'{Nh_l:>10}{Nh_r:>12}')
    ip.txt(f'{"VCE method":>24}    = ')
    ip.res(f'{vce_kind.upper():>10}\n')
    ip.txt(f'{"Order est. (p)":>19} |')
    ip.res(f'{pp:>10}{pp:>12}\n')
    ip.txt(f'{"Order bias (q)":>19} |')
    ip.res(f'{q_used:>10}{q_used:>12}\n')
    ip.txt(f'{"BW est. (h)":>19} |')
    ip.res(f'{h_l:>10.3f}{h_r:>12.3f}\n')
    ip.txt(f'{"BW bias (b)":>19} |')
    ip.res(f'{b_l:>10.3f}{b_r:>12.3f}\n')
    ip.txt(f'{"rho (h/b)":>19} |')
    ip.res(f'{h_l / b_l:>10.3f}{h_r / b_r:>12.3f}\n\n')
    ip.txt(f'Outcome: {yv}. Running variable: {xv}.\n')
    ip.txt('-' * 80 + '\n')
    ip.txt(f'{"Method":>18} |   Coef.    Std. Err.    z     P>|z|    [ 95% Conf. Interval ]\n')
    ip.txt('-' * 19 + '+' + '-' * 60 + '\n')
    ip.txt(f'{"Conventional":>18} |')
    ip.res(f'{tau_cl:>8.4f}{se_cl:>11.4f}{z_cl:>9.4f}{pv_cl:>8.3f}{tau_cl - zc * se_cl:>12.5g}{tau_cl + zc * se_cl:>12.5g}\n')
    if allopt:
        se_bc = float(r.se.iloc[1, 0])
        ip.txt(f'{"Bias-corrected":>18} |')
        ip.res(f'{tau_bc:>8.4f}{se_bc:>11.4f}{tau_bc / se_bc:>9.4f}{2 * sps.norm.sf(abs(tau_bc / se_bc)):>8.3f}{tau_bc - zc * se_bc:>12.5g}{tau_bc + zc * se_bc:>12.5g}\n')
        ip.txt(f'{"Robust":>18} |')
        ip.res(f'{tau_bc:>8.4f}{se_rb:>11.4f}{z_rb:>9.4f}{pv_rb:>8.3f}{tau_bc - zc * se_rb:>12.5g}{tau_bc + zc * se_rb:>12.5g}\n')
    else:
        ip.txt(f'{"Robust":>18} |')
        ip.res(f'{"-":>8}{"-":>11}{z_rb:>9.4f}{pv_rb:>8.3f}{tau_bc - zc * se_rb:>12.5g}{tau_bc + zc * se_rb:>12.5g}\n')
    ip.txt('-' * 80 + '\n')
    if covs is not None:
        ip.txt(f'Covariate-adjusted estimates. Additional covariates included: {covs.shape[1]}\n')
    names = ['Conventional', 'Robust'] if not allopt else ['Conventional', 'Bias-corrected', 'Robust']
    b_ = np.array([tau_cl])
    V_ = np.array([[se_cl ** 2]])
    est = {'names': ['RD_Estimate'], 'b': b_, 'V': V_, 'colinfo': None, 'df_r': None, 'depvar': yv,
           'sample': mask, 'cmd': 'rdrobust', 'stat': 'z',
           'scalars': {'N': N_l + N_r, 'N_l': N_l, 'N_r': N_r, 'N_h_l': Nh_l, 'N_h_r': Nh_r, 'c': c, 'p': pp,
                       'q': q_used, 'h_l': h_l, 'h_r': h_r, 'b_l': b_l, 'b_r': b_r, 'tau_cl': tau_cl,
                       'tau_bc': tau_bc, 'se_tau_cl': se_cl, 'se_tau_rb': se_rb, 'pv_cl': pv_cl, 'pv_rb': pv_rb,
                       'ci_l_cl': tau_cl - zc * se_cl, 'ci_r_cl': tau_cl + zc * se_cl,
                       'ci_l_rb': tau_bc - zc * se_rb, 'ci_r_rb': tau_bc + zc * se_rb,
                       'tau_cl_l': np.nan, 'tau_cl_r': np.nan, 'level': 95},
           'macros': {'cmd': 'rdrobust', 'depvar': yv, 'runningvar': xv, 'kernel': kernel.capitalize(),
                      'bwselect': bws, 'vce_select': vce_kind}}
    est['replay'] = lambda ip_: None
    ip.set_estimation(est)


@command('rdbwselect')
def cmd_rdbwselect(ip, cmd, rest, by):
    from rdpkg.rdbwselect import rdbwselect as _bw
    p = parse_command(rest)
    d = ip.data
    toks = split_top(p.body)
    yv, xv = d.resolve(toks[0]), d.resolve(toks[1])
    mask = ip.mark(p) & ~np.isnan(d.cols[yv]) & ~np.isnan(d.cols[xv])
    c = _num_opt(p.opts.get('c'), 0.0)
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        r = _bw(d.cols[yv][mask], d.cols[xv][mask], c=c)
    ip.txt(f'\nBandwidth estimators for local polynomial regression\n\n{r.bws}\n')


@command('rdplot')
def cmd_rdplot(ip, cmd, rest, by):
    from rdpkg.rdplot import rdplot as _rp
    from .graphs import Figure, apply_global_options, finish_graph, stata_color
    p = parse_command(rest)
    d = ip.data
    toks = split_top(p.body)
    yv, xv = d.resolve(toks[0]), d.resolve(toks[1])
    mask = ip.mark(p) & ~np.isnan(d.cols[yv]) & ~np.isnan(d.cols[xv])
    c = _num_opt(p.opts.get('c'), 0.0)
    pp = int(_num_opt(p.opts.get('p'), 4))
    nbins = p.opts.get('nbins')
    kw = {}
    if nbins:
        nb = [int(float(x)) for x in nbins.split()]
        kw['nbins'] = nb if len(nb) > 1 else nb[0]
    bs = (p.opts.get('binselect') or 'esmv').lower()
    kernel = (p.opts.get('kernel') or 'uni').lower()
    kernel = {'tri': 'triangular', 'uni': 'uniform', 'epa': 'epanechnikov'}.get(kernel[:3], kernel)
    if p.opts.has('h'):
        kw['h'] = float(p.opts.get('h').split()[0])
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        r = _rp(d.cols[yv][mask], d.cols[xv][mask], c=c, p=pp, binselect=bs, kernel=kernel, hide=True, **kw)
    Jl, Jr = r.J
    nl, nr = r.N
    ip.txt(f'\nRD Plot with {_binselect_text(bs)}\n\n')
    ip.txt(f'{"Cutoff c = " + g_fmt(c):>21} | Left of c  Right of c{"Number of obs":>23}  = ')
    ip.res(f'{nl + nr:>10}\n')
    ip.txt('-' * 22 + '+' + '-' * 22 + f'{"Kernel":>14}         = ')
    ip.res(f'{kernel.capitalize():>10}\n')
    ip.txt(f'{"Number of obs":>21} |')
    ip.res(f'{nl:>10}{nr:>12}\n')
    ip.txt(f'{"Eff. Number of obs":>21} |')
    ip.res(f'{int(r.N_h[0]):>10}{int(r.N_h[1]):>12}\n')
    ip.txt(f'{"Order poly. fit (p)":>21} |')
    ip.res(f'{pp:>10}{pp:>12}\n')
    ip.txt(f'{"BW poly. fit (h)":>21} |')
    ip.res(f'{float(r.h[0]):>10.3f}{float(r.h[1]):>12.3f}\n')
    ip.txt(f'{"Number of bins scale":>21} |')
    ip.res(f'{float(r.scale[0]):>10.3f}{float(r.scale[1]):>12.3f}\n\n')
    ip.txt(f'Outcome: {yv}. Running variable: {xv}.\n')
    ip.txt('-' * 45 + '\n')
    ip.txt(f'{"":>21} | Left of c  Right of c\n')
    ip.txt('-' * 22 + '+' + '-' * 22 + '\n')
    ip.txt(f'{"Bins selected":>21} |')
    ip.res(f'{Jl:>10}{Jr:>12}\n')
    ip.txt(f'{"Average bin length":>21} |')
    ip.res(f'{float(r.bin_avg[0]):>10.3f}{float(r.bin_avg[1]):>12.3f}\n')
    ip.txt(f'{"Median bin length":>21} |')
    ip.res(f'{float(r.bin_med[0]):>10.3f}{float(r.bin_med[1]):>12.3f}\n')
    ip.txt('-' * 22 + '+' + '-' * 22 + '\n')
    ip.txt(f'{"IMSE-optimal bins":>21} |')
    ip.res(f'{int(r.J_IMSE[0]):>10}{int(r.J_IMSE[1]):>12}\n')
    ip.txt(f'{"Mimicking Var. bins":>21} |')
    ip.res(f'{int(r.J_MV[0]):>10}{int(r.J_MV[1]):>12}\n')
    ip.txt('-' * 45 + '\n')
    vb = r.vars_bins
    vp = r.vars_poly
    fig = Figure(ip)
    ax = fig.ax
    ax.scatter(vb['rdplot_mean_x'], vb['rdplot_mean_y'], s=22, color=stata_color('navy'), zorder=3)
    xpl = np.asarray(vp['rdplot_x'])
    ypl = np.asarray(vp['rdplot_y'])
    left = xpl < c
    ax.plot(xpl[left], ypl[left], color=stata_color('maroon'), lw=1.8)
    ax.plot(xpl[~left], ypl[~left], color=stata_color('maroon'), lw=1.8)
    ax.axvline(c, color='black', lw=0.8)
    go = _graph_opts(p)
    gtitle = p.opts.get('graph_options') is None
    title = p.opts.get('title')
    fig.set_default_titles(title=unquote(title) if title else 'Regression function fit',
                           xtitle=unquote(p.opts.get('x_label') or '') or None,
                           ytitle=unquote(p.opts.get('y_label') or '') or None)
    apply_global_options(fig, Options(p.opts.raw))
    apply_global_options(fig, go)
    finish_graph(ip, fig, go)
    ip.set_r(scalars={'J_star_l': Jl, 'J_star_r': Jr})
    ip.set_estimation({'names': [], 'b': np.zeros(0), 'V': np.zeros((0, 0)),
                       'scalars': {'J_star_l': Jl, 'J_star_r': Jr, 'N_l': nl, 'N_r': nr},
                       'macros': {'cmd': 'rdplot'}, 'cmd': 'rdplot'})


def _binselect_text(bs):
    return {'es': 'evenly spaced number of bins using spacings estimators.',
            'espr': 'evenly spaced number of bins using polynomial regression.',
            'esmv': 'evenly spaced mimicking variance number of bins using spacings estimators.',
            'esmvpr': 'evenly spaced mimicking variance number of bins using polynomial regression.',
            'qs': 'quantile spaced number of bins using spacings estimators.',
            'qsmv': 'quantile spaced mimicking variance number of bins using spacings estimators.'}.get(bs, bs)


@command('rddensity')
def cmd_rddensity(ip, cmd, rest, by):
    from rdpkg.rddensity import rddensity as _dens
    from .graphs import Figure, apply_global_options, finish_graph, stata_color
    p = parse_command(rest)
    d = ip.data
    xv = d.resolve(split_top(p.body)[0])
    mask = ip.mark(p) & ~np.isnan(d.cols[xv])
    c = _num_opt(p.opts.get('c'), 0.0)
    x = d.cols[xv][mask]
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        r = _dens(x, c=c)
    T = float(r.test['t_jk'])
    pv = float(r.test['p_jk'])
    hl, hr = float(r.h['left']), float(r.h['right'])
    nl, nr = int(r.n['left']), int(r.n['right'])
    nel, ner = int(r.n['eff_left']), int(r.n['eff_right'])
    ip.txt('\nComputing data-driven bandwidth selectors.\n\n')
    if len(np.unique(x)) < len(x):
        ip.txt('Point estimates and standard errors have been adjusted for repeated observations.\n'
               '(Use option nomasspoints to suppress this adjustment.)\n\n')
    ip.txt('RD Manipulation test using local polynomial density estimation.\n\n')
    ip.txt(f'{"c = " + fmt_num(c, "%9.3f").strip():>20} | Left of c  Right of c{"Number of obs":>25} = ')
    ip.res(f'{nl + nr:>12}\n')
    ip.txt('-' * 20 + '+' + '-' * 22 + f'{"Model":>15}         = ')
    ip.res(f'{"unrestricted":>12}\n')
    ip.txt(f'{"Number of obs":>19} |')
    ip.res(f'{nl:>10}{nr:>12}')
    ip.txt(f'{"BW method":>19}     = ')
    ip.res(f'{"comb":>12}\n')
    ip.txt(f'{"Eff. Number of obs":>19} |')
    ip.res(f'{nel:>10}{ner:>12}')
    ip.txt(f'{"Kernel":>16}        = ')
    ip.res(f'{"triangular":>12}\n')
    ip.txt(f'{"Order est. (p)":>19} |')
    ip.res(f'{2:>10}{2:>12}')
    ip.txt(f'{"VCE method":>20}    = ')
    ip.res(f'{"jackknife":>12}\n')
    ip.txt(f'{"Order bias (q)":>19} |')
    ip.res(f'{3:>10}{3:>12}\n')
    ip.txt(f'{"BW est. (h)":>19} |')
    ip.res(f'{hl:>10.3f}{hr:>12.3f}\n\n')
    ip.txt(f'Running variable: {xv}.\n')
    ip.txt('-' * 42 + '\n')
    ip.txt(f'{"Method":>18} |      T          P>|T|\n')
    ip.txt('-' * 19 + '+' + '-' * 22 + '\n')
    ip.txt(f'{"Robust":>18} |')
    ip.res(f'{T:>11.4f}{pv:>12.4f}\n')
    ip.txt('-' * 42 + '\n')
    est = {'names': [], 'b': np.zeros(0), 'V': np.zeros((0, 0)), 'cmd': 'rddensity',
           'scalars': {'c': c, 'p': 2, 'q': 3, 'N_l': nl, 'N_r': nr, 'N_h_l': nel, 'N_h_r': ner,
                       'h_l': hl, 'h_r': hr, 'f_ql': float(r.hat['left']), 'f_qr': float(r.hat['right']),
                       'se_ql': float(r.sd_jk['left']), 'se_qr': float(r.sd_jk['right']),
                       'T_q': T, 'pv_q': pv},
           'macros': {'cmd': 'rddensity', 'runningvar': xv, 'kernel': 'triangular', 'fitmethod': 'unrestricted',
                      'bwmethod': 'comb', 'vce': 'jackknife'}}
    ip.set_estimation(est)
    if p.opts.has('plot'):
        _density_plot(ip, x, c, hl, hr, p)


def _density_plot(ip, x, c, hl, hr, p):
    from .graphs import Figure, apply_global_options, finish_graph, stata_color
    fig = Figure(ip)
    ax = fig.ax
    n = len(x)
    for side, h, col in (('l', hl, 'navy'), ('r', hr, 'maroon')):
        xs = x[x < c] if side == 'l' else x[x >= c]
        lo, hi = (max(x.min(), c - 2 * h), c) if side == 'l' else (c, min(x.max(), c + 2 * h))
        grid = np.linspace(lo, hi, 40)
        # local quadratic density via smoothed empirical CDF (lpdensity-style)
        F = np.array([np.mean(x <= g) for g in grid])
        fhat = []
        sefh = []
        for g in grid:
            u = (x - g) / h
            k = np.clip(1 - np.abs(u), 0, None)
            sel = (x < c) if side == 'l' else (x >= c)
            k = k * sel
            Fx = np.searchsorted(np.sort(x), x, side='right') / n
            X = np.column_stack([np.ones(n), x - g, (x - g) ** 2])
            W = k
            if W.sum() <= 0:
                fhat.append(np.nan)
                sefh.append(np.nan)
                continue
            A = X.T @ (X * W[:, None])
            bb = np.linalg.lstsq(A, X.T @ (W * Fx), rcond=None)[0]
            fhat.append(bb[1])
            ne = W.sum()
            sefh.append(math.sqrt(max(bb[1], 0) / max(ne * h, 1)))
        fhat = np.array(fhat)
        sefh = np.array(sefh)
        ax.fill_between(grid, fhat - 1.96 * sefh, fhat + 1.96 * sefh, color=stata_color(col), alpha=0.2, lw=0)
        ax.plot(grid, fhat, color=stata_color(col), lw=1.8)
        hist, edges = np.histogram(xs, bins=20, range=(lo, hi))
        dens = hist / (n * (edges[1] - edges[0]))
        ax.bar((edges[:-1] + edges[1:]) / 2, dens, width=edges[1] - edges[0], color=stata_color(col), alpha=0.25,
               edgecolor='white')
    ax.axvline(c, color='black', lw=0.8)
    go = _graph_opts(p, 'graph_opt')
    fig.set_default_titles(title='', xtitle='', ytitle='')
    apply_global_options(fig, go)
    finish_graph(ip, fig, go)


# ================================================================ psmatch2
def _logit_probit_ps(ip, d, tv, xvars, mask, probit, show=True):
    """Estimate pscore by running the logit/probit command (prints output like psmatch2)."""
    cmdline = f'{"probit" if probit else "logit"} {tv} {" ".join(xvars)}'
    return cmdline


@command('psmatch2')
def cmd_psmatch2(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    toks = split_top(p.body)
    tv = d.resolve(toks[0])
    xvars = toks[1:]
    outcome = d.expand_varlist(p.opts.get('outcome').split()) if p.opts.has('outcome') else []
    base = ip.mark(p) & ~np.isnan(d.cols[tv])
    for v in ('_pscore', '_treated', '_support', '_weight', '_id', '_n1', '_nn', '_pdif') + tuple('_' + o for o in outcome):
        if v in d.cols:
            d.drop_vars([v])
    for k in range(1, 20):
        if f'_n{k}' in d.cols:
            d.drop_vars([f'_n{k}'])
    if p.opts.has('pscore'):
        ps = d.cols[d.resolve(p.opts.get('pscore'))].astype(float).copy()
    else:
        probit = not p.opts.has('logit')
        line = f'{"probit" if probit else "logit"} {tv} {" ".join(xvars)}'
        if p.if_exp:
            line += f' if {p.if_exp}'
        ip.dispatch(line)
        ps = ip.est['predict'](ip, 'pr', None)
    base &= ~np.isnan(ps)
    for o in outcome:
        base &= ~np.isnan(d.cols[o])
    T = d.cols[tv]
    treated = base & (T != 0)
    control = base & (T == 0)
    support = base.copy()
    if p.opts.has('common'):
        cmin, cmax = ps[control].min(), ps[control].max()
        support &= ~(treated & ((ps < cmin) | (ps > cmax)))
    k = int(_num_opt(p.opts.get('neighbor', 1) or p.opts.get('n'), 1)) if (p.opts.has('neighbor', 1) or p.opts.has('n')) else 1
    caliper = _num_opt(p.opts.get('caliper', 3), None) if p.opts.has('caliper', 3) else None
    kernel = p.opts.has('kernel', 1)
    noreplace = p.opts.has('noreplacement', 5)
    radius = p.opts.has('radius')
    bw = _num_opt(p.opts.get('bwidth', 2), 0.06) if p.opts.has('bwidth', 2) else 0.06
    n = d.n
    weight = np.full(n, np.nan)
    cidx = np.flatnonzero(control & support)
    tidx = np.flatnonzero(treated & support)
    cps = ps[cidx]
    order = np.argsort(cps, kind='stable')
    cps_s = cps[order]
    cidx_s = cidx[order]
    used = np.zeros(len(cidx_s), dtype=bool)
    matches = {}
    nn_first = np.full(n, np.nan)
    pdif = np.full(n, np.nan)
    if kernel:
        kern = (p.opts.get('kerneltype', 5) or 'epan').lower()
        W = np.zeros((len(tidx), len(cidx)))
        for a, i in enumerate(tidx):
            u = (cps - ps[i]) / bw
            if kern.startswith('epan'):
                kw = np.where(np.abs(u) < 1, 0.75 * (1 - u * u), 0)
            elif kern.startswith('norm'):
                kw = sps.norm.pdf(u)
            elif kern.startswith('uni'):
                kw = (np.abs(u) < 1).astype(float)
            else:
                kw = np.where(np.abs(u) < 1, 0.75 * (1 - u * u), 0)
            s = kw.sum()
            if s > 0:
                W[a] = kw / s
            else:
                support[i] = False
        for a, i in enumerate(tidx):
            if support[i]:
                matches[i] = (cidx, W[a])
    else:
        for i in tidx:
            pi = ps[i]
            if radius and caliper is not None:
                sel = np.flatnonzero((np.abs(cps_s - pi) <= caliper) & ~used)
                if not len(sel):
                    support[i] = False
                    continue
                js = sel
            else:
                avail = np.flatnonzero(~used) if noreplace else np.arange(len(cps_s))
                if not len(avail):
                    support[i] = False
                    continue
                dist = np.abs(cps_s[avail] - pi)
                ordr = np.argsort(dist, kind='stable')
                js = avail[ordr[:k]]
                if caliper is not None:
                    js = js[np.abs(cps_s[js] - pi) <= caliper]
                if not len(js):
                    support[i] = False
                    continue
            if noreplace:
                used[js] = True
            matches[i] = (cidx_s[js], np.full(len(js), 1.0 / len(js)))
            nn_first[i] = cidx_s[js[0]] + 1
            pdif[i] = abs(cps_s[js[0]] - pi)
    # weights
    weight[:] = np.nan
    for i, (cj, wj) in matches.items():
        weight[i] = 1.0
        for c_, w_ in zip(cj, wj):
            weight[c_] = (0.0 if np.isnan(weight[c_]) else weight[c_]) + w_
    d.add_var('_pscore', np.where(base, ps, np.nan), 'double', label='psmatch2: Propensity Score')
    d.add_var('_treated', np.where(base, (T != 0).astype(float), np.nan), 'byte', label='psmatch2: Treatment assignment')
    d.labels['_treated'] = {0.0: 'Untreated', 1.0: 'Treated'}
    d.vallab['_treated'] = '_treated'
    d.add_var('_support', np.where(base, support.astype(float), np.nan), 'byte', label='psmatch2: Common support')
    d.labels['_support'] = {0.0: 'Off suppor', 1.0: 'On suppor'}
    d.vallab['_support'] = '_support'
    d.add_var('_weight', weight, 'double', label='psmatch2: weight of matched controls')
    d.add_var('_id', np.arange(1, n + 1, dtype=float), 'long', label='psmatch2: Identifier (ID)')
    if not kernel:
        d.add_var('_n1', nn_first, 'long', label='psmatch2: ID of nearest neighbor nr 1')
        nnv = np.full(n, np.nan)
        for i, (cj, wj) in matches.items():
            nnv[i] = len(cj)
        d.add_var('_nn', nnv, 'long', label='psmatch2: # matched neighbors')
        d.add_var('_pdif', pdif, 'double', label='psmatch2: abs(pscore - pscore(nn))')
    r_s = {}
    if outcome:
        ip.txt('-' * 88 + '\n')
        ip.txt(f'{"Variable":>16}{"Sample":>11} |{"Treated":>11}{"Controls":>13}{"Difference":>13}{"S.E.":>13}{"T-stat":>9}\n')
        ip.txt('-' * 28 + '+' + '-' * 59 + '\n')
    for o in outcome:
        y = d.cols[o]
        yt = y[treated]
        yc = y[control]
        diff_u = yt.mean() - yc.mean()
        n1, n0 = yt.size, yc.size
        sp = math.sqrt(((n1 - 1) * yt.var(ddof=1) + (n0 - 1) * yc.var(ddof=1)) / (n1 + n0 - 2))
        se_u = sp * math.sqrt(1 / n1 + 1 / n0)
        cf = np.full(n, np.nan)
        for i, (cj, wj) in matches.items():
            cf[i] = np.sum(y[cj] * wj)
        mt = np.array(list(matches.keys()), dtype=int)
        att = float(np.mean(y[mt] - cf[mt])) if len(mt) else np.nan
        m1 = float(np.mean(y[mt])) if len(mt) else np.nan
        m0 = float(np.mean(cf[mt])) if len(mt) else np.nan
        N1 = len(mt)
        cw = np.flatnonzero(control & ~np.isnan(weight) & (np.nan_to_num(weight) > 0))
        var1 = float(np.var(y[mt], ddof=1)) if N1 > 1 else np.nan
        if len(cw) > 1:
            wv = weight[cw]
            mu0 = np.sum(wv * y[cw]) / wv.sum()
            var0 = float(np.sum(wv * (y[cw] - mu0) ** 2) / wv.sum() * len(cw) / (len(cw) - 1))
            sumw2 = float(np.sum(wv ** 2))
            seatt = math.sqrt(var1 / N1 + var0 * sumw2 / N1 ** 2)
        else:
            seatt = np.nan
        ip.txt(f'{abbrev(o, 16):>16}{"Unmatched":>11} |')
        ip.res(f'{g_fmt(yt.mean(), 10):>11}{g_fmt(yc.mean(), 10):>13}{g_fmt(diff_u, 10):>13}{g_fmt(se_u, 10):>13}{diff_u / se_u:>9.2f}\n')
        ip.txt(f'{"":>16}{"ATT":>11} |')
        ip.res(f'{g_fmt(m1, 10):>11}{g_fmt(m0, 10):>13}{g_fmt(att, 10):>13}{g_fmt(seatt, 10) if seatt == seatt else ".":>13}'
               f'{(att / seatt) if seatt == seatt and seatt > 0 else float("nan"):>9.2f}\n')
        d.add_var('_' + o, cf, 'double', label=f'psmatch2: value of {o} of match(es)')
        r_s['att'] = att
        r_s['seatt'] = seatt
        r_s['att_' + o] = att
        r_s['seatt_' + o] = seatt
    if outcome:
        ip.txt('-' * 28 + '+' + '-' * 59 + '\n')
        ip.txt('Note: S.E. does not take into account that the propensity score is estimated.\n')
    # support table
    nt_on = int((treated & support).sum())
    nt_off = int((treated & ~support).sum())
    nc = int(control.sum())
    ip.txt('\n           | psmatch2:\n psmatch2: |   Common\n Treatment |  support\n')
    if nt_off:
        ip.txt('assignment | Off suppo  On suppor |     Total\n-----------+----------------------+----------\n')
        ip.txt(' Untreated |')
        ip.res(f'{0:>10}{comma_int(nc):>11} |{comma_int(nc):>10} \n')
        ip.txt('   Treated |')
        ip.res(f'{comma_int(nt_off):>10}{comma_int(nt_on):>11} |{comma_int(nt_off + nt_on):>10} \n')
        ip.txt('-----------+----------------------+----------\n     Total |')
        ip.res(f'{comma_int(nt_off):>10}{comma_int(nt_on + nc):>11} |{comma_int(nt_off + nt_on + nc):>10} \n')
    else:
        ip.txt('assignment | On suppor |     Total\n-----------+-----------+----------\n')
        ip.txt(' Untreated |')
        ip.res(f'{comma_int(nc):>10} |{comma_int(nc):>10} \n')
        ip.txt('   Treated |')
        ip.res(f'{comma_int(nt_on):>10} |{comma_int(nt_on):>10} \n')
        ip.txt('-----------+-----------+----------\n     Total |')
        ip.res(f'{comma_int(nt_on + nc):>10} |{comma_int(nt_on + nc):>10} \n')
    ip.set_r(scalars=r_s, clear=True)
    ip.psm_last = {'treat': tv, 'xvars': xvars}


@command('pstest')
def cmd_pstest(ip, cmd, rest, by):
    from .graphs import Figure, finish_graph, stata_color, apply_global_options
    p = parse_command(rest)
    d = ip.data
    vars_ = d.expand_varlist(split_top(p.body)) if p.body.strip() else d.expand_varlist(ip.psm_last['xvars'])
    T = d.cols['_treated'] if '_treated' in d.cols else d.cols[d.resolve(p.opts.get('treated'))]
    w = d.cols['_weight'] if '_weight' in d.cols else np.full(d.n, np.nan)
    sup = d.cols['_support'] if '_support' in d.cols else np.ones(d.n)
    both = p.opts.has('both')
    treated = (T == 1)
    control = (T == 0)
    mt = treated & (np.nan_to_num(w) > 0) & (sup == 1)
    mc = control & (np.nan_to_num(w) > 0)
    rows = []
    ip.txt('\n' + '-' * 77 + '\n')
    ip.txt(f'{"":>24}|       Mean               |     t-test    |  V(T)/\n')
    ip.txt(f'{"Variable":<15}{"Matched" if both else "":>8} | Treated Control    %bias |    t    p>|t| |  V(C)\n')
    ip.txt('-' * 24 + '+' + '-' * 26 + '+' + '-' * 15 + '+' + '-' * 10 + '\n')
    biasU, biasM = [], []
    for v in vars_:
        x = d.cols[v]
        x1u, x0u = x[treated], x[control]
        v1, v0 = x1u.var(ddof=1), x0u.var(ddof=1)
        bu = 100 * (x1u.mean() - x0u.mean()) / math.sqrt((v1 + v0) / 2)
        x1m = x[mt]
        wc = w[mc]
        m0m = np.sum(wc * x[mc]) / wc.sum()
        bm = 100 * (x1m.mean() - m0m) / math.sqrt((v1 + v0) / 2)
        red = 100 * (abs(bu) - abs(bm)) / abs(bu) if bu != 0 else np.nan
        tu = sps.ttest_ind(x1u, x0u)
        # weighted regression t-test for matched
        sel = mt | mc
        ww = np.where(mt, 1.0, np.nan_to_num(w))[sel]
        xx = x[sel]
        tt = treated[sel].astype(float)
        Xr = np.column_stack([np.ones(sel.sum()), tt])
        A = np.linalg.inv(Xr.T @ (Xr * ww[:, None]))
        bb = A @ Xr.T @ (ww * xx)
        e = xx - Xr @ bb
        s2 = np.sum(ww * e ** 2) / (ww.sum() - 2)
        tm = bb[1] / math.sqrt(s2 * A[1, 1])
        pm = 2 * sps.t.sf(abs(tm), ww.sum() - 2)
        vr_u = v1 / v0
        var0m = np.sum(wc * (x[mc] - m0m) ** 2) / wc.sum()
        vr_m = x1m.var(ddof=1) / var0m if var0m > 0 else np.nan
        biasU.append(abs(bu))
        biasM.append(abs(bm))
        if both:
            ip.txt(f'{abbrev(v, 18):<18}{"U":>4}  |')
            ip.res(f'{g_fmt(x1u.mean(), 7):>8}{g_fmt(x0u.mean(), 7):>9}{bu:>9.1f} |{tu.statistic:>7.2f}{tu.pvalue:>7.3f} |{vr_u:>7.2f}\n')
            ip.txt(f'{"":<18}{"M":>4}  |')
            ip.res(f'{g_fmt(x1m.mean(), 7):>8}{g_fmt(m0m, 7):>9}{bm:>9.1f} |{tm:>7.2f}{pm:>7.3f} |{vr_m:>7.2f}\n')
            ip.txt(f'{"":>24}|{"":>17}{red:>8.1f}  |{"":>15}|\n' if False else '')
        else:
            ip.txt(f'{abbrev(v, 22):<22}  |')
            ip.res(f'{g_fmt(x1m.mean(), 7):>8}{g_fmt(m0m, 7):>9}{bm:>9.1f} |{tm:>7.2f}{pm:>7.3f} |{vr_m:>7.2f}\n')
        rows.append((v, bu, bm))
    ip.txt('-' * 77 + '\n')
    ip.txt('* if variance ratio outside [0.95; 1.05] for U and [0.95; 1.05] for M\n\n')
    ip.txt(f'MeanBias (U) = {np.mean(biasU):.1f}   MeanBias (M) = {np.mean(biasM):.1f}   MedBias (M) = {np.median(biasM):.1f}\n')
    ip.set_r(scalars={'meanbiasbef': np.mean(biasU), 'meanbiasaft': np.mean(biasM),
                      'medbiasbef': np.median(biasU), 'medbiasaft': np.median(biasM)})
    if p.opts.has('graph'):
        fig = Figure(ip)
        ax = fig.ax
        yy = np.arange(len(rows))[::-1]
        ax.scatter([r[1] for r in rows], yy, marker='o', color=stata_color('navy'), label='Unmatched')
        ax.scatter([r[2] for r in rows], yy, marker='x', color=stata_color('maroon'), label='Matched')
        ax.set_yticks(yy)
        ax.set_yticklabels([r[0] for r in rows])
        ax.axvline(0, color='gray', lw=0.8)
        ax.set_xlabel('Standardized % bias across covariates')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
        finish_graph(ip, fig, Options(''))


@command('psgraph')
def cmd_psgraph(ip, cmd, rest, by):
    from .graphs import Figure, apply_global_options, finish_graph, stata_color
    p = parse_command(rest)
    d = ip.data
    ps = d.cols['_pscore']
    T = d.cols['_treated']
    sup = d.cols['_support']
    nb = int(_num_opt(p.opts.get('bin'), 20))
    ok = ~np.isnan(ps)
    edges = np.linspace(np.nanmin(ps), np.nanmax(ps), nb + 1)
    fig = Figure(ip)
    ax = fig.ax
    cen = (edges[:-1] + edges[1:]) / 2
    wdt = edges[1] - edges[0]
    hu, _ = np.histogram(ps[ok & (T == 0)], edges)
    ht_on, _ = np.histogram(ps[ok & (T == 1) & (sup == 1)], edges)
    ht_off, _ = np.histogram(ps[ok & (T == 1) & (sup == 0)], edges)
    ax.bar(cen, -hu, width=wdt * 0.95, color=stata_color('navy'), label='Untreated')
    ax.bar(cen, ht_on, width=wdt * 0.95, color=stata_color('maroon'), label='Treated: On support')
    if ht_off.sum():
        ax.bar(cen, ht_off, bottom=ht_on, width=wdt * 0.95, color=stata_color('forest_green'), label='Treated: Off support')
    ax.axhline(0, color='black', lw=0.6)
    fig.set_default_titles(xtitle='Propensity Score', ytitle='')
    fig.legend_default = True
    apply_global_options(fig, p.opts)
    finish_graph(ip, fig, p.opts)


# ================================================================ teffects psmatch
@command('teffects')
def cmd_teffects(ip, cmd, rest, by):
    rest = rest.strip()
    sub = rest.split()[0]
    if sub not in ('psmatch', 'nnmatch'):
        raise StataError(f'teffects {sub} not supported in web emulator', 198)
    body = rest[len(sub):]
    p = parse_command(body)
    d = ip.data
    groups = re.findall(r'\(([^()]*(?:\([^()]*\))?[^()]*)\)', p.body)
    yv = d.resolve(groups[0].split()[0])
    tspec = groups[1].split(',')
    ttoks = tspec[0].split()
    tv = d.resolve(ttoks[0])
    xvars = ttoks[1:]
    model = 'logit'
    if len(tspec) > 1 and 'probit' in tspec[1]:
        model = 'probit'
    atet = p.opts.has('atet')
    M = int(_num_opt(p.opts.get('nneighbor', 2), 1)) if p.opts.has('nneighbor', 2) else 1
    base = ip.mark(p) & ~np.isnan(d.cols[yv]) & ~np.isnan(d.cols[tv])
    q0 = ip.quiet
    ip.quiet += 1
    try:
        line = f'{model} {tv} {" ".join(xvars)}'
        if p.if_exp:
            line += f' if {p.if_exp}'
        ip.dispatch(line)
        ps = ip.est['predict'](ip, 'pr', None)
        ps_V = ip.est['V']
    finally:
        ip.quiet = q0
    base &= ~np.isnan(ps)
    y = d.cols[yv][base]
    T = (d.cols[tv][base] != 0)
    s = ps[base]
    # matching on logit? Stata teffects psmatch matches on the estimated propensity score
    n = len(y)
    N1 = int(T.sum())
    idx1 = np.flatnonzero(T)
    idx0 = np.flatnonzero(~T)

    def match(from_idx, to_idx, m):
        """For each i in from_idx, list of matches in to_idx (ties included)."""
        out = []
        st = s[to_idx]
        order = np.argsort(st)
        ss = st[order]
        for i in from_idx:
            dist = np.abs(ss - s[i])
            kth = np.partition(dist, min(m, len(dist)) - 1)[min(m, len(dist)) - 1]
            sel = np.flatnonzero(dist <= kth + 1e-12)
            out.append(to_idx[order[sel]])
        return out
    targets = idx1 if atet else np.arange(n)
    m_to0 = match(idx1, idx0, M)
    imputed0 = {i: y[js].mean() for i, js in zip(idx1, m_to0)}
    if not atet:
        m_to1 = match(idx0, idx1, M)
        imputed1 = {i: y[js].mean() for i, js in zip(idx0, m_to1)}
    if atet:
        tau = float(np.mean([y[i] - imputed0[i] for i in idx1]))
    else:
        y1 = np.array([y[i] if T[i] else imputed1[i] for i in range(n)])
        y0 = np.array([imputed0[i] if T[i] else y[i] for i in range(n)])
        tau = float(np.mean(y1 - y0))
    # ---- Abadie-Imbens variance with the AI (2016) correction for the estimated propensity score
    L = 2
    vm = re.search(r'nn\(\s*(\d+)\s*\)', p.opts.get('vce') or '')
    if vm:
        L = max(2, int(vm.group(1)))

    def H(i, grp):
        """i plus its L-1 nearest units (ties included) in its own treatment group."""
        dist = np.abs(s[grp] - s[i])
        dist[grp == i] = np.inf
        m = min(L - 1, len(grp) - 1)
        kth = np.partition(dist, m - 1)[m - 1]
        return np.r_[i, grp[dist <= kth + 1e-12]]

    def Hother(i, grp):
        dist = np.abs(s[grp] - s[i])
        kth = np.partition(dist, min(L, len(grp)) - 1)[min(L, len(grp)) - 1]
        return grp[dist <= kth + 1e-12]
    # treatment-model design (same columns as the logit/probit fit)
    from .estim import prepare_linear, assemble, detect_omitted
    pl = parse_command(' '.join(xvars) + (f' if {p.if_exp}' if p.if_exp else ''))
    _, _, pcols, pmask, _, _ = prepare_linear(ip, pl, d, split_top(' '.join(xvars)), tv)
    Xall, _, pinfo = assemble(pcols, pmask, cons=True)
    act = detect_omitted(Xall, pinfo, cons=True)
    Xf = np.full((d.n, len(act)), np.nan)
    Xf[pmask] = Xall[:, act]
    Xc = Xf[base]
    Vt = np.asarray(ps_V)[np.ix_(act, act)]
    if model == 'probit':
        fden = sps.norm.pdf(sps.norm.ppf(np.clip(s, 1e-15, 1 - 1e-15)))
    else:
        fden = s * (1 - s)
    sig2 = np.zeros(n)
    C1 = np.zeros((n, Xc.shape[1]))
    C0 = np.zeros_like(C1)
    for i in range(n):
        own, oth = (idx1, idx0) if T[i] else (idx0, idx1)
        h = H(i, own)
        sig2[i] = np.var(y[h], ddof=1)
        for grp, C, h_ in ((own, C1 if T[i] else C0, h), (oth, C0 if T[i] else C1, Hother(i, oth))):
            Xs, ys = Xc[h_], y[h_]
            if len(h_) > 1:
                C[i] = ((Xs - Xs.mean(0)) * (ys - ys.mean())[:, None]).sum(0) / (len(h_) - 1)
    Kw = np.zeros(n)
    Kw2 = np.zeros(n)
    for i, js in zip(idx1, m_to0):
        Kw[js] += 1.0 / len(js)
        Kw2[js] += 1.0 / len(js) ** 2
    if atet:
        m_c = match(idx0, idx1, M)
        y1a = y.copy()
        for i, js in zip(idx0, m_c):
            y1a[i] = y[js].mean()
        y0a = y.copy()
        for i in idx1:
            y0a[i] = imputed0[i]
        V = np.sum((y[idx1] - y0a[idx1] - tau) ** 2) + np.sum((Kw[idx0] ** 2 - Kw2[idx0]) * sig2[idx0])
        V /= N1 ** 2
        pbar = N1 / n
        ct = ((C1 + (s / (1 - s))[:, None] * C0) * fden[:, None]).mean(0) / pbar
        dt = (Xc * (fden * (y1a - y0a - tau))[:, None]).mean(0) / pbar
        V = V - ct @ Vt @ ct + dt @ Vt @ dt
    else:
        for i, js in zip(idx0, m_to1):
            Kw[js] += 1.0 / len(js)
            Kw2[js] += 1.0 / len(js) ** 2
        V = np.sum((y1 - y0 - tau) ** 2) + np.sum((Kw ** 2 + (2 * M - 1) / M * Kw) * sig2)
        V /= n ** 2
        c = ((C1 / s[:, None] + C0 / (1 - s)[:, None]) * fden[:, None]).mean(0)
        V = V - c @ Vt @ c
    V = max(V, 0.0)
    se = math.sqrt(V)
    tl = d.value_text(tv, 1)
    cl = d.value_text(tv, 0)
    stat = 'ATET' if atet else 'ATE'
    est = {'names': [f'{stat}:r1vs0.{tv}'], 'b': np.array([tau]), 'V': np.array([[V]]), 'colinfo': None,
           'df_r': None, 'depvar': yv, 'sample': base, 'cmd': 'teffects', 'stat': 'z',
           'scalars': {'N': n, 'n1': N1, 'n0': n - N1, 'k_nneighbor': M},
           'macros': {'cmd': 'teffects', 'subcmd': sub, 'stat': stat.lower(), 'tmodel': model, 'depvar': yv,
                      'tvar': tv, 'vce': 'robust', 'vcetype': 'AI robust'}}
    ip.set_estimation(est)
    ip.txt('\nTreatment-effects estimation' + f'{"Number of obs":>32}      = ')
    ip.res(f'{comma_int(n):>10}\n')
    ip.txt('Estimator      : propensity-score matching     Matches: requested = ')
    ip.res(f'{M:>10}\n')
    mm_ = [len(js) for js in m_to0] + ([] if atet else [len(js) for js in m_to1])
    ip.txt('Outcome model  : matching' + ' ' * 37 + 'min = ')
    ip.res(f'{min(mm_):>10}\n')
    ip.txt(('Treatment model: ' + model).ljust(62) + 'max = ')
    ip.res(f'{max(mm_):>10}\n')
    ip.txt('-' * 78 + '\n')
    ip.txt('             |              AI robust\n')
    ip.txt(f'{abbrev(yv):>12} | Coefficient  std. err.      z    P>|z|     [95% conf. interval]\n')
    ip.txt('-' * 13 + '+' + '-' * 64 + '\n')
    ip.txt(f'{stat:<12} |\n')
    ip.txt(f'{abbrev(tv):>12} |\n')
    if tl and cl:
        ip.txt(f'{"(" + tl:>11}  |\n{"vs":>11}  |\n{cl + ")":>11}  |')
    else:
        ip.txt(f'{"(1 vs 0)":>11}  |')
    z = tau / se if se > 0 else np.nan
    pz = 2 * sps.norm.sf(abs(z))
    ip.res(f'{g_fmt(tau, 9):>11}{g_fmt(se, 9):>11}{z:9.2f}{pz:8.3f}{g_fmt(tau - 1.959964 * se, 9):>13}{g_fmt(tau + 1.959964 * se, 9):>12}\n')
    ip.txt('-' * 78 + '\n')


# ================================================================ kmatch
@command('kmatch')
def cmd_kmatch(ip, cmd, rest, by):
    from .graphs import Figure, finish_graph, stata_color
    rest = rest.strip()
    sub = rest.split()[0]
    d = ip.data
    if sub in ('csummarize', 'cdensity', 'ccumul', 'cbox', 'summarize', 'density', 'cumul', 'box'):
        km = getattr(ip, 'kmatch_last', None)
        if km is None:
            raise StataError('kmatch results not found', 301)
        return _kmatch_post(ip, sub, km)
    p = parse_command(rest[len(sub):])
    m = re.match(r'^\s*(\S+)\s+(.*?)\s*\(([^()]*)\)\s*$', p.body)
    if not m:
        raise StataError('kmatch syntax: kmatch ps tvar xvars (ovar)', 198)
    tv = d.resolve(m.group(1))
    xvars = m.group(2).split()
    yvars = d.expand_varlist(m.group(3).split())
    base = ip.mark(p) & ~np.isnan(d.cols[tv])
    for v in yvars:
        base &= ~np.isnan(d.cols[v])
    q0 = ip.quiet
    ip.quiet += 1
    try:
        ip.dispatch(f'logit {tv} {" ".join(xvars)}')
        ps = ip.est['predict'](ip, 'pr', None)
    finally:
        ip.quiet = q0
    base &= ~np.isnan(ps)
    T = d.cols[tv] != 0
    t_idx = np.flatnonzero(base & T)
    c_idx = np.flatnonzero(base & ~T)
    pt, pc = ps[t_idx], ps[c_idx]
    # pair-matching based bandwidth: 1.5 * 90th percentile of 1:1 matching distances (kmatch "pm")
    srt = np.sort(pc)
    pos = np.clip(np.searchsorted(srt, pt), 1, len(srt) - 1)
    dist = np.minimum(np.abs(pt - srt[pos - 1]), np.abs(pt - srt[pos]))
    if len(srt) == 1:
        dist = np.abs(pt - srt[0])
    # kmatch: 1.5 x 90th percentile (mm_quantile, Stata definition) of nonzero pair-matching distances
    from .cmds_more import _pctiles
    nz = dist[dist > 0]
    bw = float(_pctiles(nz, np.ones(len(nz)), [90.0])[0] * 1.5) if len(nz) else 0.0
    bw = bw + np.finfo(float).eps * max(bw, 1.0) + np.finfo(float).tiny
    U = (pc[None, :] - pt[:, None]) / bw
    K = np.where(np.abs(U) < 1, 0.75 * (1 - U * U), 0.0)
    rs = K.sum(axis=1)
    on = rs > 0
    W = np.zeros_like(K)
    W[on] = K[on] / rs[on, None]
    wc = W[on].sum(axis=0)
    res = []
    for yv in yvars:
        y = d.cols[yv]
        yt = y[t_idx][on]
        cf = W[on] @ y[c_idx]
        att = float(np.mean(yt - cf))
        n1 = int(on.sum())
        yc = y[c_idx]
        W0 = wc.sum()
        b0 = np.sum(wc * yc) / W0 if W0 > 0 else np.nan
        # kmatch influence function (no regression adjustment): -total- of IF / N
        IF = np.zeros(len(t_idx) + len(c_idx))
        IF[:len(t_idx)][on] = (yt - yt.mean()) / n1
        IF[len(t_idx):] = -(wc / W0) * (yc - b0)
        Nall = len(IF)
        se = math.sqrt(Nall / (Nall - 1) * np.sum((IF - IF.mean()) ** 2))
        res.append((yv, att, se))
    ip.txt('\n(computing bandwidth ... done)\n\n' + 'Propensity-score kernel matching'.ljust(48) + 'Number of obs'.ljust(18) + '= ')
    ip.res(f'{comma_int(int(base.sum())):>10}\n')
    ip.txt(' ' * 48 + 'Kernel'.ljust(18) + '= ')
    ip.res(f'{"epan":>10}\n\n')
    ip.txt(f'Treatment   : {tv} = 1\nCovariates  : {" ".join(xvars)}\nPS model    : logit (pr)\n\n')
    ip.txt('Matching statistics\n' + '-' * 78 + '\n')
    ip.txt('             |              Matched             |             Controls\n')
    ip.txt('             |       Yes        No       Total  |       Used     Unused       Total\n')
    ip.txt('-------------+-----------------------------------+--------------------------------\n')
    used = int((wc > 0).sum())
    ip.txt(f'{"Treated":>12} |')
    ip.res(f'{comma_int(on.sum()):>10}{comma_int((~on).sum()):>10}{comma_int(len(t_idx)):>12}  |{comma_int(used):>11}{comma_int(len(c_idx) - used):>11}{comma_int(len(c_idx)):>12}\n')
    ip.txt('-' * 78 + '\n')
    ip.txt(f'\nBandwidth: {bw:.6f} (pair-matching based)\n\n')
    ip.txt('Treatment-effects estimation\n' + '-' * 78 + '\n')
    ip.txt(f'{"":>12} | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]\n')
    ip.txt('-' * 13 + '+' + '-' * 64 + '\n')
    for yv, att, se in res:
        ip.txt(f'{abbrev(yv):>12} |\n{"ATT":>12} |')
        z = att / se
        dfr = int(base.sum()) - 1
        tc = sps.t.ppf(0.975, dfr)
        ip.res(f'{g_fmt(att, 9):>11}{g_fmt(se, 9):>11}{z:9.2f}{2 * sps.t.sf(abs(z), dfr):8.3f}{g_fmt(att - tc * se, 9):>13}{g_fmt(att + tc * se, 9):>12}\n')
    ip.txt('-' * 78 + '\n')
    ip.kmatch_last = {'xvars': xvars, 'tv': tv, 't_idx': t_idx, 'c_idx': c_idx, 'wc': wc, 'on': on,
                      'ps': ps, 'base': base}
    ip.set_estimation({'names': [f'ATT:{res[0][0]}'], 'b': np.array([res[0][1]]),
                       'V': np.array([[res[0][2] ** 2]]), 'colinfo': None, 'cmd': 'kmatch', 'stat': 'z',
                       'df_r': int(base.sum()) - 1,
                       'scalars': {'N': int(base.sum()), 'bwidth': bw}, 'macros': {'cmd': 'kmatch', 'subcmd': 'ps', 'kernel': 'epan', 'tvar': tv},
                       'sample': base, 'depvar': res[0][0]})


def _kmatch_post(ip, sub, km):
    from .graphs import Figure, finish_graph, stata_color
    d = ip.data
    vars_ = [d.resolve(v) for v in km['xvars']] + ['_ps_']
    t_idx, c_idx, wc, on = km['t_idx'], km['c_idx'], km['wc'], km['on']

    def col(v):
        return km['ps'] if v == '_ps_' else d.cols[v]
    if sub.startswith('cs') or sub.startswith('sum'):
        ip.txt('\nStandardized differences\n\n')
        ip.txt('                                  Raw    Matched           Raw    Matched\n')
        ip.txt('             | Std. diff.  Var. ratio                                  \n')
        ip.txt(f'{"":>12} |{"Raw":>11}{"Matched":>11}  |{"Raw":>11}{"Matched":>11}\n')
        ip.txt('-' * 13 + '+' + '-' * 24 + '+' + '-' * 24 + '\n')
        for v in vars_:
            x = col(v)
            xt = x[t_idx]
            xc = x[c_idx]
            sd = math.sqrt((xt.var(ddof=1) + xc.var(ddof=1)) / 2)
            raw = (xt.mean() - xc.mean()) / sd
            xtm = xt[on]
            mc = np.sum(wc * xc) / wc.sum()
            vc = np.sum(wc * (xc - mc) ** 2) / wc.sum()
            mat = (xtm.mean() - mc) / sd
            ip.txt(f'{abbrev(v if v != "_ps_" else "_ps"):>12} |')
            ip.res(f'{raw:>11.4f}{mat:>11.4f}  |{xt.var(ddof=1) / xc.var(ddof=1):>11.4f}{xtm.var(ddof=1) / vc:>11.4f}\n')
        ip.txt('-' * 62 + '\n')
        return
    x = km['ps']
    fig = Figure(ip, nrows=1, ncols=2)
    for k, (ax, lab) in enumerate(zip(fig.axes, ('Raw', 'Matched'))):
        xt = x[t_idx] if lab == 'Raw' else x[t_idx][on]
        xc = x[c_idx]
        wcc = None if lab == 'Raw' else wc
        if sub.startswith('cdens') or sub == 'density':
            from .graphs import kdensity
            gx, gy = kdensity(xt)
            ax.plot(gx, gy, color=stata_color('navy'), label='Treated')
            gx, gy = kdensity(xc, wcc)
            ax.plot(gx, gy, color=stata_color('maroon'), label='Untreated')
        elif sub.startswith('ccum') or sub == 'cumul':
            for arr, ww, c_, lb in ((xt, None, 'navy', 'Treated'), (xc, wcc, 'maroon', 'Untreated')):
                o = np.argsort(arr)
                cw = np.cumsum(np.ones(len(arr)) if ww is None else ww[o])
                ax.step(arr[o], cw / cw[-1], color=stata_color(c_), label=lb, where='post')
        else:
            ax.boxplot([xt, xc if wcc is None else np.repeat(xc, np.maximum(1, np.round(wcc * 10).astype(int)))])
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Treated', 'Untreated'])
        ax.set_title(lab, fontsize=10)
        if k == 0 and not (sub.startswith('cbox') or sub == 'box'):
            ax.legend(fontsize=8)
    fig.fig.suptitle('Propensity score', fontsize=11)
    finish_graph(ip, fig, Options(''))
