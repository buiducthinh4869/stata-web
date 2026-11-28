"""ardl (Kripfganz & Schneider, SSC): ARDL models in levels or error-correction form with optimal lag selection
by AIC/BIC, plus estat ectest (Kripfganz-Schneider critical values and p-values) and estat btest (Pesaran,
Shin & Smith 2001 tables).  Ported from ardl.ado, ardlbounds.ado and ardl_estat.ado."""
import itertools
import math

import numpy as np
from scipy import stats as sps

from .design import coef_table, Col, right_lines
from .errors import StataError
from .estim import prepare_linear, ols_core, make_ols_est, ESTAT_EXT
from .fmt import fmt_num, g_fmt, comma_int
from .interp import command
from .syntax import parse_command, split_top, parse_numlist
from .tsuni import need_ts, series
from . import tsvar as TV

MAXLAG_DEFAULT = 4


def _nums(s, n, name):
    vals = s.split()
    out = []
    for v in vals:
        out.append(None if v == '.' else int(float(v)))
    if len(out) == 1 and n > 1:
        out = out * n
    if len(out) != n:
        raise StataError(f'{name}(): number of elements must equal the number of variables', 125)
    return out


@command('ardl')
def cmd_ardl(ip, cmd, rest, by):
    p = parse_command(rest)
    toks = TV._toks(ip, p.body)
    if not toks:
        if ip.est and ip.est.get('cmd') == 'ardl':
            return ip.est['replay'](ip, p)
        raise StataError('last estimates not found', 301)
    d = ip.data
    need_ts(d)
    keys = [x for x in (d.ts.get('panel'), d.ts.get('time')) if x]
    if d.sortedby[:len(keys)] != keys:
        d.sort(keys)
    nv = len(toks)
    ec, ec1 = p.opts.has('ec'), p.opts.has('ec1')
    if ec and ec1:
        raise StataError("Options 'ec' and 'ec1' are mutually exclusive.", 198)
    if nv == 1 and ec1:
        ec, ec1 = True, False
    nocons = p.opts.has('noconstant', 4)
    trendvar = (p.opts.get('trendvar', 2) or '').strip()
    restricted = p.opts.has('restricted', 3)
    exog = TV._toks(ip, p.opts.get('exog', 1)) if p.opts.get('exog', 1) else []
    if p.opts.has('minlag1', 7):
        raise StataError('option minlag1 is not available in Stata Web', 198)
    case = None
    if not (ec or ec1):
        if restricted:
            raise StataError("Option 'restricted' only allowed in conjunction with one of options 'ec' or 'ec1'.", 198)
    else:
        if nocons:
            if trendvar:
                raise StataError("Option 'trendvar' requires a constant in the model.", 198)
            if restricted:
                raise StataError("Option 'restricted' requires deterministic terms in the model.", 198)
            case = 1
        else:
            case = (2 if restricted else 3) if not trendvar else (4 if restricted else 5)
    lg = p.opts.get('lags', 2)
    if lg not in (None, ''):
        lags = _nums(lg, nv, 'lags')
    else:
        lags = [None] * nv
    ml = p.opts.get('maxlags', 1)
    maxlags = [MAXLAG_DEFAULT if v is None else v for v in _nums(ml, nv, 'maxlags')] if ml not in (None, '') \
        else [MAXLAG_DEFAULT] * nv
    if lags[0] == 0 or maxlags[0] == 0:
        raise StataError('the dependent variable must have at least one lag', 125)
    search = any(l is None for l in lags)
    ic = 'aic' if (p.opts.has('aic') and not p.opts.has('bic')) else 'bic'
    # ---- sample
    Ys = [series(ip, v)[0] for v in toks]
    mask = ip.mark(p, d)
    tv = d.cols[d.ts['time']].astype(float)
    mask &= ~np.isnan(tv)
    for i, y in enumerate(Ys):
        mask &= ~np.isnan(y)
        L = lags[i] if lags[i] is not None else maxlags[i]
        for l in range(1, L + 1):
            mask &= ~np.isnan(ip._lag(d, y, l))
    for v in exog:
        mask &= ~np.isnan(series(ip, v)[0])
    trv = None
    if trendvar:
        trv = series(ip, trendvar)[0]
        mask &= ~np.isnan(trv)
    if d.ts.get('panel'):
        pp = d.cols[d.ts['panel']].astype(float)
        if len(np.unique(pp[mask])) > 1:
            raise StataError('sample may not include multiple panels', 459)
    idx = np.flatnonzero(mask)
    N = idx.size
    numcombs = None
    if search:
        ranges = []
        for i in range(nv):
            if lags[i] is not None:
                ranges.append([lags[i]])
            elif i == 0:
                ranges.append(list(range(1, maxlags[i] + 1)))
            else:
                ranges.append(list(range(0, maxlags[i] + 1)))
        numcombs = int(np.prod([len(r) for r in ranges]))
        mc = p.opts.get('maxcombs', 5)
        maxcombs = int(mc) if mc not in (None, '') else 100000
        if numcombs > maxcombs:
            raise StataError(f"# of lag permutations ({numcombs}) exceeds setting of 'maxcombs' ({maxcombs})", 9)
        yv = Ys[0][idx]
        base = [series(ip, v)[0][idx] for v in exog] + ([trv[idx]] if trendvar else []) + \
            ([np.ones(N)] if not nocons else [])
        lagcache = {}

        def col(i, l):
            k_ = (i, l)
            if k_ not in lagcache:
                lagcache[k_] = (ip._lag(d, Ys[i], l) if l else Ys[i])[idx]
            return lagcache[k_]
        best, bestv = None, np.inf
        crit = []
        for comb in itertools.product(*ranges):
            X = list(base)
            for i, q in enumerate(comb):
                X += [col(i, l) for l in range(1 if i == 0 else 0, q + 1)]
            X = np.column_stack(X)
            k = X.shape[1]
            if N <= k:
                continue
            b = np.linalg.lstsq(X, yv, rcond=None)[0]
            e = yv - X @ b
            s2 = float(e @ e) / N
            ll2 = N * math.log(2 * math.pi) + N * math.log(s2) + N
            val = ll2 + (2 * k if ic == 'aic' else k * math.log(N))
            crit.append(list(comb) + [val])
            if val < bestv:
                bestv, best = val, comb
        if best is None:
            raise StataError('insufficient observations', 2001)
        opt = list(best)
    else:
        opt = list(lags)
        crit = None
    # ---- regression specification
    dep = toks[0]
    xv = toks[1:]
    lr_x, sr_terms = [], []
    if ec or ec1:
        q = [o - 1 for o in opt]            # lags in differences
        regdep = f'D.{dep}'
        if q[0] > 0:
            sr_terms.append(f'L(1/{q[0]})D.{dep}')
        x_zero = []
        for i, v in enumerate(xv, start=1):
            if q[i] >= 0:
                sr_terms.append(f'L(0/{q[i]})D.{v}')
            else:
                x_zero.append(v)
        if ec:
            lr_x = list(xv)
        else:
            lr_x = [f'L.{v}' if v not in x_zero else v for v in xv]
        rhs = [f'L.{dep}'] + lr_x + sr_terms
    else:
        regdep = dep
        rhs = [f'L(1/{opt[0]}).{dep}'] + [f'L(0/{opt[i]}).{v}' for i, v in enumerate(xv, start=1)]
    rhs = rhs + exog + ([trendvar] if trendvar else [])
    q0 = ip.quiet
    ip.quiet += 1
    try:
        yname, y, cols, m2, w, wtype = prepare_linear(ip, p, d, rhs, regdep, extra_mask=mask)
        fit = ols_core(ip, d, y, cols, m2, None, None, 'ols', None, nocons)
        fit.depvar = yname
        est = make_ols_est(ip, fit, 'ardl', p)
    finally:
        ip.quiet = q0
    if int(m2.sum()) != N:
        raise StataError('collinear or missing regressors changed the estimation sample', 459)
    if any(getattr(c, 'omitted', False) for c in fit.info):
        raise StataError('Collinear variables detected.', 9)
    lagstr = ','.join(str(o) for o in opt)
    title = f'ARDL({lagstr}) regression'
    sc = est['scalars']
    names = list(fit.names)
    b_r, V_r = fit.b.copy(), fit.V.copy()
    df_r = sc['df_r']
    t0, t1 = TV._tsfmt_range(d, float(tv[idx].min()), float(tv[idx].max()))
    delta = d.ts.get('delta', 1) or 1
    gaps = int(np.sum(np.round(np.diff(tv[idx]) / delta) > 1))
    info = {'N_gaps': gaps, 'tmins': t0, 'tmaxs': t1}
    from .expr import Matrix
    extra_sc = {'N_gaps': gaps, 'tmin': float(tv[idx].min()), 'tmax': float(tv[idx].max())}
    if numcombs is not None:
        extra_sc['numcombs'] = numcombs
    macros = {'cmd': 'ardl', 'title': title + (', EC representation' if (ec or ec1) else ''),
              'model': 'ec' if (ec or ec1) else 'level', 'depvar': regdep, 'tmins': t0, 'tmaxs': t1,
              'predict': 'ardl_p', 'estat_cmd': 'ardl_estat'}
    mats = {'lags': Matrix([[float(o) for o in opt]], None, toks),
            'maxlags': Matrix([[float(m) if search and lags[i] is None else np.nan
                                for i, m in enumerate(maxlags)]], None, toks)}
    if crit is not None and p.opts.get('matcrit', 4):
        ip.matrices[p.opts.get('matcrit', 4).strip()] = Matrix(crit, None, toks + [ic])
    if ec or ec1:
        lrdet = {2: ['_cons'], 4: [trendvar]}.get(case, [])
        lrvars = [f'L.{dep}'] + lr_x
        # map regression coefficients -> (ADJ, LR, SR) coefficients
        pos = {canon(n): j for j, n in enumerate(names)}
        iL = pos[canon(f'L.{dep}')]
        lrsrc = [pos[canon(v)] for v in lr_x + lrdet]
        lrnames = [('L.' + v[2:] if v.startswith('L.') else v) for v in lr_x] if ec1 else list(lr_x)
        if ec1:
            lrnames = [f'L.{v}' for v in xv]
        lrnames = lrnames + lrdet
        srsrc, srnames = [], []
        for j, nm in enumerate(names):
            if j == iL or j in lrsrc:
                continue
            srsrc.append(j)
            srnames.append(nm)
        if ec1:
            # zero-lag regressors in ec1: their level coefficient also appears as D.x in the SR part
            add = []
            for v in xv:
                if v in lr_x:
                    jj = pos[canon(v)]
                    add.append((jj, f'D.{v}'))
            # place after the depvar SR lags, before other SR terms (ordering as in ardl.ado)
            srsrc_new, srn_new = [], []
            for j, nm in zip(srsrc, srnames):
                srsrc_new.append(j)
                srn_new.append(nm)
            for jj, nm in add:
                srsrc_new.append(jj)
                srn_new.append(nm)
            srsrc, srnames = srsrc_new, srn_new

        def g(bv):
            a = bv[iL]
            return np.array([a] + [-bv[j] / a for j in lrsrc] + [bv[j] for j in srsrc])
        bn = g(b_r)
        J = np.zeros((bn.size, b_r.size))
        J[0, iL] = 1
        a = b_r[iL]
        for r_, j in enumerate(lrsrc, start=1):
            J[r_, j] = -1 / a
            J[r_, iL] = b_r[j] / a ** 2
        for r_, j in enumerate(srsrc, start=1 + len(lrsrc)):
            J[r_, j] = 1
        Vn = J @ V_r @ J.T
        eqs = ['ADJ'] + ['LR'] * len(lrsrc) + ['SR'] * len(srsrc)
        nmlist = [f'L.{dep}'] + lrnames + srnames
        # PSS statistics
        t_pss = b_r[iL] / math.sqrt(V_r[iL, iL])
        tix = [iL] + lrsrc
        bb = b_r[tix]
        F_pss = float(bb @ np.linalg.solve(V_r[np.ix_(tix, tix)], bb)) / len(tix)
        k_lr = len(xv)
        est2 = {'cmd': 'ardl', 'names': [f'{q_}:{n}' for q_, n in zip(eqs, nmlist)], 'b': bn, 'V': Vn,
                'colinfo': None, 'df_r': df_r, 'depvar': regdep, 'sample': est['sample'], 'stat': 't',
                'fit': fit, 'nomargins': True, 'tokens': None, 'mask': est.get('mask')}
        sc2 = dict(sc)
        sc2.update({'t_pss': t_pss, 'F_pss': F_pss, 'case': case})
        sc2.update(extra_sc)
        macros.update({'lrxvars': ' '.join(lrnames[:len(lr_x)]), 'lrdet': ' '.join(lrdet),
                       'srvars': ' '.join(sr_terms), 'exogvars': ' '.join(exog),
                       'det': ' '.join({3: ['_cons'], 4: ['_cons'], 5: [trendvar, '_cons']}.get(case, []))})
        est2['scalars'] = sc2
        est2['macros'] = macros
        est2['matrices'] = mats
        est2['ardl'] = {'case': case, 'k': k_lr, 'eqs': eqs, 'nm': nmlist, 'opt': opt}
        res = est2
    else:
        sc.update(extra_sc)
        est['macros'].update(macros)
        est['matrices'] = mats
        est['ardl'] = {'case': None, 'opt': opt}
        res = est
    res['title'] = title
    res['F'] = (sc['F'], sc['df_m'], df_r)
    res['info'] = info

    def replay(ip_, pp=None):
        _display(ip_, res, pp)
    res['replay'] = replay
    res['predict'] = None
    res['predict_multi'] = lambda ip_, pp, typ, tk: _ardl_predict(ip_, res, pp, typ, tk)
    ip.set_estimation(res)
    if not p.opts.has('noctable', 3) or not p.opts.has('noheader', 4):
        replay(ip, p)


def canon(n):
    n = n.replace('L1.', 'L.').replace('L1D.', 'LD.')
    return n


def _display(ip, e, pp):
    sc = e['scalars']
    noh = pp is not None and pp.opts.has('noheader', 4)
    noc = pp is not None and pp.opts.has('noctable', 3)
    if not noh:
        ip.txt('\n')
        F, dfm, dfr = e['F']
        s = f"Sample: {e['info']['tmins']} thru {e['info']['tmaxs']}"
        if e['info']['N_gaps']:
            s += ', but with gaps' if e['info']['N_gaps'] > 1 else ', but with a gap'
        pF = sps.f.sf(F, dfm, dfr) if F == F else np.nan
        right_lines(ip, [e['title'], '', s, '', '', '', '', f"Log likelihood = {g_fmt(sc['ll'], 10).strip()}"],
                    [None, None, ('Number of obs', comma_int(sc['N'])),
                     (f'F({dfm}, {dfr})', fmt_num(F, '%10.2f').strip()),
                     ('Prob > F', f'{pF:.4f}'), ('R-squared', f"{sc['r2']:.4f}"),
                     ('Adj R-squared', f"{sc['r2_a']:.4f}"), ('Root MSE', g_fmt(sc['rmse'], 9).strip())])
        ip.txt('\n')
    if noc:
        return
    if e['ardl']['case'] is None:
        f = e['fit']
        sub = {'names': f.names, 'b': f.b, 'V': f.V, 'colinfo': f.info, 'df_r': e['df_r']}
        coef_table(ip, sub, 't', depvar=e['depvar'])
        return
    A = e['ardl']
    from .tsops import split_name
    groups = []
    for q in ('ADJ', 'LR', 'SR'):
        ix = [i for i, x in enumerate(A['eqs']) if x == q]
        if ix:
            groups.append((q, ix))
    for gi, (q, ix) in enumerate(groups):
        info = []
        for i in ix:
            nm = A['nm'][i]
            ch, base = split_name(nm)
            c = Col(name=nm, values=None, term=0, group='', level='', kind='cons' if nm == '_cons' else 'cont',
                    vars=[])
            c.tsvar, c.tschain = (base, ch) if ch else (None, '')
            info.append(c)
        sub = {'names': [A['nm'][i] for i in ix], 'b': e['b'][ix], 'V': e['V'][np.ix_(ix, ix)], 'colinfo': info,
               'df_r': e['df_r']}
        coef_table(ip, sub, 't', depvar=e['depvar'], header=(gi == 0), footer=(gi == len(groups) - 1), eq_title=q)


def _ardl_predict(ip, e, p, typ, toks):
    d = ip.data
    if not toks:
        raise StataError('varlist required', 100)
    f = e['fit']
    out = np.full(d.n, np.nan)
    if p.opts.has('ec'):
        if e['ardl']['case'] is None:
            raise StataError("option ec only allowed after ardl with options 'ec' or 'ec1'", 198)
        A = e['ardl']
        dep = e['macros']['depvar'][2:]
        y = series(ip, f'L.{dep}')[0]
        val = y.copy()
        for i, (q, nm) in enumerate(zip(A['eqs'], A['nm'])):
            if q != 'LR':
                continue
            if nm == '_cons':
                val = val - e['b'][i]
            else:
                x = series(ip, nm if not nm.startswith('L.') else nm)[0]
                val = val - e['b'][i] * x
        out = val
        lab = 'Error-correction term'
    else:
        from .design import build_design
        res = f.e if hasattr(f, 'e') else None
        fitted = np.full(d.n, np.nan)
        fitted[f.mask] = f.y - f.e
        if p.opts.has('residuals', 1):
            out[f.mask] = f.e
            lab = 'Residuals'
        else:
            out = fitted
            lab = 'Linear prediction'
    out = np.where(ip.mark(p), out, np.nan)
    d.add_var(toks[0], out, typ or 'float')
    d.varlab[toks[0]] = lab
    nm_ = int(np.isnan(out).sum())
    if nm_:
        ip.txt(f'({comma_int(nm_)} missing value{"s" if nm_ != 1 else ""} generated)\n')


# ---------------------------------------------------------------- bounds tests
def ks_cv(stat, case, k, n, sr, p_list):
    """Kripfganz-Schneider critical values (I0, I1) for significance levels p_list (in %); n None = asymptotic."""
    from .ardl_cv import KS, KS_COLS
    c = case + 1 if stat == 't' and case in (2, 4) else case
    out = {}
    tabs = {}
    for I in (0, 1):
        rows = KS[f'{stat}{c}{I}']
        th = np.array([r[1:] for r in rows])
        P = np.array([r[0] for r in rows])
        col = {nm: th[:, j] for j, nm in enumerate(KS_COLS)}
        cv = np.zeros(len(rows))
        if n is not None:
            cv += (col['0_2_0'] + col['0_2_1'] * sr) / n ** 2 + (col['0_3_0'] + col['0_3_1'] * sr) / n ** 3
        for j in range(5):
            cv += col[f'{j}_0_0'] / (k + 1) ** j
            if n is not None:
                cv += col[f'{j}_1_0'] / ((k + 1) ** j * n) + col[f'{j}_1_1'] * sr / ((k + 1) ** j * n)
        tabs[I] = (P, cv)
        for pl in p_list:
            out[(pl, I)] = float(cv[np.flatnonzero(P == int(round(pl * 100)))[0]])
    return out, tabs


def ks_pvalue(stat, case, k, n, sr, val, tabs):
    df1 = 1 + k + (case in (2, 4))
    df2 = None if n is None else n - (1 + k + sr + (case >= 2) + (case >= 4))
    res = []
    for I in (0, 1):
        P, cv = tabs[I]
        ad = np.abs(cv - val)
        mp = int(np.flatnonzero(ad == ad.min()).mean())
        sel = np.arange(len(cv))
        tu = (sel >= mp - 4) & (sel <= mp + 4)
        pp = P[tu] / 10000
        if n is not None:
            it = sps.f.isf(pp, df1, df2) if stat == 'F' else -sps.t.isf(pp, df2)
        else:
            it = sps.chi2.isf(pp, df1) if stat == 'F' else sps.norm.ppf(pp)
        Xr = np.column_stack([np.ones(tu.sum()), cv[tu], cv[tu] ** 2])
        bb = np.linalg.lstsq(Xr, it, rcond=None)[0]
        crit = bb[0] + bb[1] * val + bb[2] * val ** 2
        if n is not None:
            pv = sps.f.sf(crit, df1, df2) if stat == 'F' else 1 - sps.t.sf(crit, df2)
        else:
            pv = sps.chi2.sf(crit, df1) if stat == 'F' else sps.norm.cdf(crit)
        res.append(float(pv))
    return res


def _need_ec(ip, what):
    e = ip.est
    if not e or e.get('cmd') != 'ardl':
        raise StataError('last estimates not found', 301)
    if e['ardl']['case'] is None:
        raise StataError(f"estat {what} is only applicable when option 'ec' or 'ec1' was used.", 198)
    return e


def estat_ectest(ip, p):
    e = _need_ec(ip, 'ectest')
    asym = p.opts.has('asymptotic', 3)
    sl = p.opts.get('siglevels', 3)
    levels = [float(v) for v in parse_numlist(sl)] if sl else [10.0, 5.0, 1.0]
    A = e['ardl']
    case, k = A['case'], A['k']
    sc = e['scalars']
    sr = int(sc['df_m'] - (k + 1) - (case >= 4))
    n = None if asym else int(sc['N'])
    if n is not None:
        df2 = n - (1 + k + sr + (case >= 2) + (case >= 4))
        if df2 < n / 2:
            raise StataError('There must be at least twice as many observations than coefficients.', 198)
    rows = []
    for stat, val in (('F', sc['F_pss']), ('t', sc['t_pss'])):
        cvs, tabs = ks_cv(stat, case, k, n, sr, levels)
        pv = ks_pvalue(stat, case, k, n, sr, val, tabs)
        rows.append((stat, [cvs[(l, I)] for l in levels for I in (0, 1)] + pv))
    ip.txt('\nPesaran, Shin, and Smith (2001) bounds test\n\n')
    col = 66
    ip.txt(f'{"H0: no level relationship":<{col - 1}}F =')
    ip.res(f"{sc['F_pss']:10.3f}\n")
    ip.txt('Case ')
    ip.res(f'{case}')
    ip.txt(' ' * (col - 1 - 6) + 't =')
    ip.res(f"{sc['t_pss']:10.3f}\n")
    if asym:
        ip.txt('\nAsymptotic (')
        ip.res(f'{k}')
        ip.txt(' variables)\n')
    else:
        ip.txt('\nFinite sample (')
        ip.res(f'{k}')
        ip.txt(' variables, ')
        ip.res(f'{n}')
        ip.txt(' observations, ')
        ip.res(f'{sr}')
        ip.txt(' short-run coefficients)\n')
    ip.txt('\nKripfganz and Schneider (2018) critical values and approximate p-values\n\n')
    hd = []
    for l in levels:
        hd.append(fmt_num(l, '%4.0f' if l == int(l) else '%4.1f').strip() + '%')
    hd.append('p-value')
    line1 = '  |'
    for h in hd:
        line1 += f'{h:^15}' + ' |'
    line2 = '  |' + ''.join('  I(0)    I(1)  |'[:16] + ' ' for _ in hd)
    ip.txt(line1.rstrip('|').rstrip() + '\n')
    ip.txt('  |' + ''.join(f'{"I(0)":>7} {"I(1)":>7} |' for _ in hd) + '\n')
    ip.txt('--+' + '+'.join(['-' * 16] * len(hd)) + '\n')
    for stat, vals in rows:
        ip.txt(f'{stat:>1} |')
        s = ''
        for j in range(0, len(vals), 2):
            ip.res(f'{vals[j]:7.3f} {vals[j + 1]:7.3f}')
            ip.txt(' |')
        ip.txt('\n')
    ip.txt('\ndo not reject H0 if\n    both F and t are closer to zero than critical values for I(0) variables\n'
           '      (if p-values > desired level for I(0) variables)\nreject H0 if\n'
           '    both F and t are more extreme than critical values for I(1) variables\n'
           '      (if p-values < desired level for I(1) variables)\n')
    from .expr import Matrix
    cn = []
    for h in hd:
        cn += [f'{h}:I(0)', f'{h}:I(1)']
    ip.set_r(scalars={'F_pss': sc['F_pss'], 't_pss': sc['t_pss'], 'case': case, 'k': k,
                      **({} if asym else {'sr': sr, 'N': n})},
             macros={'siglevels': ' '.join(fmt_num(l, '%9.0g').strip() for l in levels)},
             matrices={'cvmat': Matrix([r[1] for r in rows], ['F', 't'], cn)})


def estat_btest(ip, p):
    from .ardl_cv import PSS_F, PSS_T
    e = _need_ec(ip, 'btest')
    A = e['ardl']
    case, k = A['case'], A['k']
    if k > 10:
        raise StataError(f'Model has k={k} weakly exogenous variables, but Pesaran/Shin/Smith (2001) critical '
                         'values are only tabulated up to k=10.', 9)
    sc = e['scalars']
    Fr = PSS_F[case][k]
    Tr = PSS_T[case + 1 if case in (2, 4) else case][k]
    ip.txt('\nnote: estat btest has been superseded by estat ectest\n'
           '      as the prime procedure to test for a levels relationship.\n')
    ip.txt('\nPesaran/Shin/Smith (2001) ARDL Bounds Test\n')
    ip.txt('H0: no levels relationship             F =  ')
    ip.res(f"{sc['F_pss']:5.3f}\n")
    ip.txt('                                       t = ')
    ip.res(f"{sc['t_pss']:5.3f}\n")

    def tab(title, r):
        ip.txt(f'\n{title}\n\n')
        ip.txt(f'{"":5} |{"[I_0]":>8}{"[I_1]":>9} |{"[I_0]":>8}{"[I_1]":>9} |{"[I_0]":>8}{"[I_1]":>9} |'
               f'{"[I_0]":>8}{"[I_1]":>9}\n')
        ip.txt(f'{"":5} |{"L_1":>8}{"L_1":>9} |{"L_05":>8}{"L_05":>9} |{"L_025":>8}{"L_025":>9} |'
               f'{"L_01":>8}{"L_01":>9}\n')
        ip.txt('-' * 6 + '+' + '+'.join(['-' * 18] * 4) + '\n')
        ip.txt(f'{"k_" + str(k):>5} |')
        ip.res(''.join(f'{r[j]:8.2f}{r[j + 1]:9.2f}' + (' |' if j < 6 else '') for j in range(0, 8, 2)) + '\n')
    tab(f'Critical Values (0.1-0.01), F-statistic, Case {case}', Fr)
    ip.txt('accept if F < critical value for I(0) regressors\nreject if F > critical value for I(1) regressors\n')
    tab(f'Critical Values (0.1-0.01), t-statistic, Case {case}', Tr)
    ip.txt('accept if t > critical value for I(0) regressors\nreject if t < critical value for I(1) regressors\n')
    ip.txt('\nk: # of non-deterministic regressors in long-run relationship\n'
           'Critical values from Pesaran/Shin/Smith (2001)\n')
    from .expr import Matrix
    cn = ['[I_0]:L_1', '[I_1]:L_1', '[I_0]:L_05', '[I_1]:L_05', '[I_0]:L_025', '[I_1]:L_025', '[I_0]:L_01',
          '[I_1]:L_01']
    ip.set_r(scalars={'case': case, 'F_pss': sc['F_pss'], 't_pss': sc['t_pss']},
             matrices={'F_critval': Matrix([Fr], [f'k_{k}'], cn), 't_critval': Matrix([Tr], [f'k_{k}'], cn)})


ESTAT_EXT.extend([
    ('ect', 'ectest', estat_ectest),
    ('bt', 'btest', estat_btest),
])
