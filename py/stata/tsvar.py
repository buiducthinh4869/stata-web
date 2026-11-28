"""VAR family: var, varsoc, vargranger, varstable, varlmar, varnorm (ported from Stata's var.ado, _qsur.ado,
varsoc.ado, vargranger.ado, varstable_w.ado, varlmar.ado, varnorm.ado)."""
import math

import numpy as np
from scipy import stats as sps

from .design import coef_table, Col
from .errors import StataError
from .fmt import fmt_num, abbrev
from .interp import command
from .syntax import parse_command, split_top, parse_numlist
from .tsuni import need_ts, series
from .tsops import expand_token, split_name, canonical
from .stab import Tab


# ---------------------------------------------------------------- helpers
def _toks(ip, s):
    out = []
    for t in split_top(s or ''):
        for t2 in expand_token(t):
            out.append(t2)
    d = ip.data
    res = []
    for t in out:
        ch, base = split_name(t)
        if not ch:
            for v in d.expand_varlist([t]):
                res.append(v)
        else:
            res.append(t)
    return res


def _lagname(l, tok):
    ch, base = split_name(tok)
    return canonical(f'L{l}' + (ch or ''), base), f'L{l}' + (ch or ''), base


def _tsfmt_range(d, t0, t1):
    tf = d.formats.get(d.ts['time'], '%9.0g')
    return fmt_num(float(t0), tf).strip(), fmt_num(float(t1), tf).strip()


def _sample_line(ip, est):
    t0, t1 = est['tmins'], est['tmaxs']
    g = est['N_gaps']
    s = f'Sample: {t0} thru {t1}' + (', but with gaps' if g > 1 else (', but with a gap' if g == 1 else ''))
    return s


def _hdr(ip, left, key, val, fmt='%9.8g', eqpad=2):
    v = fmt_num(val, fmt) if not isinstance(val, str) else val
    ip.txt(f'{left:<48}{key:<18}=' + ' ' * (eqpad - 1))
    ip.res(f'{v}\n')


class VarSpec:
    """Data for a VAR: endogenous series, regressors (lags of every endog, exog, constant) and the sample."""

    def __init__(self, ip, p, endog, lags, exog, nocons, base_mask=None, zlags=False):
        d = ip.data
        need_ts(d)
        keys = [x for x in (d.ts.get('panel'), d.ts.get('time')) if x]
        if d.sortedby[:len(keys)] != keys:
            d.sort(keys)
        self.endog = endog
        self.lags = [] if zlags else list(lags)
        self.exog = exog
        self.nocons = nocons
        Ys = [series(ip, v)[0] for v in endog]
        mask = (ip.mark(p, d) if base_mask is None else base_mask.copy())
        tv = d.cols[d.ts['time']].astype(float)
        mask &= ~np.isnan(tv)
        cols, names, info = [], [], []
        for v, y in zip(endog, Ys):
            for l in self.lags:
                x = ip._lag(d, y, l)
                cols.append(x)
                nm, ch, base = _lagname(l, v)
                names.append(nm)
                info.append((base, ch))
        for v in exog:
            x, nm = series(ip, v)
            cols.append(x)
            ch, base = split_name(nm)
            names.append(nm)
            info.append((base if ch else None, ch or ''))
        for y in Ys:
            mask &= ~np.isnan(y)
        for c in cols:
            mask &= ~np.isnan(c)
        if d.ts.get('panel'):
            pp = d.cols[d.ts['panel']].astype(float)
            mask &= ~np.isnan(pp)
            if len(np.unique(pp[mask])) > 1:
                raise StataError('sample may not include multiple panels', 459)
        self.mask = mask
        self.idx = np.flatnonzero(mask)
        self.T = self.idx.size
        self.Y = np.column_stack([y[self.idx] for y in Ys]) if Ys else np.zeros((self.T, 0))
        X = [c[self.idx] for c in cols]
        if not nocons:
            X.append(np.ones(self.T))
            names.append('_cons')
            info.append((None, ''))
        self.X = np.column_stack(X) if X else np.zeros((self.T, 0))
        self.xnames = names
        self.xinfo = info
        t = tv[self.idx]
        self.tmin, self.tmax = (float(t.min()), float(t.max())) if t.size else (np.nan, np.nan)
        delta = d.ts.get('delta', 1) or 1
        self.N_gaps = int(np.sum(np.round(np.diff(t) / delta) > 1)) if t.size > 1 else 0
        self.tmins, self.tmaxs = _tsfmt_range(d, self.tmin, self.tmax) if t.size else ('.', '.')


def fit_var(S, dfk=False, small=False):
    Y, X = S.Y, S.X
    T, K = Y.shape
    m = X.shape[1]
    if T <= m:
        raise StataError('insufficient observations', 2001)
    if m:
        XtX = X.T @ X
        try:
            XtXi = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtXi = np.linalg.pinv(XtX)
        B = XtXi @ (X.T @ Y)
        E = Y - X @ B
    else:
        XtXi = np.zeros((0, 0))
        B = np.zeros((0, K))
        E = Y.copy()
    S_ml = E.T @ E / T
    det_ml = float(np.linalg.det(S_ml))
    Sig = S_ml * T / (T - m) if dfk else S_ml
    det_s = float(np.linalg.det(Sig))
    ll = -0.5 * T * (K * math.log(2 * math.pi) + math.log(det_ml) + K)
    V = np.kron(Sig, XtXi)
    cons = not S.nocons
    eq = []
    for i in range(K):
        e = E[:, i]
        ssr = float(e @ e)
        y = Y[:, i]
        tss = float(np.sum((y - y.mean()) ** 2)) if cons else float(y @ y)
        nc = [j for j in range(m) if S.xnames[j] != '_cons']
        b = B[nc, i]
        Vi = Sig[i, i] * XtXi[np.ix_(nc, nc)]
        chi2 = float(b @ np.linalg.solve(Vi, b)) if nc else np.nan
        df_m = len(nc)
        rec = {'k': m, 'rmse': math.sqrt(ssr / (T - m)), 'r2': 1 - ssr / tss if tss > 0 else np.nan,
               'chi2': chi2, 'df_m': df_m, 'obs': T, 'df_r': T - m,
               'll': -0.5 * T * (math.log(2 * math.pi) + math.log(ssr / T) + 1)}
        rec['F'] = chi2 / df_m if df_m else np.nan
        rec['p'] = float(sps.chi2.sf(chi2, df_m)) if df_m else np.nan
        rec['pF'] = float(sps.f.sf(rec['F'], df_m, T - m)) if df_m else np.nan
        eq.append(rec)
    tparms = K * m
    fpe = ((T + m) / (T - m)) ** K * det_ml
    return {'B': B, 'E': E, 'XtXi': XtXi, 'S_ml': S_ml, 'Sigma': Sig, 'det_ml': det_ml, 'det': det_s, 'll': ll,
            'V': V, 'eq': eq, 'T': T, 'K': K, 'm': m, 'tparms': tparms, 'fpe': fpe}


def info_crit(F, nlags, lutstats):
    T, K, ll = F['T'], F['K'], F['ll']
    if lutstats:
        ar = nlags * K * K
        ld = math.log(F['det_ml'])
        return ld + 2 * ar / T, ld + 2 * math.log(math.log(T)) / T * ar, ld + math.log(T) / T * ar
    tp = F['tparms']
    return (-2 * ll / T + 2 * tp / T, -2 * ll / T + 2 * math.log(math.log(T)) / T * tp,
            -2 * ll / T + math.log(T) / T * tp)


def _eqname(v):
    return v.replace('.', '_')


def companion(A, K):
    p = len(A)
    C = np.zeros((K * p, K * p))
    for l in range(p):
        C[:K, l * K:(l + 1) * K] = A[l]
    if p > 1:
        C[K:, :K * (p - 1)] = np.eye(K * (p - 1))
    return C


def lag_matrices(est):
    """A_1..A_maxlag (K x K) from a var estimation (zeros for lags not included)."""
    K = len(est['endog'])
    lags = est['lags']
    mlag = max(lags) if lags else 0
    B = est['Bmat']
    A = [np.zeros((K, K)) for _ in range(mlag)]
    names = est['xnames']
    for j, v in enumerate(est['endog']):
        for l in lags:
            nm = _lagname(l, v)[0]
            r = names.index(nm)
            for i in range(K):
                A[l - 1][i, j] = B[r, i]
    return A


# ---------------------------------------------------------------- var
def _var_options(ip, p):
    lg = p.opts.get('lags', 2)
    lags = sorted(set(int(v) for v in parse_numlist(lg))) if lg not in (None, '') else [1, 2]
    if any(l < 1 for l in lags):
        raise StataError('lags() must contain positive integers', 198)
    exog = _toks(ip, p.opts.get('exog', 2)) if p.opts.get('exog', 2) else []
    return lags, exog


def _display_var(ip, est, title='Vector autoregression'):
    ip.txt(f'\n{title}\n\n')
    _hdr(ip, _sample_line(ip, est), 'Number of obs', f"{est['T']:>10,}", eqpad=2)
    luts = '(lutstats)' if est['lutstats'] else ''
    ip.txt('Log likelihood =  ')
    ip.res(fmt_num(est['ll'], '%9.8g'))
    ip.txt(' ' * (37 - 18 - 9) + f'{luts:<11}' + f'{"AIC":<18}=  ')
    ip.res(fmt_num(est['aic'], '%9.8g') + '\n')
    for lab, val, k2, v2 in (('FPE', est['fpe'], 'HQIC', est['hqic']),
                             ('Det(Sigma_ml)', est['det_ml'], 'SBIC', est['sbic'])):
        ip.txt(f'{lab:<15}=  ')
        ip.res(fmt_num(val, '%9.8g'))
        ip.txt(' ' * (48 - 15 - 3 - 9) + f'{k2:<18}=  ')
        ip.res(fmt_num(v2, '%9.8g') + '\n')
    ip.txt('\n')
    _eq_table(ip, est)
    ip.txt('\n')
    _coef_display(ip, est)


def _eq_table(ip, est, key=''):
    small = est['small']
    t = Tab(ip, '17 8 11 10 11 7', lmargin=0)
    t.set_numfmt(None, '%6.0f', '%8.7g', '%7.4f', '%9.8g', '%7.4f')
    t.set_pad(None, None, 2, 1, None, None)
    t.set_strfmt('%-16s', None, None, None, None, None)
    t.set_titlefmt('%-16s', None, None, None, None, None)
    t.titles('Equation', 'Parms', 'RMSE ', 'R-sq  ', 'F    ' if small else 'chi2   ', '  P > F' if small else ' P>chi2')
    t.sep('top')
    for i, v in enumerate(est['eqnames']):
        r = est['eqstats'][i]
        t.row(abbrev(v, 16), r['k'], r['rmse'], r['r2'], r['F'] if small else r['chi2'], r['pF'] if small else r['p'])
    t.sep('bot')


def _coef_display(ip, est):
    names = est['xnames']
    K = len(est['endog'])
    m = len(names)
    stat = 't' if est['small'] else 'z'
    for i, eqn in enumerate(est['eqnames']):
        ix = list(range(i * m, (i + 1) * m))
        info = []
        for nm, (base, ch) in zip(names, est['xinfo']):
            c = Col(name=nm, values=None, term=0, group='', level='', kind='cons' if nm == '_cons' else 'cont',
                    vars=[])
            c.tsvar, c.tschain = base, ch
            info.append(c)
        sub = {'names': names, 'b': est['b'][ix], 'V': est['V'][np.ix_(ix, ix)], 'colinfo': info,
               'df_r': est['df_r'] if est['small'] else None}
        coef_table(ip, sub, stat, depvar=' ', header=(i == 0), footer=(i == K - 1), eq_title=eqn)


def var_estimate(ip, p, endog, lags, exog, nocons, dfk, small, lutstats, base_mask=None, zlags=False):
    S = VarSpec(ip, p, endog, lags, exog, nocons, base_mask=base_mask, zlags=zlags)
    if S.T <= 2:
        raise StataError(f'cannot fit a model with {max(lags) if lags else 0} lags on the current sample', 498)
    F = fit_var(S, dfk, small)
    aic, hqic, sbic = info_crit(F, 0 if zlags else len(lags), lutstats)
    eqnames = [_eqname(v) for v in endog]
    full = [f'{q}:{nm}' for q in eqnames for nm in S.xnames]
    b = F['B'].T.reshape(-1)
    est = {'cmd': 'var', 'names': full, 'b': b, 'V': F['V'], 'colinfo': None, 'depvar': ' '.join(endog),
           'sample': S.mask, 'stat': 't' if small else 'z', 'df_r': (F['T'] - F['m']) if small else None,
           'nomargins': True, 'endog': endog, 'eqnames': eqnames, 'lags': [] if zlags else lags, 'exog': exog,
           'nocons': nocons, 'xnames': S.xnames, 'xinfo': S.xinfo, 'Bmat': F['B'], 'Sigma': F['Sigma'],
           'S_ml': F['S_ml'], 'T': F['T'], 'll': F['ll'], 'aic': aic, 'hqic': hqic, 'sbic': sbic, 'fpe': F['fpe'],
           'det_ml': F['det_ml'], 'det': F['det'], 'eqstats': F['eq'], 'small': small, 'dfk': dfk,
           'lutstats': lutstats, 'tmin': S.tmin, 'tmax': S.tmax, 'tmins': S.tmins, 'tmaxs': S.tmaxs,
           'N_gaps': S.N_gaps, 'XtXi': F['XtXi'], 'E': F['E'], 'idx': S.idx, 'X': S.X, 'Y': S.Y, 'm': F['m']}
    return est


def _post_var(ip, est):
    K = len(est['endog'])
    sc = {'N': est['T'], 'll': est['ll'], 'aic': est['aic'], 'hqic': est['hqic'], 'sbic': est['sbic'],
          'fpe': est['fpe'], 'detsig_ml': est['det_ml'], 'detsig': est['det'], 'neqs': K, 'k_eq': K, 'k_dv': K,
          'df_eq': est['m'], 'k': K * est['m'], 'mlag': max(est['lags']) if est['lags'] else 0,
          'tmin': est['tmin'], 'tmax': est['tmax'], 'N_gaps': est['N_gaps'], 'rank': K * est['m']}
    if est['small']:
        sc['df_r'] = est['df_r']
    for i, r in enumerate(est['eqstats']):
        j = i + 1
        sc.update({f'k_{j}': r['k'], f'obs_{j}': r['obs'], f'rmse_{j}': r['rmse'], f'r2_{j}': r['r2'],
                   f'll_{j}': r['ll'], f'df_m{j}': r['df_m']})
        if est['small']:
            sc[f'F_{j}'] = r['F']
            sc[f'df_r{j}'] = r['df_r']
        else:
            sc[f'chi2_{j}'] = r['chi2']
    est['scalars'] = sc
    from .expr import Matrix
    est['matrices'] = {'Sigma': Matrix(est['Sigma'], est['endog'], est['endog'])}
    est['macros'] = {'cmd': 'var', 'title': 'Vector autoregression', 'endog': ' '.join(est['endog']),
                     'depvar': ' '.join(est['endog']), 'eqnames': ' '.join(est['eqnames']),
                     'lags': ' '.join(map(str, est['lags'])), 'exog': ' '.join(est['exog']),
                     'small': 'small' if est['small'] else '', 'dfk': 'dfk' if est['dfk'] else '',
                     'lutstats': 'lutstats' if est['lutstats'] else '', 'nocons': 'nocons' if est['nocons'] else '',
                     'timevar': ip.data.ts['time'], 'tsfmt': ip.data.formats.get(ip.data.ts['time'], '%9.0g')}
    est['replay'] = lambda ip_: _display_var(ip_, est)
    est['predict'] = None
    est['predict_multi'] = lambda ip_, pp, typ, tk: _var_predict_store(ip_, est, pp, typ, tk)
    ip.set_estimation(est)


def _var_predict_store(ip, est, p, typ, toks):
    from .fmt import comma_int
    if not toks:
        raise StataError('varlist required', 100)
    d = ip.data
    vals, lab = _var_predict_multi(ip, est, p, toks[0])
    vals = np.where(ip.mark(p), vals, np.nan)
    d.add_var(toks[0], vals, typ or 'float')
    d.varlab[toks[0]] = lab
    nm = int(np.isnan(d.cols[toks[0]]).sum())
    if nm:
        ip.txt(f'({comma_int(nm)} missing value{"s" if nm != 1 else ""} generated)\n')


def _var_predict_multi(ip, est, p, newvar):
    d = ip.data
    eq = (p.opts.get('equation', 2) or '').strip()
    names = est['eqnames']
    if not eq:
        i = 0
    elif eq.startswith('#'):
        i = int(eq[1:]) - 1
        if not 0 <= i < len(names):
            raise StataError(f'equation {eq} not found', 303)
    else:
        e2 = _eqname(eq)
        if e2 not in names:
            raise StataError(f'equation {eq} not found', 303)
        i = names.index(e2)
    S = VarSpec(ip, _NoIf(), est['endog'], est['lags'], est['exog'], est['nocons'],
                base_mask=np.ones(d.n, bool))
    xb = np.full(d.n, np.nan)
    xb[S.idx] = S.X @ est['Bmat'][:, i]
    if p.opts.has('residuals', 1):
        y = series(ip, est['endog'][i])[0]
        out = y - xb
        lab = 'Residuals'
    elif p.opts.has('stdp'):
        m = est['m']
        Vi = est['V'][i * m:(i + 1) * m, i * m:(i + 1) * m]
        out = np.full(d.n, np.nan)
        out[S.idx] = np.sqrt(np.einsum('ij,jk,ik->i', S.X, Vi, S.X))
        lab = 'S.E. of the prediction'
    else:
        out = xb
        lab = 'Linear prediction'
    return out, f'{lab}, {est["endog"][i]}'


class _NoIf:
    class _O:
        def get(self, *a):
            return None

        def has(self, *a):
            return False
    opts = _O()
    body = ''
    if_exp = None
    in_range = None


@command('var')
def cmd_var(ip, cmd, rest, by):
    p = parse_command(rest)
    toks = _toks(ip, p.body)
    if not toks:
        if ip.est and ip.est.get('cmd') == 'var':
            return ip.est['replay'](ip)
        raise StataError('last estimates not found', 301)
    if p.opts.get('constraints', 5):
        raise StataError('constraints() not available in Stata Web', 198)
    lags, exog = _var_options(ip, p)
    est = var_estimate(ip, p, toks, lags, exog, p.opts.has('noconstant', 4), p.opts.has('dfk'),
                       p.opts.has('small', 2), p.opts.has('lutstats', 3))
    _post_var(ip, est)
    _display_var(ip, est)


# ---------------------------------------------------------------- varsoc
@command('varsoc')
def cmd_varsoc(ip, cmd, rest, by):
    p = parse_command(rest)
    toks = _toks(ip, p.body)
    lutstats = p.opts.has('lutstats', 3)
    nocons = p.opts.has('noconstant', 4)
    exog = _toks(ip, p.opts.get('exog', 2)) if p.opts.get('exog', 2) else []
    mx = p.opts.get('maxlag', 4)
    base = None
    if not toks:
        e = ip.est
        if not e or e.get('cmd') not in ('var', 'svar'):
            raise StataError('either specify a varlist or have var or svar estimates in memory', 198)
        toks = e['endog']
        exog = exog or e['exog']
        nocons = nocons or e['nocons']
        lutstats = lutstats or e['lutstats']
        maxlag = int(mx) if mx not in (None, '') else (max(e['lags']) if e['lags'] else 4)
        base = e['sample'].copy()
    else:
        maxlag = int(mx) if mx not in (None, '') else 4
    if maxlag < 2:
        raise StataError('maxlag() must be at least 2', 198)
    lv = p.opts.get('level', 1)
    level = float(lv) if lv not in (None, '') else 95.0
    big = var_estimate(ip, p, toks, list(range(1, maxlag + 1)), exog, nocons, False, False, lutstats,
                       base_mask=base)
    smp = big['sample']
    rows = []
    llt = None
    K = len(toks)
    blag = 1 if (nocons and not exog) else 0
    for i in range(blag, maxlag):
        e = var_estimate(ip, p, toks, list(range(1, i + 1)), exog, nocons, False, False, lutstats,
                         base_mask=smp, zlags=(i == 0))
        if i > 0 and llt is not None:
            lr = 2 * (e['ll'] - llt)
            df = K * K
            pv = float(sps.chi2.sf(lr, df)) if lr > 0 else np.nan
        else:
            lr = df = pv = None
        llt = e['ll']
        rows.append([i, e['ll'], lr, df, pv, e['fpe'], e['aic'], e['hqic'], e['sbic']])
    lr = 2 * (big['ll'] - llt) if llt is not None else None
    df = K * K
    pv = (float(sps.chi2.sf(lr, df)) if lr > 0 else np.nan) if lr is not None else None
    rows.append([maxlag, big['ll'], lr, df if lr is not None else None, pv, big['fpe'], big['aic'], big['hqic'],
                 big['sbic']])
    n = len(rows)
    size = (100 - level) / 100

    def pv_ok(i):
        v = rows[i][4]
        return v is not None and v == v and v <= size
    lrs = n - 1 if pv_ok(n - 1) else -1
    best = {5: n - 1, 6: n - 1, 7: n - 1, 8: n - 1}
    for i in range(n - 2, -1, -1):
        if lrs < 0:
            lrs = i if pv_ok(i) else -1
        for c in best:
            if rows[i][c] < rows[best[c]][c]:
                best[c] = i
    ip.txt('\n')
    ip.txt('   Lag-order selection criteria' + (' (lutstats)' if lutstats else '') + '\n\n')
    s = f"   Sample: {big['tmins']} thru {big['tmaxs']}" + (', but with gaps' if big['N_gaps'] else '')
    ip.txt(f'{s:<48}Number of obs{"":6}= ')
    ip.res(f"{big['T']:>9.0f}\n")
    t = Tab(ip, '|4|9 8 1 5 7 8 1 9 1 9 1 9 1 1|')
    t.set_numfmt('%3.0f', '%8.7g', '%7.6g', None, '%4.0f', '%6.3f', '%8.7g', None, '%8.7g', None, '%8.7g', None,
                 '%8.7g', None, None)
    t.set_pad(None, 1, 1, None, None, None, 0, 1, 1, None, 1, None, 1, None, None)
    t.set_numcolor('green', *[None] * 14)
    t.sep('top')
    t.titles('Lag', 'LL  ', 'LR  ', '', 'df', 'p  ', 'FPE ', '', 'AIC ', '', 'HQIC ', '', 'SBIC ', '', '')
    t.sep('mid')
    for i, r in enumerate(rows):
        st = lambda c: '*' if best[c] == i else ''  # noqa: E731
        t.row(float(r[0]), r[1], r[2], '*' if i == lrs else '', r[3], r[4], r[5], st(5), r[6], st(6) or ' ', r[7],
              st(7) or ' ', r[8], st(8) or ' ', '')
    t.sep('bot')
    ip.txt('   * optimal lag\n')
    ip.txt(f"   Endogenous: {' '.join(toks)}\n")
    ex = ' '.join(exog + ([] if nocons else ['_cons']))
    ip.txt(f'    Exogenous: {ex}\n')
    from .expr import Matrix
    ip.set_r(scalars={'N': big['T'], 'mlag': maxlag, 'tmin': big['tmin'], 'tmax': big['tmax'],
                      'N_gaps': big['N_gaps']},
             macros={'endog': ' '.join(toks), 'exog': ex, 'lutstats': 'lutstats' if lutstats else ''},
             matrices={'stats': Matrix([[np.nan if v is None else v for v in r] for r in rows], None,
                                       ['lag', 'LL', 'LR', 'df', 'p', 'FPE', 'AIC', 'HQIC', 'SBIC'])})


# ---------------------------------------------------------------- post-estimation
def _need_var(ip, what):
    e = ip.est
    if not e or e.get('cmd') not in ('var', 'svar'):
        raise StataError(f'{what} only works with estimates from var or svar', 198)
    return e.get('varest', e)


@command('vargranger')
def cmd_vargranger(ip, cmd, rest, by):
    e = _need_var(ip, 'vargranger')
    p = parse_command(rest)
    endog, eqn, names = e['endog'], e['eqnames'], e['xnames']
    m = len(names)
    small = e['small']
    rows = []
    for i, eq in enumerate(endog):
        allidx = []
        for v in endog:
            if v == eq:
                continue
            ix = [i * m + names.index(_lagname(l, v)[0]) for l in e['lags']]
            allidx += ix
            rows.append((eqn[i], v, ix))
        rows.append((eqn[i], 'ALL', allidx))
    sep = p.opts.get('separator', 3)
    sep = int(sep) if sep not in (None, '') else len(endog)
    ip.txt('\n   Granger causality Wald tests\n')
    if small:
        t = Tab(ip, '|19 19| 9 6 8 10|', separator=sep)
        t.set_numfmt(None, None, '%7.6g', '%5.0f', '%7.0f', '%6.4f')
        t.set_pad(None, None, 1, None, None, 2)
        t.sep('top')
        t.titles('Equation', 'Excluded', 'F  ', 'df', 'df_r', 'Prob > F')
    else:
        t = Tab(ip, '|19 19| 9 6 12|', separator=sep)
        t.set_numfmt(None, None, '%7.6g', '%5.0f', '%6.3f')
        t.set_pad(None, None, 1, None, 2)
        t.sep('top')
        t.titles('Equation', 'Excluded', 'chi2 ', 'df', 'Prob > chi2')
    t.sep('mid')
    res = []
    for eqa, vna, ix in rows:
        b = e['b'][ix]
        V = e['V'][np.ix_(ix, ix)]
        chi2 = float(b @ np.linalg.solve(V, b))
        q = len(ix)
        if small:
            F = chi2 / q
            pv = float(sps.f.sf(F, q, e['df_r']))
            t.row(abbrev(eqa, 17), abbrev(vna, 17), F, q, e['df_r'], pv)
            res.append([F, q, e['df_r'], pv])
        else:
            pv = float(sps.chi2.sf(chi2, q))
            t.row(abbrev(eqa, 17), abbrev(vna, 17), chi2, q, pv)
            res.append([chi2, q, pv])
    t.sep('bot')
    from .expr import Matrix
    ip.set_r(matrices={'gstats': Matrix(res, [r[1] for r in rows],
                                        ['F', 'df', 'df_r', 'p'] if small else ['chi2', 'df', 'p'])})


@command('varstable')
def cmd_varstable(ip, cmd, rest, by):
    e = _need_var(ip, 'varstable')
    p = parse_command(rest)
    K = len(e['endog'])
    A = lag_matrices(e)
    C = companion(A, K)
    ev = np.linalg.eigvals(C)
    ev = sorted(ev, key=lambda z: (-round(abs(z), 12), -z.imag))
    re_, im_, mod = [z.real for z in ev], [z.imag for z in ev], [abs(z) for z in ev]
    ip.txt('\n   Eigenvalue stability condition\n')
    t1 = Tab(ip, '|26|13|')
    t2 = Tab(ip, '|12 14|13|')
    t2.set_numfmt('%10.7g', None, '%8.7g')
    t2.set_pad(1, None, 2)
    t2.set_strcolor(None, 'yellow', None)
    t1.sep('top')
    t1.titles('Eigenvalue       ', 'Modulus  ')
    t1.sep('mid')
    for r, c, md in zip(re_, im_, mod):
        if abs(c) > 1e-10 * max(1.0, abs(c)) and abs(c) > 1e-13:
            s = ('-' if c < 0 else '+') + fmt_num(abs(c), '%10.7g') + '{it:i}'
            t2.row(r, s, md)
        else:
            t2.row(r, '', md)
    t1.sep('bot')
    if max(mod) < 1:
        ip.txt('   All the eigenvalues lie inside the unit circle.\n   VAR satisfies stability condition.\n')
    else:
        ip.txt('   At least one eigenvalue is at least 1.0.\n   VAR does not satisfy stability condition.\n')
    from .expr import Matrix
    ip.set_r(matrices={'Re': Matrix([re_]), 'Im': Matrix([im_]), 'Modulus': Matrix([mod])})
    if p.opts.has('graph', 2):
        _stable_graph(ip, re_, im_, p)


def _stable_graph(ip, re_, im_, p):
    from .graphs import Figure, finish_graph, stata_color, apply_global_options
    fig = Figure(ip, figsize=(6.2, 5.8))
    ax = fig.ax
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=(0.6, 0.6, 0.6), lw=1)
    ax.scatter(re_, im_, color=stata_color('navy'), s=22, zorder=3)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    fig.set_default_titles(title='Roots of the companion matrix', xtitle='Real', ytitle='Imaginary')
    apply_global_options(fig, p.opts)
    finish_graph(ip, fig, p.opts)


@command('varlmar')
def cmd_varlmar(ip, cmd, rest, by):
    e = _need_var(ip, 'varlmar')
    p = parse_command(rest)
    ml = p.opts.get('mlag', 2)
    mlag = int(ml) if ml not in (None, '') else 2
    d = ip.data
    K = len(e['endog'])
    smp = e['sample']
    # residuals over the full data, lagged, missing->0 within sample
    S = VarSpec(ip, _NoIf(), e['endog'], e['lags'], e['exog'], e['nocons'], base_mask=np.ones(d.n, bool))
    resid = np.full((d.n, K), np.nan)
    resid[S.idx] = S.Y - S.X @ e['Bmat']
    resid[~smp] = np.nan
    rows = []
    for j in range(1, mlag + 1):
        R = []
        for i in range(K):
            r = ip._lag(d, resid[:, i], j)
            r[np.isnan(r) & smp] = 0.0
            R.append(r)
        S2 = VarSpec(ip, _NoIf(), e['endog'], e['lags'], e['exog'], e['nocons'], base_mask=smp)
        X2 = np.column_stack([S2.X[:, :-1] if not e['nocons'] else S2.X] + [r[S2.idx] for r in R]
                             + ([np.ones((S2.T, 1))] if not e['nocons'] else []))
        Bb = np.linalg.lstsq(X2, S2.Y, rcond=None)[0]
        E2 = S2.Y - X2 @ Bb
        sig2 = E2.T @ E2 / S2.T
        mdf = e['T'] - X2.shape[1] - .5
        lm = mdf * math.log(e['det_ml'] / np.linalg.det(sig2))
        df = K * K
        rows.append([j, lm, df, float(sps.chi2.sf(lm, df))])
    ip.txt('\n   Lagrange-multiplier test\n')
    t = Tab(ip, '|6|12 6 13|')
    t.set_numcolor('green', 'yellow', 'yellow', 'yellow')
    t.set_numfmt('%4.0f', '%8.4f', '%4.0f', '%6.5f')
    t.set_pad(None, 2, None, 3)
    t.sep('top')
    t.titles('lag ', 'chi2 ', 'df ', 'Prob > chi2')
    t.sep('mid')
    for r in rows:
        t.row(*r)
    t.sep('bot')
    ip.txt('   H0: no autocorrelation at lag order\n')
    from .expr import Matrix
    ip.set_r(matrices={'lm': Matrix(rows, [str(r[0]) for r in rows], ['j', 'chi2', 'df', 'p'])})


@command('varnorm')
def cmd_varnorm(ip, cmd, rest, by):
    e0 = ip.est
    e = _need_var(ip, 'varnorm')
    p = parse_command(rest)
    jb, sk, ku = p.opts.has('jbera', 1), p.opts.has('skewness', 1), p.opts.has('kurtosis', 1)
    if not (jb or sk or ku):
        jb = sk = ku = True
    T = e['T']
    K = len(e['endog'])
    Sig = e['Sigma']
    if e0.get('cmd') == 'svar' and not p.opts.has('cholesky', 4):
        Pinv = np.linalg.inv(e0['Bs']) @ e0['As']
    else:
        Pinv = np.linalg.inv(np.linalg.cholesky(Sig))
    W = e['E'] @ Pinv.T
    b1 = np.mean(W ** 3, axis=0)
    b2 = np.mean(W ** 4, axis=0) - 3
    c1 = T / 6 * b1 ** 2
    c2 = T / 24 * b2 ** 2
    jbs = T * (b1 ** 2 / 6 + b2 ** 2 / 24)
    l1, l2 = float(np.sum(c1)), float(np.sum(c2))
    l3 = l1 + l2
    names = e['eqnames'] + ['ALL']
    from .expr import Matrix

    def table(title, head, rows):
        ip.txt(f'\n   {title} test\n')
        t = Tab(ip, '|20|10 8 4 13|')
        t.set_strcolor('green', None, None, None, None)
        t.set_numcolor(None, 'yellow', 'yellow', 'yellow', 'yellow')
        t.set_numfmt(None, '%7.6g', '%7.3f', '%3.0f', '%6.5f')
        t.set_pad(None, 1, 0, None, 3)
        t.sep('top')
        t.titles('Equation', head, 'chi2 ', 'df', 'Prob > chi2')
        t.sep('mid')
        for r in rows:
            t.row(*r)
        t.sep('bot')
    mats = {}
    if jb:
        rows = [(names[i], ' ', jbs[i], 2.0, float(sps.chi2.sf(jbs[i], 2))) for i in range(K)]
        rows.append(('ALL', ' ', l3, float(2 * K), float(sps.chi2.sf(l3, 2 * K))))
        table('Jarque-Bera', '', rows)
        mats['jb'] = Matrix([list(r[2:]) for r in rows], names, ['Jarque-Bera', 'df', 'p'])
    rs = [(names[i], b1[i], c1[i], 1.0, float(sps.chi2.sf(c1[i], 1))) for i in range(K)]
    rs.append(('ALL', None, l1, float(K), float(sps.chi2.sf(l1, K))))
    if sk:
        table('Skewness', 'Skewness', rs)
    rk = [(names[i], b2[i] + 3, c2[i], 1.0, float(sps.chi2.sf(c2[i], 1))) for i in range(K)]
    rk.append(('ALL', None, l2, float(K), float(sps.chi2.sf(l2, K))))
    if ku:
        table('Kurtosis', 'Kurtosis', rk)
    if e['dfk']:
        ip.txt('   dfk estimator used in computations\n')
    mats['skewness'] = Matrix([[np.nan if v is None else v for v in r[1:]] for r in rs], names,
                              ['skewness', 'chi2', 'df', 'p'])
    mats['kurtosis'] = Matrix([[np.nan if v is None else v for v in r[1:]] for r in rk], names,
                              ['kurtosis', 'chi2', 'df', 'p'])
    ip.set_r(matrices=mats, macros={'dfk': 'dfk' if e['dfk'] else ''})
