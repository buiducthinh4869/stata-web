"""svar: structural VAR, short-run (A, B) and long-run (C) restrictions (ported from Stata's svar.ado, _svard2.ado,
_svarlrd2.ado).  ML with the expected information matrix, as ml d2 with negh in Stata."""
import math

import numpy as np
from scipy import stats as sps

from .errors import StataError
from .fmt import fmt_num, g_fmt
from .interp import command
from .syntax import parse_command
from .tsvar import _toks, _var_options, var_estimate, _display_var, lag_matrices
from . import tsvar as TV


def _commutation(m, n):
    K = np.zeros((m * n, m * n))
    for i in range(m):
        for j in range(n):
            K[i * n + j, j * m + i] = 1
    return K


def _vec(M):
    return M.reshape(-1, order='F')


def _get_mat(ip, raw, name, K):
    raw = (raw or '').strip()
    if not raw:
        return None
    M = ip.get_matrix(raw) if hasattr(ip, 'get_matrix') else ip.matrices.get(raw)
    if M is None:
        raise StataError(f'{name}({raw}) does not define a matrix', 198)
    a = M.a if hasattr(M, 'a') else np.asarray(M)
    if a.shape != (K, K):
        raise StataError(f'{name} matrix does not have the correct dimensions', 198)
    return a


class Restr:
    """Linear map theta -> vec(M) for one K x K matrix from eq() (values/missing) or cns() (patterns)."""

    def __init__(self, K, eq=None, cns=None, letter='A'):
        self.K = K
        self.letter = letter
        n = K * K
        self.fixed = {}           # vec index -> value
        self.tie = {}             # vec index -> representative vec index
        self.cns_txt = []
        if eq is not None:
            for i in range(K):
                for j in range(K):
                    v = eq[i, j]
                    if not np.isnan(v):
                        self.fixed[j * K + i] = float(v)
                        self.cns_txt.append(f'[/{letter}]{i + 1}_{j + 1} = {g_fmt(v, 9).strip()}')
        if cns is not None:
            groups = {}
            for i in range(K):
                for j in range(K):
                    v = cns[i, j]
                    if np.isnan(v):
                        continue
                    if v != int(v):
                        raise StataError(f'non-integer element in matrix defining constraints on {letter}', 198)
                    if v == 0:
                        self.fixed[j * K + i] = 0.0
                        self.cns_txt.append(f'[/{letter}]{i + 1}_{j + 1} = 0')
                    else:
                        groups.setdefault(v, []).append((i, j))
            for v, mem in groups.items():
                i0, j0 = mem[0]
                for (i, j) in mem[1:]:
                    self.tie[j * K + i] = j0 * K + i0
                    self.cns_txt.append(f'[/{letter}]{i0 + 1}_{j0 + 1} - [/{letter}]{i + 1}_{j + 1} = 0')
        self.free = [k for k in range(n) if k not in self.fixed and k not in self.tie]
        self.M = np.zeros((n, len(self.free)))
        self.c = np.zeros(n)
        for c_, k in enumerate(self.free):
            self.M[k, c_] = 1
        for k, v in self.fixed.items():
            self.c[k] = v
        for k, r in self.tie.items():
            self.M[k] = self.M[r]
            self.c[k] = self.c[r]

    def full(self, th):
        return self.M @ th + self.c


def svar_fit(Sig, T, K, rA, rB, lr_abar=None, from_=None, maxit=300):
    """Returns theta, ll, info inverse, log, build function."""
    n2 = K * K
    if lr_abar is None:
        MA, cA, MB, cB = rA.M, rA.c, rB.M, rB.c
        nA = MA.shape[1]
        M = np.zeros((2 * n2, nA + MB.shape[1]))
        M[:n2, :nA] = MA
        M[n2:, nA:] = MB
        c = np.r_[cA, cB]
    else:
        IA = np.kron(np.eye(K), lr_abar)
        M = np.zeros((2 * n2, rA.M.shape[1]))
        M[n2:] = IA @ rA.M
        c = np.r_[_vec(np.eye(K)), IA @ rA.c]

    def build(th):
        z = M @ th + c
        return z[:n2].reshape(K, K, order='F'), z[n2:].reshape(K, K, order='F')
    Kmm = _commutation(K, K)
    I2 = np.eye(n2)

    def ll_g_h(th, todo=2):
        A, B = build(th)
        try:
            BI = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            return -np.inf, None, None
        Km = BI @ A
        dK = np.linalg.det(Km)
        if not np.isfinite(dK) or dK == 0:
            return -np.inf, None, None
        ll = -.5 * T * K * math.log(2 * math.pi) + .5 * T * math.log(dK ** 2) - .5 * T * np.trace(Km.T @ Km @ Sig)
        if todo == 0:
            return ll, None, None
        KIP = np.linalg.inv(Km.T)
        Q1 = T * _vec(KIP) - T * (_vec(Km) @ np.kron(Sig, np.eye(K)))
        BIP = np.linalg.inv(B.T)
        Q2 = np.hstack([np.kron(np.eye(K), BI), -np.kron(A.T @ BIP, BI)])
        g = Q1 @ Q2
        KI = KIP.T
        Q4 = np.vstack([np.kron(KI, BIP), -np.kron(np.eye(K), BIP)])
        h = T * Q4 @ (I2 + Kmm) @ Q4.T
        return ll, M.T @ g, M.T @ h @ M
    nf = M.shape[1]
    if from_ is not None:
        th = np.asarray(from_, float)
    else:
        st = 1 + np.arange(1, 2 * n2 + 1) / 100.0
        if lr_abar is not None:
            st = st[:n2]
            th = np.linalg.lstsq(rA.M, st - rA.c, rcond=None)[0]
        else:
            th = np.linalg.lstsq(M, st - c, rcond=None)[0]
    ll, g, H = ll_g_h(th)
    if not np.isfinite(ll):
        raise StataError('initial values not feasible', 1400)
    log = [(0, ll)]
    it = 0
    while it < maxit:
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ g
        lam = 1.0
        while True:
            th1 = th + lam * step
            ll1, _, _ = ll_g_h(th1, 0)
            if np.isfinite(ll1) and ll1 >= ll - 1e-12 * abs(ll):
                break
            lam /= 2
            if lam < 1e-10:
                th1, ll1 = th, ll
                break
        it += 1
        dl = abs(ll1 - ll) / (abs(ll) + 1)
        dth = np.max(np.abs(th1 - th) / (np.abs(th) + 1)) if th.size else 0
        th, ll = th1, ll1
        ll, g, H = ll_g_h(th)
        log.append((it, ll))
        try:
            nr = float(g @ np.linalg.solve(H, g))
        except np.linalg.LinAlgError:
            nr = np.inf
        if (dl < 1e-7 or dth < 1e-6) and nr < 1e-5:
            break
    # final Newton polish (silent) for accuracy
    for _ in range(200):
        try:
            th2 = th + np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        l2, g2, H2 = ll_g_h(th2)
        if not (np.isfinite(l2) and l2 >= ll - 1e-12 * abs(ll)):
            break
        done = abs(l2 - ll) <= 1e-15 * (abs(ll) + 1) and np.max(np.abs(th2 - th) / (np.abs(th) + 1e-8)) < 1e-10
        th, ll, g, H = th2, l2, g2, H2
        if done:
            break
    try:
        Vt = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        Vt = np.linalg.pinv(H)
    return th, ll, Vt, M, c, build, log, it < maxit


@command('svar')
def cmd_svar(ip, cmd, rest, by):
    p = parse_command(rest)
    toks = _toks(ip, p.body)
    if not toks:
        if ip.est and ip.est.get('cmd') == 'svar':
            return ip.est['replay'](ip)
        raise StataError('last estimates not found', 301)
    K = len(toks)
    aeq = _get_mat(ip, p.opts.get('aeq', 3), 'aeq', K)
    beq = _get_mat(ip, p.opts.get('beq', 3), 'beq', K)
    acns = _get_mat(ip, p.opts.get('acns', 4), 'acns', K)
    bcns = _get_mat(ip, p.opts.get('bcns', 4), 'bcns', K)
    lreq = _get_mat(ip, p.opts.get('lreq', 3), 'lreq', K)
    lrcns = _get_mat(ip, p.opts.get('lrcns', 3), 'lrcns', K)
    if p.opts.get('aconstraints', 2) or p.opts.get('bconstraints', 2) or p.opts.get('lrconstraints', 3):
        raise StataError('aconstraints()/bconstraints()/lrconstraints() are not available in Stata Web; '
                         'use aeq()/acns()/beq()/bcns()/lreq()/lrcns()', 198)
    long_run = lreq is not None or lrcns is not None
    if long_run and any(x is not None for x in (aeq, beq, acns, bcns)):
        raise StataError('short-run and long-run constraints may not both be specified', 198)
    if not long_run and all(x is None for x in (aeq, beq, acns, bcns)):
        raise StataError('no constraints specified', 198)
    lags, exog = _var_options(ip, p)
    small = p.opts.has('small', 2)
    dfk = p.opts.has('dfk')
    ve = var_estimate(ip, p, toks, lags, exog, p.opts.has('noconstant', 4), dfk, small, p.opts.has('lutstats', 3))
    if p.opts.has('var'):
        _display_var(ip, ve)
        ip.txt('\n')
    T = ve['T']
    Sig = ve['Sigma']
    if long_run:
        A = lag_matrices(ve)
        Abar = np.eye(K) - sum(A) if A else np.eye(K)
        try:
            np.linalg.inv(Abar)
        except np.linalg.LinAlgError:
            raise StataError('Matrix (I-A_1-A_2...-A_p) not invertible', 498)
        rC = Restr(K, lreq, lrcns, 'C')
        restr = [rC]
        th, ll, Vt, M, c, build, log, conv = svar_fit(Sig, T, K, rC, None, lr_abar=Abar)
        groups = [('C', rC)]
    else:
        rA = Restr(K, aeq, acns, 'A')
        rB = Restr(K, beq, bcns, 'B')
        restr = [rA, rB]
        th, ll, Vt, M, c, build, log, conv = svar_fit(Sig, T, K, rA, rB)
        groups = [('A', rA), ('B', rB)]
    As, Bs = build(th)
    n2 = K * K
    # full parameter vector and V (constrained elements have zero variance)
    if long_run:
        full = rC.full(th)
        Mg = rC.M
        Vfull = Mg @ Vt @ Mg.T
    else:
        full = np.r_[_vec(As), _vec(Bs)]
        Vfull = M @ Vt @ M.T
    nfree = th.size
    N_cns = (n2 if long_run else 2 * n2) - nfree
    oid_df = N_cns - (n2 if long_run else 2 * n2) + K * (K + 1) / 2
    ll_ref = ve['ll'] if not dfk else -0.5 * T * (K * math.log(2 * math.pi) + math.log(ve['det']) + K)
    chi2_oid = 2 * (ll_ref - ll) if oid_df > 0 else np.nan
    names, labels = [], []
    for letter, r in groups:
        for jcol in range(K):
            for irow in range(K):
                names.append(f'/{letter}:{irow + 1}_{jcol + 1}')
                labels.append((letter, f'{irow + 1}_{jcol + 1}', irow + jcol * K, r))
    df_r = (T - nfree) if small else None
    cns_txt = [t for _, r in groups for t in r.cns_txt]
    est = {'cmd': 'svar', 'names': names, 'b': full, 'V': Vfull, 'colinfo': None, 'sample': ve['sample'],
           'depvar': ' '.join(toks), 'nomargins': True, 'varest': ve, 'As': As, 'Bs': Bs, 'endog': toks,
           'svar_free': list(range(nfree)), 'svar_theta': th, 'svar_Vfree': Vt,
           'svar_build': build, 'long_run': long_run}
    sc = {'N': T, 'll': ll, 'oid_df': oid_df, 'N_cns': N_cns, 'neqs': K, 'k_eq': n2 * (1 if long_run else 2),
          'rank': nfree, 'converged': 1 if conv else 0, 'ic_ml': len(log) - 1,
          'll_var': ve['ll'], 'detsig_ml_var': ve['det_ml'], 'detsig_var': ve['det'], 'T_var': T,
          'mlag_var': max(lags), 'tmin': ve['tmin'], 'tmax': ve['tmax'], 'N_gaps_var': ve['N_gaps']}
    if oid_df > 0:
        sc['chi2_oid'] = chi2_oid
    if small:
        sc['df_r'] = df_r
    from .expr import Matrix
    mats = {'A': Matrix(As, toks, toks), 'B': Matrix(Bs, toks, toks), 'Sigma': Matrix(Sig, toks, toks)}
    if long_run:
        mats['C'] = Matrix(np.linalg.inv(Abar) @ Bs, toks, toks)
    est['scalars'] = sc
    est['matrices'] = mats
    est['macros'] = {'cmd': 'svar', 'title': 'Structural vector autoregression', 'endog_var': ' '.join(toks),
                     'depvar_var': ' '.join(toks), 'lags_var': ' '.join(map(str, lags)),
                     'small': 'small' if small else '', 'dfk_var': 'dfk' if dfk else '',
                     'lrmodel': 'lrmodel' if long_run else ''}

    def replay(ip_, show_log=False):
        if show_log:
            ip_.txt('Estimating long-run parameters\n' if long_run else 'Estimating short-run parameters\n')
            from .tsml import print_log
            print_log(ip_, log)
        ip_.txt('Structural vector autoregression\n\n')
        if cns_txt:
            for k, t_ in enumerate(cns_txt, start=1):
                ip_.txt(f' ({k:>2})  {t_}\n')
            ip_.txt('\n')
        s = TV._sample_line(ip_, ve)
        ip_.txt(f'{s:<48}{"Number of obs":<18}= ')
        ip_.res(f'{T:>10,}\n')
        ip_.txt(f'{"Exactly identified model" if oid_df < 1 else "Overidentified model":<48}'
                f'{"Log likelihood":<18}=  ')
        ip_.res(fmt_num(ll, '%9.8g') + '\n\n')
        _svar_table(ip_, groups, full, Vfull, df_r, K)
        if oid_df >= 1:
            chi = g_fmt(chi2_oid, 7).strip()
            ip_.txt(f'LR test of identifying restrictions: chi2(')
            ip_.res(f'{oid_df:.0f}')
            ip_.txt(') = ')
            ip_.res(chi)
            ln = len(f'LR test of identifying restrictions: chi2({oid_df:.0f}) = {chi}')
            ip_.txt(' ' * max(59 - ln, 1) + 'Prob > chi2' + ' ' * 1 + '=')
            ip_.res(f'{sps.chi2.sf(chi2_oid, oid_df):6.3f}\n')
    est['replay'] = lambda ip_: replay(ip_, False)
    est['predict'] = None
    est['predict_multi'] = lambda ip_, pp, typ, tk: TV._var_predict_store(ip_, ve, pp, typ, tk)
    ip.set_estimation(est)
    replay(ip, True)


def _svar_table(ip, groups, b, V, df_r, K):
    se = np.sqrt(np.maximum(np.diag(V), 0))
    stat = 't' if df_r is not None else 'z'
    crit = sps.t.ppf(.975, df_r) if df_r is not None else sps.norm.ppf(.975)
    ip.txt('-' * 78 + '\n')
    ip.txt(f'{"":>12} | Coefficient  Std. err.      {stat}    P>|{stat}|     [95% conf. interval]\n')
    k = 0
    for letter, r in groups:
        ip.txt('-' * 13 + '+' + '-' * 64 + '\n')
        ip.res(f'{"/" + letter:<12}')
        ip.txt(' |\n')
        for jcol in range(K):
            for irow in range(K):
                vi = irow + jcol * K
                lab = f'{irow + 1}_{jcol + 1}'
                ip.txt(f'{lab:>12} |')
                if se[k] == 0:
                    ip.res(f'{g_fmt(b[k], 9):>11}')
                    ip.txt('  (constrained)\n')
                else:
                    t = b[k] / se[k] if se[k] > 0 else np.nan
                    pv = 2 * (sps.t.sf(abs(t), df_r) if df_r is not None else sps.norm.sf(abs(t)))
                    ip.res(f'{g_fmt(b[k], 9):>11}{g_fmt(se[k], 9):>11}{t:9.2f}{pv:8.3f}'
                           f'{g_fmt(b[k] - crit * se[k], 9):>13}{g_fmt(b[k] + crit * se[k], 9):>12}\n')
                k += 1
    ip.txt('-' * 78 + '\n')
