"""xtabond2 (Roodman): difference and system GMM for dynamic panels.  Ported from xtabond2.ado/xtabond2.mata
(version 03.06.00): instrument construction (iv/gmm styles, laglimits, collapse, eq(), passthru), first-difference
or forward-orthogonal-deviation transforms, one- and two-step estimation with robust (Windmeijer-corrected)
standard errors, Arellano-Bond AR tests, Sargan/Hansen and difference-in-Sargan/Hansen tests."""
import math
import re

import numpy as np
from scipy import stats as sps

from .design import coef_table, Col, build_design
from .errors import StataError
from .fmt import fmt_num, comma_int, g_fmt
from .interp import command
from .syntax import parse_command, split_top
from .tsuni import series
from .tsops import split_name


def invsym(A, tol=1e-13):
    """Stata invsym(): generalized inverse that drops (zeroes) linearly dependent rows/columns in order."""
    n = A.shape[0]
    keep = []
    for j in range(n):
        ajj = A[j, j]
        if not ajj > 0 and not ajj < 0:
            continue
        if keep:
            K = keep
            try:
                c = np.linalg.solve(A[np.ix_(K, K)], A[K, j])
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(A[np.ix_(K, K)], A[K, j], rcond=None)[0]
            d = ajj - A[j, K] @ c
        else:
            d = ajj
        if abs(d) > tol * abs(ajj) and abs(ajj) > 1e-300:
            keep.append(j)
    out = np.zeros_like(A)
    if keep:
        out[np.ix_(keep, keep)] = np.linalg.inv(A[np.ix_(keep, keep)])
    return out, n - len(keep)


def _parse_style(raw, kind):
    """split 'varlist, opts' inside iv()/gmm()."""
    if ',' in raw:
        a, b = raw.split(',', 1)
    else:
        a, b = raw, ''
    p = parse_command(', ' + b if b.strip() else '')
    return a.strip(), p.opts


def _eqcode(s):
    s = (s or '').strip().lower()
    if not s or 'both'.startswith(s):
        return 2
    if 'level'.startswith(s):
        return 0
    if 'diff'.startswith(s):
        return 1
    raise StataError(f'equation({s}) invalid.', 198)


class Panel:
    """Rectangular N x T layout of the estimation panel (xtabond2.mata 'Fill')."""

    def __init__(self, ip, idsample):
        d = ip.data
        self.d = d
        self.ivar, self.tvar = d.ts['panel'], d.ts['time']
        self.delta = d.ts.get('delta', 1) or 1
        rows = np.flatnonzero(idsample)
        tv = d.cols[self.tvar][rows].astype(float)
        iv = d.cols[self.ivar][rows].astype(float)
        self.tmin = float(tv.min())
        self.ids = np.unique(iv)
        self.N = self.ids.size
        tt = np.round((tv - self.tmin) / self.delta).astype(int)
        self.T = int(tt.max()) + 1
        self.gi = np.searchsorted(self.ids, iv)
        self.ti = tt
        self.rows = rows

    def rect(self, x):
        out = np.full((self.N, self.T), np.nan)
        out[self.gi, self.ti] = np.asarray(x, float)[self.rows]
        return out


def _diff(R):
    out = np.full(R.shape, np.nan)
    out[:, 1:] = R[:, 1:] - R[:, :-1]
    return out


def _orthog(R, complete):
    """forward orthogonal deviations stored one period later (xtabond2.mata _Orthog, forward=1)."""
    N, T = R.shape[:2]
    out = np.full(R.shape, np.nan)
    for i in range(N):
        n = 0
        s = np.zeros(R.shape[2:]) if R.ndim == 3 else 0.0
        for t in range(T - 1, -1, -1):
            if n and t + 1 < T:
                out[i, t + 1] = math.sqrt(1 - 1 / (n + 1)) * (R[i, t] - s / n)
            if complete[i, t]:
                n += 1
                s = s + R[i, t]
    return out


def _shift(R, l):
    """R[:, t] -> value at t-l (lag l>0, lead l<0), zero outside."""
    N, T = R.shape[:2]
    out = np.zeros(R.shape)
    if l >= 0:
        if l < T:
            out[:, l:] = R[:, :T - l]
    else:
        if -l < T:
            out[:, :T + l] = R[:, -l:]
    return out


@command('xtabond2')
def cmd_xtabond2(ip, cmd, rest, by):
    p = parse_command(rest)
    if not p.body.strip():
        if ip.est and ip.est.get('cmd') == 'xtabond2':
            return ip.est['replay'](ip)
        raise StataError('last estimates not found', 301)
    d = ip.data
    if not d.ts or not d.ts.get('panel') or not d.ts.get('time'):
        raise StataError('You must xtset the data to specify the panel and time variables.', 459)
    if p.weight:
        raise StataError('weights are not available with xtabond2 in Stata Web', 101)
    for o in ('pca', 'cluster', 'svmat', 'svvar'):
        if p.opts.has(o, 2 if o == 'cl' else len(o)):
            raise StataError(f'option {o} is not available in Stata Web', 198)
    toks = split_top(p.body)
    full = np.ones(d.n, bool)
    cols = _bd(ip, d, toks[1:])
    yname, y = _series_any(ip, toks[0])
    arlevels = p.opts.has('arlevels', 3)
    small = p.opts.has('small', 2)
    steps = 2 if p.opts.has('twostep', 3) else 1
    robust = p.opts.has('robust', 1) or p.opts.has('vce', 3)
    if p.opts.has('vce', 3):
        v = (p.opts.get('vce', 3) or '').strip().lower()
        if not v.startswith('r'):
            raise StataError(f'vce({v}) not allowed', 198)
    onestepnonrobust = steps == 1 and not robust
    diffsargan = not p.opts.has('nodiffsargan', 6)
    orthogonal = p.opts.has('orthogonal', 2)
    hh = p.opts.get('h', 1)
    h = int(hh) if hh not in (None, '') else 3
    if h not in (1, 2, 3):
        raise StataError(f'h({h}) invalid.', 198)
    at = p.opts.get('artests', 2)
    artests = int(at) if at not in (None, '') else 2
    system = not p.opts.has('noleveleq', 4)
    consopt = system and not p.opts.has('noconstant', 4)
    if not system and arlevels:
        raise StataError('arlevels option invalid for difference GMM estimation.', 198)
    # ---- sample: marksample on the varlist (levels), if/in
    touse_full = ip.mark(p, d) & ~np.isnan(y)
    for c in cols:
        touse_full &= ~np.isnan(c.values)
    touse_full &= ~np.isnan(d.cols[d.ts['panel']].astype(float))
    pid = d.cols[d.ts['panel']].astype(float)
    ids_with = np.unique(pid[touse_full])
    idsample = np.isin(pid, ids_with) & ~np.isnan(pid)
    if not touse_full.any():
        raise StataError('No observations.', 2000)
    tvv = d.cols[d.ts['time']].astype(float)
    if np.isnan(tvv[idsample]).any():
        raise StataError(f'Missing values in time variable ({d.ts["time"]}).', 459)
    P = Panel(ip, idsample)
    N, T = P.N, P.T
    Y0 = P.rect(y)
    xnames = [c.name for c in cols]
    xinfo = list(cols)
    X0 = np.stack([P.rect(c.values) for c in cols], axis=2) if cols else np.zeros((N, T, 0))
    if consopt:
        X0 = np.concatenate([X0, np.ones((N, T, 1))], axis=2)
        xnames.append('_cons')
        xinfo.append(Col(name='_cons', values=None, term=-1, group='_cons', level='', kind='cons', vars=[]))
    elif X0.shape[2] == 0:
        raise StataError('No regressors.', 481)
    touse0 = P.rect(touse_full.astype(float))
    touse0 = np.nan_to_num(touse0) > 0
    complete = ~(np.isnan(X0).any(axis=2) | np.isnan(Y0))
    # ---- instruments
    ivspecs = []
    if consopt:
        ivspecs.append({'names': ['_cons'], 'base': np.ones((N, T, 1)), 'eq': 0, 'passthru': False, 'mz': False,
                        'style': ''})
    for raw in p.opts.all('ivstyle', 2):
        vl, o = _parse_style(raw, 'iv')
        ivc = _bd(ip, d, split_top(vl))
        eq = _eqcode(o.get('equation', 2))
        sp = {'names': [c.name for c in ivc], 'base': np.stack([P.rect(c.values) for c in ivc], axis=2),
              'eq': eq, 'passthru': o.has('passthru', 1), 'mz': o.has('mz')}
        if not (system or eq):
            ip.txt('Instruments for levels equations only ignored since noleveleq specified.\n')
            continue
        if sp['passthru'] and eq == 2 and system:
            raise StataError('passthru not valid with equation(both) in system GMM.', 198)
        st_ = 'iv(' + ' '.join(sp['names'])
        if sp['mz'] or sp['passthru'] or eq != 2:
            st_ += ','
            if sp['passthru']:
                st_ += ' passthru'
            if sp['mz']:
                st_ += ' mz'
            if eq != 2:
                st_ += ' eq(' + ('diff' if eq else 'level') + ')'
        sp['style'] = st_ + ')'
        complete &= ~np.isnan(sp['base']).any(axis=2)
        ivspecs.append(sp)
    gmmspecs = []
    for raw in p.opts.all('gmmstyle', 2):
        vl, o = _parse_style(raw, 'gmm')
        if o.has('split', 2):
            raise StataError('the split suboption is not available in Stata Web', 198)
        if o.has('orthogonal', 2):
            raise StataError('the orthogonal suboption of gmm() is not available in Stata Web', 198)
        gtoks = []
        for t in split_top(vl):
            gtoks += _expand_ts(ip, t)
        base = np.stack([P.rect(_series_any(ip, t)[1]) for t in gtoks], axis=2)
        eq = _eqcode(o.get('equation', 2))
        ll = (o.get('laglimits', 1) or '').split()
        if ll:
            if len(ll) != 2:
                raise StataError(f'Laglimits({" ".join(ll)}) must have two arguments.', 198)
            a = 1 if ll[0] == '.' else int(ll[0])
            b = np.inf if ll[1] == '.' else int(ll[1])
            if a > b:
                a, b = b, a
        else:
            a, b = 1, np.inf
        sp = {'names': gtoks, 'raw': base, 'eq': eq, 'collapse': o.has('collapse', 1),
              'passthru': o.has('passthru', 1), 'lag0': (a, b)}
        if not (system or eq):
            ip.txt('Instruments for levels equations only ignored since noleveleq specified.\n')
            continue
        if system and sp['passthru'] and eq == 2:
            raise StataError('passthru not valid with equation(both) in system GMM.', 198)
        sty = 'gmm(' + ' '.join(gtoks) + ','
        if sp['collapse']:
            sty += ' collapse'
        if sp['passthru']:
            sty += ' passthru'
        if eq != 2:
            sty += ' eq(' + ('diff' if eq else 'level') + ')'
        sty += f' lag({a} {"." if b == np.inf else b}))'
        sp['style'] = sty
        gmmspecs.append(sp)
    # ---- transformed / levels data
    xform = (lambda R: _orthog(R, complete)) if orthogonal else _diff
    Xd = xform(X0)
    Yd = xform(Y0[:, :, None])[:, :, 0]
    tuse_d = _shift(touse0.astype(float), 1) > 0 if orthogonal else touse0.copy()
    tuse_l = touse0.copy()
    # IV instrument columns: list of (diff_rows (N,T) or None, level_rows or None, group index)
    Zcols_d, Zcols_l, grp_iv = [], [], []
    for gi_, sp in enumerate(ivspecs):
        B = sp['base']
        for m in range(B.shape[2]):
            bd = None
            if sp['eq']:
                bd = B[:, :, m] if sp['passthru'] else xform(B[:, :, m:m + 1])[:, :, 0]
                if sp['mz']:
                    bd = np.nan_to_num(bd)
            bl = None
            if system and sp['eq'] != 1:
                bl = np.nan_to_num(B[:, :, m]) if sp['mz'] else B[:, :, m]
            Zcols_d.append(bd)
            Zcols_l.append(bl)
            grp_iv.append(gi_)
    # rows with missing IV instruments are dropped (unless mz)
    for bd, bl in zip(Zcols_d, Zcols_l):
        if bd is not None:
            tuse_d &= ~np.isnan(bd)
        if bl is not None:
            tuse_l &= ~np.isnan(bl)
    tuse_d &= ~(np.isnan(Xd).any(axis=2) | np.isnan(Yd))
    tuse_l &= ~(np.isnan(X0).any(axis=2) | np.isnan(Y0))
    if not system:
        tuse_l = np.zeros_like(tuse_l)
    elif p.opts.has('diffsmp'):
        tuse_l &= tuse_d
    # GMM instrument columns
    Gd, Gl, grp_g, g_extra = [], [], [], []
    laglims = []
    for gi_, sp in enumerate(gmmspecs):
        R = sp['raw']
        eq = sp['eq']
        fe = eq == 0                                    # FullInstSetEq
        fdiff = not (eq or sp['passthru'])             # FullInstSetDiffed
        extra = eq == 2 and system
        lagmax = T - 1 - (1 if (fe and fdiff) else 0)
        fwdmax = T - 2 + (1 if fe else 0)
        a, b = sp['lag0']
        a = max(-fwdmax, a)
        b = int(min(lagmax, b))
        laglims.append((a, b))
        for m in range(R.shape[2]):
            x = R[:, :, m]
            base = np.nan_to_num(_diff(x[:, :, None])[:, :, 0]) if fdiff else np.nan_to_num(x)
            dbase = np.nan_to_num(_diff(x[:, :, None])[:, :, 0])
            for l in range(a, b + 1):
                sh = _shift(base, l)
                if sp['collapse']:
                    col = sh
                    (Gl if fe else Gd).append(col)
                    (Gd if fe else Gl).append(None)
                    grp_g.append(gi_)
                    g_extra.append(False)
                else:
                    for s in range(T):
                        t = s + l
                        if t < 0 or t >= T or s < (1 if fdiff else 0):
                            continue
                        col = np.zeros((N, T))
                        col[:, t] = base[:, s]
                        (Gl if fe else Gd).append(col)
                        (Gd if fe else Gl).append(None)
                        grp_g.append(gi_)
                        g_extra.append(False)
            if extra:
                if a > 0:
                    L = a - 1
                elif b > 0:
                    L = 0
                else:
                    L = b - 1
                if sp['collapse']:
                    Gd.append(None)
                    Gl.append(_shift(dbase, L))
                    grp_g.append(gi_)
                    g_extra.append(True)
                else:
                    for s in range(T):
                        t = s + L
                        if t < 0 or t >= T:
                            continue
                        col = np.zeros((N, T))
                        col[:, t] = dbase[:, s]
                        Gd.append(None)
                        Gl.append(col)
                        grp_g.append(gi_)
                        g_extra.append(True)
    j_iv, j_g = len(Zcols_d), len(Gd)
    j = j_iv + j_g
    # ---- stacked per-group matrices: rows = [diff t=0..T-1 ; level t=0..T-1]
    R2 = (2 if system else 1) * T
    k0 = X0.shape[2]
    use = np.concatenate([tuse_d, tuse_l], axis=1) if system else tuse_d
    Xs = np.concatenate([np.nan_to_num(Xd), np.nan_to_num(X0)], axis=1) if system else np.nan_to_num(Xd)
    Ys = np.concatenate([np.nan_to_num(Yd), np.nan_to_num(Y0)], axis=1) if system else np.nan_to_num(Yd)
    Xs = Xs * use[:, :, None]
    Ys = Ys * use
    Z = np.zeros((N, R2, j))
    for c, (bd, bl) in enumerate(zip(Zcols_d, Zcols_l)):
        if bd is not None:
            Z[:, :T, c] = np.nan_to_num(bd)
        if bl is not None and system:
            Z[:, T:, c] = np.nan_to_num(bl)
    for c, (bd, bl) in enumerate(zip(Gd, Gl)):
        if bd is not None:
            Z[:, :T, j_iv + c] = bd
        if bl is not None and system:
            Z[:, T:, j_iv + c] = bl
    Z = Z * use[:, :, None]
    # collinear regressors (drop), as _rmcoll
    Xflat = Xs.reshape(-1, k0)
    keep = _rmcoll(Xflat, consopt)
    dropped = [xnames[i] for i in range(k0) if i not in keep]
    for nm in dropped:
        ip.txt(f'{nm} dropped due to collinearity\n')
    Xs = Xs[:, :, keep]
    xnames = [xnames[i] for i in keep]
    xinfo = [xinfo[i] for i in keep]
    k = len(keep)
    if k == 0:
        raise StataError('No regressors.', 481)
    obs_rows = tuse_l if system else tuse_d
    NObs = int(obs_rows.sum())
    gcount = obs_rows.sum(axis=1)
    NGroups = int((gcount > 0).sum())
    if NObs == 0:
        raise StataError('No observations.', 2000)
    if j < k:
        raise StataError('Equation not identified. Regessors outnumber instruments.', 481)
    wttot = int(tuse_d.sum()) if (system and h > 1) else NObs
    # ---- H matrix
    M = _xmat(orthogonal, T)                  # rows: transformed equations, cols: levels
    if not system:
        H = np.eye(T) if h == 1 else M @ M.T
    else:
        if h == 1:
            H = np.eye(2 * T)
        elif h == 2:
            H = np.block([[M @ M.T, np.zeros((T, T))], [np.zeros((T, T)), np.eye(T)]])
        else:
            H = np.block([[M @ M.T, M], [M.T, np.eye(T)]])
    H[np.abs(H) < 1e-12] = 0
    ZX = np.einsum('irj,irk->jk', Z, Xs)
    ZY = np.einsum('irj,ir->j', Z, Ys)
    S = np.einsum('irj,rs,isk->jk', Z, H, Z)
    A1raw, nzero = invsym(S)
    XZA = ZX.T @ A1raw
    V1, _ = invsym(XZA @ ZX)
    b1 = V1 @ XZA @ ZY
    e1 = Ys - np.einsum('irk,k->ir', Xs, b1)
    errq = np.zeros(R2, bool)
    if system:
        errq[:T] = h != 1
        errq[T:] = h == 1
    else:
        errq[:] = True
    pe = e1[:, errq]
    sig2 = float(np.sum(pe ** 2)) / (2 - (1 if (orthogonal or h == 1) else 0)) / wttot
    A1 = A1raw / sig2
    V1 = V1 * sig2
    j0 = j - int(np.sum(np.diag(A1) == 0))
    warn_many = j0 > NGroups
    Ze1 = np.einsum('irj,ir->j', Z, e1)
    sargan = float(Ze1 @ A1 @ Ze1)
    sar_df = j0 - k
    sarganp = float(sps.chi2.sf(sargan, sar_df)) if sar_df > 0 else np.nan
    hansen = hansenp = None
    V1r = V2 = V2r = A2 = e2 = b2 = None
    warn_sing = False
    if onestepnonrobust:
        H = H * sig2
        S = S * sig2
    else:
        Zei = np.einsum('irj,ir->ij', Z, e1)       # N x j
        S = Zei.T @ Zei
        VXZA = V1 @ ZX.T @ A1
        if robust:
            V1r = VXZA @ S @ VXZA.T
        A2, z2 = invsym(S)
        warn_sing = z2 > 0
        XZA2 = ZX.T @ A2
        V2, _ = invsym(XZA2 @ ZX)
        VXZA2 = V2 @ XZA2
        b2 = VXZA2 @ ZY
        e2 = Ys - np.einsum('irk,k->ir', Xs, b2)
        if steps == 2:
            pe = e2[:, errq]
            sig2 = float(np.sum(pe ** 2)) / (2 - (1 if (orthogonal or h == 1) else 0)) / wttot
        Ze2 = np.einsum('irj,ir->j', Z, e2)
        A2Ze = A2 @ Ze2
        if steps == 2 and robust:
            ZXi = np.einsum('irj,irk->ijk', Z, Xs)          # N x j x k
            sc = Zei @ A2Ze                                 # N
            D = np.einsum('i,ijk->jk', sc, ZXi) + np.einsum('ij,l,ilk->jk', Zei, A2Ze, ZXi)
            D = VXZA2 @ D
            V2r = V2 + D @ V1r @ D.T + 2 * D @ V2
        hansen = float(Ze2 @ A2Ze)
        hansenp = float(sps.chi2.sf(hansen, sar_df)) if sar_df > 0 else np.nan
    if steps == 1:
        b = b1
        Vp = V1r if robust else V1
        Vnr, A_, e_ = V1, A1, e1
    else:
        b = b2
        Vp = V2r if robust else V2
        Vnr, A_, e_ = V2, A2, e2
    V = (Vp + Vp.T) / 2
    m2VZXA = (-2 * Vnr) @ ZX.T @ A_
    Var_ar = Vp
    if small:
        tmp = wttot / (wttot - k)
        V = V * (tmp if onestepnonrobust else (NObs - 1) / (NObs - k) * NGroups / (NGroups - 1))
        sig2 *= tmp
    # Wald / F excluding the constant
    sl = [i for i, nm in enumerate(xnames) if nm != '_cons']
    DFm = len(sl)
    F = Fp = Wald = Waldp = None
    DFr = None
    if small:
        DFr = (NObs - DFm if onestepnonrobust else NGroups) - (1 if consopt else 0)
    if DFm:
        Vi, _ = invsym(V[np.ix_(sl, sl)])
        q = float(b[sl] @ Vi @ b[sl])
        if small:
            F = q / DFm
            Fp = float(sps.f.sf(F, DFm, DFr))
        else:
            Wald = q
            Waldp = float(sps.chi2.sf(q, DFm))
    # ---- difference-in-Sargan/Hansen
    dsr = []
    if diffsargan:
        A1diag = np.diag(A1)
        groups = []
        if system:
            ps = [j_iv + c for c in range(j_g) if Gd[c] is not None and not g_extra[c]]
            groups.append(('GMM instruments for levels', list(range(j_iv)) + ps))
        for gi_, sp in enumerate(gmmspecs):
            mine = [j_iv + c for c in range(j_g) if grp_g[c] == gi_]
            groups.append((sp['style'], [c for c in range(j) if c not in mine]))
        for gi_, sp in enumerate(ivspecs):
            if not sp['style']:
                continue
            mine = [c for c in range(j_iv) if grp_iv[c] == gi_]
            groups.append((sp['style'], [c for c in range(j) if c not in mine]))
        base_stat = hansen if hansen is not None else sargan
        for lab, pidx in groups:
            nother = int(np.sum(A1diag[pidx] != 0))
            if nother >= k and nother < j0:
                App, _ = invsym(S[np.ix_(pidx, pidx)])
                ZXp = ZX[pidx]
                XZAp = ZXp.T @ App
                bp = invsym(XZAp @ ZXp)[0] @ (XZAp @ ZY[pidx])
                ep = Ys - np.einsum('irk,k->ir', Xs, bp)
                Zep = np.einsum('irj,ir->j', Z[:, :, pidx], ep)
                unr = float(Zep @ App @ Zep)
                ng = j0 - nother
                dsr.append((lab, unr, base_stat - unr, ng, float(sps.chi2.sf(unr, sar_df - ng)),
                            float(sps.chi2.sf(base_stat - unr, ng))))
    # ---- AR tests
    ar = _ar_tests(arlevels, artests, onestepnonrobust, h, N, T, system, orthogonal, use, tuse_d, tuse_l,
                   Xs, Z, e_, b, X0[:, :, keep], Y0, complete, sig2,
                   m2VZXA, Var_ar, M)
    # ---- post
    from .expr import Matrix
    smask = np.zeros(d.n, bool)
    rr = P.rows
    smask[rr] = obs_rows[P.gi, P.ti]
    Ti = gcount[gcount > 0]
    vcetype = ('Robust' if steps == 1 else 'Corrected') if robust else ''
    sc = {'N': NObs, 'N_g': NGroups, 'j': j0, 'j0': j, 'g_min': int(Ti.min()), 'g_max': int(Ti.max()),
          'g_avg': NObs / NGroups, 'sargan': sargan, 'sar_df': sar_df, 'sarganp': sarganp, 'sig2': sig2,
          'sigma': math.sqrt(sig2), 'artests': artests, 'df_m': DFm, 'h': h}
    if hansen is not None:
        sc.update({'hansen': hansen, 'hansen_df': sar_df, 'hansenp': hansenp})
    if small:
        sc.update({'F': F if F is not None else np.nan, 'F_p': Fp if Fp is not None else np.nan, 'df_r': DFr})
    else:
        sc.update({'chi2': Wald if Wald is not None else np.nan, 'chi2p': Waldp if Waldp is not None else np.nan})
    for i_, (z, pz) in enumerate(ar, start=1):
        sc[f'ar{i_}'] = z
        sc[f'ar{i_}p'] = pz
    est = {'cmd': 'xtabond2', 'names': xnames, 'b': b, 'V': V, 'colinfo': xinfo, 'df_r': DFr, 'depvar': yname,
           'sample': smask, 'stat': 't' if small else 'z', 'vce': 'robust' if robust else 'gmm', 'scalars': sc,
           'macros': {'cmd': 'xtabond2', 'depvar': yname, 'ivar': P.ivar, 'tvar': P.tvar,
                      'esttype': 'system' if system else 'difference', 'twostep': 'twostep' if steps == 2 else '',
                      'transform': 'orthogonal deviations' if orthogonal else 'first differences',
                      'artype': 'levels' if arlevels else 'first differences', 'vcetype': vcetype,
                      'small': 'small' if small else '', 'robust': 'robust' if robust else ''},
           'matrices': {'A1': Matrix(A1), 'Ze': Matrix(Ze1[:, None])}, 'nomargins': True}
    if dsr:
        est['matrices']['diffsargan'] = Matrix([[r[1] for r in dsr], [r[2] for r in dsr], [r[3] for r in dsr],
                                                [r[4] for r in dsr], [r[5] for r in dsr]])
    lag_disp = {gi_: laglims[gi_] for gi_ in range(len(gmmspecs))}
    warns = []
    if warn_many:
        warns.append('Warning: Number of instruments may be large relative to number of observations.')
    if warn_sing:
        warns.append('Warning: Two-step estimated covariance matrix of moments is singular.')
        warns.append('  Using a generalized inverse to calculate ' + (
            'optimal weighting matrix for two-step estimation.' if steps == 2 else
            'robust weighting matrix for Hansen test.'))
        if diffsargan:
            warns.append('  Difference-in-Sargan/Hansen statistics may be negative.')

    def replay(ip_):
        for w_ in warns:
            ip_.res(w_ + '\n')
        _display(ip_, est, steps, system, small, robust, vcetype, ivspecs, gmmspecs, lag_disp, orthogonal, ar,
                 arlevels, dsr, hansen is not None)
    est['replay'] = replay
    est['ab_internal'] = {'obs_d': int(tuse_d.sum()), 'g_d': tuse_d.sum(axis=1), 'robust': robust,
                          'steps': steps, 'system': system}

    def predict(ip_, kind, m_):
        Xf = []
        for c in xinfo:
            Xf.append(np.ones(d.n) if c.kind == 'cons' else c.values)
        xb = np.column_stack(Xf) @ b
        if kind in (None, '', 'xb'):
            return xb
        if kind in ('residuals', 'resid', 'r', 'e'):
            return y - xb
        raise StataError(f'predict option {kind} not allowed after xtabond2', 198)
    est['predict'] = predict
    ip.set_estimation(est)
    replay(ip)


def _bd(ip, d, toks):
    tk = []
    for t in toks:
        if re.search(r'[*?~]', t) or (re.match(r'^[A-Za-z_]\w*-[A-Za-z_]\w*$', t)):
            tk += d.expand_varlist([t])
        else:
            tk.append(t)
    toks = tk
    cols, m = build_design(ip, d, toks, np.ones(d.n, bool))
    for c in cols:
        c.values = np.where(m, np.asarray(c.values, float), np.nan)
    return cols


def _expand_ts(ip, tok):
    from .tsops import expand_token
    out = []
    for t in expand_token(tok):
        ch, base = split_name(t)
        if ch:
            out.append(t)
        else:
            out += ip.data.expand_varlist([t])
    return out


def _series_any(ip, tok):
    x, nm = series(ip, tok)
    return nm, x


def _xmat(orthogonal, T):
    """T x T transformation matrix: row t gives the transformed equation stored at t (D or FOD)."""
    M = np.zeros((T, T))
    if not orthogonal:
        for t in range(1, T):
            M[t, t] = 1
            M[t, t - 1] = -1
    else:
        for t in range(T - 1):
            m = T - 1 - t
            M[t + 1, t] = 1
            M[t + 1, t + 1:] = -1 / m
            M[t + 1] *= math.sqrt(m / (m + 1))
    return M


def _rmcoll(X, hascons):
    k = X.shape[1]
    if k <= 1:
        return list(range(k))
    if hascons:
        t = X[:, -1]
        Xp = X - np.outer(t, (X.T @ t) / t.sum())
        U = Xp.T @ Xp
        U = U[:-1, :-1]
    else:
        U = X.T @ X
    dg = np.sqrt(np.diag(U))
    dg[dg == 0] = 1
    Un = U / dg[:, None] / dg[None, :]
    Ui, _ = invsym(Un)
    dd = np.diag(Ui)
    keep = [i for i in range(len(dd)) if abs(dd[i]) > 1e-12]
    if hascons:
        keep.append(k - 1)
    return keep


def _ar_tests(arlevels, artests, onr, h, N, T, system, orthogonal, use, tuse_d, tuse_l, Xs, Z, e, b, X0, Y0,
              complete, sig2, m2VZXA, V, M):
    out = []
    Dm = _xmat(False, T)
    if onr:
        hh = 1 if arlevels else h
        Har = (np.eye(T) if hh == 1 else Dm @ Dm.T) * sig2
        Mr = _xmat(orthogonal, T)
        # psit as in _H(h, 0, orthogonal, 0, T): Ml'Mr with Ml = D', Mr = M'
        if h == 1:
            psit = np.eye(T) * sig2
        else:
            psit = (Dm @ Mr.T) * sig2
        if system:
            psit = np.hstack([np.zeros((T, T)), psit]) if arlevels else np.hstack([psit, np.zeros((T, T))])
    if system:
        rows = slice(T, 2 * T) if arlevels else slice(0, T)
    else:
        rows = slice(0, T)
    tuse2 = use[:, rows]
    if orthogonal and not arlevels:
        el = Y0 - np.einsum('itk,k->it', X0, b)
        wl0 = np.nan_to_num(_diff(el[:, :, None])[:, :, 0])
        pX = np.nan_to_num(_diff(X0))
        w = wl0.copy()
        wl = wl0.copy()
    else:
        w = e[:, rows].copy()
        pX = Xs[:, rows, :]
        wl = w.copy()
    for lag in range(1, artests + 1):
        wl = _shift(wl, 1)
        sw = np.sum(w * wl, axis=1)
        if not np.any(sw):
            out.append((np.nan, np.nan))
            continue
        if onr:
            wli = wl * tuse2
            wHw = float(np.einsum('it,ts,is->', wli, Har, wli))
            psiw = np.einsum('ts,it->is', psit, wli)
            ZHw = np.einsum('irj,ir->j', Z, psiw)
        else:
            wHw = float(np.sum(sw ** 2))
            ZHw = np.einsum('irj,ir,i->j', Z, e, sw)
        tmp = np.einsum('itk,it->k', pX, wl)
        den = wHw + tmp @ (m2VZXA @ ZHw + V @ tmp)
        z = float(np.sum(sw)) / math.sqrt(den) if den > 0 else np.nan
        out.append((z, 2 * sps.norm.cdf(-abs(z)) if z == z else np.nan))
    return out


def _display(ip, e, steps, system, small, robust, vcetype, ivspecs, gmmspecs, laglims, orthogonal, ar, arlevels,
             dsr, has_hansen):
    sc = e['scalars']
    ip.txt(f"\nDynamic panel-data estimation, {'one' if steps == 1 else 'two'}-step "
           f"{'system' if system else 'difference'} GMM\n" + '-' * 78 + '\n')
    m = e['macros']
    ip.txt(f"{'Group variable: ' + m['ivar'][:12]:<48}Number of obs      = ")
    ip.res(f"{sc['N']:9.0f}\n")
    ip.txt(f"{'Time variable : ' + m['tvar'][:12]:<48}Number of groups   = ")
    ip.res(f"{fmt_num(sc['N_g'], '%9.0g')}\n")
    s = f"Number of instruments = {sc['j']}"
    ip.txt(f'{s:<48}Obs per group: min = ')
    ip.res(f"{fmt_num(sc['g_min'], '%9.0g')}\n")
    if small:
        s = f"F({sc['df_m']}, {sc['df_r']})"
        ip.txt(f'{s:<14}= ')
        ip.res(fmt_num(sc['F'], '%9.2f'))
        ip.txt(' ' * (63 - 14 - 2 - 9) + 'avg = ')
        ip.res(f"{sc['g_avg']:9.2f}\n")
        ip.txt(f'{"Prob > F":<14}=')
        ip.res(fmt_num(sc['F_p'], '%10.3f'))
        ip.txt(' ' * (63 - 14 - 1 - 10) + 'max = ')
        ip.res(f"{fmt_num(sc['g_max'], '%9.0g')}\n")
    else:
        s = f"Wald chi2({sc['df_m']})"
        ip.txt(f'{s:<14}= ')
        ip.res(fmt_num(sc['chi2'], '%9.2f'))
        ip.txt(' ' * (63 - 14 - 2 - 9) + 'avg = ')
        ip.res(f"{sc['g_avg']:9.2f}\n")
        ip.txt(f'{"Prob > chi2":<14}=')
        ip.res(fmt_num(sc['chi2p'], '%10.3f'))
        ip.txt(' ' * (63 - 14 - 1 - 10) + 'max = ')
        ip.res(f"{fmt_num(sc['g_max'], '%9.0g')}\n")
    coef_table(ip, e, 't' if small else 'z', robust=vcetype or None, depvar=e['depvar'])
    if steps == 2 and vcetype != 'Corrected':
        ip.txt('Warning: Uncorrected two-step standard errors are unreliable.\n\n')
    eqname = 'orthogonal deviations' if orthogonal else 'first differences'
    for eq in range(2 if system else 1):
        shown = False
        for kind, specs in (('iv', list(reversed(ivspecs))), ('gmm', list(reversed(gmmspecs)))):
            # display order follows e(ivinsts#)/e(gmminsts#): last-specified first, _cons last
            if kind == 'iv':
                cons = [s for s in specs if s['names'] == ['_cons'] and not s.get('style')]
                specs = [s for s in specs if not any(s is c_ for c_ in cons)] + cons
            kshown = False
            for idx, sp in enumerate(specs):
                if sp['eq'] == eq:
                    continue
                if not shown:
                    ip.txt(f'Instruments for {eqname if eq == 0 else "levels"} equation\n')
                    shown = True
                if not kshown:
                    ip.txt('  Standard\n' if kind == 'iv' else
                           '  GMM-type (missing=0, separate instruments for each period unless collapsed)\n')
                    kshown = True
                vars_ = ' '.join(sp['names'])
                vv = f'({vars_})' if len(sp['names']) > 1 else vars_
                if kind == 'iv':
                    if eq or sp['passthru']:
                        line = vars_
                    else:
                        line = ('FO' if orthogonal else '') + 'D.' + vv
                    if sp['mz']:
                        line += ', missing recoded as zero'
                else:
                    gi_ = next(i_ for i_, x in enumerate(gmmspecs) if x is sp)
                    a, b = laglims[gi_]
                    sub = 1 if (eq and sp['eq']) else 0
                    l1, l2 = a - sub, b - sub
                    line = ('D' if (eq and not sp['passthru']) else '')
                    if l2 > l1 and not (eq and sp['eq']):
                        line += f'L({l1}/{l2})'
                    elif l1:
                        line += 'L' if l1 == 1 else f'L{l1}'
                    line += '.' + vv
                    if sp['collapse']:
                        line += ' collapsed'
                while line:
                    ip.txt('    ' + line[:74] + '\n')
                    line = line[74:]
        eqname = 'levels'
    ip.txt('-' * 78 + '\n')
    for i_, (z, pz) in enumerate(ar, start=1):
        s = f"Arellano-Bond test for AR({i_}) in {e['macros']['artype']}:"
        ip.txt(f'{s:<51}z = ')
        ip.res(fmt_num(z, '%6.2f'))
        ip.txt('  Pr > z = ')
        ip.res(fmt_num(pz, '%6.3f') + '\n')
    ip.txt('-' * 78 + '\n')
    s = f"Sargan test of overid. restrictions: chi2({sc['sar_df']})"
    ip.txt(f'{s:<48}=')
    ip.res(fmt_num(sc['sargan'], '%7.2f'))
    ip.txt('  Prob > chi2 = ')
    ip.res(fmt_num(sc['sarganp'], '%6.3f') + '\n')
    ip.txt('  (Not robust, but not weakened by many instruments.)\n')
    if has_hansen:
        s = f"Hansen test of overid. restrictions: chi2({sc['hansen_df']})"
        ip.txt(f'{s:<48}=')
        ip.res(fmt_num(sc['hansen'], '%7.2f'))
        ip.txt('  Prob > chi2 = ')
        ip.res(fmt_num(sc['hansenp'], '%6.3f') + '\n')
        ip.txt('  (Robust, but weakened by many instruments.)\n')
    if dsr:
        ot = 'Hansen' if has_hansen else 'Sargan'
        ip.txt(f'\nDifference-in-{ot} tests of exogeneity of instrument subsets:\n')
        for lab, unr, dif, ng, p1, p2 in dsr:
            ip.txt(f'  {lab}\n')
            s = f'    {ot} test excluding group:'
            ip.txt(f'{s:<37}' + f'chi2({sc["sar_df"] - ng})'.ljust(11) + '=')
            ip.res(fmt_num(unr, '%7.2f'))
            ip.txt('  Prob > chi2 = ')
            ip.res(fmt_num(p1, '%6.3f') + '\n')
            s = '    Difference (null H = exogenous):'
            ip.txt(f'{s:<37}' + f'chi2({ng})'.ljust(11) + '=')
            ip.res(fmt_num(dif, '%7.2f'))
            ip.txt('  Prob > chi2 = ')
            ip.res(fmt_num(p2, '%6.3f') + '\n')
    ip.txt('\n')
