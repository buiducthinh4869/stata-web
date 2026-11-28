"""Shared maximum-likelihood engine for time-series models (arima, arch):
BHHH-then-BFGS iterations as Stata's technique(bhhh 5 bfgs 10), numerical per-observation scores,
OPG / OIM / robust variance."""
import numpy as np

from .errors import StataError
from .fmt import g_fmt


def scores(llvec, th, h_rel=1e-6):
    """Per-observation scores by central differences: n x k."""
    k = th.size
    cols = []
    for j in range(k):
        h = h_rel * max(1.0, abs(th[j]))
        e = np.zeros(k)
        e[j] = h
        up, dn = llvec(th + e), llvec(th - e)
        base = None
        if not np.all(np.isfinite(up)) or not np.all(np.isfinite(dn)):
            base = llvec(th)
        if base is not None and np.all(np.isfinite(up)):
            col = (up - base) / h
        elif base is not None and np.all(np.isfinite(dn)):
            col = (base - dn) / h
        elif base is not None:
            col = np.zeros_like(base)
        else:
            col = (up - dn) / (2 * h)
        cols.append(col)
    return np.column_stack(cols)


def hessian(ll, th, h_rel=1e-4):
    """Numerical Hessian; shrinks the step where an evaluation leaves the feasible region."""
    k = th.size
    for shrink in (1.0, 0.1, 0.01, 0.001):
        H = np.zeros((k, k))
        hs = np.array([h_rel * shrink * max(1.0, abs(t)) for t in th])
        f0 = ll(th)
        okall = True
        for i in range(k):
            ei = np.zeros(k)
            ei[i] = hs[i]
            fp, fm = ll(th + ei), ll(th - ei)
            if np.isfinite(fp) and np.isfinite(fm):
                H[i, i] = (fp - 2 * f0 + fm) / hs[i] ** 2
            elif np.isfinite(fp):
                f2 = ll(th + 2 * ei)
                H[i, i] = (f2 - 2 * fp + f0) / hs[i] ** 2 if np.isfinite(f2) else np.nan
            elif np.isfinite(fm):
                f2 = ll(th - 2 * ei)
                H[i, i] = (f2 - 2 * fm + f0) / hs[i] ** 2 if np.isfinite(f2) else np.nan
            else:
                H[i, i] = np.nan
            for j in range(i + 1, k):
                ej = np.zeros(k)
                ej[j] = hs[j]
                v = [ll(th + ei + ej), ll(th + ei - ej), ll(th - ei + ej), ll(th - ei - ej)]
                if all(np.isfinite(v)):
                    H[i, j] = H[j, i] = (v[0] - v[1] - v[2] + v[3]) / (4 * hs[i] * hs[j])
                else:
                    a = [ll(th + ei + ej), ll(th + ei), ll(th + ej)]
                    if all(np.isfinite(a)):
                        H[i, j] = H[j, i] = (a[0] - a[1] - a[2] + f0) / (hs[i] * hs[j])
                    else:
                        H[i, j] = H[j, i] = np.nan
        if np.all(np.isfinite(H)):
            return H
    return H


def safe_inv(A):
    if not np.all(np.isfinite(A)):
        return np.full(A.shape, np.nan)
    try:
        return np.linalg.pinv(A)
    except np.linalg.LinAlgError:
        return np.full(A.shape, np.nan)


def maximize(llvec, th0, bhhh=5, maxit=300, log=None, bfgs_n=10):
    """Maximise sum(llvec(th)). Returns th, ll, iteration log [(it, ll, note)]."""
    th = np.array(th0, float)

    def LL(t):
        v = llvec(t)
        s = float(np.sum(v))
        return s if np.isfinite(s) else -np.inf
    ll = LL(th)
    if not np.isfinite(ll):
        raise StataError('initial values not feasible', 1400)
    it_log = [('(setting optimization to BHHH)', None)] if bhhh > 0 else [('(setting optimization to BFGS)', None)]
    it_log.append((0, ll))
    Hinv = None
    g_prev = None
    it = 0
    mode = 'bhhh' if bhhh > 0 else 'bfgs'
    switch_at = 0
    Hinv_bhhh = None
    while it < maxit:
        S = scores(llvec, th)
        g = S.sum(axis=0)
        if not np.all(np.isfinite(S)):
            S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
            g = S.sum(axis=0)
        if mode == 'bhhh':
            A = S.T @ S
            Hinv_bhhh = safe_inv(A)
            d = Hinv_bhhh @ g
        else:
            if Hinv is None:
                A = S.T @ S
                Hinv = safe_inv(A)
            elif g_prev is not None:
                s_ = step
                y_ = -(g - g_prev)
                sy = float(s_ @ y_)
                if sy > 1e-12:
                    rho = 1.0 / sy
                    I = np.eye(th.size)
                    Hinv = (I - rho * np.outer(s_, y_)) @ Hinv @ (I - rho * np.outer(y_, s_)) + rho * np.outer(s_, s_)
            d = Hinv @ g
        # step halving / doubling line search
        lam = 1.0
        best = None
        for _ in range(60):
            cand = th + lam * d
            llc = LL(cand)
            if np.isfinite(llc) and llc >= ll - 1e-12 * (1 + abs(ll)):
                best = (cand, llc)
                break
            lam /= 2
        if best is None:
            break
        # try a longer step if it helps (Stata's ml does stepping forward too)
        cand2 = th + 2 * lam * d
        ll2 = LL(cand2)
        if np.isfinite(ll2) and ll2 > best[1]:
            best = (cand2, ll2)
            lam *= 2
        step = best[0] - th
        g_prev = g
        dll = best[1] - ll
        th, ll = best
        it += 1
        it_log.append((it, ll))
        crit = abs(float(g @ (Hinv_bhhh if mode == 'bhhh' else Hinv) @ g))
        cnt = it - switch_at
        if mode == 'bhhh' and cnt >= bhhh:
            mode = 'bfgs'
            Hinv = Hinv_bhhh
            switch_at = it
            it_log.append(('(switching optimization to BFGS)', None))
        elif mode == 'bfgs' and bhhh > 0 and cnt >= bfgs_n:
            mode = 'bhhh'
            switch_at = it
            it_log.append(('(switching optimization to BHHH)', None))
        if abs(dll) < 1e-9 * (1 + abs(ll)) and crit < 1e-6:
            break
    # final Newton polishing (not logged) so that reported estimates sit at the optimum
    for _ in range(20):
        try:
            H = hessian(LL, th)
            g = scores(llvec, th).sum(axis=0)
            if not (np.all(np.isfinite(H)) and np.all(np.isfinite(g))):
                break
            d = -np.linalg.solve(H, g)
        except (np.linalg.LinAlgError, ValueError):
            break
        lam = 1.0
        ok = False
        for _ in range(30):
            cand = th + lam * d
            llc = LL(cand)
            if np.isfinite(llc) and llc >= ll - 1e-10 * (1 + abs(ll)):
                ok = True
                break
            lam /= 2
        if not ok:
            break
        done = abs(llc - ll) < 1e-12 * (1 + abs(ll)) and np.max(np.abs(cand - th)) < 1e-9
        th, ll = cand, llc
        if done:
            break
    if it_log and it_log[-1][1] is not None and abs(it_log[-1][1] - ll) > 1e-9 * (1 + abs(ll)):
        it_log.append((it + 1, ll))
    return th, ll, it_log


def print_log(ip, it_log):
    for a, b in it_log:
        if b is None:
            ip.txt(a + '\n')
        else:
            ip.txt(f'Iteration {a}:'.ljust(14) + 'Log likelihood = ')
            ip.res(f'{g_fmt(b, 10).strip()}\n')
    ip.txt('\n')


def vce(llvec, th, kind='opg'):
    """Variance matrix: opg (default), oim, robust (sandwich)."""
    S = scores(llvec, th)
    if kind == 'opg':
        return safe_inv(S.T @ S)
    H = hessian(lambda t: float(np.sum(llvec(t))), th)
    Hi = safe_inv(-H)
    if kind == 'oim':
        return Hi
    return Hi @ (S.T @ S) @ Hi
