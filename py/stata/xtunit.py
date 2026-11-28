"""xtunitroot: llc, ips, ht, breitung, fisher, hadri (ported from Stata's xtunitroot.ado, _xturllc.ado,
_xturips.ado, _xturht.ado, _xturbreitung.ado, _xturfisher.ado, _xturhadri.ado, _xturadfreg.ado)."""
import math

import numpy as np
from scipy import stats as sps

from .errors import StataError
from .fmt import fmt_num
from .interp import command
from .syntax import parse_command, split_top


# ---------------------------------------------------------------- helpers
def _ols(y, X):
    """OLS returning b, se, rss, n, k (k = #columns), ll."""
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    rss = float(e @ e)
    s2 = rss / (n - k) if n > k else np.nan
    se = np.sqrt(np.maximum(np.diag(XtXi) * s2, 0))
    ll = -0.5 * n * (1 + math.log(2 * math.pi) + math.log(rss / n)) if rss > 0 else np.inf
    return b, se, e, rss, n, k, ll


class _Panels:
    def __init__(self, ip, p, var):
        d = ip.data
        if not d.ts or not d.ts.get('panel') or not d.ts.get('time'):
            raise StataError('panel and time variables must be set; use xtset panelvar timevar', 459)
        self.ivar, self.tvar = d.ts['panel'], d.ts['time']
        keys = [self.ivar, self.tvar]
        if d.sortedby[:2] != keys:
            d.sort(keys)
        self.delta = d.ts.get('delta', 1) or 1
        from .tsuni import series
        y, nm = series(ip, var)
        self.name = var
        touse = ip.mark(p, d) & ~np.isnan(y)
        self.y_user = y
        self.touse = touse
        self.pid = d.cols[self.ivar].astype(float)
        self.tv = d.cols[self.tvar].astype(float)
        ids = np.unique(self.pid[touse])
        self.ids = ids
        self.N = ids.size
        if self.N == 0:
            raise StataError('no observations', 2000)
        self.d = d
        self.ip = ip

    def balanced(self):
        cnt = [int(np.sum(self.touse & (self.pid == i))) for i in self.ids]
        times = [tuple(self.tv[self.touse & (self.pid == i)]) for i in self.ids]
        return len(set(cnt)) == 1 and len(set(times)) == 1

    def gaps(self):
        for i in self.ids:
            t = self.tv[self.touse & (self.pid == i)]
            if t.size > 1 and np.any(np.round(np.diff(t) / self.delta) != 1):
                return True
        return False

    def depvar(self, demean):
        y = self.y_user.copy()
        if demean:
            out = np.full(y.size, np.nan)
            for t in np.unique(self.tv[self.touse]):
                s = self.touse & (self.tv == t)
                out[s] = y[s] - y[s].mean()
            return out
        return y

    def trend(self):
        """1,2,... within panel, counting all observations (as by panel: sum((t - t[_n-1])/delta))."""
        tr = np.full(self.pid.size, np.nan)
        for i in np.unique(self.pid[~np.isnan(self.pid)]):
            s = np.flatnonzero(self.pid == i)
            t = self.tv[s]
            v = np.ones(s.size)
            v[1:] = (t[1:] - t[:-1]) / self.delta
            tr[s] = np.cumsum(v)
        return tr

    def lag(self, x, k=1):
        return self.ip._lag(self.d, x, k)


def _header(ip, title, var, h0, ha, rows_left, rows_right, panels_line):
    ip.txt(f'\n{title} for ')
    ip.res(var + '\n')
    ip.txt('-' * (len(title) + 5 + len(var)) + '\n')
    for (l1, r1) in zip((h0, ha), panels_line):
        ip.txt(f'{l1:<44}')
        if r1:
            k, v, col, fmt = r1
            ip.txt(f'{k:<{col - 45}}=')
            ip.res(fmt_num(v, fmt) + '\n')
        else:
            ip.txt('\n')
    ip.txt('\n')


def _ph(N, T, unbalanced):
    if unbalanced:
        return [('Number of panels', N, 68, '%7.0g'), ('Avg. number of periods', T, 68, '%7.2f')]
    return [('Number of panels', N, 63, '%7.0g'), ('Number of periods', T, 63, '%7.0g')]


def _parse_lags(s):
    if s in (None, ''):
        return None, None
    w = s.split()
    if len(w) == 1:
        try:
            v = int(w[0])
            if v < 0:
                raise ValueError
            return v, None
        except ValueError:
            pass
    c = w[0].lower()
    if c not in ('aic', 'bic', 'hqic'):
        raise StataError(f'lags({s}) invalid', 198)
    m = int(w[1]) if len(w) > 1 else 1
    if m < 1:
        raise StataError(f'lags({s}) invalid', 198)
    return m, c


def _ic(ll, n, k, crit, cons=True):
    dfm1 = k if cons else k + 1   # e(df_m)+1
    if crit == 'aic':
        return (-2 * ll + 2 * dfm1) / n
    if crit == 'bic':
        return (-2 * ll + math.log(n) * dfm1) / n
    return (-2 * ll + 2 * math.log(math.log(n)) * dfm1) / n


def _adf_design(dy, ly, lagsdy, tr, p, cons, rows):
    cols = [ly[rows]] + [lagsdy[l][rows] for l in range(1, p + 1)]
    if tr is not None:
        cols.append(tr[rows])
    if cons:
        cols.append(np.ones(rows.size))
    return np.column_stack(cols)


def _adf_panel(dy, ly, lagsdy, tr, p, cons, idx, crit, sel_cons=None):
    """ADF regression for one panel; idx = observation indices of the panel range. Returns (p_used, fit, rows)."""
    def rows_for(pp, base):
        ok = ~np.isnan(dy[base]) & ~np.isnan(ly[base])
        for l in range(1, pp + 1):
            ok &= ~np.isnan(lagsdy[l][base])
        if tr is not None:
            ok &= ~np.isnan(tr[base])
        return base[ok]
    if crit:
        # _xturadfreg is called by _xturllc with an undefined macro for noconstant, so Stata always
        # includes a constant during (and in the final regression of) the lag search
        cons = cons if sel_cons is None else sel_cons
        rmax = rows_for(p, idx)
        if rmax.size == 0:
            return None
        lo, hi = rmax.min(), rmax.max()
        base = idx[(idx >= lo) & (idx <= hi)]
        best, pbest = np.inf, p
        for pp in range(p, -1, -1):
            r = rows_for(pp, base)
            f = _ols(dy[r], _adf_design(dy, ly, lagsdy, tr, pp, cons, r))
            v = _ic(f[6], f[4], f[5], crit, cons)
            if v < best:
                best, pbest = v, pp
        p = pbest
    r = rows_for(p, idx)
    if r.size == 0:
        return None
    return p, _ols(dy[r], _adf_design(dy, ly, lagsdy, tr, p, cons, r)), r


# ---------------------------------------------------------------- LLC
_LLC_ADJ = {1: [(25, .004, 1.049), (30, .003, 1.035), (35, .002, 1.027), (40, .002, 1.021), (45, .001, 1.017),
                (50, .001, 1.014), (60, .001, 1.011), (70, .000, 1.008), (80, .000, 1.007), (90, .000, 1.006),
                (100, .000, 1.005), (250, .000, 1.001), (np.inf, .000, 1.000)],
            2: [(25, -.554, .919), (30, -.546, .889), (35, -.541, .867), (40, -.537, .850), (45, -.533, .837),
                (50, -.531, .826), (60, -.527, .810), (70, -.524, .798), (80, -.521, .789), (90, -.520, .782),
                (100, -.518, .776), (250, -.509, .742), (np.inf, -.500, .707)],
            3: [(25, -.703, 1.003), (30, -.674, .949), (35, -.653, .906), (40, -.637, .971), (45, -.624, .842),
                (50, -.614, .818), (60, -.598, .780), (70, -.587, .751), (80, -.578, .728), (90, -.571, .710),
                (100, -.566, .695), (250, -.533, .603), (np.inf, -.500, .500)]}


def _llc_adj(t, model, nointerp):
    tab = _LLC_ADJ[model]
    row = next(i for i, r in enumerate(tab) if t <= r[0])
    if row == 0 or row == len(tab) - 1 or nointerp:
        return tab[row][1], tab[row][2]
    wc = (t - tab[row - 1][0]) / (tab[row][0] - tab[row - 1][0])
    wp = 1 - wc
    return wp * tab[row - 1][1] + wc * tab[row][1], wp * tab[row - 1][2] + wc * tab[row][2]


def _kernel_w(kernel, i, lags):
    x = i / (lags + 1)
    if kernel == 'bartlett':
        return 1 - x
    if kernel == 'parzen':
        return 1 - 6 * x ** 2 + 6 * abs(x) ** 3 if x <= .5 else 2 * (1 - abs(x)) ** 3
    x2 = 6 * math.pi * x / 5
    return (25 / (12 * (math.pi * x) ** 2)) * (math.sin(x2) / x2 - math.cos(x2))


def _hacvar(P, w, idx, lags, kernel):
    """_gethacvar: w defined on full data (nan elsewhere); sums over the panel range idx."""
    wi = w[idx]
    n = int(np.sum(~np.isnan(wi)))
    s0 = float(np.nansum(wi ** 2))
    v = 0.0
    if kernel in ('bartlett', 'parzen'):
        lags = int(lags)
        rng = range(1, lags + 1)
    else:
        rng = range(1, n + 1)
    for i in rng:
        lw = P.lag(w, i)[idx]
        prod = lw * wi
        sm = float(np.nansum(prod))
        k = 2 * (1 - i / (int(lags) + 1)) if kernel == 'bartlett' else 2 * _kernel_w(kernel, i, lags)
        v += k * sm
    return (s0 + v) / n


def _haclags(P, w, idx, method, kernel):
    if method == 'llc':
        return int(3.21 * idx.size ** (1 / 3))
    wi = w[idx]
    ok = ~np.isnan(wi)
    bigt = int(ok.sum())
    if kernel == 'bartlett':
        q, cy, intb, n = 1, 1.1447, 1, int(4 * (bigt / 100) ** (2 / 9))
    elif kernel == 'parzen':
        q, cy, intb, n = 2, 2.6614, 1, int(4 * (bigt / 100) ** (4 / 25))
    else:
        q, cy, intb, n = 2, 1.3221, 0, int(4 * (bigt / 100) ** (2 / 25))
    s0 = float(np.nansum(wi ** 2))
    sq = 0.0
    sz = s0
    for j in range(1, n + 1):
        sj = float(np.nansum(wi * P.lag(w, j)[idx]))
        sq += 2 * j ** q * sj
        sz += 2 * sj
    pp = 1 / (2 * q + 1)
    gamma = cy * ((sq / sz) ** 2) ** pp
    if intb:
        hb = int(gamma * bigt ** pp)
        return hb - 1 if hb < bigt else bigt - 2
    hb = gamma * bigt ** pp
    return hb - 1 if hb <= bigt else bigt - 1


def _kernel_opt(s):
    if not s:
        return 'bartlett', None, 'llc'
    w = s.split()
    k = w[0].lower()
    kern = next((full for full, mn in (('bartlett', 3), ('parzen', 3), ('quadraticspectral', 4))
                 if len(k) >= mn and full.startswith(k)), 'quadraticspectral' if k == 'qs' else None)
    if kern is None:
        raise StataError(f'{w[0]} is not a recognized kernel', 498)
    if len(w) < 2:
        raise StataError('must specify number of lags or lag-selection algorithm in kernel()', 198)
    if w[1].lower() in ('nwest', 'llc'):
        return kern, None, w[1].lower()
    return kern, float(w[1]), None


def _llc(ip, P, p):
    trend, nocons = p.opts.has('trend', 1), p.opts.has('noconstant', 6)
    if trend and nocons:
        raise StataError('cannot specify both noconstant and trend', 198)
    lags, crit = _parse_lags(p.opts.get('lags', 1))
    if lags is None:
        lags = 1
    kern, hlags, hsel = _kernel_opt(p.opts.get('kernel', 3))
    model = 3 if trend else (1 if nocons else 2)
    if not P.balanced():
        raise StataError('Levin-Lin-Chiu test requires strongly balanced data', 498)
    if P.gaps():
        raise StataError('Levin-Lin-Chiu test cannot have gaps in data', 498)
    dv = P.depvar(p.opts.has('demean'))
    tr = P.trend() if trend else None
    dy = np.where(P.touse, dv - P.lag(dv, 1), np.nan)
    ly = np.where(P.touse, P.lag(dv, 1), np.nan)
    maxl = lags
    lagsdy = {l: P.lag(dy, l) for l in range(1, maxl + 1)}
    E = np.full(dy.size, np.nan)
    Vv = np.full(dy.size, np.nan)
    tis, sis, adfl, hacl = [], [], [], []
    cons = not nocons
    for i in P.ids:
        idx = np.flatnonzero(P.pid == i)
        ok = P.touse[idx] & ~np.isnan(dy[idx]) & ~np.isnan(ly[idx])
        if ok.sum() == 0:
            continue
        capT = int(ok.sum()) + 1
        r = _adf_panel(dy, ly, lagsdy, tr, lags, cons, idx, crit, sel_cons=True)
        pi, f, rows = r
        adfl.append(pi)
        tis.append(f[4])
        # e: D.y on lagged differences (+trend, cons); v: L.y on the same
        Xd = [lagsdy[l][rows] for l in range(1, pi + 1)] + ([tr[rows]] if tr is not None else []) + \
            ([np.ones(rows.size)] if cons else [])
        Xd = np.column_stack(Xd) if Xd else np.zeros((rows.size, 0))
        if Xd.shape[1]:
            e = dy[rows] - Xd @ np.linalg.lstsq(Xd, dy[rows], rcond=None)[0]
            v = ly[rows] - Xd @ np.linalg.lstsq(Xd, ly[rows], rcond=None)[0]
        else:
            e, v = dy[rows].copy(), ly[rows].copy()
        bb = float(v @ e / (v @ v))
        rss = float(np.sum((e - bb * v) ** 2))
        mrssdf = rss / (capT - pi - 1)
        E[rows] = e / math.sqrt(mrssdf)
        Vv[rows] = v / math.sqrt(mrssdf)
        okr = idx[ok]
        if model == 1:
            w = np.where(P.touse & ~np.isnan(dy), dy, np.nan)
        elif model == 2:
            mu = float(np.nanmean(dy[idx]))
            w = np.full(dy.size, np.nan)
            w[okr] = dy[okr] - mu
        else:
            rr = idx[~np.isnan(dy[idx])]
            X3 = np.column_stack([tr[rr], np.ones(rr.size)])
            res = dy[rr] - X3 @ np.linalg.lstsq(X3, dy[rr], rcond=None)[0]
            w = np.full(dy.size, np.nan)
            w[rr] = res
            w[~P.touse] = np.nan
        hl = hlags if hlags is not None else _haclags(P, w, idx, hsel, kern)
        hacl.append(hl)
        sig = _hacvar(P, w, idx, hl, kern)
        sis.append(math.sqrt(sig / mrssdf))
    ok = ~np.isnan(E) & ~np.isnan(Vv)
    e, v = E[ok], Vv[ok]
    n = e.size
    delta = float(v @ e / (v @ v))
    res = e - delta * v
    rss = float(res @ res)
    se = math.sqrt(rss / (n - 1) / (v @ v)) * math.sqrt((n - 1) / n)
    td = delta / se
    tbar = float(np.mean(tis))
    nt = float(np.sum(tis))
    sbar = float(np.mean(sis))
    sig_e2 = rss / n
    capT = int(P.touse.sum()) / P.N
    mu, sa = _llc_adj(tbar, model, p.opts.has('nointerpolate', 8))
    tds = (td - nt * sbar / sig_e2 * se * mu) / sa
    p_tds = float(sps.norm.cdf(tds))
    p_td = float(sps.norm.cdf(td)) if model == 1 else None
    title = 'Levin-Lin-Chu unit-root test'
    ip.txt(f'\n{title} for ')
    ip.res(P.name + '\n')
    ip.txt('-' * (33 + len(P.name)) + '\n')
    _two(ip, 'Ho: Panels contain unit roots', 'Number of panels', P.N, 63)
    _two(ip, 'Ha: Panels are stationary', 'Number of periods', capT, 63)
    ip.txt('\nAR parameter: ')
    ip.res('Common')
    ip.txt(' ' * (44 - 20) + 'Asymptotics: ')
    ip.res('root(N)/T -> 0\n' if nocons else 'N/T -> 0\n')
    ip.txt('Panel means:  ')
    ip.res('Included\n' if model in (2, 3) else 'Not included\n')
    ip.txt('Time trend:   ')
    s = 'Included' if model == 3 else 'Not included'
    ip.res(s)
    if p.opts.has('demean'):
        ip.txt(' ' * (44 - 14 - len(s)) + 'Cross-sectional means removed')
    ip.txt('\n\n')
    if crit is None:
        ip.txt('ADF regressions: ')
        ip.res(f'{lags}')
        ip.txt(f' lag{"s" if lags != 1 else ""}\n')
    else:
        ip.txt('ADF regressions: ')
        ip.res(f'{np.mean(adfl):4.2f}')
        ip.txt(' lags average (chosen by ')
        ip.res(crit.upper())
        ip.txt(')\n')
    kn = 'Quad. Spectral' if kern == 'quadraticspectral' else kern.capitalize()
    ip.txt('LR variance:     ')
    ip.res(kn)
    ip.txt(' kernel, ')
    if hsel is None:
        hv = hacl[0]
        ip.res(f'{hv:4.2f}' if kern == 'quadraticspectral' else fmt_num(hv, '%9.0g').strip())
        ip.txt(f' lag{"s" if hv != 1 else ""}\n')
    else:
        hm = float(np.mean(hacl))
        ip.res(f'{hm:4.2f}')
        ip.txt(f' lag{"s" if hm != 1 else ""} average ')
        if hsel == 'llc':
            ip.txt('(chosen by ')
            ip.res('LLC')
            ip.txt(')\n')
        else:
            ip.txt('\n' + ' ' * 17 + '(chosen by ')
            ip.res('Newey-West')
            ip.txt(')\n')
    ip.txt('-' * 78 + '\n')
    ip.txt(' ' * 20 + 'Statistic' + ' ' * 6 + 'p-value\n')
    ip.txt('-' * 78 + '\n')
    ip.txt(' Unadjusted t' + ' ' * 7)
    ip.res(f'{td:8.4f}')
    if p_td is not None:
        ip.res(' ' * 8 + f'{p_td:6.4f}')
    ip.txt('\n Adjusted t*' + ' ' * 8)
    ip.res(f'{tds:8.4f}' + ' ' * 8 + f'{p_tds:6.4f}\n')
    ip.txt('-' * 78 + '\n')
    sc = {'N': nt, 'N_t': capT, 'N_g': P.N, 'tds': tds, 'td': td, 'ttilde': tbar, 'sbar': sbar, 'Var_ep': sig_e2,
          'se_delta': se, 'delta': delta, 'mu_adj': mu, 'sig_adj': sa, 'p_tds': p_tds}
    if p_td is not None:
        sc['p_td'] = p_td
    if crit is None:
        sc['adf_lags'] = lags
    else:
        sc['adf_lagm'] = float(np.mean(adfl))
    if hsel is None:
        sc['hac_lags'] = hacl[0]
    else:
        sc['hac_lagm'] = float(np.mean(hacl))
    ip.set_r(scalars=sc, macros={'test': 'llc', 'hac_kernel': kern,
                                 'deterministics': 'trend' if trend else ('noconstant' if nocons else 'constant'),
                                 'demean': 'demean' if p.opts.has('demean') else ''})


def _two(ip, left, key, val, col, fmt='%7.0g'):
    ip.txt(f'{left:<44}{key:<{col - 45}}=')
    ip.res(fmt_num(val, fmt) + '\n')


# ---------------------------------------------------------------- IPS
_MVT = [(6, -1.125, 0.497), (7, -1.178, 0.506), (8, -1.214, 0.506), (9, -1.244, 0.527), (10, -1.274, 0.521),
        (15, -1.349, 0.565), (20, -1.395, 0.592), (25, -1.423, 0.609), (30, -1.439, 0.623), (40, -1.463, 0.639),
        (50, -1.477, 0.656), (100, -1.504, 0.683), (500, -1.526, 0.704), (1000, -1.526, 0.702),
        (np.inf, -1.533, 0.706)]
_MV = [(6, -1.520, 1.745), (7, -1.514, 1.414), (8, -1.501, 1.228), (9, -1.501, 1.132), (10, -1.504, 1.069),
       (15, -1.514, 0.923), (20, -1.522, 0.851), (25, -1.520, 0.809), (30, -1.526, 0.789), (40, -1.523, 0.770),
       (50, -1.527, 0.760), (100, -1.532, 0.735), (500, -1.531, 0.715), (1000, -1.529, 0.707),
       (np.inf, -1.533, 0.706)]


def _ipstab1(t, nointerp):
    row = len(_MV) - 1
    for i in range(14):
        if t <= _MV[i][0]:
            row = i
            break
    if t <= 10 or t > 1000 or nointerp:
        return _MVT[row][1], _MVT[row][2], _MV[row][1], _MV[row][2]
    bot, top = _MV[row - 1][0], _MV[row][0]
    bf = 1 - (t - bot) / (top - bot)
    tf = 1 - (top - t) / (top - bot)
    return tuple(bf * a[row - 1][j] + tf * a[row][j] for a, j in ((_MVT, 1), (_MVT, 2), (_MV, 1), (_MV, 2)))


_IPS_CV = {
    '': {1: [[-3.79, -2.66, -2.54, -2.50, -2.46, -2.44, -2.43, -2.42, -2.42, -2.40, -2.40],
             [-3.45, -2.47, -2.38, -2.33, -2.32, -2.31, -2.29, -2.28, -2.28, -2.28, -2.27],
             [-3.06, -2.32, -2.24, -2.21, -2.19, -2.18, -2.16, -2.16, -2.16, -2.16, -2.15],
             [-2.79, -2.14, -2.10, -2.08, -2.07, -2.05, -2.04, -2.05, -2.04, -2.04, -2.04],
             [-2.61, -2.06, -2.02, -2.00, -1.99, -1.99, -1.98, -1.98, -1.98, -1.97, -1.97],
             [-2.51, -2.01, -1.97, -1.95, -1.94, -1.94, -1.93, -1.93, -1.93, -1.93, -1.92],
             [-2.20, -1.85, -1.83, -1.82, -1.82, -1.82, -1.81, -1.81, -1.81, -1.81, -1.81],
             [-2.00, -1.75, -1.74, -1.73, -1.73, -1.73, -1.73, -1.73, -1.73, -1.73, -1.73]],
         5: [[-2.76, -2.28, -2.21, -2.19, -2.18, -2.16, -2.16, -2.15, -2.16, -2.15, -2.15],
             [-2.57, -2.17, -2.11, -2.09, -2.08, -2.07, -2.07, -2.06, -2.06, -2.06, -2.05],
             [-2.42, -2.06, -2.02, -1.99, -1.99, -1.99, -1.98, -1.98, -1.97, -1.98, -1.97],
             [-2.28, -1.95, -1.92, -1.91, -1.90, -1.90, -1.90, -1.89, -1.89, -1.89, -1.89],
             [-2.18, -1.89, -1.87, -1.86, -1.85, -1.85, -1.85, -1.85, -1.84, -1.84, -1.84],
             [-2.11, -1.85, -1.83, -1.82, -1.82, -1.82, -1.82, -1.81, -1.81, -1.81, -1.81],
             [-1.95, -1.75, -1.74, -1.73, -1.73, -1.73, -1.73, -1.73, -1.73, -1.73, -1.73],
             [-1.84, -1.68, -1.67, -1.67, -1.67, -1.67, -1.67, -1.67, -1.67, -1.67, -1.67]],
         10: [[-2.38, -2.10, -2.06, -2.04, -2.04, -2.02, -2.02, -2.02, -2.02, -2.02, -2.01],
              [-2.27, -2.01, -1.98, -1.96, -1.95, -1.95, -1.95, -1.95, -1.94, -1.95, -1.94],
              [-2.17, -1.93, -1.90, -1.89, -1.88, -1.88, -1.88, -1.88, -1.88, -1.88, -1.88],
              [-2.06, -1.85, -1.83, -1.82, -1.82, -1.82, -1.81, -1.81, -1.81, -1.81, -1.81],
              [-2.00, -1.80, -1.79, -1.78, -1.78, -1.78, -1.78, -1.78, -1.78, -1.77, -1.77],
              [-1.96, -1.77, -1.76, -1.75, -1.75, -1.75, -1.75, -1.75, -1.75, -1.75, -1.75],
              [-1.85, -1.70, -1.69, -1.69, -1.69, -1.69, -1.68, -1.68, -1.68, -1.68, -1.69],
              [-1.77, -1.64, -1.64, -1.64, -1.64, -1.64, -1.64, -1.64, -1.64, -1.64, -1.64]]},
    'trend': {1: [[-8.12, -3.42, -3.21, -3.13, -3.09, -3.05, -3.03, -3.02, -3.00, -3.00, -2.99],
                  [-7.36, -3.20, -3.03, -2.97, -2.94, -2.93, -2.90, -2.88, -2.88, -2.87, -2.86],
                  [-6.44, -3.03, -2.88, -2.84, -2.82, -2.79, -2.78, -2.77, -2.76, -2.75, -2.75],
                  [-5.72, -2.86, -2.74, -2.71, -2.69, -2.68, -2.67, -2.65, -2.66, -2.65, -2.64],
                  [-5.54, -2.75, -2.67, -2.63, -2.62, -2.61, -2.59, -2.60, -2.59, -2.58, -2.58],
                  [-5.16, -2.69, -2.61, -2.58, -2.58, -2.56, -2.55, -2.55, -2.55, -2.54, -2.54],
                  [-4.50, -2.53, -2.48, -2.46, -2.45, -2.45, -2.44, -2.44, -2.44, -2.44, -2.43],
                  [-4.00, -2.42, -2.39, -2.38, -2.37, -2.37, -2.36, -2.36, -2.36, -2.36, -2.36]],
              5: [[-4.66, -2.98, -2.87, -2.82, -2.80, -2.79, -2.77, -2.76, -2.75, -2.75, -2.75],
                  [-4.38, -2.85, -2.76, -2.72, -2.70, -2.69, -2.68, -2.67, -2.67, -2.66, -2.66],
                  [-4.11, -2.74, -2.66, -2.63, -2.62, -2.60, -2.60, -2.59, -2.59, -2.58, -2.58],
                  [-3.88, -2.63, -2.57, -2.55, -2.53, -2.53, -2.52, -2.52, -2.52, -2.51, -2.51],
                  [-3.73, -2.56, -2.52, -2.49, -2.48, -2.48, -2.48, -2.47, -2.47, -2.46, -2.46],
                  [-3.62, -2.52, -2.48, -2.46, -2.45, -2.45, -2.44, -2.44, -2.44, -2.44, -2.43],
                  [-3.35, -2.42, -2.38, -2.38, -2.37, -2.37, -2.36, -2.36, -2.36, -2.36, -2.36],
                  [-3.13, -2.34, -2.32, -2.32, -2.31, -2.31, -2.31, -2.31, -2.31, -2.31, -2.31]],
              10: [[-3.73, -2.77, -2.70, -2.67, -2.65, -2.64, -2.63, -2.62, -2.63, -2.62, -2.62],
                   [-3.60, -2.68, -2.62, -2.59, -2.58, -2.57, -2.57, -2.56, -2.56, -2.55, -2.55],
                   [-3.45, -2.59, -2.54, -2.52, -2.51, -2.51, -2.50, -2.50, -2.50, -2.49, -2.49],
                   [-3.33, -2.52, -2.47, -2.46, -2.45, -2.45, -2.44, -2.44, -2.44, -2.44, -2.44],
                   [-3.26, -2.47, -2.44, -2.42, -2.41, -2.41, -2.41, -2.40, -2.40, -2.40, -2.40],
                   [-3.18, -2.44, -2.40, -2.39, -2.39, -2.38, -2.38, -2.38, -2.38, -2.38, -2.38],
                   [-3.02, -2.36, -2.33, -2.33, -2.33, -2.32, -2.32, -2.32, -2.32, -2.32, -2.32],
                   [-2.90, -2.30, -2.29, -2.28, -2.28, -2.28, -2.28, -2.28, -2.28, -2.28, -2.28]]}}


def _ips_cv(N, T, trend):
    row = 7
    for k, lim in enumerate((5, 7, 10, 15, 20, 25, 50)):
        if N <= lim:
            row = k
            break
    col = 10
    for k, lim in enumerate((5, 10, 15, 20, 25, 30, 40, 50, 60, 70)):
        if T <= lim:
            col = k
            break
    tab = _IPS_CV['trend' if trend else '']
    return tab[1][row][col], tab[5][row][col], tab[10][row][col]


_ADF_MV = {
    'mean': [[-1.504, -1.514, -1.522, -1.520, -1.526, -1.523, -1.527, -1.519, -1.524, -1.532],
             [-1.488, -1.503, -1.516, -1.514, -1.519, -1.520, -1.524, -1.519, -1.522, -1.530],
             [-1.319, -1.387, -1.428, -1.443, -1.460, -1.476, -1.493, -1.490, -1.498, -1.514],
             [-1.306, -1.366, -1.413, -1.433, -1.453, -1.471, -1.489, -1.486, -1.495, -1.512],
             [-1.171, -1.260, -1.329, -1.363, -1.394, -1.428, -1.454, -1.458, -1.470, -1.495],
             [np.nan, np.nan, -1.313, -1.351, -1.384, -1.421, -1.451, -1.454, -1.467, -1.494],
             [np.nan, np.nan, np.nan, -1.289, -1.331, -1.380, -1.418, -1.427, -1.444, -1.476],
             [np.nan, np.nan, np.nan, -1.273, -1.319, -1.371, -1.411, -1.423, -1.441, -1.474],
             [np.nan, np.nan, np.nan, -1.212, -1.266, -1.329, -1.377, -1.393, -1.415, -1.456]],
    'var': [[1.069, 0.923, 0.851, 0.809, 0.789, 0.770, 0.760, 0.749, 0.736, 0.735],
            [1.255, 1.011, 0.915, 0.861, 0.831, 0.803, 0.781, 0.770, 0.753, 0.745],
            [1.421, 1.078, 0.969, 0.905, 0.865, 0.830, 0.798, 0.789, 0.766, 0.754],
            [1.759, 1.181, 1.037, 0.952, 0.907, 0.858, 0.819, 0.802, 0.782, 0.761],
            [2.080, 1.279, 1.097, 1.005, 0.946, 0.886, 0.842, 0.819, 0.801, 0.771],
            [np.nan, np.nan, 1.171, 1.055, 0.980, 0.912, 0.863, 0.839, 0.814, 0.781],
            [np.nan, np.nan, np.nan, 1.114, 1.023, 0.942, 0.886, 0.858, 0.834, 0.795],
            [np.nan, np.nan, np.nan, 1.164, 1.062, 0.968, 0.910, 0.875, 0.851, 0.806],
            [np.nan, np.nan, np.nan, 1.217, 1.105, 0.996, 0.929, 0.896, 0.871, 0.818]],
    'meantrend': [[-2.166, -2.167, -2.168, -2.167, -2.172, -2.173, -2.176, -2.174, -2.174, -2.177],
                  [-2.173, -2.169, -2.172, -2.172, -2.173, -2.177, -2.180, -2.178, -2.176, -2.179],
                  [-1.914, -1.999, -2.047, -2.074, -2.095, -2.120, -2.137, -2.143, -2.146, -2.158],
                  [-1.922, -1.977, -2.032, -2.065, -2.091, -2.117, -2.137, -2.142, -2.146, -2.158],
                  [-1.750, -1.823, -1.911, -1.968, -2.009, -2.057, -2.091, -2.103, -2.114, -2.135],
                  [np.nan, np.nan, -1.888, -1.955, -1.998, -2.051, -2.087, -2.101, -2.111, -2.135],
                  [np.nan, np.nan, np.nan, -1.868, -1.923, -1.995, -2.042, -2.065, -2.081, -2.113],
                  [np.nan, np.nan, np.nan, -1.851, -1.912, -1.986, -2.036, -2.063, -2.079, -2.112],
                  [np.nan, np.nan, np.nan, -1.761, -1.835, -1.925, -1.987, -2.204, -2.046, -2.088]],
    'vartrend': [[1.132, 0.869, 0.763, 0.713, 0.690, 0.655, 0.633, 0.621, 0.610, 0.597],
                 [1.453, 0.975, 0.845, 0.769, 0.734, 0.687, 0.654, 0.641, 0.627, 0.605],
                 [1.627, 1.036, 0.882, 0.796, 0.756, 0.702, 0.661, 0.653, 0.634, 0.613],
                 [2.482, 1.214, 0.983, 0.861, 0.808, 0.735, 0.688, 0.674, 0.650, 0.625],
                 [3.947, 1.332, 1.052, 0.913, 0.845, 0.759, 0.705, 0.685, 0.662, 0.629],
                 [np.nan, np.nan, 1.165, 0.991, 0.899, 0.792, 0.730, 0.705, 0.673, 0.638],
                 [np.nan, np.nan, np.nan, 1.055, 0.945, 0.828, 0.753, 0.725, 0.689, 0.650],
                 [np.nan, np.nan, np.nan, 1.145, 1.009, 0.872, 0.786, 0.747, 0.713, 0.661],
                 [np.nan, np.nan, np.nan, 1.208, 1.063, 0.902, 0.808, 0.766, 0.728, 0.670]]}


def _adftmeanvar(lag, capt, trend, nointerp):
    if lag < 0 or lag > 8:
        return np.nan, np.nan
    mt = _ADF_MV['meantrend' if trend else 'mean']
    vt = _ADF_MV['vartrend' if trend else 'var']
    heads = [10, 15, 20, 25, 30, 40, 50, 60, 70]
    col = 9
    for k, lim in enumerate(heads):
        if capt <= lim:
            col = k
            break
    heads = heads + [100]
    row = lag
    if nointerp or col == 0 or capt >= 100 or capt in heads:
        return mt[row][col], vt[row][col]
    bot, top = heads[col - 1], heads[col]
    bf = 1 - (capt - bot) / (top - bot)
    tf = 1 - (top - capt) / (top - bot)
    return bf * mt[row][col - 1] + tf * mt[row][col], bf * vt[row][col - 1] + tf * vt[row][col]


def _ips(ip, P, p):
    trend = p.opts.has('trend', 1)
    lags, crit = _parse_lags(p.opts.get('lags', 1))
    if P.gaps():
        raise StataError('Im-Pesaran-Shin test cannot have gaps in data', 498)
    unbal = not P.balanced()
    dv = P.depvar(p.opts.has('demean'))
    tr = P.trend() if trend else None
    dy = dv - P.lag(dv, 1)
    ly = P.lag(dv, 1)
    Ti = np.array([int(np.sum(P.touse & (P.pid == i))) for i in P.ids])
    minT = int(Ti.min())
    capT = float(Ti.mean())
    nointerp = p.opts.has('nointerpolate', 8)
    obs = 0
    if lags is None:
        acc = np.zeros(6)
        for i, Tn in zip(P.ids, Ti):
            idx = np.flatnonzero((P.pid == i) & P.touse)
            ok = idx[~np.isnan(dy[idx]) & ~np.isnan(ly[idx]) & (True if tr is None else ~np.isnan(tr[idx]))]
            X = np.column_stack([ly[ok]] + ([tr[ok]] if tr is not None else []) + [np.ones(ok.size)])
            b, se, e, rss, n, k, ll = _ols(dy[ok], X)
            obs += n
            rmse = math.sqrt(rss / (n - k))
            sd = float(np.std(dy[idx][~np.isnan(dy[idx])], ddof=1))
            mt, vt, m, v = _ipstab1(Tn, nointerp)
            acc += [b[0] / se[0], b[0] / (se[0] / rmse * sd), m, v, mt, vt]
        ti, tti, mui, vari, muti, varti = acc / P.N
        zt = math.sqrt(P.N) * (ti - mui) / math.sqrt(vari)
        ztt = math.sqrt(P.N) * (tti - muti) / math.sqrt(varti)
        if not unbal:
            cv1, cv5, cv10 = _ips_cv(P.N, capT, trend)
            pz = float(sps.norm.cdf(ztt)) if ((minT > 5 and not trend) or (minT > 6 and trend)) else np.nan
        else:
            pz = float(sps.norm.cdf(ztt)) if minT > 9 else np.nan
    else:
        maxl = lags
        lagsdy = {l: P.lag(dy, l) for l in range(1, maxl + 1)}
        dyt = np.where(P.touse, dy, np.nan)
        lyt = ly
        ti = mui = vari = 0.0
        pls = []
        for i, Tn in zip(P.ids, Ti):
            idx = np.flatnonzero((P.pid == i) & P.touse)
            r = _adf_panel(dyt, lyt, lagsdy, tr, lags, True, idx, crit)
            pi, f, rows = r
            b, se = f[0], f[1]
            ti += b[0] / se[0]
            obs += f[4]
            capt = f[4] if p.opts.has('usedfkadj', 9) else Tn
            m, v = _adftmeanvar(pi, capt, trend, nointerp)
            if m != m:
                raise StataError(f'mean and variance of t-statistic not available for T = {capt} with {pi} lags',
                                 498)
            mui += m
            vari += v
            pls.append(pi)
        ti, mui, vari = ti / P.N, mui / P.N, vari / P.N
        wtbar = math.sqrt(P.N) * (ti - mui) / math.sqrt(vari)
    # ---- display
    ip.txt('\nIm-Pesaran-Shin unit-root test for ')
    ip.res(P.name + '\n')
    ip.txt('-' * (35 + len(P.name)) + '\n')
    ph = _ph(P.N, capT, unbal)
    _two(ip, 'Ho: All panels contain unit roots', ph[0][0], P.N, ph[0][2], '%7.0g')
    _two(ip, 'Ha: Some panels are stationary', ph[1][0], capT, ph[1][2], ph[1][3])
    ip.txt('\nAR parameter: ')
    ip.res('Panel-specific')
    ip.txt(' ' * (44 - 28) + 'Asymptotics: ')
    ip.res('T,N -> Infinity\n')
    ip.txt('Panel means:  ')
    ip.res('Included')
    ip.res(' ' * (62 - 22) + 'sequentially\n')
    ip.txt('Time trend:   ')
    s = 'Included' if trend else 'Not included'
    ip.res(s)
    if p.opts.has('demean'):
        ip.txt(' ' * (44 - 14 - len(s)) + 'Cross-sectional means removed')
    ip.txt('\n\nADF regressions: ')
    if lags is None:
        ip.res('No lags included\n')
    elif crit is None:
        ip.res(f'{lags}')
        ip.txt(f' lag{"s" if lags != 1 else ""}\n')
    else:
        ip.res(f'{np.mean(pls):4.2f}')
        ip.txt(' lags average (chosen by ')
        ip.res(crit.upper())
        ip.txt(')\n')
    ip.txt('-' * 78 + '\n')
    sc = {'N_g': P.N, 'N_t': capT, 'N': obs}
    if lags is None:
        ip.txt(' ' * 46 + 'Fixed-N exact critical values\n')
        ip.txt(' ' * 20 + 'Statistic' + ' ' * 6 + 'p-value' + ' ' * 9 + '1%' + ' ' * 6 + '5%' + ' ' * 6 + '10%\n')
        ip.txt('-' * 78 + '\n')
        ip.txt(' t-bar' + ' ' * 14)
        ip.res(f'{ti:8.4f}')
        if unbal:
            ip.res(' ' * 26 + '(Not available)\n')
        else:
            ip.res(' ' * 21 + f'{cv1:6.3f}  {cv5:6.3f}  {cv10:6.3f}\n')
        ip.txt(' t-tilde-bar' + ' ' * 8)
        ip.res(f'{tti:8.4f}\n')
        ip.txt(' Z-t-tilde-bar' + ' ' * 6)
        ip.res(f'{ztt:8.4f}')
        ip.res(' ' * 8 + f'{pz:6.4f}\n' if pz == pz else ' ' * 10 + '*\n')
        sc.update({'tbar': ti, 'ttildebar': tti, 'zt': zt, 'zttildebar': ztt, 'p_zttildebar': pz})
        if not unbal:
            sc.update({'cv_1': cv1, 'cv_5': cv5, 'cv_10': cv10})
    else:
        ip.txt(' ' * 20 + 'Statistic' + ' ' * 6 + 'p-value\n')
        ip.txt('-' * 78 + '\n')
        ip.txt(' W-t-bar' + ' ' * 12)
        ip.res(f'{wtbar:8.4f}' + ' ' * 8 + f'{sps.norm.cdf(wtbar):6.4f}\n')
        sc.update({'wtbar': wtbar, 'p_wtbar': float(sps.norm.cdf(wtbar))})
        if crit:
            sc['lagm'] = float(np.mean(pls))
        else:
            sc['lags'] = lags
    ip.txt('-' * 78 + '\n')
    if lags is None and pz != pz:
        if unbal:
            ip.res('*')
            ip.txt(' Normality of Z-t-tilde-bar requires at least 10 observations\n  per panel with unbalanced data.\n')
        elif trend:
            ip.res('*')
            ip.txt(' Normality of Z-t-tilde-bar requires at least 7 observations\n  per panel with balanced data and a '
                   'time trend.\n')
        else:
            ip.res('*')
            ip.txt(' Normality of Z-t-tilde-bar requires at least 6 observations\n  per panel with balanced data and no '
                   'time trend.\n')
    ip.set_r(scalars=sc, macros={'test': 'ips', 'deterministics': 'trend' if trend else 'constant',
                                 'demean': 'demean' if p.opts.has('demean') else ''})


# ---------------------------------------------------------------- HT
def _ht(ip, P, p):
    trend, nocons = p.opts.has('trend', 1), p.opts.has('noconstant', 6)
    if trend and nocons:
        raise StataError('cannot specify both noconstant and trend', 198)
    model = 3 if trend else (1 if nocons else 2)
    if not P.balanced():
        raise StataError('Harris-Tzavalis test requires strongly balanced data', 498)
    if P.gaps():
        raise StataError('Harris-Tzavalis test cannot have gaps in data', 498)
    Ti = [int(np.sum(P.touse & (P.pid == i))) for i in P.ids]
    capT = min(Ti)
    capN = int(P.touse.sum()) / capT
    dv = P.depvar(p.opts.has('demean'))
    ly = P.lag(dv, 1)
    ok = P.touse & ~np.isnan(ly)
    if model == 1:
        rho = float(np.sum(dv[ok] * ly[ok]) / np.sum(ly[ok] ** 2))
    else:
        Y, L = [], []
        for i in P.ids:
            s = np.flatnonzero(ok & (P.pid == i))
            y_, l_ = dv[s], ly[s]
            if model == 2:
                Y.append(y_ - y_.mean())
                L.append(l_ - l_.mean())
            else:
                tt = np.flatnonzero(P.pid == i)
                tr = (np.arange(tt.size) + 1.0)[np.isin(tt, s)]
                X = np.column_stack([tr, np.ones(s.size)])
                Y.append(y_ - X @ np.linalg.lstsq(X, y_, rcond=None)[0])
                L.append(l_ - X @ np.linalg.lstsq(X, l_, rcond=None)[0])
        Y, L = np.concatenate(Y), np.concatenate(L)
        rho = float(Y @ L / (L @ L))
    T = capT - 1 if p.opts.has('altt') else capT
    if model == 1:
        mean, var = 1, (1 / capN) * (2 / (T * (T - 1)))
    elif model == 2:
        mean = 1 - 3 / (T + 1)
        var = (1 / capN) * (3 * (17 * T ** 2 - 20 * T + 17)) / (5 * (T - 1) * (T + 1) ** 3)
    else:
        mean = 1 - 15 / (2 * (T + 2))
        var = (1 / capN) * (15 * (193 * T ** 2 - 728 * T + 1147)) / (112 * (T + 2) ** 3 * (T - 2))
    z = (rho - mean) / math.sqrt(var)
    pv = float(sps.norm.cdf(z))
    ip.txt('\nHarris-Tzavalis unit-root test for ')
    ip.res(P.name + '\n')
    ip.txt('-' * (35 + len(P.name)) + '\n')
    _two(ip, 'Ho: Panels contain unit roots', 'Number of panels', capN, 63)
    _two(ip, 'Ha: Panels are stationary', 'Number of periods', capT, 63)
    ip.txt('\nAR parameter: ')
    ip.res('Common')
    ip.txt(' ' * (44 - 20) + 'Asymptotics: ')
    ip.res('N -> Infinity\n')
    ip.txt('Panel means:  ')
    s = 'Not included' if model == 1 else 'Included'
    ip.res(s + ' ' * (57 - 14 - len(s)) + 'T Fixed\n')
    ip.txt('Time trend:   ')
    s = 'Included' if model == 3 else 'Not included'
    ip.res(s)
    if p.opts.has('demean'):
        ip.txt(' ' * (44 - 14 - len(s)) + 'Cross-sectional means removed')
    ip.txt('\n')
    if p.opts.has('altt'):
        ip.txt('Small-sample adjustment to T applied\n')
    ip.txt('-' * 78 + '\n')
    ip.txt(' ' * 20 + 'Statistic' + ' ' * 9 + 'z' + ' ' * 9 + 'p-value\n')
    ip.txt('-' * 78 + '\n')
    ip.txt(' rho' + ' ' * 16)
    ip.res(f'{rho:8.4f}' + ' ' * 6 + f'{z:8.4f}' + ' ' * 7 + f'{pv:6.4f}\n')
    ip.txt('-' * 78 + '\n')
    ip.set_r(scalars={'N': capN * capT, 'N_g': capN, 'N_t': capT, 'rho': rho, 'mean_rho': mean, 'Var_rho': var,
                      'z': z, 'p': pv},
             macros={'test': 'ht', 'deterministics': 'trend' if trend else ('noconstant' if nocons else 'constant')})


# ---------------------------------------------------------------- Breitung
def _breitung(ip, P, p):
    trend, nocons = p.opts.has('trend', 1), p.opts.has('noconstant', 6)
    if p.opts.has('robust', 1):
        raise StataError('the robust option of the Breitung test is not available in Stata Web', 198)
    lg = p.opts.get('lags', 1)
    lags = int(lg) if lg not in (None, '') else 0
    if not P.balanced():
        raise StataError('Breitung test requires strongly balanced data', 498)
    if P.gaps():
        raise StataError('Breitung test cannot have gaps in data', 498)
    dv = P.depvar(p.opts.has('demean'))
    Ti = [int(np.sum(P.touse & (P.pid == i))) for i in P.ids]
    capT = int(P.touse.sum()) / P.N
    s10 = s11 = 0.0
    for i in P.ids:
        s = np.flatnonzero((P.pid == i) & P.touse)
        y = dv[s]
        T = y.size
        ly = np.r_[np.nan, y[:-1]]
        dy = np.r_[np.nan, np.diff(y)]
        ldy = {l: np.r_[np.full(l, np.nan), dy[:-l]] for l in range(1, lags + 1)}
        if not trend:
            if not nocons:
                ly = ly - y[lags]
            if lags > 0:
                ok = ~np.isnan(dy)
                for l in range(1, lags + 1):
                    ok &= ~np.isnan(ldy[l])
                X = np.column_stack([ldy[l][ok] for l in range(1, lags + 1)])
                d2 = np.full(T, np.nan)
                l2 = np.full(T, np.nan)
                d2[ok] = dy[ok] - X @ np.linalg.lstsq(X, dy[ok], rcond=None)[0]
                okl = ok & ~np.isnan(ly)
                Xl = np.column_stack([ldy[l][okl] for l in range(1, lags + 1)])
                l2[okl] = ly[okl] - Xl @ np.linalg.lstsq(Xl, ly[okl], rcond=None)[0]
                dy, ly = d2, l2
            sig2 = float(np.nansum(dy ** 2)) / (capT - lags - 1 - 1)
            last = T - 1
        else:
            if lags > 0:
                ok = ~np.isnan(dy)
                for l in range(1, lags + 1):
                    ok &= ~np.isnan(ldy[l])
                X = np.column_stack([ldy[l][ok] for l in range(1, lags + 1)] + [np.ones(ok.sum())])
                beta = np.linalg.lstsq(X, dy[ok], rcond=None)[0]
                du = dy.copy()
                ut = y.copy()
                for j in range(1, lags + 1):
                    du = du - beta[j - 1] * ldy[j]
                    lyj = np.r_[np.full(j, np.nan), y[:-j]]
                    ut = ut - beta[j - 1] * lyj
                u = np.r_[np.nan, ut[:-1]]
            else:
                du, u = dy.copy(), ly.copy()
            meandu = float(np.nansum(du)) / (capT - lags - 1)
            sig2 = float(np.nansum((du - meandu) * du)) / (capT - lags - 1 - 1)
            n_ = np.arange(1, T + 1)
            ly = u - u[lags + 1] - (n_ - lags - 2) * meandu
            cs = np.nancumsum(du)
            fwd = (cs[-1] - cs) / np.maximum(T - n_, 1e-300)
            fwd[T - n_ == 0] = np.nan
            dy = (du - fwd) * np.sqrt((capT - n_) / (capT - n_ + 1))
            last = T - 2
        prod10 = ly * dy
        prod11 = ly * ly
        s10 += float(np.nansum(prod10[:last + 1])) / sig2
        s11 += float(np.nansum(prod11[:last + 1])) / sig2
    lam = s10 / math.sqrt(s11)
    pv = float(sps.norm.cdf(lam))
    ip.txt('\nBreitung unit-root test for ')
    ip.res(P.name + '\n')
    ip.txt('-' * (30 + len(P.name)) + '\n')
    _two(ip, 'Ho: Panels contain unit roots', 'Number of panels', P.N, 63)
    _two(ip, 'Ha: Panels are stationary', 'Number of periods', capT, 63)
    ip.txt('\nAR parameter: ')
    ip.res('Common')
    ip.txt(' ' * (44 - 20) + 'Asymptotics:  ')
    ip.res('T,N -> Infinity\n')
    ip.txt('Panel means:  ')
    s = 'Not included' if nocons else 'Included'
    ip.res(s + ' ' * (62 - 14 - len(s)) + 'sequentially\n')
    ip.txt('Time trend:   ')
    s = 'Included' if trend else 'Not included'
    ip.res(s)
    ip.txt(' ' * (44 - 14 - len(s)) + 'Prewhitening: ')
    if lags > 0:
        ip.res(f'{lags}')
        ip.txt(f' lag{"s" if lags > 1 else ""}\n')
    else:
        ip.res('Not performed\n')
    if p.opts.has('demean'):
        ip.txt(' ' * 44 + 'Cross-sectional means removed\n')
    ip.txt('-' * 78 + '\n')
    ip.txt(' ' * 20 + 'Statistic' + ' ' * 6 + 'p-value\n')
    ip.txt('-' * 78 + '\n')
    ip.txt(' lambda' + ' ' * 13)
    ip.res(f'{lam:8.4f}' + ' ' * 8 + f'{pv:6.4f}\n')
    ip.txt('-' * 78 + '\n')
    ip.set_r(scalars={'N': P.N * capT, 'N_g': P.N, 'N_t': capT, 'lambda': lam, 'p': pv, 'lags': lags},
             macros={'test': 'breitung', 'deterministics': 'trend' if trend else ('noconstant' if nocons else
                                                                                'constant')})


# ---------------------------------------------------------------- Fisher
def _fisher(ip, P, p):
    dfu, ppe = p.opts.has('dfuller', 3), p.opts.has('pperron', 2)
    if dfu + ppe == 0:
        raise StataError('must specify one of dfuller or pperron', 198)
    if dfu + ppe > 1:
        raise StataError('can specify only one of dfuller or pperron', 198)
    testcmd = 'dfuller' if dfu else 'pperron'
    if ppe and p.opts.has('drift', 2):
        raise StataError('cannot specify drift with pperron', 198)
    if p.opts.has('noconstant', 3):
        raise StataError('cannot specify noconstant because p-values are not available in this case', 498)
    lg = p.opts.get('lags', 1)
    if lg in (None, ''):
        raise StataError('must specify lags()', 198)
    opts = ' '.join(o for o, on in (('trend', p.opts.has('trend', 2)), ('drift', p.opts.has('drift', 2))) if on) + \
        f' lags({lg})'
    unbal = not P.balanced()
    dv = P.depvar(p.opts.has('demean'))
    d = P.d
    tmp = '__xtur_dv'
    d.add_var(tmp, dv, 'double')
    pid_tmp = '__xtur_id'
    d.add_var(pid_tmp, np.where(P.touse, P.pid, np.nan), 'double')
    ps, pv_by_id, ids_ok = [], [], []
    q0 = ip.quiet
    ip.quiet += 1
    keep_ts = dict(d.ts)
    try:
        for i in P.ids:
            try:
                ip.dispatch(f'{testcmd} {tmp} if {pid_tmp} == {float(i)!r}, {opts}')
                ps.append(float(ip.r.scalars.get('p')))
                pv_by_id.append(max(ps[-1], np.finfo(float).eps))
                ids_ok.append(i)
            except StataError:
                ip.quiet = q0
                ip.txt(f'could not compute test for panel {i:g}\n')
                ip.quiet += 1
    finally:
        ip.quiet = q0
        d.ts = keep_ts
        d.drop_vars([tmp, pid_tmp])
    pv = np.maximum(np.array(ps), np.finfo(float).eps)
    n = pv.size
    if n == 0:
        raise StataError('could not compute unit-root test for any panel', 2000)
    with np.errstate(divide='ignore', invalid='ignore'):
        Pst = -2 * float(np.sum(np.log(pv)))
        Pdf = 2 * n
        zz = sps.norm.ppf(pv)
        zz = zz[np.isfinite(zz)]       # invnormal(1) is missing in Stata and dropped by summarize
        Z = float(np.sum(zz)) / math.sqrt(zz.size)
        lgt = np.log(pv / (1 - pv))
        lgt = lgt[np.isfinite(lgt)]
        nl = lgt.size
        k = (3 * (5 * nl + 3)) / (math.pi ** 2 * nl * (5 * nl + 2))
        L = math.sqrt(k) * float(np.sum(lgt))
        Ldf = 5 * nl + 4
    Pm = float(np.sum(-2 * np.log(pv) - 2)) / (2 * math.sqrt(n))
    capT = int(P.touse.sum()) / P.N
    res = {'P': Pst, 'p_P': float(sps.chi2.sf(Pst, Pdf)), 'df_P': Pdf, 'Z': Z, 'p_Z': float(sps.norm.cdf(Z)),
           'L': L, 'p_L': float(sps.t.cdf(L, Ldf)), 'df_L': Ldf, 'Pm': Pm, 'p_Pm': float(1 - sps.norm.cdf(Pm)),
           'N_g': P.N, 'N_t': capT, 'N': round(capT * P.N)}
    ip.txt('\nFisher-type unit-root test for ')
    ip.res(P.name + '\n')
    l2 = 'Based on augmented Dickey-Fuller tests' if dfu else 'Based on Phillips-Perron tests'
    ip.txt(l2 + '\n')
    ip.txt('-' * max(len('Fisher-type unit-root test for ' + P.name), len(l2)) + '\n')
    ph = _ph(P.N, capT, unbal)
    _two(ip, 'Ho: All panels contain unit roots', ph[0][0], P.N, ph[0][2], '%7.0g')
    _two(ip, 'Ha: At least one panel is stationary', ph[1][0], capT, ph[1][2], ph[1][3])
    col = 15 if dfu else 18

    def lab(t, v, extra=None):
        ip.txt(t + ' ' * (col - 1 - len(t)))
        ip.res(v)
        if extra:
            ip.txt(' ' * (44 - (col - 1) - len(v)) + extra[0])
            if len(extra) > 1:
                ip.res(extra[1])
                if len(extra) > 2:
                    ip.txt(extra[2])
        ip.txt('\n')
    ip.txt('\n')
    lab('AR parameter:', 'Panel-specific', ('Asymptotics: ', 'T -> Infinity'))
    lab('Panel means:', 'Included')
    lab('Time trend:', 'Included' if p.opts.has('trend', 2) else 'Not included',
        ('Cross-sectional means removed',) if p.opts.has('demean') else None)
    s = '' if str(lg).strip() == '1' else 's'
    if dfu:
        lab('Drift term:', 'Included' if p.opts.has('drift', 2) else 'Not included',
            ('ADF regressions: ', str(lg).strip(), f' lag{s}'))
    else:
        ip.txt('Newey-West lags:' + ' ' * (col - 1 - 16))
        ip.res(str(lg).strip())
        ip.txt(f' lag{s}\n')
    ip.txt('-' * 78 + '\n')
    ip.txt(' ' * 34 + 'Statistic' + ' ' * 6 + 'p-value\n')
    ip.txt('-' * 78 + '\n')
    for t1, t2, s1, s2 in ((f' Inverse chi-squared({Pdf})', 'P', 'P', 'p_P'), (' Inverse normal', 'Z', 'Z', 'p_Z'),
                           (f' Inverse logit t({Ldf})', 'L*', 'L', 'p_L'),
                           (' Modified inv. chi-squared', 'Pm', 'Pm', 'p_Pm')):
        ip.txt(f'{t1:<27}{t2:<7}')
        ip.res(f'{res[s1]:9.4f}' + ' ' * 7 + f'{res[s2]:6.4f}\n')
    ip.txt('-' * 78 + '\n')
    ip.txt(' P statistic requires number of panels to be finite.\n'
           ' Other statistics are suitable for finite or infinite number of panels.\n')
    ip.txt('-' * 78 + '\n')
    gen = p.opts.get('generate', 3)
    if gen:
        if d.has(gen):
            d.drop_vars([gen])
        gv = np.full(d.N, np.nan)
        for i, pval in zip(ids_ok, pv_by_id):
            first = np.flatnonzero(P.touse & (P.pid == i))[0]
            gv[first] = pval
        d.add_var(gen, gv, 'double')
    ip.set_r(scalars=res, macros={'test': 'fisher', 'urtest': testcmd, 'options': opts.strip()})


# ---------------------------------------------------------------- dispatcher
@command('xtunitroot')
def cmd_xtunitroot(ip, cmd, rest, by):
    sub, _, tail = rest.strip().partition(' ')
    test = sub.lower()
    p = parse_command(tail)
    toks = split_top(p.body)
    if len(toks) != 1:
        raise StataError('varname required', 100)
    P = _Panels(ip, p, toks[0])
    if test == 'llc':
        return _llc(ip, P, p)
    if test == 'ips':
        return _ips(ip, P, p)
    if test == 'ht':
        return _ht(ip, P, p)
    if test in ('breitung', 'bre'):
        return _breitung(ip, P, p)
    if test in ('fisher', 'fis'):
        return _fisher(ip, P, p)
    if test == 'hadri':
        return _hadri(ip, P, p)
    raise StataError(f'{sub} not a valid test; choose llc, ips, ht, breitung, fisher, or hadri', 198)


def _hadri(ip, P, p):
    trend, robust = p.opts.has('trend', 1), p.opts.has('robust', 1)
    ks = p.opts.get('kernel', 3)
    kern = hl = None
    if ks:
        kern, hl, hsel = _kernel_opt(ks)
        if hsel is not None:
            raise StataError('kernel() lag length invalid', 198)
        if robust:
            raise StataError('cannot specify both kernel() and robust', 198)
    if not P.balanced():
        raise StataError('Hadri LM test requires strongly balanced data', 498)
    if P.gaps():
        raise StataError('Hadri LM test cannot have gaps in data', 498)
    dv = P.depvar(p.opts.has('demean'))
    capT = int(P.touse.sum()) / P.N
    nodfk = p.opts.has('nodfkadjust')
    div = capT if nodfk else capT - 1 - (1 if trend else 0)
    E = []
    for i in P.ids:
        s_ = np.flatnonzero((P.pid == i) & P.touse)
        y = dv[s_]
        X = np.column_stack(([np.arange(1, y.size + 1.0)] if trend else []) + [np.ones(y.size)])
        E.append(y - X @ np.linalg.lstsq(X, y, rcond=None)[0])
    S2 = [float(np.sum(np.cumsum(e) ** 2)) for e in E]
    if not robust and not kern:
        sig = sum(float(e @ e) for e in E) / (P.N * div)
        LM = sum(S2) / (P.N * capT ** 2) / sig
    elif robust:
        LM = sum(s2 / (float(e @ e) / div) for s2, e in zip(S2, E)) / (P.N * capT ** 2)
    else:
        lr = []
        L = int(hl)
        for e in E:
            v = float(e @ e) / capT
            for l in range(1, L + 1):
                z = l / (hl + 1)
                if kern == 'bartlett':
                    kw = 1 - z
                elif kern == 'parzen':
                    kw = 1 - 6 * z ** 2 + 6 * z ** 3 if z <= .5 else 2 * (1 - z) ** 3
                else:
                    th = 6 * math.pi * z / 5
                    kw = 1 if z == 1 else 3 * (math.sin(th) / th - math.cos(th)) / th ** 2
                v += 2 * kw * float(e[l:] @ e[:-l]) / capT
            lr.append(v)
        LM = sum(S2) / float(np.mean(lr)) / (P.N * capT ** 2)
    mu, var = (1 / 15, 11 / 6300) if trend else (1 / 6, 1 / 45)
    Z = math.sqrt(P.N) * (LM - mu) / math.sqrt(var)
    pv = float(sps.norm.cdf(-Z))
    ip.txt('\nHadri LM test for ')
    ip.res(P.name + '\n')
    ip.txt('-' * (20 + len(P.name)) + '\n')
    _two(ip, 'Ho: All panels are stationary', 'Number of panels', P.N, 63)
    _two(ip, 'Ha: Some panels contain unit roots', 'Number of periods', capT, 63)
    ip.txt('\nTime trend:         ')
    s = 'Included' if trend else 'Not included'
    ip.res(s)
    ip.txt(' ' * (44 - 20 - len(s)) + 'Asymptotics: ')
    ip.res('T, N -> Infinity\n')
    ip.txt('Heteroskedasticity: ')
    s = 'Robust' if (robust or kern) else 'Not robust'
    ip.res(s + ' ' * (62 - 20 - len(s)) + 'sequentially\n')
    dm = p.opts.has('demean')
    if kern:
        kn = 'Quad. Spectral' if kern == 'quadraticspectral' else kern.capitalize()
        hv = f'{hl:4.2f}' if kern == 'quadraticspectral' else fmt_num(hl, '%9.0g').strip()
        sfx = 's' if hl != 1 else ''
        ip.txt('LR variance:        ')
        ip.res(kn)
        ip.txt(' kernel, ')
        used = 20 + len(kn) + 9
        if kern == 'quadraticspectral' or hl > 9:
            ip.txt((' ' * max(44 - used, 1) + 'Cross-sectional means removed') if dm else '')
            ip.txt('\n' + ' ' * 20)
            ip.res(hv)
            ip.txt(f' lag{sfx}\n')
        else:
            ip.res(hv)
            ip.txt(f' lag{sfx}')
            used += len(hv) + 4 + len(sfx)
            ip.txt((' ' * max(44 - used, 1) + 'Cross-sectional means removed\n') if dm else '\n')
    else:
        ip.txt('LR variance:        ')
        ip.res('(not used)')
        ip.txt((' ' * (44 - 30) + 'Cross-sectional means removed\n') if dm else '\n')
    ip.txt('-' * 78 + '\n')
    ip.txt(' ' * 20 + 'Statistic' + ' ' * 6 + 'p-value\n')
    ip.txt('-' * 78 + '\n')
    ip.txt(' z' + ' ' * 18)
    ip.res(f'{Z:8.4f}' + ' ' * 8 + f'{pv:6.4f}\n')
    ip.txt('-' * 78 + '\n')
    sc = {'N': P.N * capT, 'N_g': P.N, 'N_t': capT, 'z': Z, 'p': pv, 'mu': mu, 'var': var}
    mc = {'test': 'hadri', 'deterministics': 'trend' if trend else 'constant', 'robust': 'robust' if robust else ''}
    if kern:
        sc['lags'] = hl
        mc['kernel'] = kern
    ip.set_r(scalars=sc, macros=mc)
