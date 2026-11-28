"""Official xtabond and xtdpdsys, implemented through the xtabond2 engine with the equivalent instrument
specification (as built by Stata's _xtab_parser.ado for xtdpd, which corresponds to xtabond2 with h(2)), with
Stata's own output (xtdpd.ado HDisplay) and estat abond / estat sargan (xtdpd_estat.ado)."""
import numpy as np
from scipy import stats as sps

from .design import coef_table
from .errors import StataError
from .estim import ESTAT_EXT
from .fmt import fmt_num, comma_int
from .interp import command
from .syntax import parse_command, split_top
from .stab import Tab


def _parse_lagstruct(raw, endog):
    if ',' in raw:
        vl, o = raw.split(',', 1)
    else:
        vl, o = raw, ''
    po = parse_command(', ' + o) if o.strip() else parse_command('')
    if po.opts.has('endogenous', 3):
        endog = True
    ls = po.opts.get('lagstruct', 3)
    if ls:
        a, b = [x.strip() for x in ls.split(',')]
        lags = int(a)
        maxl = None if b == '.' else int(b)
    else:
        lags, maxl = 0, None
    first = 2 if endog else 1
    if endog and maxl is not None:
        maxl += 1
    return vl.split(), lags, maxl, first


def _run(ip, cmd, rest, which):
    p = parse_command(rest)
    if not p.body.strip():
        if ip.est and ip.est.get('cmd') == which:
            return ip.est['replay'](ip)
        raise StataError('last estimates not found', 301)
    if which == 'xtdpdsys' and p.opts.has('robust', 1):
        raise StataError('robust not allowed', 198)
    toks = split_top(p.body)
    dep, indeps = toks[0], toks[1:]
    lg = p.opts.get('lags', 2)
    lags = int(lg) if lg not in (None, '') else 1
    if lags < 1:
        raise StataError('lags of dependent variable must >=1', 198)
    md = p.opts.get('maxldep', 4)
    maxldep = (int(md) + 1) if md not in (None, '') else None
    ml = p.opts.get('maxlags', 4)
    gmaxl = int(ml) if ml not in (None, '') else None
    vce = (p.opts.get('vce', 3) or '').strip().lower()
    robust = p.opts.has('robust', 1) or vce.startswith('r')
    if vce and not (vce.startswith('r') or vce == 'gmm'):
        raise StataError(f'vce({vce}) not allowed', 198)
    twostep = p.opts.has('twostep', 3)
    nocons = p.opts.has('noconstant', 4)
    at = p.opts.get('artests', 2)
    artests = int(at) if at not in (None, '') else 2
    pre = []
    for raw in p.opts.all('pre', 3):
        pre.append(_parse_lagstruct(raw, False))
    for raw in p.opts.all('endogenous', 3):
        pre.append(_parse_lagstruct(raw, True))
    if p.opts.get('diffvars', 5):
        raise StataError('diffvars() is not available in Stata Web', 198)
    reg = [f'L(0/{lags}).{dep}'.replace('L(0/', 'L(1/')]
    gmm = [f'gmm({dep}, lag(2 {maxldep if maxldep else "."}) eq(diff))']
    lgm = []
    dg_disp = [(dep, 2, maxldep)]
    lg_disp = []
    for vars_, lgs, maxl, first in pre:
        if gmaxl is not None:
            maxl = gmaxl + (1 if first == 2 else 0)
        for v in vars_:
            reg.append(f'L(0/{lgs}).{v}')
            gv = f'L{lgs}.{v}' if lgs else v
            gmm.append(f'gmm({gv}, lag({first} {maxl if maxl else "."}) eq(diff))')
            dg_disp.append((gv, first, maxl))
            if which == 'xtdpdsys':
                lgm.append(f'gmm({gv}, lag({first - 1} {first - 1}) eq(level))')
                lg_disp.append((gv, first - 1))
    if which == 'xtdpdsys':
        lgm.insert(0, f'gmm({dep}, lag(1 1) eq(level))')
        lg_disp.insert(0, (dep, 1))
    ivs = []
    std_d = []
    if indeps:
        ivs.append(f'iv({" ".join(indeps)}, eq(diff))')
        std_d += [('D', t) for t in indeps]
    inst = p.opts.get('inst', 4)
    if inst:
        ivs.append(f'iv({inst}, eq(diff) passthru)')
        std_d += [('', t) for t in inst.split()]
    line = f'{dep} {" ".join(reg + indeps)}'
    if p.if_exp:
        line += f' if {p.if_exp}'
    if p.in_range:
        line += f' in {p.in_range}'
    line += ', ' + ' '.join(gmm + lgm + ivs) + ' h(2)'
    if nocons and which == 'xtabond' and not lgm:
        line += ' noleveleq'
    elif nocons:
        line += ' noconstant'
    if which == 'xtabond' and not nocons:
        line += ' diffsmp'
    if twostep:
        line += ' twostep'
    if robust:
        line += ' robust'
    line += f' artests({artests}) nodiffsargan'
    q0 = ip.quiet
    ip.quiet += 1
    try:
        from .xtab2 import cmd_xtabond2
        cmd_xtabond2(ip, 'xtabond2', line, None)
    finally:
        ip.quiet = q0
    e = ip.est
    sc = e['scalars']
    e['cmd'] = which
    e['macros']['cmd'] = which
    sys_ = which == 'xtdpdsys' or not nocons
    e['macros']['system'] = 'system' if which == 'xtdpdsys' else ''
    e['macros']['vce'] = 'robust' if robust else 'gmm'
    if which == 'xtabond':
        ab = e['ab_internal']
        g = ab['g_d'][ab['g_d'] > 0]
        sc.update({'N': ab['obs_d'], 'N_g': int(g.size), 'g_min': int(g.min()), 'g_max': int(g.max()),
                   'g_avg': ab['obs_d'] / g.size})
    sc['zrank'] = sc['j']
    if not twostep and not robust:
        # xtdpd estimates sigma^2 with a degrees-of-freedom correction (N - k) in the one-step GMM case
        wt = e['ab_internal']['obs_d']
        kk = len(e['names'])
        f = wt / (wt - kk)
        e['V'] = e['V'] * f
        sc['sargan'] = sc['sargan'] / f
        for i in range(1, 50):
            if f'ar{i}' in sc:
                sc[f'ar{i}'] = sc[f'ar{i}'] / np.sqrt(f)
        sl = [i for i, nm in enumerate(e['names']) if nm != '_cons']
        if sl:
            bb = e['b'][sl]
            sc['chi2'] = float(bb @ np.linalg.solve(e['V'][np.ix_(sl, sl)], bb))
    sc['arm1'] = sc.get('ar1', np.nan)
    sc['arm2'] = sc.get('ar2', np.nan)
    if twostep:
        sc['sargan'] = sc.get('hansen', sc['sargan'])
    title = {'xtabond': 'Arellano-Bond dynamic panel-data estimation',
             'xtdpdsys': 'System dynamic panel-data estimation'}[which]
    vlab = ('WC-Robust' if twostep else 'Robust') if robust else None

    def replay(ip_):
        ip_.txt(f'\n{title:<48}{"Number of obs":<18}= ')
        ip_.res(f"{sc['N']:10,.0f}\n")
        s = 'Group variable: ' + e['macros']['ivar'][:12]
        ip_.txt(f'{s:<48}{"Number of groups":<18}= ')
        ip_.res(f"{sc['N_g']:10,.0f}\n")
        ip_.txt('Time variable: ' + e['macros']['tvar'][:12] + '\n')
        ip_.txt(' ' * 48 + 'Obs per group:\n')
        ip_.txt(' ' * 62 + 'min = ')
        ip_.res(f"{sc['g_min']:10,.0f}\n")
        ip_.txt(' ' * 62 + 'avg =  ')
        ip_.res(fmt_num(sc['g_avg'], '%9.0gc') + '\n')
        ip_.txt(' ' * 62 + 'max = ')
        ip_.res(f"{sc['g_max']:10,.0f}\n\n")
        s = 'Number of instruments =  '
        ip_.txt(s)
        ip_.res(fmt_num(sc['zrank'], '%5.0g'))
        lab = f"Wald chi2({sc['df_m']})"
        ip_.txt(' ' * (48 - len(s) - 5) + f'{lab:<18}=  ')
        ip_.res(f"{sc.get('chi2', np.nan):9.2f}\n")
        ip_.txt(' ' * 48 + f'{"Prob > chi2":<18}=' + ' ' * 5)
        ip_.res(f"{sps.chi2.sf(sc.get('chi2', np.nan), sc['df_m']):6.4f}\n")
        ip_.txt('Two-step results\n' if twostep else 'One-step results\n')
        if robust:
            msg = f"(Std. err. adjusted for clustering on {e['macros']['ivar']})"
            ip_.txt(f'{msg:>78}\n')
        coef_table(ip_, e, 'z', robust=vlab, depvar=e['depvar'])
        if twostep and not robust:
            ip_.txt('Warning: gmm two-step standard errors are biased; robust standard\n'
                    '         errors are recommended.\n')
        gd = ' '.join(f'L({a}/{b if b else "."}).{_norm(v)}' for v, a, b in dg_disp)
        ip_.txt('Instruments for differenced equation\n')
        _wrap(ip_, 'GMM-type: ', gd)
        sd = ' '.join(_tsnorm(('D.' if pre_ == 'D' else '') + v) for pre_, v in std_d)
        if sd:
            _wrap(ip_, 'Standard: ', sd)
        if sys_:
            ip_.txt('Instruments for level equation\n')
            if lg_disp:
                _wrap(ip_, 'GMM-type: ', ' '.join(_tsnorm((('LD.' if f == 1 else f'L{f}D.') if f else 'D.') + v) for v, f in lg_disp))
            if not nocons:
                _wrap(ip_, 'Standard: ', '_cons')
    e['replay'] = replay
    e['artests'] = artests
    e['onestep_sys_gmm'] = (which == 'xtdpdsys' and not twostep and not robust)
    replay(ip)


def _norm(v):
    return _tsnorm(v)


def _tsnorm(v):
    from .tsops import split_name, canonical
    ch, base = split_name(v)
    if not ch:
        return v
    return canonical(ch, base)


def _wrap(ip, lead, text):
    width = 78 - 18
    words = text.split()
    line = ''
    first = True
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            ip.txt((' ' * 8 + lead if first else ' ' * 18))
            ip.res(line + '\n')
            first = False
            line = w
        else:
            line = (line + ' ' + w).strip()
    if line:
        ip.txt((' ' * 8 + lead if first else ' ' * 18))
        ip.res(line + '\n')


@command('xtabond')
def cmd_xtabond(ip, cmd, rest, by):
    _run(ip, cmd, rest, 'xtabond')


@command('xtdpdsys')
def cmd_xtdpdsys(ip, cmd, rest, by):
    _run(ip, cmd, rest, 'xtdpdsys')


def estat_abond(ip, p):
    e = ip.est
    if not e or e.get('cmd') not in ('xtabond', 'xtdpdsys'):
        raise StataError('estat abond only valid after xtabond or xtdpdsys', 301)
    if e.get('onestep_sys_gmm'):
        ip.txt('artests not computed for one-step system estimator with vce(gmm)\n')
        raise StataError('', 198)
    sc = e['scalars']
    n = e.get('artests', 2)
    at = p.opts.get('artests', 2)
    if at not in (None, '') and int(at) > n:
        ip.txt('note: only the AR tests computed during estimation are displayed\n')
    ip.txt('\nArellano-Bond test for zero autocorrelation in first-differenced errors\n')
    t = Tab(ip, '|6|8 8|')
    t.set_strcolor('green', None, None)
    t.set_numcolor('green', 'yellow', 'yellow')
    t.set_numfmt('%4.0f', '%7.6g', '%6.4f')
    t.set_pad(0, 0, 1)
    t.sep('top')
    t.titles('Order', 'z    ', 'Prob > z')
    t.sep('mid')
    rows = []
    for i in range(1, n + 1):
        z = sc.get(f'ar{i}', np.nan)
        pv = 2 * sps.norm.cdf(-abs(z))
        t.row(float(i), z, pv)
        rows.append([i, z, pv])
    t.sep('bot')
    ip.txt('   H0: no autocorrelation \n')
    from .expr import Matrix
    ip.set_r(matrices={'arm': Matrix(rows, None, ['order', 'arstat', 'p-values'])})


def estat_sargan(ip, p):
    e = ip.est
    if not e or e.get('cmd') not in ('xtabond', 'xtdpdsys'):
        raise StataError('estat sargan only valid after xtabond or xtdpdsys', 301)
    sc = e['scalars']
    ip.txt('Sargan test of overidentifying restrictions\n')
    ip.txt('        H0: overidentifying restrictions are valid\n')
    if e['macros'].get('vce') == 'robust':
        ip.txt('        cannot calculate Sargan test with vce(robust)\n')
    df = int(sc['zrank'] - len(e['names']))
    ip.txt(f'\n        chi2(')
    ip.res(str(df))
    ip.txt(')' + ' ' * max(21 - 14 - len(str(df)), 0) + '= ')
    ip.res(fmt_num(sc['sargan'], '%9.8g') + '\n')
    ip.txt('        Prob > chi2  = ')
    ip.res('   ' + fmt_num(sps.chi2.sf(sc['sargan'], df), '%6.4f') + '\n')
    ip.set_r(scalars={'chi2': sc['sargan'], 'df': df, 'p': float(sps.chi2.sf(sc['sargan'], df))})


ESTAT_EXT.extend([('ab', 'abond', estat_abond), ('sar', 'sargan', estat_sargan)])
