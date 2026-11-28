"""Stata date functions and %t display formats.

Conventions (as in Stata): %td = days since 01jan1960, %tw = weeks (52 per year) since 1960w1,
%tm = months since 1960m1, %tq = quarters since 1960q1, %th = half-years since 1960h1, %ty = year.
"""
import datetime as _dt
import re

import numpy as np

_EPOCH = _dt.date(1960, 1, 1).toordinal()
_MON = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
_MONTH = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
          'November', 'December']
_DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']


def _isarr(x):
    return isinstance(x, np.ndarray)


def _vec(f, *args):
    """Apply scalar f elementwise; missing (nan) in -> nan out."""
    if any(_isarr(a) for a in args):
        n = max(len(a) for a in args if _isarr(a))
        cols = [a if _isarr(a) else [a] * n for a in args]
        return np.array([_safe(f, *vals) for vals in zip(*cols)], dtype=float)
    return _safe(f, *args)


def _safe(f, *vals):
    try:
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
            return np.nan
        r = f(*vals)
        return np.nan if r is None else float(r)
    except (ValueError, OverflowError, TypeError):
        return np.nan


# ------------------------------------------------------------------ elementary conversions
def mdy(m, d, y):
    if m != int(m) or d != int(d) or y != int(y):
        return None
    return _dt.date(int(y), int(m), int(d)).toordinal() - _EPOCH


def _date(td):
    return _dt.date.fromordinal(int(np.floor(td)) + _EPOCH)


def year(td):
    return _date(td).year


def month(td):
    return _date(td).month


def day(td):
    return _date(td).day


def quarter(td):
    return (_date(td).month - 1) // 3 + 1


def halfyear(td):
    return 1 if _date(td).month <= 6 else 2


def dow(td):
    return (_date(td).weekday() + 1) % 7  # 0 = Sunday


def doy(td):
    return _date(td).timetuple().tm_yday


def week(td):
    return min((doy(td) - 1) // 7 + 1, 52)


def yq(y, q):
    return (y - 1960) * 4 + q - 1 if 1 <= q <= 4 else None


def ym(y, m):
    return (y - 1960) * 12 + m - 1 if 1 <= m <= 12 else None


def yh(y, h):
    return (y - 1960) * 2 + h - 1 if 1 <= h <= 2 else None


def yw(y, w):
    return (y - 1960) * 52 + w - 1 if 1 <= w <= 52 else None


def dofq(q):
    q = int(np.floor(q))
    return mdy((q % 4) * 3 + 1, 1, 1960 + q // 4)


def qofd(td):
    d = _date(td)
    return yq(d.year, (d.month - 1) // 3 + 1)


def dofm(m):
    m = int(np.floor(m))
    return mdy(m % 12 + 1, 1, 1960 + m // 12)


def mofd(td):
    d = _date(td)
    return ym(d.year, d.month)


def dofh(h):
    h = int(np.floor(h))
    return mdy((h % 2) * 6 + 1, 1, 1960 + h // 2)


def hofd(td):
    d = _date(td)
    return yh(d.year, 1 if d.month <= 6 else 2)


def dofw(w):
    w = int(np.floor(w))
    return mdy(1, 1, 1960 + w // 52) + (w % 52) * 7


def wofd(td):
    return yw(year(td), week(td))


def dofy(y):
    return mdy(1, 1, int(y))


def yofd(td):
    return year(td)


# ------------------------------------------------------------------ string -> date
def _year_fix(y, topyear):
    if y >= 100:
        return y
    if topyear is None:
        return None
    top = int(topyear)
    base = top - top % 100
    cand = base + y
    return cand if cand <= top else cand - 100


def _tokens(s):
    s = s.strip().lower()
    toks = re.findall(r'[a-z]+|\d+', s)
    out = []
    for t in toks:
        if t.isdigit():
            out.append(int(t))
        else:
            mi = next((i for i, mn in enumerate(_MON) if t.startswith(mn)), None)
            if mi is None:
                return None
            out.append(('m', mi + 1))
    # split run-together digits like 20100115 only when no separators were used
    return out


def _parse_daily(s, mask, topyear=None):
    if not isinstance(s, str) or not s.strip():
        return None
    cm = re.search(r'(\d\d)Y', mask.upper())
    century = int(cm.group(1)) if cm else None
    mask = re.sub(r'[^DMY]', '', re.sub(r'\d\dY', 'Y', mask.upper()))
    toks = _tokens(s)
    if toks is None:
        return None
    if len(toks) == 1 and isinstance(toks[0], int) and len(s.strip()) in (6, 8):
        raw = s.strip()
        parts, i = [], 0
        for ch in mask:
            ln = (4 if len(raw) == 8 else 2) if ch == 'Y' else 2
            parts.append(int(raw[i:i + ln]))
            i += ln
        toks = parts
    if len(toks) != 3:
        return None
    vals = {}
    for ch, t in zip(mask, toks):
        if isinstance(t, tuple):
            if ch != 'M':
                return None
            vals['M'] = t[1]
        else:
            vals[ch] = t
    if set(vals) != {'D', 'M', 'Y'}:
        return None
    y = vals['Y']
    if y < 100 and century is not None:
        y = century * 100 + y
    else:
        y = _year_fix(y, topyear)
    if y is None:
        return None
    return mdy(vals['M'], vals['D'], y)


def _parse_period(s, mask, per, topyear=None):
    """monthly("2010m3","YM"), quarterly("2010q1","YQ"), halfyearly, weekly."""
    if not isinstance(s, str) or not s.strip():
        return None
    mask = re.sub(r'[^YMQHW]', '', mask.upper())
    st = s.strip().lower()
    toks = []
    for t in re.findall(r'[a-z]+|\d+', st):
        if t.isdigit():
            toks.append(int(t))
        elif t in ('m', 'q', 'h', 'w'):
            continue
        else:
            mi = next((i for i, mn in enumerate(_MON) if t.startswith(mn)), None)
            if mi is None:
                return None
            toks.append(mi + 1)
    if len(toks) != 2:
        return None
    vals = dict(zip(mask, toks))
    y = _year_fix(vals.get('Y', -1), topyear)
    if y is None or y < 0:
        return None
    p = vals.get(per)
    if p is None:
        return None
    return {'M': ym, 'Q': yq, 'H': yh, 'W': yw}[per](y, p)


def literal(kind, raw):
    """tq(2010q1), tm(2010m3), td(15jan2010), th(2010h1), tw(2010w5), ty(2010)."""
    s = raw.strip().lower()
    if kind == 'td':
        return _parse_daily(s, 'DMY')
    m = re.fullmatch(r'(-?\d+)\s*([qmhw])\s*(\d+)', s)
    if kind == 'ty':
        return float(s)
    if not m:
        return None
    y, k, p = int(m.group(1)), m.group(2), int(m.group(3))
    if {'tq': 'q', 'tm': 'm', 'th': 'h', 'tw': 'w'}[kind] != k:
        return None
    return {'q': yq, 'm': ym, 'h': yh, 'w': yw}[k](y, p)


# ------------------------------------------------------------------ expression functions
def _num(a):
    return a.astype(float) if _isarr(a) else (np.nan if a is None else float(a))


def make_funcs():
    def f1(fn):
        return lambda c, a: _vec(fn, _num(a[0]))

    def f2(fn):
        return lambda c, a: _vec(fn, _num(a[0]), _num(a[1]))

    def strconv(kind):
        def g(c, a):
            s, mask = a[0], a[1]
            top = _num(a[2]) if len(a) > 2 else None
            top = top if top is None or not _isarr(top) else top[0]
            if kind == 'D':
                fn = lambda x: _parse_daily(x, mask, top)  # noqa: E731
            else:
                fn = lambda x: _parse_period(x, mask, kind, top)  # noqa: E731
            if _isarr(s):
                return np.array([_safe(fn, x) for x in s], dtype=float)
            return _safe(fn, s)
        return g
    return {
        'mdy': lambda c, a: _vec(mdy, _num(a[0]), _num(a[1]), _num(a[2])),
        'year': f1(year), 'month': f1(month), 'day': f1(day), 'quarter': f1(quarter),
        'halfyear': f1(halfyear), 'dow': f1(dow), 'doy': f1(doy), 'week': f1(week),
        'yq': f2(yq), 'ym': f2(ym), 'yh': f2(yh), 'yw': f2(yw),
        'dofq': f1(dofq), 'qofd': f1(qofd), 'dofm': f1(dofm), 'mofd': f1(mofd),
        'dofh': f1(dofh), 'hofd': f1(hofd), 'dofw': f1(dofw), 'wofd': f1(wofd),
        'dofy': f1(dofy), 'yofd': f1(yofd),
        'date': strconv('D'), 'daily': strconv('D'),
        'monthly': strconv('M'), 'quarterly': strconv('Q'), 'halfyearly': strconv('H'), 'weekly': strconv('W'),
        'today': lambda c, a: float(_dt.date.today().toordinal() - _EPOCH),
    }


# ------------------------------------------------------------------ display formats
_TFMT = re.compile(r'^%(-)?t([dwmqhy])(.*)$')


def is_tfmt(fmt):
    return bool(fmt) and bool(_TFMT.match(fmt.strip()))


def _custom(td_or_none, codes, kind, val):
    """Render a custom %t format (subset of Stata codes)."""
    if kind == 'd':
        d = _date(val)
        y, mo, dd = d.year, d.month, d.day
    elif kind == 'm':
        y, mo, dd = 1960 + int(val) // 12, int(val) % 12 + 1, 1
    elif kind == 'q':
        y, mo, dd = 1960 + int(val) // 4, (int(val) % 4) * 3 + 1, 1
    elif kind == 'h':
        y, mo, dd = 1960 + int(val) // 2, (int(val) % 2) * 6 + 1, 1
    elif kind == 'w':
        y, mo, dd = 1960 + int(val) // 52, 1, 1
    else:
        y, mo, dd = int(val), 1, 1
    out = []
    i = 0
    tok = [('CCYY', None), ('Month', None), ('month', None), ('Mon', None), ('mon', None), ('CC', None),
           ('cc', None), ('YY', None), ('yy', None), ('NN', None), ('nn', None), ('DD', None), ('dd', None),
           ('Dayname', None), ('Dan', None), ('q', None), ('h', None), ('ww', None)]
    while i < len(codes):
        ch = codes[i]
        if ch == '!' and i + 1 < len(codes):
            out.append(codes[i + 1])
            i += 2
            continue
        if ch == '_':
            out.append(' ')
            i += 1
            continue
        hit = None
        for t, _ in tok:
            if codes.startswith(t, i):
                hit = t
                break
        if hit is None:
            out.append(ch)
            i += 1
            continue
        i += len(hit)
        if hit == 'CCYY':
            out.append(f'{y:04d}')
        elif hit in ('CC', 'cc'):
            out.append(f'{y // 100:02d}' if hit == 'CC' else str(y // 100))
        elif hit in ('YY', 'yy'):
            out.append(f'{y % 100:02d}' if hit == 'YY' else str(y % 100))
        elif hit == 'Month':
            out.append(_MONTH[mo - 1])
        elif hit == 'month':
            out.append(_MONTH[mo - 1].lower())
        elif hit == 'Mon':
            out.append(_MONTH[mo - 1][:3])
        elif hit == 'mon':
            out.append(_MON[mo - 1])
        elif hit == 'NN':
            out.append(f'{mo:02d}')
        elif hit == 'nn':
            out.append(str(mo))
        elif hit == 'DD':
            out.append(f'{dd:02d}')
        elif hit == 'dd':
            out.append(str(dd))
        elif hit == 'Dayname':
            out.append(_DAYS[dow(val)] if kind == 'd' else '')
        elif hit == 'Dan':
            out.append(_DAYS[dow(val)][:3] if kind == 'd' else '')
        elif hit == 'q':
            out.append(str((mo - 1) // 3 + 1))
        elif hit == 'h':
            out.append('1' if mo <= 6 else '2')
        elif hit == 'ww':
            out.append(f'{int(val) % 52 + 1:02d}')
    return ''.join(out)


def fmt_t(x, fmt):
    m = _TFMT.match(fmt.strip())
    left, kind, rest = m.groups()
    if x is None or (isinstance(x, float) and np.isnan(x)):
        s = '.'
    else:
        try:
            v = float(x)
            if rest:
                s = _custom(None, rest, kind, v)
            elif kind == 'd':
                d = _date(v)
                s = f'{d.day:02d}{_MON[d.month - 1]}{d.year}'
            elif kind == 'm':
                s = f'{1960 + int(np.floor(v)) // 12}m{int(np.floor(v)) % 12 + 1}'
            elif kind == 'q':
                s = f'{1960 + int(np.floor(v)) // 4}q{int(np.floor(v)) % 4 + 1}'
            elif kind == 'h':
                s = f'{1960 + int(np.floor(v)) // 2}h{int(np.floor(v)) % 2 + 1}'
            elif kind == 'w':
                s = f'{1960 + int(np.floor(v)) // 52}w{int(np.floor(v)) % 52 + 1}'
            else:
                s = f'{int(v)}'
        except (ValueError, OverflowError):
            s = '.'
    w = max(len(s), 9 if kind == 'd' and not rest else len(s))
    return s.ljust(w) if left else s.rjust(w)


def tin_mask(ip, d, raw, within=False):
    """tin(d1, d2) / twithin(d1, d2): time variable in [d1, d2] (or open interval for twithin);
    d1/d2 are written in the time variable's display units (e.g. 1978q1); either may be blank."""
    from .errors import StataError
    if not d.ts or not d.ts.get('time'):
        raise StataError('time variable not set, use tsset varname ...', 111)
    tv = d.ts['time']
    t = d.cols[tv].astype(float)
    parts = raw.split(',')
    if len(parts) != 2:
        raise StataError('tin() requires two arguments', 198)
    fmt = d.formats.get(tv, '%9.0g').strip().lstrip('%').lstrip('-')
    kind = fmt[:2] if fmt[:1] == 't' else None

    def conv(s):
        s = s.strip()
        if not s:
            return None
        if kind in ('tq', 'tm', 'th', 'tw', 'td'):
            v = literal(kind, s)
            if v is None:
                raise StataError(f'{s} invalid date', 198)
            return float(v)
        return float(s)
    a, b = conv(parts[0]), conv(parts[1])
    lo = -np.inf if a is None else a
    hi = np.inf if b is None else b
    if within:
        m = (t > lo) & (t < hi)
    else:
        m = (t >= lo) & (t <= hi)
    return np.where(np.isnan(t), 0.0, m.astype(float))
