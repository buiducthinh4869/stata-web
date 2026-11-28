"""Stata-style number/string formatting (%fmt directives and %w.0g display)."""
import math
import re

import numpy as np

_FMT_RE = re.compile(r'^%(-)?(0)?(\d+)?(?:\.(\d+))?([fgecs])(c)?$')


def is_missing(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def _strip0(s):
    # drop leading zero: 0.5 -> .5, -0.5 -> -.5
    if s.startswith('0.'):
        return s[1:]
    if s.startswith('-0.'):
        return '-' + s[2:]
    return s


def g_fmt(x, w=9):
    """Emulate Stata %w.0g: at most w chars, max digits, no trailing zeros."""
    if is_missing(x):
        return '.'
    x = float(x)
    if x == 0:
        return '0'
    if math.isinf(x):
        return '.'
    neg = x < 0
    ax = abs(x)
    room = w - 1  # leave room for sign like Stata
    best = None
    # fixed notation candidates
    if ax < 10 ** room:
        intdigits = len(str(int(ax))) if ax >= 1 else 0
        for dec in range(max(0, room - intdigits - (0 if ax >= 1 else 0)), -1, -1):
            s = f'{ax:.{dec}f}'
            if '.' in s:
                s = s.rstrip('0').rstrip('.')
            s = _strip0(s)
            if len(s) <= room:
                best = s
                break
        if best is not None:
            # check significant digits retained
            digits = re.sub(r'[^0-9]', '', best).lstrip('0')
            if ax < 1 and len(digits) < 3 and ax < 1e-4:
                best = None
            elif ax >= 1 and float(best) == 0:
                best = None
    if best is None or (ax >= 1e8 and '.' not in (best or '')) and len(str(int(ax))) > room:
        for dec in range(room, -1, -1):
            s = f'{ax:.{dec}e}'
            mant, ex = s.split('e')
            if '.' in mant:
                mant = mant.rstrip('0').rstrip('.')
            exn = int(ex)
            s = f'{mant}e{"+" if exn >= 0 else "-"}{abs(exn):02d}'
            if len(s) <= room:
                best = s
                break
    if best is None:
        best = f'{ax:.2e}'
    return ('-' + best) if neg else best


def _commas(intpart):
    return f'{int(intpart):,}'


def fmt_num(x, fmt):
    """Apply a Stata %fmt to a number; returns string padded to width."""
    from .dates import is_tfmt, fmt_t
    if is_tfmt(fmt) and not isinstance(x, str):
        return fmt_t(x, fmt)
    m = _FMT_RE.match(fmt.strip())
    if not m:
        return g_fmt(x)
    left, zero, w, d, typ, comma = m.groups()
    w = int(w) if w else 0
    if typ == 's':
        s = str(x)
    elif is_missing(x):
        s = '.'
    else:
        x = float(x)
        if typ == 'f':
            dd = int(d) if d is not None else 0
            if comma:
                s = f'{x:,.{dd}f}'
            else:
                s = f'{x:.{dd}f}'
            if s in ('-0', ) or re.fullmatch(r'-0\.0*', s):
                s = s[1:]
            if dd > 0 and not comma and w and len(s) > w:
                s = _strip0(s)
            if len(s) > w and w and dd > 0:
                # Stata falls back to g when it does not fit
                s = g_fmt(x, w)
        elif typ == 'g':
            dd = int(d) if d else 0
            if dd == 0:
                s = g_fmt(x, w if w else 9)
                if comma:
                    try:
                        if float(s) == int(float(s)) and 'e' not in s:
                            s = f'{int(float(s)):,}'
                    except ValueError:
                        pass
            else:
                # %w.dg: at most d significant digits, then shown as %w.0g
                s = g_fmt(float(f'{x:.{dd}g}'), w if w else 9)
        elif typ == 'e':
            dd = int(d) if d is not None else 2
            s = f'{x:.{dd}e}'
        else:
            s = str(x)
    if zero and not left and typ == 'f':
        return s.rjust(w, '0') if w else s
    if left:
        return s.ljust(w)
    return s.rjust(w)


def fmt_any(v, fmt=None):
    if isinstance(v, str):
        if fmt:
            m = _FMT_RE.match(fmt)
            if m and m.group(5) == 's':
                return fmt_num(v, fmt)
        return v
    if fmt:
        return fmt_num(v, fmt)
    return g_fmt(v, 9).strip()


def num_to_macro(x):
    """String form of a number stored in a macro / displayed by `=exp'."""
    if is_missing(x):
        return '.'
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    x = float(x)
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    s = repr(x)
    if 'e' in s:
        mant, ex = s.split('e')
        exn = int(ex)
        if -5 <= exn < 0:
            s = f'{x:.{17}f}'.rstrip('0')
        else:
            return f'{mant}e{"+" if exn >= 0 else "-"}{abs(exn):02d}'
    return _strip0(s)


def display_num(x):
    """Default numeric display in `display` (Stata uses %10.0g-like, max 8 sig digits)."""
    if is_missing(x):
        return '.'
    x = float(x)
    if x == int(x) and abs(x) < 1e10:
        return str(int(x))
    return g_fmt(x, 10).strip()


def abbrev(name, w=12):
    """Stata abbreviation of a name to width w using ~."""
    if len(name) <= w:
        return name
    return name[: w - 2] + '~' + name[-1:]


def comma_int(n):
    try:
        return f'{int(n):,}'
    except (TypeError, ValueError):
        return '.'
