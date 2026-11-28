"""Stata time-series operators: L. F. D. S. with powers, chains (LD., L2D.) and numlists (L(1/3).)."""
import re

import numpy as np

from .errors import StataError

_OPS_ONLY = re.compile(r'^(?:[LDFSldfs]\d*)+$')
_RANGE = re.compile(r'^([LDFSldfs])\(\s*(-?\d+)\s*(?:/\s*(-?\d+)\s*)?\)$')
_PREFIX = re.compile(r'^((?:[LDFSldfs](?:\d+|\(\s*-?\d+\s*(?:/\s*-?\d+\s*)?\))?)+)\.(.+)$')


def has_tsop(tok):
    m = _PREFIX.match(tok.strip())
    return bool(m) and not re.match(r'^(i|c|o|b|ib)', tok.strip())


def _parse_chain(s):
    """'L2D' -> {'L':2,'D':1}; supports repeated ops (LL = L2)."""
    out = {'L': 0, 'D': 0, 'S': []}
    for op, k in re.findall(r'([LDFS])(\d*)', s.upper()):
        k = int(k) if k else 1
        if op == 'L':
            out['L'] += k
        elif op == 'F':
            out['L'] -= k
        elif op == 'D':
            out['D'] += k
        else:
            out['S'].append(k)
    out['S'] = tuple(out['S'])
    return out


def _split_groups(prefix):
    """Split a prefix like 'L(1/3)D' or 'L2D' into op groups, expanding numlists."""
    parts = re.findall(r'[LDFSldfs](?:\(\s*-?\d+\s*(?:/\s*-?\d+\s*)?\)|\d*)', prefix)
    alts = [['']]
    for p in parts:
        m = _RANGE.match(p)
        if m:
            op, a = m.group(1).upper(), int(m.group(2))
            b = int(m.group(3)) if m.group(3) is not None else a
            step = 1 if b >= a else -1
            alts.append([f'{op}{k}' for k in range(a, b + step, step)])
        else:
            alts.append([p.upper()])
    combos = ['']
    for a in alts:
        combos = [c + x for c in combos for x in a]
    return combos


def expand_token(tok):
    """Expand one varlist token into canonical tokens (strings).
    'L(1/3).D.y' -> ['LD.y','L2D.y','L3D.y'];  'L(0/1).x' -> ['x','L.x'];  'LD.y' -> ['LD.y']."""
    t = tok.strip()
    # allow successive groups: L(1/3).D.y
    pre = ''
    while True:
        m = _PREFIX.match(t)
        if not m or re.match(r'^(i|c|o|b|ib)\.', t):
            break
        pre += m.group(1)
        t = m.group(2)
    if not pre:
        return [tok]
    out = []
    for chain in _split_groups(pre):
        out.append(canonical(chain, t))
    return out


def canonical(chain, var):
    c = _parse_chain(chain)
    return (opname(c) + '.' + var) if opname(c) else var


def opname(c):
    s = ''
    if c['L'] > 0:
        s += 'L' + (str(c['L']) if c['L'] != 1 else '')
    elif c['L'] < 0:
        s += 'F' + (str(-c['L']) if c['L'] != -1 else '')
    if c['D']:
        s += 'D' + (str(c['D']) if c['D'] != 1 else '')
    for k in c['S']:
        s += 'S' + (str(k) if k != 1 else '')
    return s


def display_label(chain_or_name):
    """Row label used in coefficient tables: L1. D1. LD. L2D. F1. --."""
    if not chain_or_name:
        return '--.'
    c = _parse_chain(chain_or_name)
    parts = []
    single = (c['L'] != 0) + (c['D'] != 0) + len(c['S']) == 1
    if c['L'] > 0:
        parts.append('L' + (str(c['L']) if (c['L'] != 1 or single) else ''))
    elif c['L'] < 0:
        parts.append('F' + (str(-c['L']) if (c['L'] != -1 or single) else ''))
    if c['D']:
        parts.append('D' + (str(c['D']) if (c['D'] != 1 or single) else ''))
    for k in c['S']:
        parts.append('S' + (str(k) if (k != 1 or single) else ''))
    return ''.join(parts) + '.'


def split_name(name):
    """'L2D.y' -> ('L2D', 'y');  'y' -> ('', 'y')."""
    m = re.match(r'^((?:[LDFS]\d*)+)\.([A-Za-z_]\w*)$', name)
    if m:
        return m.group(1), m.group(2)
    return '', name


def apply(ip, d, chain, var):
    """Values of chain.var respecting tsset (panel-aware)."""
    if not d.ts or not d.ts.get('time'):
        raise StataError('time variable not set, use tsset varname ...', 111)
    var = d.resolve(var)
    if d.isstr(var):
        raise StataError(f'{var}: time-series operators not allowed with string variables', 109)
    x = d.cols[var].astype(float)
    c = _parse_chain(chain)
    for _ in range(c['D']):
        x = x - ip._lag(d, x, 1)
    for k in c['S']:
        x = x - ip._lag(d, x, k)
    if c['L']:
        x = ip._lag(d, x, c['L'])
    return x
