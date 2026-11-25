"""Generic Stata command-line parsing helpers."""
import re

from .errors import StataError


def split_top(s, sep=None):
    """Split on whitespace (or a separator char) at top level (outside quotes/parens)."""
    out, cur, depth, i, n = [], [], 0, 0, len(s)
    while i < n:
        c = s[i]
        if c == '`' and i + 1 < n and s[i + 1] == '"':
            j = find_cq_end(s, i)
            cur.append(s[i:j])
            i = j
            continue
        if c == '"':
            j = s.find('"', i + 1)
            j = n if j < 0 else j + 1
            cur.append(s[i:j])
            i = j
            continue
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
        if depth == 0 and ((sep is None and c.isspace()) or (sep is not None and c == sep)):
            if cur or sep is not None:
                out.append(''.join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    if cur or (sep is not None and out):
        out.append(''.join(cur))
    if sep is None:
        out = [x for x in out if x != '']
    return out


def find_cq_end(s, i):
    """Given s[i:i+2]=='`\"', return index after matching \"' (handles nesting)."""
    depth = 0
    j = i
    n = len(s)
    while j < n:
        if s.startswith('`"', j):
            depth += 1
            j += 2
            continue
        if s.startswith('"\'', j):
            depth -= 1
            j += 2
            if depth == 0:
                return j
            continue
        j += 1
    return n


def find_top(s, target, start=0):
    """Index of first top-level occurrence of a char."""
    depth, i, n = 0, start, len(s)
    while i < n:
        c = s[i]
        if c == '`' and i + 1 < n and s[i + 1] == '"':
            i = find_cq_end(s, i)
            continue
        if c == '"':
            j = s.find('"', i + 1)
            i = n if j < 0 else j + 1
            continue
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
        elif c == target and depth == 0:
            return i
        i += 1
    return -1


def unquote(s):
    s = s.strip()
    if s.startswith('`"') and s.endswith('"\''):
        return s[2:-2]
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


def quoted_strings(s):
    """Extract all quoted strings ("..." or `"..."') from an option argument."""
    out = []
    i, n = 0, len(s)
    while i < n:
        if s.startswith('`"', i):
            j = find_cq_end(s, i)
            out.append(s[i + 2:j - 2])
            i = j
        elif s[i] == '"':
            j = s.find('"', i + 1)
            if j < 0:
                break
            out.append(s[i + 1:j])
            i = j + 1
        else:
            i += 1
    return out


class Options:
    def __init__(self, raw=''):
        self.raw = raw
        self.items = []
        for tok in split_top(raw):
            m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*(\((.*)\))?$', tok, re.S)
            if m:
                self.items.append((m.group(1).lower() if False else m.group(1), m.group(3) if m.group(2) else None))
            else:
                # name(arg) with trailing content or weird token; keep raw
                k = tok.find('(')
                if k > 0 and tok.endswith(')'):
                    self.items.append((tok[:k], tok[k + 1:-1]))
                else:
                    self.items.append((tok, None))

    def _match(self, name, full, minlen):
        return full.startswith(name) and len(name) >= minlen if minlen else name == full

    def find(self, full, minlen=None):
        for n, a in self.items:
            if n == full or (minlen and full.startswith(n) and len(n) >= minlen):
                return (n, a)
        return None

    def has(self, full, minlen=None):
        return self.find(full, minlen) is not None

    def get(self, full, minlen=None, default=None):
        r = self.find(full, minlen)
        if r is None:
            return default
        return r[1] if r[1] is not None else ''

    def all(self, full, minlen=None):
        return [a for n, a in self.items
                if n == full or (minlen and full.startswith(n) and len(n) >= minlen)]

    def names(self):
        return [n for n, _ in self.items]


class Parsed:
    def __init__(self):
        self.body = ''
        self.if_exp = None
        self.in_range = None
        self.weight = None
        self.using = None
        self.opts = Options('')


_KW_RE = re.compile(r'(?<![A-Za-z0-9_.])(if|in|using)(?![A-Za-z0-9_])')


def parse_command(rest):
    p = Parsed()
    k = find_top(rest, ',')
    main = rest if k < 0 else rest[:k]
    p.opts = Options(rest[k + 1:] if k >= 0 else '')
    # weights [w=exp] at top level
    wm = None
    i = 0
    depth = 0
    while i < len(main):
        c = main[i]
        if c == '"':
            j = main.find('"', i + 1)
            i = len(main) if j < 0 else j + 1
            continue
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == '[' and depth == 0:
            m = re.match(r'\[\s*(pweight|pw|aweight|aw|fweight|fw|iweight|iw|weight)\s*=\s*([^\]]+)\]',
                         main[i:])
            if m:
                wm = (i, i + m.end(), m.group(1), m.group(2).strip())
                break
        i += 1
    if wm:
        wt = {'pw': 'pweight', 'aw': 'aweight', 'fw': 'fweight', 'iw': 'iweight'}.get(wm[2], wm[2])
        p.weight = (wt, wm[3])
        main = main[:wm[0]] + ' ' + main[wm[1]:]
    # tokenize keywords at top level
    segs = _split_keywords(main)
    p.body = segs.get('', '').strip()
    if 'if' in segs:
        p.if_exp = segs['if'].strip()
    if 'in' in segs:
        p.in_range = segs['in'].strip()
    if 'using' in segs:
        p.using = unquote(segs['using'].strip())
    return p


def _split_keywords(s):
    """Split main part into body / if / in / using segments (top level only)."""
    marks = []
    depth = 0
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == '`' and i + 1 < n and s[i + 1] == '"':
            i = find_cq_end(s, i)
            continue
        if c == '"':
            j = s.find('"', i + 1)
            i = n if j < 0 else j + 1
            continue
        if c in '([':
            depth += 1
        elif c in ')]':
            depth -= 1
        elif depth == 0:
            m = _KW_RE.match(s, i)
            if m and (i == 0 or s[i - 1].isspace()):
                marks.append((i, m.group(1), m.end()))
                i = m.end()
                continue
        i += 1
    segs = {}
    prev_key, prev_end = '', 0
    for pos, key, end in marks:
        segs[prev_key] = segs.get(prev_key, '') + ' ' + s[prev_end:pos]
        prev_key, prev_end = key, end
    segs[prev_key] = segs.get(prev_key, '') + ' ' + s[prev_end:]
    return segs


# ---------------------------------------------------------------- factor vars
_FV_RE = re.compile(r'^(?:(i|c|ib(?:n|\d+|\((?:first|last|freq|#\d+|\d+)\))?|o|b)\.)?(.+)$')


class Comp:
    """One component of a (possibly interacted) term."""
    __slots__ = ('kind', 'var', 'base', 'tsop')

    def __init__(self, kind, var, base=None, tsop=None):
        self.kind = kind      # 'i' or 'c'
        self.var = var
        self.base = base      # None default, 'none' (ibn), number, 'last', 'freq'
        self.tsop = tsop

    def key(self):
        return (self.kind, self.var)


def parse_comp(tok, in_interaction):
    tok = tok.strip()
    m = re.match(r'^(i|c|ibn|ib\d+|ib\(\w+\)|ib\(#\d+\))\.(.+)$', tok)
    if m:
        pre, var = m.group(1), m.group(2)
        if pre == 'c':
            return Comp('c', var)
        base = None
        if pre == 'ibn':
            base = 'none'
        elif pre.startswith('ib('):
            base = pre[3:-1]
        elif pre.startswith('ib'):
            base = float(pre[2:])
        return Comp('i', var, base)
    tm = re.match(r'^([LDF])(\d*)\.(.+)$', tok)
    if tm:
        return Comp('c', tm.group(3), tsop=(tm.group(1), int(tm.group(2) or 1)))
    return Comp('i' if in_interaction else 'c', tok)


def parse_terms(tokens):
    """Expand varlist tokens with factor notation into a list of terms (lists of Comp)."""
    terms = []
    for tok in tokens:
        if '##' in tok:
            parts = tok.split('##')
            comps = [parse_comp(p, True) for p in parts]
            # all subsets in order: a, b, a#b
            from itertools import combinations
            k = len(comps)
            for r in range(1, k + 1):
                for sub in combinations(range(k), r):
                    terms.append([comps[j] for j in sub])
        elif '#' in tok:
            comps = [parse_comp(p, True) for p in tok.split('#')]
            terms.append(comps)
        else:
            terms.append([parse_comp(tok, False)])
    # de-duplicate keeping first occurrence
    seen = set()
    out = []
    for t in terms:
        key = tuple(c.key() for c in t)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def parse_numlist(s):
    """Parse a Stata numlist: '1/5', '2011(1)2020', '0.4 0.6', '4(-1)2', '1(1)`n''."""
    out = []
    for tok in split_top(s.replace(',', ' ')):
        m = re.match(r'^(-?[\d.]+)\((-?[\d.]+)\)(-?[\d.]+)$', tok)
        if m:
            a, st, b = float(m.group(1)), float(m.group(2)), float(m.group(3))
            if st == 0:
                raise StataError('invalid numlist', 121)
            x = a
            cnt = 0
            while (st > 0 and x <= b + 1e-9) or (st < 0 and x >= b - 1e-9):
                out.append(round(x, 10))
                cnt += 1
                x = a + cnt * st
            continue
        m = re.match(r'^(-?[\d.]+)/(-?[\d.]+)$', tok)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            step = 1 if b >= a else -1
            x = a
            while (step > 0 and x <= b + 1e-9) or (step < 0 and x >= b - 1e-9):
                out.append(x)
                x += step
            continue
        try:
            out.append(float(tok))
        except ValueError:
            raise StataError(f'invalid numlist: {tok}', 121)
    return out
