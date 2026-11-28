"""Stata expression tokenizer, parser and vectorised evaluator."""
import math
import re

import numpy as np
from scipy import stats as sps
from scipy import special as spf

from .errors import StataError
from .fmt import fmt_num, num_to_macro

MISS = float('nan')

# ----------------------------------------------------------------------- tokens
_TOK_RE = re.compile(r'''
   (?P<ws>\s+)
 | (?P<cstr>`"(?:.|\n)*?"')
 | (?P<str>"[^"]*")
 | (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)
 | (?P<miss>\.[a-z]?(?![0-9A-Za-z_]))
 | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
 | (?P<op>==|!=|~=|>=|<=|\^|[-+*/<>&|!~()\[\],:=])
''', re.X)


class Tok:
    __slots__ = ('kind', 'val', 'pos', 'end')

    def __init__(self, kind, val, pos, end):
        self.kind, self.val, self.pos, self.end = kind, val, pos, end

    def __repr__(self):
        return f'{self.kind}:{self.val}'


def tokenize(s, lenient=False):
    toks = []
    i = 0
    n = len(s)
    while i < n:
        m = _TOK_RE.match(s, i)
        if not m:
            if lenient:
                toks.append(Tok('bad', s[i], i, i + 1))
                break
            raise StataError(f'invalid syntax near "{s[i:i+20]}"', 198)
        kind = m.lastgroup
        if kind != 'ws':
            val = m.group(kind)
            if kind == 'cstr':
                kind, val = 'str', val[2:-2]
            elif kind == 'str':
                val = val[1:-1]
            toks.append(Tok(kind, val, m.start(), m.end()))
        i = m.end()
    return toks


# ----------------------------------------------------------------------- AST
class Node:
    pass


class Num(Node):
    def __init__(self, v):
        self.v = v


class Str(Node):
    def __init__(self, v):
        self.v = v


class Name(Node):
    def __init__(self, n):
        self.n = n


class Call(Node):
    def __init__(self, f, args, raw=None):
        self.f, self.args, self.raw = f, args, raw


class Index(Node):
    def __init__(self, base, idx):
        self.base, self.idx = base, idx


class Coef(Node):
    def __init__(self, kind, name):
        self.kind, self.name = kind, name


class Un(Node):
    def __init__(self, op, x):
        self.op, self.x = op, x


class Bin(Node):
    def __init__(self, op, a, b):
        self.op, self.a, self.b = op, a, b


class TsOp(Node):
    def __init__(self, op, lag, var):
        self.op, self.lag, self.var = op, lag, var


_PREC = {'|': 1, '&': 2, '==': 3, '!=': 3, '~=': 3, '>': 3, '<': 3, '>=': 3, '<=': 3,
         '+': 4, '-': 4, '*': 5, '/': 5, '^': 7}

_RAWARG = {'e', 'r', 'c', 's', 'tq', 'tm', 'td', 'th', 'tw', 'ty', 'tin', 'twithin'}


class Parser:
    def __init__(self, s, lenient=False):
        self.s = s
        self.toks = tokenize(s, lenient)
        self.i = 0

    def peek(self, k=0):
        j = self.i + k
        if j < len(self.toks) and self.toks[j].kind != 'bad':
            return self.toks[j]
        return None

    def next(self):
        t = self.peek()
        self.i += 1
        return t

    def expect(self, val):
        t = self.next()
        if t is None or t.val != val:
            raise StataError('invalid syntax', 198)
        return t

    def parse_full(self):
        node = self.expr(0)
        if self.peek() is not None:
            raise StataError(f'invalid syntax: unexpected "{self.peek().val}"', 198)
        return node

    def parse_prefix(self):
        """Parse an expression and return (node, remaining text)."""
        node = self.expr(0)
        rest = self.s[self.toks[self.i].pos:] if self.i < len(self.toks) else ''
        return node, rest

    def expr(self, minp):
        left = self.unary()
        while True:
            t = self.peek()
            if t is None or t.kind != 'op' or t.val not in _PREC:
                break
            p = _PREC[t.val]
            if p < minp:
                break
            self.next()
            if t.val == '^':
                right = self.expr(p)       # right assoc
            else:
                right = self.expr(p + 1)
            op = '!=' if t.val == '~=' else t.val
            left = Bin(op, left, right)
        return left

    def unary(self):
        t = self.peek()
        if t is not None and t.kind == 'op' and t.val in ('-', '+', '!', '~'):
            self.next()
            x = self.expr(6) if t.val in '-+' else self.expr(3)
            if t.val == '+':
                return x
            return Un('-' if t.val == '-' else '!', x)
        return self.postfix(self.atom())

    def postfix(self, node):
        while True:
            t = self.peek()
            if t is not None and t.val == '[' and isinstance(node, (Name, Call)):
                # subscript (but not after whitespace for safety)
                self.next()
                idx = [self.expr(0)]
                while self.peek() is not None and self.peek().val == ',':
                    self.next()
                    idx.append(self.expr(0))
                self.expect(']')
                node = Index(node, idx)
            else:
                return node

    def atom(self):
        t = self.next()
        if t is None:
            raise StataError('invalid syntax', 198)
        if t.kind == 'num':
            return Num(float(t.val))
        if t.kind == 'miss':
            return Num(MISS)
        if t.kind == 'str':
            return Str(t.val)
        if t.kind == 'op' and t.val == '(':
            e = self.expr(0)
            self.expect(')')
            return e
        if t.kind == 'name':
            name = t.val
            nt = self.peek()
            # _b[...] / _se[...]
            if name in ('_b', '_se', '_coef') and nt is not None and nt.val == '[':
                close = self.s.index(']', nt.end)
                raw = self.s[nt.end:close].strip()
                while self.peek() is not None and self.peek().pos < close:
                    self.next()
                self.next()  # ]
                return Coef('_se' if name == '_se' else '_b', raw)
            # time-series operators L.x  L2.x  D.x F.x
            m = re.fullmatch(r'([LDF])(\d*)', name)
            if m and nt is not None and nt.val == '.' and False:
                pass
            if nt is not None and nt.val == '(' and nt.pos == t.end:
                self.next()
                if name in _RAWARG:
                    depth = 1
                    j = nt.end
                    while j < len(self.s):
                        if self.s[j] == '(':
                            depth += 1
                        elif self.s[j] == ')':
                            depth -= 1
                            if depth == 0:
                                break
                        j += 1
                    raw = self.s[nt.end:j].strip()
                    while self.peek() is not None and self.peek().pos < j:
                        self.next()
                    self.next()
                    return Call(name, [], raw=raw)
                args = []
                if self.peek() is not None and self.peek().val == ')':
                    self.next()
                    return Call(name, args)
                args.append(self.expr(0))
                while self.peek() is not None and self.peek().val == ',':
                    self.next()
                    args.append(self.expr(0))
                self.expect(')')
                return Call(name, args)
            return Name(name)
        raise StataError(f'invalid syntax near "{t.val}"', 198)


_TS_RE = re.compile(r'(?<![\w.])((?:[LDFS]\d*)+)\.([A-Za-z_][A-Za-z0-9_]*)')


_COEF_RE = re.compile(r'_(b|se|coef)\[([^\]]*)\]')
_COEF_STORE = {}


_COEF_IDS = {}


def _b36(n):
    a = '0123456789abcdefghijklmnopqrstuvwxyz'
    out = ''
    while True:
        out = a[n % 36] + out
        n //= 36
        if not n:
            return out


def _coef_sub(s):
    """Replace _b[...] by same-length identifiers so string offsets are preserved."""
    def rep(m):
        val = ('_se' if m.group(1) == 'se' else '_b', m.group(2).strip())
        if val not in _COEF_IDS:
            _COEF_IDS[val] = _b36(len(_COEF_IDS))
        key = '__cf' + _COEF_IDS[val]
        L = len(m.group(0))
        if len(key) > L:
            key = 'Q' + _COEF_IDS[val]
        key = key.ljust(L, '_')
        _COEF_STORE[key] = val
        return key
    return _COEF_RE.sub(rep, s)


def parse(s):
    s = _coef_sub(s)
    s = _TS_RE.sub(lambda m: f'__ts_{m.group(1)}__{m.group(2)}', s)
    return Parser(s).parse_full()


def parse_prefix(s):
    return Parser(_coef_sub(s), lenient=True).parse_prefix()


# ----------------------------------------------------------------------- eval
class Ctx:
    """Evaluation context. data may be None (scalar context)."""

    def __init__(self, interp, data=None, by=None, sample=None):
        self.ip = interp
        self.data = data
        self.by = by            # (group_index_array, start_array, size_array, pos_array)
        self.N = data.n if data is not None else None

    def n_array(self):
        if self.by is not None:
            return self.by[3] + 1.0
        return np.arange(1, self.N + 1, dtype=float)

    def bigN_array(self):
        if self.by is not None:
            return self.by[2].astype(float)
        return np.full(self.N, float(self.N))


def is_arr(x):
    return isinstance(x, np.ndarray)


def _isstr(x):
    if isinstance(x, str):
        return True
    if is_arr(x) and x.dtype == object:
        return True
    return False


def truth(x):
    """Stata truth: nonzero (missing counts as true)."""
    if is_arr(x):
        return x != 0
    return x != 0


def to_num(x):
    if is_arr(x):
        return x.astype(float)
    return float(x)


def _cmp_prep(a):
    if is_arr(a) and a.dtype != object:
        return np.where(np.isnan(a), np.inf, a)
    if isinstance(a, float) and math.isnan(a):
        return math.inf
    return a


def _bool_out(r):
    if is_arr(r):
        return r.astype(float)
    return 1.0 if r else 0.0


def _clean(x):
    if is_arr(x) and x.dtype != object:
        x[np.isinf(x)] = np.nan
        return x
    if isinstance(x, float) and math.isinf(x):
        return MISS
    return x


def evaluate(node, ctx):
    t = type(node)
    if t is Num:
        return node.v
    if t is Str:
        return node.v
    if t is Name:
        return _eval_name(node.n, ctx)
    if t is Un:
        x = evaluate(node.x, ctx)
        if node.op == '-':
            if _isstr(x):
                raise StataError('type mismatch', 109)
            return -to_num(x) if not is_arr(x) else -x
        # logical not
        if _isstr(x):
            raise StataError('type mismatch', 109)
        if is_arr(x):
            return (x == 0).astype(float)
        return 1.0 if x == 0 else 0.0
    if t is Bin:
        return _eval_bin(node, ctx)
    if t is Call:
        return _eval_call(node, ctx)
    if t is Index:
        return _eval_index(node, ctx)
    if t is Coef:
        return ctx.ip.get_coef(node.kind, node.name)
    raise StataError('invalid expression', 198)


def _eval_name(n, ctx):
    ip = ctx.ip
    if n in _COEF_STORE:
        k, raw = _COEF_STORE[n]
        return ip.get_coef(k, raw)
    if n.startswith('__ts_'):
        m = re.match(r'__ts_((?:[LDFS]\d*)+)__(.+)', n)
        from .tsops import apply as _ts_apply
        return _ts_apply(ip, ctx.data if ctx.data is not None else ip.data, m.group(1), m.group(2))
    d = ctx.data
    if n == '_n':
        if d is None:
            raise StataError('_n not allowed', 198)
        return ctx.n_array()
    if n == '_N':
        if d is None:
            return float(ip.data.n)
        return ctx.bigN_array()
    if n == '_pi':
        return math.pi
    if n == '_rc':
        return float(ip.rc)
    if d is not None and n in d.cols:
        return d.cols[n]
    if n in ip.scalars:
        return ip.scalars[n]
    if d is not None:
        v = d.resolve(n, required=False)
        if v is not None:
            return d.cols[v]
    if d is None and ip.data is not None:
        v = ip.data.resolve(n, required=False)
        if v is not None:
            # scalar context referencing variable -> first observation
            col = ip.data.cols[v]
            return col[0] if len(col) else MISS
    raise StataError(f'{n} not found', 111)


def _eval_bin(node, ctx):
    op = node.op
    a = evaluate(node.a, ctx)
    b = evaluate(node.b, ctx)
    sa, sb = _isstr(a), _isstr(b)
    if sa or sb:
        if not (sa and sb):
            raise StataError('type mismatch', 109)
        if op == '+':
            if is_arr(a) or is_arr(b):
                return np.array([str(x) + str(y) for x, y in np.broadcast(a, b)], dtype=object) \
                    if (is_arr(a) and is_arr(b)) else (
                        np.array([str(x) + str(b) for x in a], dtype=object) if is_arr(a)
                        else np.array([str(a) + str(y) for y in b], dtype=object))
            return str(a) + str(b)
        if op in ('==', '!=', '<', '>', '<=', '>='):
            fn = {'==': lambda x, y: x == y, '!=': lambda x, y: x != y, '<': lambda x, y: x < y,
                  '>': lambda x, y: x > y, '<=': lambda x, y: x <= y, '>=': lambda x, y: x >= y}[op]
            if is_arr(a) or is_arr(b):
                aa = a if is_arr(a) else np.full(len(b), a, dtype=object)
                bb = b if is_arr(b) else np.full(len(a), b, dtype=object)
                return np.array([1.0 if fn(str(x), str(y)) else 0.0 for x, y in zip(aa, bb)])
            return 1.0 if fn(a, b) else 0.0
        raise StataError('type mismatch', 109)
    if op in ('&', '|'):
        ta, tb = truth(a), truth(b)
        r = (ta & tb) if op == '&' else (ta | tb)
        if not is_arr(r):
            r = bool(r)
        return _bool_out(r)
    if op in ('==', '!=', '<', '>', '<=', '>='):
        x, y = _cmp_prep(a), _cmp_prep(b)
        with np.errstate(invalid='ignore'):
            if op == '==':
                r = x == y
            elif op == '!=':
                r = x != y
            elif op == '<':
                r = x < y
            elif op == '>':
                r = x > y
            elif op == '<=':
                r = x <= y
            else:
                r = x >= y
        return _bool_out(r)
    with np.errstate(all='ignore'):
        if op == '+':
            r = a + b
        elif op == '-':
            r = a - b
        elif op == '*':
            r = a * b
        elif op == '/':
            if is_arr(b):
                r = np.where(b == 0, np.nan, a / np.where(b == 0, 1, b))
            else:
                r = (a / b) if b != 0 else (np.full_like(a, np.nan) if is_arr(a) else MISS)
        elif op == '^':
            r = np.power(a, b) if (is_arr(a) or is_arr(b)) else _pow(a, b)
        else:
            raise StataError('invalid operator', 198)
    if is_arr(r):
        r = np.asarray(r, dtype=float)
        r[~np.isfinite(r)] = np.nan
        return r
    return _clean(float(r))


def _pow(a, b):
    try:
        r = a ** b
        if isinstance(r, complex):
            return MISS
        return float(r)
    except (OverflowError, ZeroDivisionError):
        return MISS


def _eval_index(node, ctx):
    ip = ctx.ip
    base = node.base
    if isinstance(base, Name) and len(node.idx) == 2:
        M = ip.get_matrix(base.n)
        return _mat_el(M, node.idx, ctx)
    if isinstance(base, Call):
        val = evaluate(base, ctx)
        if isinstance(val, Matrix):
            return _mat_el(val, node.idx, ctx)
        raise StataError('invalid subscript', 198)
    if isinstance(base, Name):
        if base.n in ip.matrices and ctx.data is None:
            return _mat_el(ip.matrices[base.n], node.idx + [Num(1)], ctx)
        d = ctx.data if ctx.data is not None else ip.data
        var = d.resolve(base.n)
        col = d.cols[var]
        k = evaluate(node.idx[0], ctx)
        if ctx.data is None:
            k = int(k) if not math.isnan(k) else -1
            if 1 <= k <= d.n:
                return col[k - 1]
            return '' if d.isstr(var) else MISS
        k = np.broadcast_to(np.asarray(k, dtype=float), (d.n,))
        if ctx.by is not None:
            start, size = ctx.by[1], ctx.by[2]
        else:
            start, size = np.zeros(d.n, dtype=int), np.full(d.n, d.n)
        valid = (~np.isnan(k)) & (k >= 1) & (k <= size)
        kk = np.where(valid, k, 1).astype(int)
        absidx = start + kk - 1
        if d.isstr(var):
            out = np.array([col[j] if v else '' for j, v in zip(absidx, valid)], dtype=object)
        else:
            out = np.where(valid, col[absidx], np.nan)
        return out
    raise StataError('invalid subscript', 198)


class Matrix:
    def __init__(self, arr, rownames=None, colnames=None):
        self.a = np.atleast_2d(np.asarray(arr, dtype=float))
        r, c = self.a.shape
        self.rownames = list(rownames) if rownames else [f'r{i+1}' for i in range(r)]
        self.colnames = list(colnames) if colnames else [f'c{i+1}' for i in range(c)]


def _mat_el(M, idx, ctx):
    if not isinstance(M, Matrix):
        raise StataError('matrix not found', 111)
    vals = []
    for k, ix in enumerate(idx[:2]):
        if isinstance(ix, Str) or (isinstance(ix, Name) and False):
            names = M.rownames if k == 0 else M.colnames
            vals.append(names.index(ix.v) + 1)
        else:
            vals.append(evaluate(ix, ctx))
    i, j = int(vals[0]), int(vals[1])
    if i < 1 or j < 1 or i > M.a.shape[0] or j > M.a.shape[1]:
        return MISS
    return float(M.a[i - 1, j - 1])


# ----------------------------------------------------------------------- functions
def _vec(ctx, *xs):
    return xs


def _nlen(ctx, *xs):
    for x in xs:
        if is_arr(x):
            return len(x)
    return None


def _apply1(f, x):
    with np.errstate(all='ignore'):
        if is_arr(x):
            r = f(x.astype(float))
            r = np.asarray(r, dtype=float)
            r[~np.isfinite(r)] = np.nan
            return r
        if isinstance(x, float) and math.isnan(x):
            return MISS
        try:
            r = float(f(np.float64(x)))
        except (ValueError, OverflowError):
            return MISS
        return MISS if not math.isfinite(r) else r


def _round(x, y=1.0):
    with np.errstate(all='ignore'):
        if is_arr(x) or is_arr(y):
            r = np.floor(np.asarray(x, float) / y + 0.5) * y
            return r
        if math.isnan(x) or math.isnan(y):
            return MISS
        if y == 0:
            return x
        return math.floor(x / y + 0.5) * y


def _missing(ctx, args):
    out = None
    for a in args:
        if is_arr(a):
            m = (a == '') if a.dtype == object else np.isnan(a)
        else:
            m = (a == '') if isinstance(a, str) else (isinstance(a, float) and math.isnan(a))
        out = m if out is None else (out | m)
    return _bool_out(out)


def _rng_n(ctx):
    return ctx.N if ctx.data is not None else None


def _minmax(args, fn):
    arrs = [np.asarray(a, dtype=float) for a in args]
    st = np.stack(np.broadcast_arrays(*arrs))
    with np.errstate(all='ignore'), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        r = fn(st, axis=0)
    if r.ndim == 0:
        return float(r)
    return r


def _eval_call(node, ctx):
    f = node.f
    ip = ctx.ip
    if node.raw is not None:
        if f in ('tin', 'twithin'):
            from .dates import tin_mask
            return tin_mask(ip, ctx.data if ctx.data is not None else ip.data, node.raw, f == 'twithin')
        if f in ('tq', 'tm', 'td', 'th', 'tw', 'ty'):
            from .dates import literal
            v = literal(f, node.raw)
            if v is None:
                raise StataError(f'{node.raw} invalid date literal', 198)
            return float(v)
        return ip.get_return(f, node.raw)
    if f in ('rowsof', 'colsof', 'el') and node.args and isinstance(node.args[0], Name):
        M = ip.get_matrix(node.args[0].n)
        if f == 'rowsof':
            return float(M.a.shape[0])
        if f == 'colsof':
            return float(M.a.shape[1])
        i = int(evaluate(node.args[1], ctx))
        j = int(evaluate(node.args[2], ctx))
        return float(M.a[i - 1, j - 1])
    args = [evaluate(a, ctx) for a in node.args]
    n = _rng_n(ctx)
    F = FUNCS.get(f)
    if F is not None:
        return F(ctx, args)
    if f in ('runiform', 'uniform', 'rnormal', 'rbinomial', 'rchi2', 'rpoisson', 'runiformint', 'rexponential', 'rt', 'rlogistic', 'rgamma', 'rbeta', 'rnbinomial'):
        return ip.rand(f, args, n)
    if f == 'missing' or f == 'mi':
        return _missing(ctx, args)
    if f == 'scalar':
        nm = node.args[0].n if isinstance(node.args[0], Name) else None
        if nm and nm in ip.scalars:
            return ip.scalars[nm]
        return args[0]
    raise StataError(f'unknown function {f}()', 133)


def _cond(ctx, a):
    c, x, y = a[0], a[1], a[2]
    if is_arr(c) or is_arr(x) or is_arr(y):
        cc = np.asarray(c, dtype=float)
        res_str = _isstr(x) or _isstr(y)
        if res_str:
            n = max(len(z) for z in (c, x, y) if is_arr(z))
            cc = np.broadcast_to(cc, (n,))
            xx = np.broadcast_to(np.asarray(x, dtype=object), (n,))
            yy = np.broadcast_to(np.asarray(y, dtype=object), (n,))
            return np.array([xi if ci != 0 else yi for ci, xi, yi in zip(cc, xx, yy)], dtype=object)
        r = np.where(cc != 0, x, y).astype(float)
        if len(a) > 3:
            r = np.where(np.isnan(cc), a[3], r)
        return r
    if len(a) > 3 and isinstance(c, float) and math.isnan(c):
        return a[3]
    return x if c != 0 else y


def _inlist(ctx, a):
    x = a[0]
    if is_arr(x):
        out = np.zeros(len(x), dtype=bool)
        for v in a[1:]:
            out |= (x == v)
        return out.astype(float)
    return 1.0 if any(x == v for v in a[1:]) else 0.0


def _irecode(ctx, a):
    x = a[0]
    cuts = [float(v) for v in a[1:]]
    xa = np.asarray(x, float) if is_arr(x) else np.array([float(x)])
    out = np.full(xa.shape, np.nan)
    ok = ~np.isnan(xa)
    out[ok] = np.searchsorted(np.array(cuts), xa[ok], side='left')
    return out if is_arr(x) else float(out[0])


def _inrange(ctx, a):
    x, lo, hi = a
    x2 = _cmp_prep(x)
    with np.errstate(invalid='ignore'):
        if is_arr(x2):
            lo_ = -np.inf if (isinstance(lo, float) and math.isnan(lo)) else lo
            hi_ = np.inf if (isinstance(hi, float) and math.isnan(hi)) else hi
            r = (x2 >= lo_) & (x2 <= hi_) & ~np.isinf(x2) if not (isinstance(hi, float) and math.isnan(hi)) else (x2 >= lo_)
            return r.astype(float)
        return 1.0 if lo <= x2 <= hi else 0.0


def _sum_running(ctx, a):
    x = np.asarray(a[0], dtype=float)
    x = np.where(np.isnan(x), 0.0, x)
    if ctx.by is not None:
        out = np.empty_like(x)
        g = ctx.by[0]
        start = ctx.by[1]
        cs = np.cumsum(x)
        before = np.where(start > 0, cs[np.maximum(start - 1, 0)], 0.0)
        before = np.where(start > 0, before, 0.0)
        out = cs - before
        return out
    return np.cumsum(x)


def _strmap(fn):
    def f(ctx, a):
        x = a[0]
        if is_arr(x):
            return np.array([fn(str(v), *a[1:]) for v in x], dtype=object)
        return fn(str(x), *a[1:])
    return f


def _numstr(fn):
    def f(ctx, a):
        x = a[0]
        if is_arr(x):
            return np.array([fn(str(v), *a[1:]) for v in x], dtype=float)
        return float(fn(str(x), *a[1:]))
    return f


def _substr(s, n1, n2=None):
    n1 = int(n1)
    L = len(s)
    if n1 < 0:
        n1 = L + n1 + 1
    if n1 < 1:
        return ''
    if n2 is None or (isinstance(n2, float) and math.isnan(n2)):
        return s[n1 - 1:]
    return s[n1 - 1:n1 - 1 + int(n2)]


def _string(ctx, a):
    x = a[0]
    fmt = a[1] if len(a) > 1 else None

    def one(v):
        if isinstance(v, str):
            return v
        if fmt:
            return fmt_num(v, fmt).strip()
        return num_to_macro(v)
    if is_arr(x):
        return np.array([one(v) for v in x], dtype=object)
    return one(x)


def _real(s):
    try:
        return float(s.strip())
    except ValueError:
        return MISS


def _word(s, n):
    w = s.split()
    n = int(n)
    if n < 0:
        n = len(w) + n + 1
    return w[n - 1] if 1 <= n <= len(w) else ''


def _subinstr(s, a, b, n):
    if isinstance(n, float) and math.isnan(n):
        return s.replace(a, b)
    return s.replace(a, b, int(n))


def _strpos(s, t):
    return s.find(t) + 1


def _regexm(s, p):
    return 1.0 if re.search(p, s) else 0.0


def _fn2(f):
    def g(ctx, a):
        with np.errstate(all='ignore'):
            x = a[0]
            y = a[1]
            if is_arr(x) or is_arr(y):
                r = np.asarray(f(np.asarray(x, float), np.asarray(y, float)), dtype=float)
                r[~np.isfinite(r)] = np.nan
                return r
            try:
                r = float(f(x, y))
            except (ValueError, ZeroDivisionError):
                return MISS
            return r if math.isfinite(r) else MISS
    return g


def _fn3(f):
    def g(ctx, a):
        with np.errstate(all='ignore'):
            if any(is_arr(v) for v in a):
                r = np.asarray(f(*[np.asarray(v, float) for v in a]), dtype=float)
                r[~np.isfinite(r)] = np.nan
                return r
            r = float(f(*a))
            return r if math.isfinite(r) else MISS
    return g


def _mod(x, y):
    return x - y * np.floor(x / y)


from .dates import make_funcs as _date_funcs  # noqa: E402

FUNCS = {
    'abs': lambda c, a: _apply1(np.abs, a[0]),
    'ln': lambda c, a: _apply1(lambda v: np.where(v > 0, np.log(np.where(v > 0, v, 1)), np.nan), a[0]),
    'log': lambda c, a: _apply1(lambda v: np.where(v > 0, np.log(np.where(v > 0, v, 1)), np.nan), a[0]),
    'log10': lambda c, a: _apply1(lambda v: np.where(v > 0, np.log10(np.where(v > 0, v, 1)), np.nan), a[0]),
    'exp': lambda c, a: _apply1(np.exp, a[0]),
    'sqrt': lambda c, a: _apply1(lambda v: np.where(v >= 0, np.sqrt(np.abs(v)), np.nan), a[0]),
    'sin': lambda c, a: _apply1(np.sin, a[0]),
    'cos': lambda c, a: _apply1(np.cos, a[0]),
    'tan': lambda c, a: _apply1(np.tan, a[0]),
    'atan': lambda c, a: _apply1(np.arctan, a[0]),
    'floor': lambda c, a: _apply1(np.floor, a[0]),
    'ceil': lambda c, a: _apply1(np.ceil, a[0]),
    'int': lambda c, a: _apply1(np.trunc, a[0]),
    'trunc': lambda c, a: _apply1(np.trunc, a[0]),
    'round': lambda c, a: _round(a[0], a[1] if len(a) > 1 else 1.0),
    'mod': _fn2(_mod),
    'min': lambda c, a: _minmax(a, np.nanmin),
    'max': lambda c, a: _minmax(a, np.nanmax),
    'cond': _cond,
    'inlist': _inlist,
    'irecode': _irecode,
    'inrange': _inrange,
    'sum': _sum_running,
    'normal': lambda c, a: _apply1(sps.norm.cdf, a[0]),
    'normalden': lambda c, a: (_apply1(sps.norm.pdf, a[0]) if len(a) == 1 else
                               _fn3(lambda x, m, s: sps.norm.pdf(x, m, s))(c, a)),
    'invnormal': lambda c, a: _apply1(sps.norm.ppf, a[0]),
    'logit': lambda c, a: _apply1(lambda p: np.log(p / (1 - p)), a[0]),
    'invlogit': lambda c, a: _apply1(lambda x: 1 / (1 + np.exp(-x)), a[0]),
    'ttail': _fn2(lambda df, t: sps.t.sf(t, df)),
    'invttail': _fn2(lambda df, p: sps.t.isf(p, df)),
    't': _fn2(lambda df, t: sps.t.cdf(t, df)),
    'invt': _fn2(lambda df, p: sps.t.ppf(p, df)),
    'chi2': _fn2(lambda df, x: sps.chi2.cdf(x, df)),
    'chi2tail': _fn2(lambda df, x: sps.chi2.sf(x, df)),
    'invchi2': _fn2(lambda df, p: sps.chi2.ppf(p, df)),
    'invchi2tail': _fn2(lambda df, p: sps.chi2.isf(p, df)),
    'F': _fn3(lambda d1, d2, f: sps.f.cdf(f, d1, d2)),
    'Ftail': _fn3(lambda d1, d2, f: sps.f.sf(f, d1, d2)),
    'invFtail': _fn3(lambda d1, d2, p: sps.f.isf(p, d1, d2)),
    'lngamma': lambda c, a: _apply1(spf.gammaln, a[0]),
    'comb': _fn2(lambda n, k: spf.comb(n, k)),
    'float': lambda c, a: _apply1(lambda v: np.float32(v).astype(np.float64), a[0]),
    'string': _string,
    'strofreal': _string,
    'real': _numstr(_real),
    'substr': _strmap(_substr),
    'usubstr': _strmap(_substr),
    'strlen': _numstr(len),
    'length': _numstr(len),
    'ustrlen': _numstr(len),
    'upper': _strmap(lambda s: s.upper()),
    'strupper': _strmap(lambda s: s.upper()),
    'lower': _strmap(lambda s: s.lower()),
    'strlower': _strmap(lambda s: s.lower()),
    'trim': _strmap(lambda s: s.strip()),
    'strtrim': _strmap(lambda s: s.strip()),
    'ltrim': _strmap(lambda s: s.lstrip()),
    'rtrim': _strmap(lambda s: s.rstrip()),
    'word': _strmap(_word),
    'wordcount': _numstr(lambda s: len(s.split())),
    'subinstr': _strmap(_subinstr),
    'strpos': _numstr(_strpos),
    'regexm': _numstr(_regexm),
    'strmatch': _numstr(lambda s, p: 1.0 if re.fullmatch(p.replace('*', '.*').replace('?', '.'), s) else 0.0),
    'char': lambda c, a: chr(int(a[0])),
    'proper': _strmap(lambda s: s.title()),
    'reverse': _strmap(lambda s: s[::-1]),
    'sign': lambda c, a: _apply1(np.sign, a[0]),
}


def eval_expr(s, ctx):
    return evaluate(parse(s), ctx)

FUNCS.update(_date_funcs())
