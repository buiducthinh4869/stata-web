"""Core interpreter: do-file preprocessing, macros, control flow, dispatch."""
import datetime
import math
import os
import re

import numpy as np

from . import expr as E
from .data import Data
from .errors import StataError, ExitDo
from .fmt import fmt_num, num_to_macro, display_num
from .syntax import split_top, find_top, unquote, find_cq_end

COMMANDS = {}
ALIASES = {}


def command(*names):
    def deco(fn):
        for n in names:
            COMMANDS[n] = fn
        return fn
    return deco


def alias(full, *shorts):
    for s in shorts:
        ALIASES[s] = full


# ---------------------------------------------------------------- output
class Output:
    def __init__(self):
        self.chunks = []
        self.listeners = []
        self.log = None

    def write(self, text, style='txt'):
        if not text:
            return
        if self.chunks and self.chunks[-1][0] == style:
            self.chunks[-1] = (style, self.chunks[-1][1] + text)
        else:
            self.chunks.append((style, text))
        for f in self.listeners:
            f(style, text)
        if self.log is not None:
            self.log.append(text)

    def text(self):
        return ''.join(t for _, t in self.chunks)

    def take(self):
        c = self.chunks
        self.chunks = []
        return c


# ---------------------------------------------------------------- returns
class Results:
    def __init__(self):
        self.scalars = {}
        self.macros = {}
        self.matrices = {}

    def clear(self):
        self.scalars.clear()
        self.macros.clear()
        self.matrices.clear()


# ---------------------------------------------------------------- preprocessing
class Line:
    __slots__ = ('text', 'raw', 'lineno')

    def __init__(self, text, raw, lineno):
        self.text, self.raw, self.lineno = text, raw, lineno


def strip_block_comments(src):
    out = []
    i, n, depth = 0, len(src), 0
    inq = False
    while i < n:
        if depth == 0:
            c = src[i]
            if c == '\n':
                inq = False
                out.append(c)
                i += 1
                continue
            if c == '"' and not inq:
                # only track quotes for /* detection in obvious strings
                inq = True
                out.append(c)
                i += 1
                continue
            if c == '"' and inq:
                inq = False
                out.append(c)
                i += 1
                continue
            if not inq and src.startswith('/*', i):
                depth = 1
                i += 2
                continue
            out.append(c)
            i += 1
        else:
            if src.startswith('/*', i):
                depth += 1
                i += 2
            elif src.startswith('*/', i):
                depth -= 1
                i += 2
            else:
                if src[i] == '\n':
                    out.append('\n')
                i += 1
    return ''.join(out)


def _strip_line_comment(line):
    """Remove // comments (preceded by whitespace or at start), respecting quotes."""
    if line.lstrip().startswith('//'):
        return '', False
    i, n = 0, len(line)
    inq = False
    while i < n:
        c = line[i]
        if c == '"':
            inq = not inq
        elif not inq and line.startswith('//', i) and (i == 0 or line[i - 1].isspace()):
            if line.startswith('///', i):
                return line[:i], True
            return line[:i], False
        i += 1
    return line, False


def preprocess(src):
    src = src.replace('\r\n', '\n').replace('\r', '\n')
    if src.startswith('﻿'):
        src = src[1:]
    src = strip_block_comments(src)
    rawlines = src.split('\n')
    lines = []
    buf, rawbuf, start = '', [], None
    delim_semicolon = False
    for no, ln in enumerate(rawlines, 1):
        if not buf and ln.strip().startswith('*'):
            continue
        stripped = ln.strip()
        if re.match(r'^#d(elimit)?\s+;', stripped):
            delim_semicolon = True
            continue
        if re.match(r'^#d(elimit)?\s+cr', stripped):
            delim_semicolon = False
            continue
        code, cont = _strip_line_comment(ln)
        if start is None:
            start = no
        rawbuf.append(ln)
        if delim_semicolon:
            buf += ' ' + code
            while ';' in buf:
                k = buf.index(';')
                piece = buf[:k].strip()
                if piece:
                    lines.append(Line(piece, '\n'.join(rawbuf), start))
                buf = buf[k + 1:]
                rawbuf, start = [], None
            continue
        if cont:
            buf += code.rstrip() + ' '
            continue
        buf += code
        if buf.strip():
            lines.append(Line(buf.strip(), '\n'.join(r for r in rawbuf), start))
        buf, rawbuf, start = '', [], None
    if buf.strip():
        lines.append(Line(buf.strip(), '\n'.join(rawbuf), start or 0))
    return lines


# ---------------------------------------------------------------- interpreter
class Interp:
    def __init__(self, workdir='/work', echo=True):
        self.data = Data()
        self.globals = {'S_level': '95'}
        self.frames = [{}]
        self.scalars = {}
        self.matrices = {}
        self.r = Results()
        self.e = Results()
        self.est = None              # current estimation object
        self.stored = {}             # estimates store
        self.eststo_order = []
        self.rc = 0
        self.out = Output()
        self.quiet = 0
        self.noisy = 0
        self.preserved = []
        self.rng = np.random.default_rng(123456789)
        self.seed = 123456789
        self.linesize = 80
        self.graphs = {}
        self.graph_order = []
        self.current_graph = None
        self.graph_listeners = []
        self.files = {}
        self.postfiles = {}
        self.tmpcount = 0
        self.workdir = workdir
        self.echo = echo
        self.depth = 0
        self.log_file = None
        self.label_store = {}
        self.scheme = 's2color'
        self.loop_depth = 0
        os.makedirs(workdir, exist_ok=True)
        try:
            os.chdir(workdir)
        except OSError:
            pass

    # ---------------- locals
    @property
    def locals(self):
        return self.frames[-1]

    # ---------------- output helpers
    def write(self, text, style='txt'):
        if self.quiet and not self.noisy:
            return
        self.out.write(text, style)

    def txt(self, text):
        self.write(text, 'txt')

    def res(self, text):
        self.write(text, 'res')

    def err(self, text):
        self.out.write(text, 'err')

    def nl(self):
        self.write('\n')

    # ---------------- macro expansion
    def expand(self, s, depth=0):
        if depth > 50:
            raise StataError('macro expansion too deep', 198)
        out = s
        for _ in range(200):
            new = self._expand_once(out)
            if new == out:
                break
            out = new
        return out

    _GLOB_RE = re.compile(r'\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))')

    def _expand_once(self, s):
        # innermost local macro: ` followed by non-` non-' chars then '
        i = 0
        n = len(s)
        res = []
        changed = False
        while i < n:
            c = s[i]
            if c == '`' and i + 1 < n and s[i + 1] == '"':
                # compound quote opening: copy marker
                res.append('`"')
                i += 2
                continue
            if c == '`':
                # find closing ' with no intervening `
                j = i + 1
                while j < n and s[j] not in "`'":
                    j += 1
                if j < n and s[j] == "'":
                    name = s[i + 1:j]
                    if name.startswith('"'):
                        res.append(c)
                        i += 1
                        continue
                    res.append(self._local_ref(name))
                    i = j + 1
                    changed = True
                    continue
                res.append(c)
                i += 1
                continue
            if c == '$':
                m = self._GLOB_RE.match(s, i)
                if m:
                    nm = m.group(1) or m.group(2)
                    res.append(self.globals.get(nm, ''))
                    i = m.end()
                    changed = True
                    continue
            res.append(c)
            i += 1
        return ''.join(res) if changed else s

    def _local_ref(self, name):
        if name.startswith('='):
            v = E.eval_expr(name[1:], E.Ctx(self))
            return v if isinstance(v, str) else num_to_macro(v)
        if name.startswith(':'):
            return self.extended_fn(name[1:].strip())
        if name.startswith('++'):
            nm = name[2:]
            v = self._macro_num(nm) + 1
            self.locals[nm] = num_to_macro(v)
            return self.locals[nm]
        if name.startswith('--'):
            nm = name[2:]
            v = self._macro_num(nm) - 1
            self.locals[nm] = num_to_macro(v)
            return self.locals[nm]
        if name.endswith('++'):
            nm = name[:-2]
            old = self.locals.get(nm, '')
            self.locals[nm] = num_to_macro(self._macro_num(nm) + 1)
            return old
        return self.locals.get(name, '')

    def _macro_num(self, nm):
        v = self.locals.get(nm, '0') or '0'
        try:
            return float(v)
        except ValueError:
            return 0.0

    def extended_fn(self, s):
        m = re.match(r'^word\s+count\s*(.*)$', s, re.S)
        if m:
            return str(len(split_top(m.group(1))))
        m = re.match(r'^word\s+(\d+)\s+of\s*(.*)$', s, re.S)
        if m:
            w = split_top(m.group(2))
            k = int(m.group(1))
            return unquote(w[k - 1]) if 1 <= k <= len(w) else ''
        m = re.match(r'^(rownames|colnames)\s+(\S+)', s)
        if m:
            M = self.get_matrix(m.group(2))
            return ' '.join(M.rownames if m.group(1) == 'rownames' else M.colnames)
        m = re.match(r'^(rowsof|colsof)\s+(\S+)', s)
        if m:
            M = self.get_matrix(m.group(2))
            return str(M.a.shape[0 if m.group(1) == 'rowsof' else 1])
        m = re.match(r'^var(?:iable)?\s+label\s+(\S+)', s)
        if m:
            v = self.data.resolve(m.group(1))
            return self.data.varlab.get(v, '')
        m = re.match(r'^val(?:ue)?\s+label\s+(\S+)', s)
        if m:
            v = self.data.resolve(m.group(1))
            return self.data.vallab.get(v, '')
        m = re.match(r'^label\s+(\S+)\s+(\S+)', s)
        if m:
            lab = self.data.labels.get(m.group(1), {})
            try:
                key = float(m.group(2))
            except ValueError:
                return m.group(2)
            return lab.get(key, m.group(2))
        m = re.match(r'^type\s+(\S+)', s)
        if m:
            return self.data.types[self.data.resolve(m.group(1))]
        m = re.match(r'^format\s+(\S+)', s)
        if m:
            return self.data.formats[self.data.resolve(m.group(1))]
        m = re.match(r'^list\s+sizeof\s+(\S+)', s)
        if m:
            return str(len(self.locals.get(m.group(1), '').split()))
        m = re.match(r'^list\s+uniq\s+(\S+)', s)
        if m:
            seen = []
            for w in self.locals.get(m.group(1), '').split():
                if w not in seen:
                    seen.append(w)
            return ' '.join(seen)
        m = re.match(r'^piece\s+(\d+)\s+(\d+)\s+of\s+(.*)$', s)
        if m:
            k, w, txt = int(m.group(1)), int(m.group(2)), unquote(m.group(3))
            return txt[(k - 1) * w:k * w]
        m = re.match(r'^(?:e|r)\((\w+)\)', s)
        if m:
            v = self.get_return(s[0], m.group(1))
            return v if isinstance(v, str) else num_to_macro(v)
        raise StataError(f'extended macro function not supported: {s}', 198)

    # ---------------- returns
    def get_return(self, kind, raw):
        if kind == 'c':
            return self._cvalue(raw)
        if kind == 's':
            return self.globals.get('S_' + raw, '')
        R = self.r if kind == 'r' else self.e
        if raw in R.scalars:
            return R.scalars[raw]
        if raw in R.macros:
            return R.macros[raw]
        if raw in R.matrices:
            return R.matrices[raw]
        if kind == 'e' and raw == 'sample':
            return self.e_sample()
        return E.MISS if raw not in ('cmd', 'depvar', 'title', 'vce', 'clustvar') else ''

    def e_sample(self):
        s = self.e.macros.get('__sample__')
        if s is None or self.est is None:
            return np.zeros(self.data.n)
        m = self.est.get('sample')
        if m is None or len(m) != self.data.n:
            return np.zeros(self.data.n)
        return m.astype(float)

    def _cvalue(self, raw):
        now = datetime.datetime.now()
        vals = {
            'current_date': now.strftime('%e %b %Y').strip(),
            'current_time': now.strftime('%H:%M:%S'),
            'pi': math.pi, 'N': float(self.data.n), 'k': float(len(self.data.vars)),
            'version': 18.0, 'stata_version': 18.0, 'linesize': float(self.linesize),
            'pwd': os.getcwd(), 'level': 95.0, 'seed': str(self.seed), 'filename': self.data.filename,
            'maxvar': 5000.0, 'alpha': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'os': 'Unix', 'dirsep': '/',
            'mindouble': -8.9884656743e+307, 'maxdouble': 8.9884656743e+307, 'epsdouble': 2.220446e-16,
            'rc': float(self.rc), 'scheme': self.scheme, 'more': 'off', 'changed': 0.0,
        }
        if raw in vals:
            return vals[raw]
        return E.MISS

    def get_coef(self, kind, name):
        if self.est is None:
            raise StataError('last estimates not found', 301)
        names = self.est['names']
        key = self.match_coef(name)
        if key is None:
            raise StataError(f'[{name}] not found', 111)
        j = names.index(key)
        if kind == '_b':
            return float(self.est['b'][j])
        V = self.est['V']
        v = V[j, j]
        return float(math.sqrt(v)) if v > 0 else 0.0

    def match_coef(self, name, est=None):
        est = est or self.est
        names = est['names']
        name = name.strip()
        if ':' in name:
            name = name.split(':', 1)[1]
        if name in names:
            return name
        canon = canon_coef(name)
        for n in names:
            if canon_coef(n) == canon:
                return n
        if self.data is not None and '#' not in name and '.' not in name:
            v = self.data.resolve(name, required=False)
            if v and v in names:
                return v
        return None

    def get_matrix(self, name):
        if name in self.matrices:
            return self.matrices[name]
        raise StataError(f'matrix {name} not found', 111)

    # ---------------- random numbers
    def rand(self, f, args, n):
        size = n
        g = self.rng
        a = [float(x) if not isinstance(x, np.ndarray) else x for x in args]
        if f in ('runiform', 'uniform'):
            if len(a) == 2:
                r = g.uniform(a[0], a[1], size)
            else:
                r = g.random(size)
        elif f == 'runiformint':
            r = g.integers(int(a[0]), int(a[1]) + 1, size).astype(float)
        elif f == 'rnormal':
            m = a[0] if len(a) > 0 else 0.0
            s = a[1] if len(a) > 1 else 1.0
            r = g.normal(m, s, size)
        elif f == 'rbinomial':
            nn = np.asarray(a[0])
            p = np.asarray(a[1], dtype=float)
            p = np.clip(np.nan_to_num(p, nan=0.0), 0, 1)
            r = g.binomial(nn.astype(int), p, size).astype(float)
        elif f == 'rchi2':
            r = g.chisquare(a[0], size)
        elif f == 'rpoisson':
            r = g.poisson(a[0], size).astype(float)
        elif f == 'rexponential':
            r = g.exponential(a[0], size)
        elif f == 'rt':
            r = g.standard_t(a[0], size)
        elif f == 'rlogistic':
            m = a[0] if a else 0.0
            s = a[1] if len(a) > 1 else 1.0
            r = g.logistic(m, s, size)
        elif f == 'rgamma':
            r = g.gamma(a[0], a[1] if len(a) > 1 else 1.0, size)
        elif f == 'rbeta':
            r = g.beta(a[0], a[1], size)
        elif f == 'rnbinomial':
            # Stata: number of failures before the n-th success, success probability p
            r = g.negative_binomial(a[0], np.clip(np.asarray(a[1], dtype=float), 1e-12, 1), size).astype(float)
        else:
            raise StataError(f'unknown function {f}()', 133)
        if size is None:
            return float(r)
        return np.asarray(r, dtype=float)

    # ---------------- time series operators
    def ts_op(self, ctx, op, k, var):
        d = ctx.data if ctx.data is not None else self.data
        if not d.ts or not d.ts.get('time'):
            raise StataError('time variable not set, use tsset varname ...', 111)
        var = d.resolve(var)
        x = d.cols[var].astype(float)
        if op == 'S':
            op = 'D'
        if op == 'D':
            out = x.copy()
            for _ in range(k):
                out = out - self._lag(d, out, 1)
            return out
        lag = k if op == 'L' else -k
        return self._lag(d, x, lag)

    def _lag(self, d, x, lag):
        tv = d.cols[d.ts['time']].astype(float)
        delta = d.ts.get('delta', 1.0)
        pv = d.cols[d.ts['panel']].astype(float) if d.ts.get('panel') else np.zeros(d.n)
        big = 1e7
        key = pv * big + tv
        target = pv * big + (tv - lag * delta)
        order = np.argsort(key, kind='stable')
        sk = key[order]
        pos = np.searchsorted(sk, target)
        pos = np.clip(pos, 0, len(sk) - 1)
        found = (sk[pos] == target) & ~np.isnan(target)
        out = np.full(d.n, np.nan)
        out[found] = x[order[pos[found]]]
        return out

    # ---------------- by groups
    def by_groups(self, byvars, data=None):
        d = data or self.data
        n = d.n
        if n == 0:
            z = np.zeros(0, dtype=int)
            return (z, z, z, z)
        keys = []
        for v in byvars:
            a = d.cols[v]
            if d.isstr(v):
                keys.append(np.array([str(x) for x in a], dtype=object))
            else:
                keys.append(np.where(np.isnan(a), np.inf, a))
        newgrp = np.zeros(n, dtype=bool)
        newgrp[0] = True
        for kk in keys:
            if kk.dtype == object:
                newgrp[1:] |= kk[1:] != kk[:-1]
            else:
                newgrp[1:] |= kk[1:] != kk[:-1]
        gid = np.cumsum(newgrp) - 1
        starts_g = np.flatnonzero(newgrp)
        sizes_g = np.diff(np.append(starts_g, n))
        start = starts_g[gid]
        size = sizes_g[gid]
        pos = np.arange(n) - start
        return (gid, start, size, pos)

    def check_sorted(self, byvars):
        d = self.data
        if d.sortedby[:len(byvars)] == list(byvars):
            return
        # verify actual order
        prev = None
        ok = True
        keys = [d.cols[v] for v in byvars]
        rows = list(zip(*[(np.where(np.isnan(k), np.inf, k) if k.dtype != object else k) for k in keys]))
        for r in rows:
            if prev is not None and r < prev:
                ok = False
                break
            prev = r
        if not ok:
            raise StataError('not sorted', 5)

    # ---------------- expressions
    def eval(self, s, data=None, by=None):
        ctx = E.Ctx(self, data, by)
        return E.eval_expr(s, ctx)

    def eval_scalar(self, s):
        v = E.eval_expr(s, E.Ctx(self))
        if isinstance(v, np.ndarray):
            v = v[0] if len(v) else E.MISS
        return v

    def mark(self, p, data=None, by=None, novars=None):
        """Sample mask from parsed if/in."""
        d = data or self.data
        m = np.ones(d.n, dtype=bool)
        if p.if_exp:
            v = E.eval_expr(p.if_exp, E.Ctx(self, d, by))
            if isinstance(v, np.ndarray):
                m &= (v != 0)
            else:
                m &= bool(v != 0)
        if p.in_range:
            m &= self.in_mask(p.in_range, d)
        return m

    def in_mask(self, rng, d=None):
        d = d or self.data
        rng = rng.strip()
        m = np.zeros(d.n, dtype=bool)

        def conv(t):
            t = t.strip()
            if t in ('f', 'F'):
                return 1
            if t in ('l', 'L'):
                return d.n
            v = int(float(t))
            return v if v > 0 else d.n + v + 1
        if '/' in rng:
            a, b = rng.split('/', 1)
            a, b = conv(a), conv(b)
            m[max(a, 1) - 1:min(b, d.n)] = True
        else:
            a = conv(rng)
            if 1 <= a <= d.n:
                m[a - 1] = True
        return m

    # ---------------- execution
    def run(self, src, echo=None):
        """Run a do-file text. Returns rc."""
        lines = preprocess(src)
        old = self.echo
        if echo is not None:
            self.echo = echo
        try:
            self.exec_block(lines, 0, len(lines), top=True)
            return 0
        except ExitDo:
            return 0
        except StataError as e:
            self.err(e.msg + '\n')
            self.out.write(f'r({e.rc});\n\n', 'err')
            self.rc = e.rc
            return e.rc
        except Exception as e:  # noqa: BLE001  internal error -> Stata-like failure
            import traceback
            self.err(f'internal error: {type(e).__name__}: {e}\n')
            self.internal_tb = traceback.format_exc()
            self.out.write('r(198);\n\n', 'err')
            self.rc = 198
            return 198
        finally:
            self.echo = old

    def echo_line(self, line, prefix='. '):
        if not self.echo or self.quiet:
            return
        raw = line.raw.split('\n')
        txt = prefix + raw[0].strip() + '\n'
        for r in raw[1:]:
            txt += '> ' + r.strip() + '\n'
        self.out.write(txt, 'cmd')

    def find_block_end(self, lines, i):
        """lines[i] opens a block; return index of the closing '}' line."""
        depth = 0
        for j in range(i, len(lines)):
            t = lines[j].text.strip()
            opens = t.endswith('{')
            closes = t.startswith('}')
            if j == i:
                depth = 1
                continue
            if closes:
                depth -= 1
                if depth == 0:
                    return j
            if opens:
                depth += 1
        raise StataError('unexpected end of file (missing })', 198)

    def exec_block(self, lines, i0, i1, top=False):
        i = i0
        while i < i1:
            ln = lines[i]
            t = ln.text.strip()
            if t == '}' or t.startswith('}') and t[1:].strip() == '':
                i += 1
                continue
            if t.endswith('{'):
                j = self.find_block_end(lines, i)
                if top:
                    self.echo_block(lines, i, j)
                i = self.exec_compound(lines, i, j, top)
                continue
            if re.match(r'^(input|inpu|inp)\s', t) and not re.search(r'\bend\s*$', t):
                j = i + 1
                while j < i1 and lines[j].text.strip() != 'end':
                    j += 1
                if top:
                    self.echo_line(ln)
                    if self.echo and not self.quiet:
                        for x in range(i + 1, min(j + 1, i1)):
                            self.out.write(f'{x - i:>3}. {lines[x].text.strip()}\n', 'cmd')
                from .cmds_data import do_input
                do_input(self, self.expand(t.split(None, 1)[1]), [lines[x].text for x in range(i + 1, min(j, i1))])
                if top and self.echo and not self.quiet:
                    self.out.write('\n', 'txt')
                i = j + 1
                continue
            if top:
                self.echo_line(ln)
            self.exec_line(ln.text)
            if top and self.echo and not self.quiet:
                self.out.write('\n', 'txt')
            i += 1

    def echo_block(self, lines, i, j):
        if not self.echo or self.quiet:
            return
        self.echo_line(lines[i])
        k = 2
        for x in range(i + 1, j + 1):
            for r in lines[x].raw.split('\n'):
                self.out.write(f'{k:>3}. {r.strip()}\n', 'cmd')
                k += 1

    def exec_compound(self, lines, i, j, top):
        header = lines[i].text.strip()[:-1].strip()
        after = j + 1
        # handle "} else {" on closing line
        closing = lines[j].text.strip()
        m = re.match(r'^(quietly|qui|quietl|quiet|noisily|noi|capture|cap|capt)\s*:?\s*$', header)
        if m:
            kind = m.group(1)
            if kind.startswith('qui'):
                self.quiet += 1
                try:
                    self.exec_block(lines, i + 1, j)
                finally:
                    self.quiet -= 1
            elif kind.startswith('noi'):
                self.noisy += 1
                try:
                    self.exec_block(lines, i + 1, j)
                finally:
                    self.noisy -= 1
            else:
                self.quiet += 1
                try:
                    self.exec_block(lines, i + 1, j)
                    self.rc = 0
                except StataError as e:
                    self.rc = e.rc
                finally:
                    self.quiet -= 1
            return after
        hx = self.expand(header)
        word = hx.split(None, 1)[0] if hx.split() else ''
        if word in ('foreach',):
            self.do_foreach(hx, lines, i + 1, j)
            return after
        if word in ('forvalues', 'forval', 'forv', 'forva', 'forvalu', 'forvalue'):
            self.do_forvalues(hx, lines, i + 1, j)
            return after
        if word == 'while':
            cond = hx[len(word):].strip()
            guard = 0
            while True:
                c = self.eval_scalar(self.expand(header[len(word):].strip()))
                if not (c != 0):
                    break
                self.loop_depth += 1
                try:
                    self.exec_block(lines, i + 1, j)
                finally:
                    self.loop_depth -= 1
                guard += 1
                if guard > 1000000:
                    raise StataError('infinite loop', 198)
            return after
        if word == 'if' or (word == 'else' and hx.split()[1:2] == ['if']):
            return self.do_if(lines, i, j)
        if word == 'else':
            # stray else (executed as part of if)
            return after
        if word in ('program',):
            return after
        # unknown block like "mata {"
        return after

    def do_if(self, lines, i, j):
        header = lines[i].text.strip()[:-1].strip()
        hx = self.expand(header)
        cond_s = re.sub(r'^(else\s+)?if\s*', '', hx)
        c = self.eval_scalar(cond_s)
        taken = (c != 0)
        if taken:
            self.exec_block(lines, i + 1, j)
        k = j + 1
        # else chain: closing line could be "} else {" or next line starts with else
        nxt = lines[j].text.strip()
        if nxt.startswith('}') and nxt[1:].strip().startswith('else'):
            # same-line else
            rest = nxt[1:].strip()
            # treat like an else block starting at j
            j2 = self.find_block_end(lines, j)
            if not taken:
                if re.match(r'^else\s+if\b', rest):
                    fake = lines[j]
                    lines[j] = Line(rest, fake.raw, fake.lineno)
                    try:
                        return self.do_if(lines, j, j2)
                    finally:
                        lines[j] = fake
                self.exec_block(lines, j + 1, j2)
            return j2 + 1
        while k < len(lines):
            t = lines[k].text.strip()
            if not t.startswith('else'):
                break
            if t.endswith('{'):
                j2 = self.find_block_end(lines, k)
                if not taken:
                    if re.match(r'^else\s+if\b', t):
                        return self.do_if(lines, k, j2)
                    self.exec_block(lines, k + 1, j2)
                    return j2 + 1
                k = j2 + 1
                if re.match(r'^else\s+if\b', t):
                    continue
                return k
            else:
                if not taken:
                    self.exec_line(t[4:].strip())
                return k + 1
        return k

    def do_foreach(self, hx, lines, a, b):
        m = re.match(r'^foreach\s+(\w+)\s+(in|of\s+(?:local|global|varlist|newlist|numlist))\s*(.*)$', hx, re.S)
        if not m:
            raise StataError('invalid syntax', 198)
        name, kind, rest = m.group(1), m.group(2), m.group(3)
        kind = kind.split()[-1]
        if kind == 'in':
            items = [unquote(x) for x in split_top(rest)]
        elif kind == 'local':
            items = [unquote(x) for x in split_top(self.locals.get(rest.strip(), ''))]
        elif kind == 'global':
            items = [unquote(x) for x in split_top(self.globals.get(rest.strip(), ''))]
        elif kind == 'varlist':
            items = self.data.expand_varlist(split_top(rest))
        elif kind == 'numlist':
            from .syntax import parse_numlist
            items = [num_to_macro(x) for x in parse_numlist(rest)]
        else:
            items = split_top(rest)
        self.loop_depth += 1
        try:
            for it in items:
                self.locals[name] = it
                self.exec_block(lines, a, b)
        finally:
            self.loop_depth -= 1

    def do_forvalues(self, hx, lines, a, b):
        m = re.match(r'^\w+\s+(\w+)\s*=\s*(.+)$', hx)
        if not m:
            raise StataError('invalid syntax', 198)
        from .syntax import parse_numlist
        rng = m.group(2).strip()
        vals = parse_numlist(rng)
        self.loop_depth += 1
        try:
            for v in vals:
                self.locals[m.group(1)] = num_to_macro(v)
                self.exec_block(lines, a, b)
        finally:
            self.loop_depth -= 1

    # ---------------- single command
    def exec_line(self, text):
        s = self.expand(text).strip()
        if not s:
            return
        self.dispatch(s)

    def dispatch(self, s, by=None):
        s = s.strip()
        if not s:
            return
        # prefixes
        m = re.match(r'^(quietly|quietl|quiet|quie|qui|noisily|noisil|noisi|nois|noi|capture|captur|captu|capt|cap)\b\s*:?\s*(.*)$', s, re.S)
        if m:
            kind, rest = m.group(1), m.group(2)
            if kind.startswith('qui'):
                self.quiet += 1
                try:
                    self.dispatch(rest, by)
                finally:
                    self.quiet -= 1
            elif kind.startswith('noi'):
                self.noisy += 1
                try:
                    self.dispatch(rest, by)
                finally:
                    self.noisy -= 1
            else:
                self.quiet += 1
                try:
                    self.dispatch(rest, by)
                    self.rc = 0
                except StataError as e:
                    self.rc = e.rc
                    if re.match(r'^(noisily|noisil|noisi|nois|noi)\b', rest):
                        self.err(e.msg + '\n')
                        self.out.write(f'r({e.rc});\n', 'err')
                except Exception:  # noqa: BLE001
                    self.rc = 198
                finally:
                    self.quiet -= 1
            return
        m = re.match(r'^(bysort|bys|by)\s+([^:]*):\s*(.*)$', s, re.S)
        if m:
            kind, spec, rest = m.group(1), m.group(2), m.group(3)
            spec = spec.split(',')[0]
            pm = re.match(r'^([^()]*)(?:\(([^)]*)\))?\s*$', spec.strip())
            byvars = self.data.expand_varlist(pm.group(1).split())
            sortvars = self.data.expand_varlist(pm.group(2).split()) if pm.group(2) else []
            if kind != 'by':
                self.data.sort(byvars + sortvars)
            else:
                self.check_sorted(byvars + sortvars)
            return self.dispatch(rest, by=byvars)
        m = re.match(r'^xi\s*:\s*(.*)$', s, re.S)
        if m:
            return self.dispatch(m.group(1), by)
        m = re.match(r'^version\s+[\d.]+\s*:\s*(.*)$', s, re.S)
        if m:
            return self.dispatch(m.group(1), by)
        m = re.match(r'^eststo\s+(\w+)\s*:\s*(.*)$', s, re.S)
        if m:
            self.dispatch(m.group(2), by)
            self.store_estimates(m.group(1), eststo=True)
            return
        m = re.match(r'^eststo\s*:\s*(.*)$', s, re.S)
        if m:
            self.dispatch(m.group(1), by)
            self.store_estimates(f'est{len(self.eststo_order)+1}', eststo=True)
            return
        if s.startswith('ritest ') and find_top(s, ':') > 0:
            k = find_top(s, ':')
            return COMMANDS['ritest'](self, 'ritest', s[7:k], None, inner=s[k + 1:].strip())
        if s.startswith('if ') or s.startswith('if('):
            node, rest = E.parse_prefix(s[2:].strip())
            c = E.evaluate(node, E.Ctx(self))
            if isinstance(c, np.ndarray):
                c = c[0] if len(c) else 0
            if c != 0:
                self.dispatch(rest, by)
            return
        if s.startswith('else'):
            return
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)(.*)$', s, re.S)
        if not m:
            if s.startswith('}') or s.startswith('{'):
                return
            raise StataError(f'unrecognized command: {s.split()[0]}', 199)
        cmd, rest = m.group(1), m.group(2)
        if rest and rest[0] not in ' \t,(":=' and cmd not in ('di', 'display'):
            # e.g. "gen x" glued forms are fine; for weird tokens keep going
            pass
        full = ALIASES.get(cmd, cmd)
        fn = COMMANDS.get(full)
        if fn is None:
            # unique abbreviation of known command
            cands = [c for c in COMMANDS if c.startswith(cmd) and len(cmd) >= 3]
            if len(cands) == 1:
                full = cands[0]
                fn = COMMANDS[full]
        if fn is None:
            raise StataError(f'command {cmd} is unrecognized', 199)
        return fn(self, full, rest, by)

    # ---------------- estimates
    def store_estimates(self, name, eststo=False, title=None):
        if self.est is None:
            raise StataError('last estimation results not found, nothing to store', 301)
        import copy as _c
        est = dict(self.est)
        est['e_scalars'] = dict(self.e.scalars)
        est['e_macros'] = dict(self.e.macros)
        est['e_matrices'] = {k: v for k, v in self.e.matrices.items()}
        est['_name'] = name
        if title:
            est['_title'] = title
        self.stored[name] = est
        if eststo and name not in self.eststo_order:
            self.eststo_order.append(name)

    def restore_estimates(self, name):
        if name not in self.stored:
            raise StataError(f'estimation result {name} not found', 111)
        est = self.stored[name]
        self.est = est
        self.e = Results()
        self.e.scalars.update(est.get('e_scalars', {}))
        self.e.macros.update(est.get('e_macros', {}))
        self.e.matrices.update(est.get('e_matrices', {}))

    def set_estimation(self, est):
        """Install new estimation results; est is a dict with names,b,V,sample, scalars, macros."""
        self.est = est
        self.e = Results()
        names = est.get('names', [])
        if names:
            b = np.asarray(est['b'], float)
            self.e.matrices['b'] = E.Matrix(b.reshape(1, -1), ['y1'], names)
            self.e.matrices['V'] = E.Matrix(np.asarray(est['V'], float), names, names)
        for k, v in est.get('scalars', {}).items():
            self.e.scalars[k] = float(v) if v is not None else E.MISS
        for k, v in est.get('macros', {}).items():
            self.e.macros[k] = v
        for k, v in est.get('matrices', {}).items():
            self.e.matrices[k] = v
        self.e.macros['__sample__'] = '1'

    def set_r(self, scalars=None, macros=None, matrices=None, clear=True):
        if clear:
            self.r = Results()
        for k, v in (scalars or {}).items():
            self.r.scalars[k] = float(v) if v is not None and not isinstance(v, str) else (E.MISS if v is None else v)
        for k, v in (macros or {}).items():
            self.r.macros[k] = v
        for k, v in (matrices or {}).items():
            self.r.matrices[k] = v

    # ---------------- files
    def path(self, fn, must_exist=False, ext=None):
        fn = fn.strip().strip('"')
        fn = fn.replace('\\', '/')
        if ext and not os.path.splitext(fn)[1]:
            fn += ext
        if fn.startswith('http://') or fn.startswith('https://'):
            return fn
        if must_exist and not os.path.exists(fn):
            base = os.path.basename(fn)
            for cand in (os.path.join(self.workdir, base), base, os.path.join('/examples/data', base)):
                if os.path.exists(cand):
                    return cand
            # search workdir recursively
            for root, _, files in os.walk(self.workdir):
                if base in files:
                    return os.path.join(root, base)
            raise StataError(f'file {fn} not found', 601)
        if not must_exist and re.match(r'^[A-Za-z]:/', fn):
            # windows absolute path: map into workdir keeping relative part after last known dir
            fn = os.path.join(self.workdir, os.path.basename(fn))
        return fn

    def tempname(self, prefix='__'):
        self.tmpcount += 1
        return f'{prefix}{self.tmpcount:06d}'


def canon_coef(name):
    """Canonical coefficient name for matching: strip b/o/n markers and c. on continuous."""
    parts = name.split('#')
    out = []
    for p in parts:
        p = p.strip()
        m = re.match(r'^(\d+(?:\.\d+)?)[bno]*\.(.+)$', p)
        if m:
            lv = float(m.group(1))
            out.append(f'{num_to_macro(lv)}.{m.group(2)}')
            continue
        p = re.sub(r'^(?:o\.|co\.|c\.)', '', p)
        tm = re.match(r'^((?:[LDFSldfs]\d*)+)\.(.+)$', p)
        if tm:
            from .tsops import opname, _parse_chain
            ch = opname(_parse_chain(tm.group(1)))
            p = f'{ch}.{tm.group(2)}' if ch else tm.group(2)
        out.append(p)
    return '#'.join(out)


def load_commands():
    from . import cmds_core, cmds_data, cmds_stats, estim, causal, causal2, tables, graphs, cmds_more, cmds_ext, models, tsuni, tsreg, tsarima, tsarch, tsvar, tsirf, tssvar, tsvec, tsardl, xtdiag, xtgls, xtab2, xtdpdoff, xtunit  # noqa: F401
