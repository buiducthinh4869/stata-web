"""Core programming commands."""
import math
import os
import re

import numpy as np

from . import expr as E
from .errors import StataError, ExitDo
from .fmt import fmt_num, num_to_macro, display_num, g_fmt
from .interp import command, alias, Results
from .syntax import split_top, unquote, find_top, parse_command, find_cq_end, parse_numlist

alias('display', 'di', 'dis', 'disp', 'displ', 'displa')
alias('local', 'loc', 'loca')
alias('global', 'gl', 'glo', 'glob', 'globa')
alias('matrix', 'mat', 'matr', 'matri')
alias('scalar', 'sca', 'scal', 'scala')
alias('estimates', 'est', 'esti', 'estim', 'estima', 'estimat', 'estimate')
alias('forvalues', 'forval', 'forv')
alias('ereturn', 'eret', 'eretu', 'eretur')
alias('return', 'ret', 'retu', 'retur')


# ------------------------------------------------------------------ SMCL
_SMCL_RE = re.compile(r'\{([a-z_]+)(?:\s+([^}:]*))?(?::([^{}]*))?\}')


def smcl(text, default='txt'):
    """Very small SMCL renderer -> list of (style, text)."""
    out = []
    style = default
    pos = 0
    for m in _SMCL_RE.finditer(text):
        if m.start() > pos:
            out.append((style, text[pos:m.start()]))
        tag, arg, inner = m.group(1), m.group(2), m.group(3)
        if tag == 'hline':
            n = int(arg) if arg and arg.strip().isdigit() else 78
            out.append((style, '-' * n))
        elif tag in ('txt', 'text'):
            style = 'txt'
            if inner:
                out.append(('txt', inner))
        elif tag in ('res', 'result'):
            if inner:
                out.append(('res', inner))
            else:
                style = 'res'
        elif tag in ('err', 'error'):
            if inner:
                out.append(('err', inner))
            else:
                style = 'err'
        elif tag in ('bf', 'it', 'cmd', 'inp', 'input', 'sf', 'ul'):
            out.append((style if tag not in ('cmd', 'inp', 'input') else 'res', inner or ''))
        elif tag in ('space',):
            out.append((style, ' ' * (int(arg) if arg else 1)))
        elif tag == 'c':
            ch = (arg or '').strip()
            out.append((style, {'|': '|', '-': '-', '+': '+', 'TT': '-', 'BT': '-'}.get(ch, ch)))
        elif tag in ('col', 'column'):
            out.append((style, '\x00COL' + (arg or '1').strip() + '\x00'))
        elif tag in ('p', 'p_end', 'pstd', 'title', 'browse', 'help', 'newline'):
            if tag == 'newline':
                out.append((style, '\n'))
            elif inner:
                out.append((style, inner))
        elif tag == 'dup':
            out.append((style, (inner or '') * int(arg or 1)))
        else:
            out.append((style, m.group(0)))
        pos = m.end()
    if pos < len(text):
        out.append((style, text[pos:]))
    return out


# ------------------------------------------------------------------ display
def _display_items(ip, s):
    """Parse display arguments; returns list of (style, text)."""
    pieces = []
    style = 'txt'
    fmt = None
    i = 0
    n = len(s)
    col = 0
    cont = False

    def emit(st, t):
        nonlocal col
        if '\x00COL' in t:
            parts = re.split(r'\x00COL(\d+)\x00', t)
            for k, part in enumerate(parts):
                if k % 2 == 1:
                    target = int(part) - 1
                    if target > col:
                        pieces.append((st, ' ' * (target - col)))
                        col = target
                else:
                    if part:
                        pieces.append((st, part))
                        col = col + len(part) if '\n' not in part else len(part.split('\n')[-1])
            return
        pieces.append((st, t))
        if '\n' in t:
            col = len(t.split('\n')[-1])
        else:
            col += len(t)

    while i < n:
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break
        rest = s[i:]
        m = re.match(r'(as|in)\s+(text|txt|result|res|error|err|input|inp|red|green|yellow|white|blue)\b', rest)
        if m:
            w = m.group(2)
            style = {'text': 'txt', 'txt': 'txt', 'result': 'res', 'res': 'res', 'error': 'err', 'err': 'err',
                     'input': 'res', 'inp': 'res', 'red': 'err', 'green': 'txt', 'yellow': 'res', 'white': 'res',
                     'blue': 'res'}[w]
            i += m.end()
            continue
        m = re.match(r'(_newline|_n)(?:\((\d+)\))?(?![A-Za-z0-9_])', rest)
        if m:
            emit(style, '\n' * int(m.group(2) or 1))
            i += m.end()
            continue
        m = re.match(r'(_column|_col)\((\d+)\)', rest)
        if m:
            target = int(m.group(2)) - 1
            if target > col:
                emit(style, ' ' * (target - col))
            i += m.end()
            continue
        m = re.match(r'_skip\((\d+)\)', rest)
        if m:
            emit(style, ' ' * int(m.group(1)))
            i += m.end()
            continue
        m = re.match(r'_dup\((\d+)\)\s*', rest)
        if m:
            k = int(m.group(1))
            i += m.end()
            node, r2 = E.parse_prefix(s[i:])
            v = E.evaluate(node, E.Ctx(ip))
            emit(style, str(v) * k)
            i = n - len(r2)
            continue
        if rest.startswith('_continue') or rest.startswith('_c '):
            cont = True
            i += 9 if rest.startswith('_continue') else 2
            continue
        m = re.match(r'%-?\d*(?:\.\d+)?[fgecs]c?', rest)
        if m:
            fmt = m.group(0)
            i += m.end()
            continue
        if rest.startswith('_request'):
            i += len(rest)
            continue
        # expression (string or numeric)
        try:
            node, r2 = E.parse_prefix(rest)
        except StataError:
            raise
        v = E.evaluate(node, E.Ctx(ip))
        if isinstance(v, np.ndarray):
            v = v[0] if len(v) else E.MISS
        if isinstance(v, str):
            txt = v if not fmt else fmt_num(v, fmt)
            for st, t in smcl(txt, style):
                emit(st, t)
        else:
            txt = fmt_num(v, fmt) if fmt else display_num(v)
            emit(style if style != 'txt' or not isinstance(node, (E.Num,)) else style, txt)
        fmt = None
        i = n - len(r2)
    return pieces, cont


@command('display')
def cmd_display(ip, cmd, rest, by):
    pieces, cont = _display_items(ip, rest)
    for st, t in pieces:
        if st == 'err':
            if not (ip.quiet and not ip.noisy):
                ip.out.write(t, 'err')
        else:
            ip.write(t, st)
    if not cont:
        ip.write('\n')


# ------------------------------------------------------------------ macros
def _macro_value(ip, rest):
    """Value for `local name ...` definitions (rest after the name)."""
    r = rest
    if r.startswith('='):
        v = E.eval_expr(r[1:].strip(), E.Ctx(ip))
        if isinstance(v, np.ndarray):
            v = v[0] if len(v) else E.MISS
        return v if isinstance(v, str) else num_to_macro(v)
    if r.startswith(':'):
        return ip.extended_fn(r[1:].strip())
    r = r.strip()
    if r.startswith('`"') and r.endswith('"\'') and find_cq_end(r, 0) == len(r):
        return r[2:-2]
    if len(r) >= 2 and r[0] == '"' and r[-1] == '"' and r.count('"') == 2:
        return r[1:-1]
    return r


def _def_macro(ip, rest, store):
    rest = rest.strip()
    m = re.match(r'^(\+\+|--)([A-Za-z_]\w*)$', rest)
    if m:
        cur = store.get(m.group(2), '0') or '0'
        try:
            v = float(cur)
        except ValueError:
            v = 0.0
        store[m.group(2)] = num_to_macro(v + (1 if m.group(1) == '++' else -1))
        return
    m = re.match(r'^([A-Za-z_]\w*)\s*(.*)$', rest, re.S)
    if not m:
        raise StataError('invalid syntax', 198)
    name, val = m.group(1), m.group(2)
    if val.startswith('=') or val.startswith(':'):
        store[name] = _macro_value(ip, val)
    else:
        store[name] = _macro_value(ip, val)


@command('local')
def cmd_local(ip, cmd, rest, by):
    # local expansion already happened; but `local x : ...` uses raw
    _def_macro(ip, rest, ip.locals)


@command('global')
def cmd_global(ip, cmd, rest, by):
    _def_macro(ip, rest, ip.globals)


@command('tempfile')
def cmd_tempfile(ip, cmd, rest, by):
    os.makedirs('/tmp/st_tmp', exist_ok=True)
    for nm in rest.split():
        fn = f'/tmp/st_tmp/{ip.tempname("ST_")}.dta'
        if os.path.exists(fn):
            os.remove(fn)
        ip.locals[nm] = fn


@command('tempvar', 'tempname')
def cmd_tempvar(ip, cmd, rest, by):
    for nm in rest.split():
        ip.locals[nm] = ip.tempname('__')


@command('scalar')
def cmd_scalar(ip, cmd, rest, by):
    rest = rest.strip()
    m = re.match(r'^(?:define\s+)?([A-Za-z_]\w*)\s*=\s*(.+)$', rest, re.S)
    if m:
        v = E.eval_expr(m.group(2), E.Ctx(ip))
        if isinstance(v, np.ndarray):
            v = v[0] if len(v) else E.MISS
        ip.scalars[m.group(1)] = v if isinstance(v, str) else float(v)
        return
    m = re.match(r'^(list|dir)\s*(.*)$', rest)
    if m:
        names = m.group(2).split() or sorted(ip.scalars)
        for nm in names:
            if nm in ('_all',):
                continue
            v = ip.scalars.get(nm)
            ip.txt(f'{nm:>13} = ')
            ip.res((v if isinstance(v, str) else fmt_num(v, '%9.0g').strip()) + '\n')
        return
    m = re.match(r'^drop\s+(.*)$', rest)
    if m:
        for nm in m.group(1).split():
            if nm == '_all':
                ip.scalars.clear()
            ip.scalars.pop(nm, None)
        return
    raise StataError('invalid syntax', 198)


@command('matrix')
def cmd_matrix(ip, cmd, rest, by):
    rest = rest.strip()
    m = re.match(r'^list\s+(\S+)(.*)$', rest)
    if m:
        M = ip.get_matrix(m.group(1))
        _list_matrix(ip, m.group(1), M)
        return
    m = re.match(r'^drop\s+(.*)$', rest)
    if m:
        for nm in m.group(1).split():
            if nm == '_all':
                ip.matrices.clear()
            ip.matrices.pop(nm, None)
        return
    m = re.match(r'^(?:define\s+)?([A-Za-z_]\w*)\s*\[\s*(.+?)\s*,\s*(.+?)\s*\]\s*=\s*(.+)$', rest)
    if m:
        M = ip.get_matrix(m.group(1))
        i = int(ip.eval_scalar(m.group(2)))
        j = int(ip.eval_scalar(m.group(3)))
        M.a[i - 1, j - 1] = float(ip.eval_scalar(m.group(4)))
        return
    m = re.match(r'^(?:define\s+)?([A-Za-z_]\w*)\s*=\s*(.+)$', rest, re.S)
    if m:
        ip.matrices[m.group(1)] = _matrix_expr(ip, m.group(2).strip())
        return
    m = re.match(r'^(rownames|colnames)\s+(\w+)\s*=\s*(.*)$', rest)
    if m:
        M = ip.get_matrix(m.group(2))
        names = [unquote(x) for x in split_top(m.group(3))]
        if m.group(1) == 'rownames':
            M.rownames = names
        else:
            M.colnames = names
        return
    raise StataError('matrix syntax not supported', 198)


def _matrix_expr(ip, s):
    m = re.match(r'^J\(\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*\)$', s)
    if m:
        r = int(ip.eval_scalar(m.group(1)))
        c = int(ip.eval_scalar(m.group(2)))
        v = ip.eval_scalar(m.group(3))
        return E.Matrix(np.full((r, c), float(v)))
    m = re.match(r'^(e|r)\((\w+)\)$', s)
    if m:
        v = ip.get_return(m.group(1), m.group(2))
        if isinstance(v, E.Matrix):
            return E.Matrix(v.a.copy(), v.rownames, v.colnames)
        return E.Matrix(np.array([[float(v)]]))
    if s.startswith('('):
        body = s[1:-1]
        rows = [r for r in body.split('\\')]
        arr = [[float(ip.eval_scalar(x)) for x in r.split(',')] for r in rows]
        return E.Matrix(np.array(arr))
    m = re.match(r"^([A-Za-z_]\w*)'$", s)
    if m:
        M = ip.get_matrix(m.group(1))
        return E.Matrix(M.a.T.copy(), M.colnames, M.rownames)
    if s in ip.matrices:
        M = ip.matrices[s]
        return E.Matrix(M.a.copy(), M.rownames, M.colnames)
    v = ip.eval_scalar(s)
    return E.Matrix(np.array([[float(v)]]))


def _list_matrix(ip, name, M):
    r, c = M.a.shape
    ip.txt(f'\n{name}[{r},{c}]\n')
    for j0 in range(0, c, 6):
        cols = range(j0, min(c, j0 + 6))
        ip.txt(' ' * 13 + ''.join(f'{M.colnames[j][:10]:>11}' for j in cols) + '\n')
        for i in range(r):
            ip.txt(f'{M.rownames[i][:12]:>12} ')
            ip.res(''.join(f'{fmt_num(M.a[i, j], "%10.0g"):>11}' for j in cols) + '\n')


@command('macro')
def cmd_macro(ip, cmd, rest, by):
    rest = rest.strip()
    if rest.startswith('drop'):
        for nm in rest[4:].split():
            if nm == '_all':
                ip.globals.clear()
            ip.globals.pop(nm, None)
        return
    if rest.startswith('list') or rest.startswith('dir') or not rest:
        for k, v in ip.globals.items():
            ip.txt(f'{k + ":":<13}{v}\n')
        for k, v in ip.locals.items():
            ip.txt(f'{"_" + k + ":":<13}{v}\n')


@command('return')
def cmd_return(ip, cmd, rest, by):
    rest = rest.strip()
    if rest.startswith('li'):
        _list_results(ip, ip.r, 'r')
    elif rest.startswith('clear'):
        ip.r = Results()


@command('ereturn')
def cmd_ereturn(ip, cmd, rest, by):
    rest = rest.strip()
    if rest.startswith('li'):
        _list_results(ip, ip.e, 'e')
    elif rest.startswith('clear'):
        ip.e = Results()
        ip.est = None


def _list_results(ip, R, k):
    if R.scalars:
        ip.txt('\nscalars:\n')
        for n, v in R.scalars.items():
            ip.txt(f'{k + "(" + n + ")":>22} = ')
            ip.res((v if isinstance(v, str) else fmt_num(v, '%10.0g').strip()) + '\n')
    macros = {n: v for n, v in R.macros.items() if not n.startswith('__')}
    if macros:
        ip.txt('\nmacros:\n')
        for n, v in macros.items():
            ip.txt(f'{k + "(" + n + ")":>22} : ')
            ip.res(f'"{v}"\n')
    if R.matrices:
        ip.txt('\nmatrices:\n')
        for n, v in R.matrices.items():
            ip.txt(f'{k + "(" + n + ")":>22} :  ')
            ip.res(f'{v.a.shape[0]} x {v.a.shape[1]}\n')
    if k == 'e' and ip.est is not None:
        ip.txt('\nfunctions:\n')
        ip.txt(f'{"e(sample)":>22}\n')


# ------------------------------------------------------------------ settings / misc
@command('set')
def cmd_set(ip, cmd, rest, by):
    parts = rest.split()
    if not parts:
        return
    k = parts[0]
    v = parts[1] if len(parts) > 1 else ''
    if k == 'obs':
        n = int(ip.eval_scalar(' '.join(parts[1:])))
        ip.data.set_obs(n)
    elif k == 'seed':
        try:
            ip.seed = int(float(v))
        except ValueError:
            ip.seed = abs(hash(v)) % (2 ** 31)
        ip.rng = np.random.default_rng(ip.seed)
    elif k == 'linesize':
        ip.linesize = int(v)
    elif k == 'scheme':
        ip.scheme = v
    # everything else (more, maxvar, matsize, varabbrev, type...) silently accepted


@command('version', 'clear', 'pause', 'sleep', 'discard', 'timer', 'mata', 'program', 'end',
         'search', 'update', 'query', 'cls', 'window', 'adopath', 'sysdir', 'creturn', 'translate',
         'graph_set_dummy', 'exit_dummy', 'compress', 'notes_dummy')
def cmd_noop(ip, cmd, rest, by):
    if cmd == 'clear':
        r = rest.strip()
        from .data import Data
        if r in ('', 'all', '*', '_all', 'results', 'matrix', 'mata', 'programs', 'ado', 'frames'):
            if r in ('', 'all', '*', '_all', 'frames'):
                ip.data = Data()
            if r in ('all', '*', '_all', 'results'):
                ip.est = None
                ip.e = Results()
                ip.r = Results()
                ip.stored.clear()
                ip.eststo_order = []
                if r != 'results':
                    ip.scalars.clear()
                    ip.matrices.clear()
            if r == 'matrix':
                ip.matrices.clear()
        return
    if cmd == 'compress':
        return


@command('exit')
def cmd_exit(ip, cmd, rest, by):
    raise ExitDo()


@command('which')
def cmd_which(ip, cmd, rest, by):
    from .interp import COMMANDS, ALIASES
    nm = rest.strip().split(',')[0].strip()
    if nm in COMMANDS or nm in ALIASES:
        ip.txt(f'{nm}.ado (built into this Stata web emulator)\n')
        return
    raise StataError(f'command {nm} not found as either built-in or ado-file', 111)


@command('net', 'ssc', 'ado')
def cmd_net(ip, cmd, rest, by):
    ip.txt(f'(note: {cmd} {rest.strip()[:60]} -- packages are built in; nothing to install)\n')


@command('more')
def cmd_more(ip, cmd, rest, by):
    pass


@command('mkdir')
def cmd_mkdir(ip, cmd, rest, by):
    p = ip.path(unquote(rest.split(',')[0].strip()))
    if os.path.exists(p):
        raise StataError(f'could not create directory {p}', 693)
    os.makedirs(p, exist_ok=True)


@command('erase', 'rm')
def cmd_erase(ip, cmd, rest, by):
    p = ip.path(unquote(rest.strip()))
    if os.path.exists(p):
        os.remove(p)
    else:
        raise StataError(f'file {p} not found', 601)


@command('cd')
def cmd_cd(ip, cmd, rest, by):
    p = unquote(rest.strip())
    if not p:
        ip.txt(os.getcwd() + '\n')
        return
    try:
        os.chdir(ip.path(p))
    except OSError:
        raise StataError(f'unable to change to {p}', 170)


@command('pwd')
def cmd_pwd(ip, cmd, rest, by):
    ip.txt(os.getcwd() + '\n')


@command('dir', 'ls')
def cmd_dir(ip, cmd, rest, by):
    for f in sorted(os.listdir('.')):
        sz = os.path.getsize(f) if os.path.isfile(f) else 0
        ip.txt(f'{sz / 1024:8.1f}k  {f}\n')


@command('log')
def cmd_log(ip, cmd, rest, by):
    p = parse_command(rest)
    body = p.body.strip()
    if body.startswith('using'):
        body = body[5:]
    if p.using is not None or body.startswith('using'):
        fn = p.using
        ip.log_file = ip.path(fn)
        os.makedirs(os.path.dirname(ip.log_file) or '.', exist_ok=True)
        ip.out.log = []
        ip.txt('-' * 79 + '\n')
        ip.txt(f'      name:  <unnamed>\n       log:  {ip.log_file}\n  log type:  text\n')
        return
    if body.startswith('close') or body.startswith('off'):
        if ip.out.log is not None and ip.log_file:
            with open(ip.log_file, 'w', encoding='utf-8') as f:
                f.write(''.join(ip.out.log))
            ip.txt(f'      log:  {ip.log_file}\n')
        elif not ip.quiet:
            raise StataError('no log file open', 606)
        ip.out.log = None
        ip.log_file = None
        return


@command('cmdlog')
def cmd_cmdlog(ip, cmd, rest, by):
    pass


@command('do', 'run', 'include')
def cmd_do(ip, cmd, rest, by):
    fn = unquote(rest.strip().split(',')[0])
    p = ip.path(fn, must_exist=True, ext='.do')
    with open(p, encoding='utf-8', errors='replace') as f:
        src = f.read()
    ip.frames.append({})
    try:
        from .interp import preprocess
        lines = preprocess(src)
        ip.exec_block(lines, 0, len(lines), top=ip.echo and cmd == 'do')
    except ExitDo:
        pass
    finally:
        ip.frames.pop()


@command('confirm')
def cmd_confirm(ip, cmd, rest, by):
    parts = rest.split()
    kind = parts[0]
    if kind in ('variable', 'var', 'numeric', 'string', 'new'):
        if kind == 'new':
            names = parts[2:]
            for n in names:
                if ip.data.resolve(n, allow_abbrev=False, required=False):
                    raise StataError(f'variable {n} already defined', 110)
            return
        names = parts[1:] if kind in ('variable', 'var') else parts[2:]
        for n in names:
            v = ip.data.resolve(n, allow_abbrev=False, required=False)
            if v is None:
                raise StataError(f'variable {n} not found', 111)
            if kind == 'numeric' and ip.data.isstr(v):
                raise StataError(f'{n} found where numeric variable expected', 7)
            if kind == 'string' and not ip.data.isstr(v):
                raise StataError(f'{n} found where string variable expected', 7)
        return
    if kind == 'file':
        fn = unquote(' '.join(parts[1:]))
        if not os.path.exists(ip.path(fn)):
            raise StataError(f'file {fn} not found', 601)
        return
    if kind in ('number', 'integer'):
        try:
            float(parts[-1])
        except ValueError:
            raise StataError(f'{parts[-1]} found where number expected', 7)
        return
    if kind == 'matrix':
        ip.get_matrix(parts[1])
        return
    if kind == 'scalar':
        if parts[1] not in ip.scalars:
            raise StataError(f'scalar {parts[1]} not found', 111)


@command('assert')
def cmd_assert(ip, cmd, rest, by):
    p = parse_command(rest)
    v = ip.eval(p.body, ip.data)
    m = ip.mark(p)
    if isinstance(v, np.ndarray):
        bad = int(np.sum((v == 0) & m))
    else:
        bad = 0 if v != 0 else ip.data.n
    if bad:
        raise StataError(f'{bad} contradiction{"s" if bad > 1 else ""} in {int(m.sum())} observations\nassertion is false', 9)


@command('error')
def cmd_error(ip, cmd, rest, by):
    rc = int(ip.eval_scalar(rest))
    if rc:
        raise StataError('', rc)


# ------------------------------------------------------------------ preserve / restore
@command('preserve')
def cmd_preserve(ip, cmd, rest, by):
    ip.preserved.append(ip.data.copy())


@command('restore')
def cmd_restore(ip, cmd, rest, by):
    r = rest.strip()
    if not ip.preserved:
        raise StataError('nothing to restore', 622)
    if 'not' in r:
        ip.preserved.pop()
        return
    if 'preserve' in r:
        ip.data = ip.preserved[-1].copy()
        return
    ip.data = ip.preserved.pop()


# ------------------------------------------------------------------ file write
@command('file')
def cmd_file(ip, cmd, rest, by):
    rest = rest.strip()
    m = re.match(r'^open\s+(\w+)\s+using\s+(.+?)\s*,\s*(.*)$', rest)
    if m:
        h, fn, opts = m.group(1), unquote(m.group(2)), m.group(3)
        if h in ip.files:
            raise StataError(f'file handle {h} already exists', 110)
        path = ip.path(fn)
        d = os.path.dirname(path)
        if d and not os.path.isdir(d):
            raise StataError(f'file {fn} could not be opened', 603)
        mode = 'w' if 'write' in opts else 'r'
        if 'append' in opts:
            mode = 'a'
        ip.files[h] = open(path, mode, encoding='utf-8')
        return
    m = re.match(r'^close\s+(\S+)', rest)
    if m:
        h = m.group(1)
        if h == '_all':
            for f in ip.files.values():
                f.close()
            ip.files.clear()
            return
        if h not in ip.files:
            raise StataError(f'handle {h} not found', 601)
        ip.files.pop(h).close()
        return
    m = re.match(r'^write\s+(\w+)\s*(.*)$', rest, re.S)
    if m:
        h = m.group(1)
        if h not in ip.files:
            raise StataError(f'file handle {h} not found', 601)
        ip.files[h].write(_file_write_text(ip, m.group(2)))
        return
    m = re.match(r'^read\s+(\w+)\s+(\w+)', rest)
    if m:
        line = ip.files[m.group(1)].readline()
        ip.locals[m.group(2)] = line.rstrip('\n')
        ip.r.scalars['eof'] = 0.0 if line else 1.0
        return
    raise StataError('file subcommand not supported', 198)


def _file_write_text(ip, s):
    out = []
    i = 0
    n = len(s)
    fmt = None
    while i < n:
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break
        rest = s[i:]
        m = re.match(r'_n(?:ewline)?(?:\((\d+)\))?(?![A-Za-z0-9_])', rest)
        if m:
            out.append('\n' * int(m.group(1) or 1))
            i += m.end()
            continue
        m = re.match(r'_tab(?:\((\d+)\))?', rest)
        if m:
            out.append('\t' * int(m.group(1) or 1))
            i += m.end()
            continue
        m = re.match(r'_col(?:umn)?\((\d+)\)', rest)
        if m:
            cur = ''.join(out).split('\n')[-1]
            pad = int(m.group(1)) - 1 - len(cur)
            out.append(' ' * max(0, pad))
            i += m.end()
            continue
        m = re.match(r'_skip\((\d+)\)', rest)
        if m:
            out.append(' ' * int(m.group(1)))
            i += m.end()
            continue
        m = re.match(r'%-?\d*(?:\.\d+)?[fgecs]c?', rest)
        if m:
            fmt = m.group(0)
            i += m.end()
            continue
        node, r2 = E.parse_prefix(rest)
        v = E.evaluate(node, E.Ctx(ip))
        if isinstance(v, np.ndarray):
            v = v[0] if len(v) else E.MISS
        if isinstance(v, str):
            out.append(fmt_num(v, fmt) if fmt else v)
        else:
            out.append(fmt_num(v, fmt) if fmt else display_num(v))
        fmt = None
        i = n - len(r2)
    return ''.join(out)


# ------------------------------------------------------------------ postfile
@command('postfile')
def cmd_postfile(ip, cmd, rest, by):
    m = re.match(r'^(\w+)\s+(.+?)\s+using\s+(\S+)(?:\s*,\s*(.*))?$', rest.strip())
    if not m:
        raise StataError('invalid postfile syntax', 198)
    h = m.group(1)
    spec = m.group(2).split()
    names, types = [], []
    cur_type = 'float'
    for t in spec:
        if t in ('byte', 'int', 'long', 'float', 'double') or re.match(r'^str\d+$', t) or t == 'strL':
            cur_type = t
            continue
        names.append(t)
        types.append(cur_type)
        cur_type = 'float'
    ip.postfiles[h] = {'names': names, 'types': types, 'rows': [], 'file': unquote(m.group(3))}


@command('post')
def cmd_post(ip, cmd, rest, by):
    m = re.match(r'^(\w+)\s+(.*)$', rest.strip(), re.S)
    h = m.group(1)
    pf = ip.postfiles[h]
    vals = []
    s = m.group(2).strip()
    i = 0
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        if s[i] != '(':
            raise StataError('post: expressions must be in parentheses', 198)
        depth = 0
        j = i
        while j < len(s):
            if s[j] == '(':
                depth += 1
            elif s[j] == ')':
                depth -= 1
                if depth == 0:
                    break
            elif s[j] == '"':
                j = s.find('"', j + 1)
            j += 1
        v = E.eval_expr(s[i + 1:j], E.Ctx(ip))
        if isinstance(v, np.ndarray):
            v = v[0] if len(v) else E.MISS
        vals.append(v)
        i = j + 1
    if len(vals) != len(pf['names']):
        raise StataError('post: wrong number of values', 198)
    pf['rows'].append(vals)


@command('postclose')
def cmd_postclose(ip, cmd, rest, by):
    from .data import Data, write_dta
    h = rest.strip()
    pf = ip.postfiles.pop(h)
    d = Data()
    d.n = len(pf['rows'])
    for k, (nm, tp) in enumerate(zip(pf['names'], pf['types'])):
        col = [r[k] for r in pf['rows']]
        if tp.startswith('str'):
            d.add_var(nm, np.array([str(c) for c in col], dtype=object), 'str%d' % max(1, max((len(str(c)) for c in col), default=1)))
        else:
            d.add_var(nm, np.array([float(c) if not isinstance(c, str) else np.nan for c in col]), tp)
    path = ip.path(pf['file'], ext='.dta')
    write_dta(d, path)
    ip.label_store[path] = d


@command('estimates')
def cmd_estimates(ip, cmd, rest, by):
    from .tables import estimates_table
    rest = rest.strip()
    parts = rest.split(None, 1)
    sub = parts[0] if parts else ''
    arg = parts[1] if len(parts) > 1 else ''
    if sub in ('store', 'sto'):
        nm = arg.split(',')[0].strip()
        ip.store_estimates(nm)
    elif sub in ('restore', 'res'):
        ip.restore_estimates(arg.strip())
    elif sub in ('clear',):
        ip.stored.clear()
    elif sub in ('drop',):
        for nm in arg.split():
            ip.stored.pop(nm, None)
    elif sub in ('dir',):
        for nm in ip.stored:
            ip.txt(f'{nm}\n')
    elif sub.startswith('tab'):
        estimates_table(ip, arg)
    elif sub in ('replay', 'rep'):
        names = arg.split() or [None]
        for nm in names:
            if nm:
                ip.restore_estimates(nm)
            if ip.est and ip.est.get('replay'):
                ip.est['replay'](ip)
    elif sub in ('save', 'use', 'describe', 'title', 'notes', 'esample', 'stats'):
        ip.txt(f'(estimates {sub} not available in web emulator)\n')
    else:
        raise StataError(f'estimates {sub} not supported', 198)


@command('label')
def cmd_label(ip, cmd, rest, by):
    from .cmds_data import do_label
    do_label(ip, rest)


alias('label', 'la', 'lab', 'labe')


@command('notes', 'note')
def cmd_notes(ip, cmd, rest, by):
    r = rest.strip()
    if r.startswith(':'):
        ip.data.notes.append(r[1:].strip())
    elif r.startswith('list') or not r:
        for k, n in enumerate(ip.data.notes, 1):
            ip.txt(f'  {k}.  {n}\n')


@command('type')
def cmd_type(ip, cmd, rest, by):
    fn = unquote(rest.split(',')[0].strip())
    p = ip.path(fn, must_exist=True)
    with open(p, encoding='utf-8', errors='replace') as f:
        ip.txt(f.read() + '\n')


HELP = {
    'Dữ liệu': 'use, save, import excel/delimited, export excel/delimited, webuse, merge, append, collapse, reshape*, describe, codebook, list, browse, count, misstable, xtset, tsset',
    'Biến': 'generate, replace, egen, drop, keep, rename, order, sort, gsort, label, recode, encode, decode, destring, tostring, split, format, duplicates, expand, fillin, contract, tsfill, winsor2, egen concat/rowfirst/rowlast/rowmedian/anymatch/anycount',
    'Mô tả': 'summarize [, detail], tabulate [, row col chi2], tabstat, ttest, mean, correlate, pwcorr, xtsum, xtdescribe',
    'Đồ thị': 'histogram, kdensity, scatter, twoway (scatter/line/connected/lfit/qfit/lfitci/rcap/rarea/bar/function), graph box/bar/matrix/combine/export, coefplot, marginsplot, rvfplot, pnorm, qnorm',
    'Hồi quy': 'regress, predict, xtreg fe/re/be, reghdfe, areg, logit, probit, test, testparm, lincom, margins, hausman, estat hettest/ovtest/vif/ic',
    'Biến rời rạc': 'poisson [, irr exposure() offset()], nbreg, ppmlhdfe (gravity, absorb()), tobit [, ll() ul()], ologit, oprobit, mlogit [, baseoutcome() rrr]',
    'IV': 'ivregress 2sls, ivreg2, estat firststage/endogenous/overid, weakivtest',
    'RDD': 'rdrobust, rdbwselect, rdplot, rddensity',
    'PSM': 'psmatch2, pstest, psgraph, teffects psmatch, kmatch',
    'DID': 'didregress, xtdidregress, estat ptrend/trendplots/granger, csdid (+ estat event), sdid',
    'SCM': 'synth, synth2',
    'RCT': 'ritest, ttest, regress',
    'Bảng kết quả': 'eststo, esttab, estout, estpost, estimates store/restore/table, outreg2',
    'Lập trình': 'local, global, scalar, matrix, display, foreach, forvalues, while, if/else, quietly, capture, by/bysort, preserve/restore, tempfile, postfile, file, do, log',
}


@command('help')
def cmd_help(ip, cmd, rest, by):
    out = ip
    r = rest.strip().split()
    if r:
        name = r[0]
        full = ALIASES_get(name)
        if full in COMMANDS_get():
            out.txt(f"{full}: lệnh có trong Stata Web. Cú pháp theo Stata gốc; xem tài liệu Stata: https://www.stata.com/manuals/\n")
        else:
            out.err(f"help for {name} not found\n")
        return
    out.txt('Stata Web — các lệnh được hỗ trợ (theo nhóm)\n' + '-' * 78 + '\n')
    for k, v in HELP.items():
        out.res(f'{k:>14}') ; out.txt(' │ ')
        line, first = '', True
        for w in v.split(', '):
            if len(line) + len(w) > 58:
                out.txt(('' if first else ' ' * 17) + line.rstrip(', ') + '\n'); line, first = '', False
            line += w + ', '
        out.txt(('' if first else ' ' * 17) + line.rstrip(', ') + '\n')
    out.txt('-' * 78 + '\n(* = hỗ trợ một phần.) Gõ  help <lệnh>  để kiểm tra một lệnh.\n')


def ALIASES_get(n):
    from .interp import ALIASES
    return ALIASES.get(n, n)


def COMMANDS_get():
    from .interp import COMMANDS
    return COMMANDS
