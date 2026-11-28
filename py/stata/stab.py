"""Emulation of Stata's ._tab class (base/_/_tab.class), used by many official table displays.

Column layout follows _reset_columns: column i starts at lmargin+1+sum(width_j + vbar_j, j<i); a vertical
bar, when present before column i, occupies that column's first position.  Titles and strings use %{w-1}s and
numbers %{w-1}.0g unless set; numbers get pad spaces before the formatted value."""
import re

from .fmt import fmt_num

_SMCL = re.compile(r'\{it:([^}]*)\}|\{bf:([^}]*)\}')


def _visible(s):
    return _SMCL.sub(lambda m: m.group(1) or m.group(2) or '', s)


class Tab:
    def __init__(self, ip, widths, lmargin=2, separator=0, ignore=None):
        self.ip = ip
        self.lmargin = lmargin
        self.sep_every = separator
        self.rows = 0
        self.w, self.vbar = [], {}
        i = 0
        for tok in re.findall(r'\||\d+', widths):
            if tok == '|':
                self.vbar[i] = True
            else:
                self.w.append(int(tok))
                i += 1
        n = len(self.w)
        self.n = n
        self.vb = [bool(self.vbar.get(k)) for k in range(n + 1)]
        self.start = []
        cur = lmargin + 1
        for k in range(n):
            self.start.append(cur)
            cur += self.w[k] + (1 if self.vb[k] else 0)
        self.start.append(cur)
        self.numfmt = [f'%{max(wk - 1, 1)}.0g' for wk in self.w]
        self.strfmt = [f'%{max(wk - 1, 1)}s' for wk in self.w]
        self.titlefmt = [f'%{max(wk - 1, 1)}s' for wk in self.w]
        self.pad = [0] * n
        self.strcolor = ['txt'] * n
        self.numcolor = ['res'] * n
        self.ignore = ignore

    # --- setters accept a list with None meaning "keep"
    def _set(self, attr, vals):
        cur = getattr(self, attr)
        for k, v in enumerate(vals):
            if v is not None and v != '.':
                cur[k] = v

    def set_numfmt(self, *v):
        self._set('numfmt', v)

    def set_strfmt(self, *v):
        self._set('strfmt', v)

    def set_titlefmt(self, *v):
        self._set('titlefmt', v)

    def set_pad(self, *v):
        self._set('pad', [int(x) if x not in (None, '.') else None for x in v])

    def set_strcolor(self, *v):
        self._set('strcolor', [{'green': 'txt', 'yellow': 'res'}.get(x, x) if x else None for x in v])

    def set_numcolor(self, *v):
        self._set('numcolor', [{'green': 'txt', 'yellow': 'res'}.get(x, x) if x else None for x in v])

    # --- output
    def _emit(self, segs):
        for txt, sty in segs:
            if sty == 'res':
                self.ip.res(txt)
            else:
                self.ip.txt(txt)
        self.ip.txt('\n')

    def sep(self, kind='mid'):
        left, mid, right = {'top': ('+', '+', '+'), 'mid': ('|', '+', '|'), 'bot': ('+', '+', '+')}[kind]
        s = ' ' * (self.start[0] - 1)
        if self.vb[0]:
            s += left
        s += '-' * self.w[0]
        for k in range(1, self.n):
            if self.vb[k]:
                s += mid
            s += '-' * self.w[k]
        if self.vb[self.n]:
            s += right
        self.ip.txt(s + '\n')

    def _line(self, toks, kinds):
        segs = []
        pos = [0]

        def put(t, sty):
            segs.append((t, sty))
            pos[0] += len(_visible(t))

        def col(c):
            if pos[0] < c - 1:
                put(' ' * (c - 1 - pos[0]), 'txt')
        for k in range(self.n):
            tok, kind = toks[k], kinds[k]
            if self.vb[k]:
                col(self.start[k])
                put('|', 'txt')
            elif tok not in (None, ''):
                col(self.start[k])
            if tok is None or tok == '':
                continue
            if kind == 'title':
                put(_fmt_s(tok, self.titlefmt[k]), 'txt')
            elif kind == 'str':
                put(_fmt_s(tok, self.strfmt[k]), self.strcolor[k])
            else:
                if self.ignore is not None and tok == self.ignore:
                    continue
                if self.pad[k]:
                    put(' ' * self.pad[k], self.numcolor[k])
                put(fmt_num(tok, self.numfmt[k]), self.numcolor[k])
        if self.vb[self.n]:
            col(self.start[self.n])
            put('|', 'txt')
        self._emit(segs)

    def titles(self, *toks):
        self._line(list(toks), ['title'] * self.n)

    def row(self, *toks):
        if self.sep_every and self.rows and self.rows % self.sep_every == 0:
            self.sep('mid')
        kinds = ['str' if isinstance(t, str) else 'num' for t in toks]
        self._line(list(toks), kinds)
        self.rows += 1


def _fmt_s(s, fmt):
    m = re.match(r'%(-)?(\d+)s', fmt)
    w = int(m.group(2)) if m else 0
    raw = s
    if len(raw) >= w:
        return _render(raw)
    padn = w - len(raw)
    return _render(raw + ' ' * padn) if (m and m.group(1)) else _render(' ' * padn + raw)


def _render(s):
    return _visible(s)
