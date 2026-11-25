"""In-memory Stata dataset."""
import copy
import fnmatch
import re

import numpy as np

from .errors import StataError

NUM_TYPES = ('byte', 'int', 'long', 'float', 'double')


def is_str_type(t):
    return t.startswith('str')


def store_cast(arr, typ):
    """Round values to the storage precision of a Stata type."""
    if is_str_type(typ):
        return arr
    a = np.asarray(arr, dtype=float)
    if typ == 'float':
        with np.errstate(over='ignore', invalid='ignore'):
            return a.astype(np.float32).astype(np.float64)
    if typ in ('byte', 'int', 'long'):
        out = a.copy()
        ok = ~np.isnan(out)
        out[ok] = np.trunc(out[ok])
        return out
    return a


def promote_type(typ, arr):
    """Promotion used by replace: integer types become float/double if needed."""
    if is_str_type(typ):
        return typ
    a = np.asarray(arr, dtype=float)
    ok = a[~np.isnan(a)]
    if typ in ('byte', 'int', 'long'):
        if ok.size and np.any(ok != np.trunc(ok)):
            return 'float' if typ in ('byte', 'int') else 'double'
        if ok.size:
            mx, mn = ok.max(), ok.min()
            if typ == 'byte' and (mx > 100 or mn < -127):
                typ = 'int'
            if typ == 'int' and (mx > 32740 or mn < -32767):
                typ = 'long'
            if typ == 'long' and (mx > 2147483620 or mn < -2147483647):
                typ = 'double'
    return typ


class Data:
    def __init__(self):
        self.vars = []
        self.cols = {}
        self.types = {}
        self.varlab = {}
        self.vallab = {}      # var -> label name
        self.formats = {}
        self.labels = {}      # label name -> {value: text}
        self.sortedby = []
        self.notes = []
        self.label = ''
        self.chars = {}
        self.ts = None        # dict(panel=, time=, delta=1)
        self.n = 0
        self.filename = ''

    # ------------------------------------------------------------------
    def copy(self):
        d = Data()
        d.vars = list(self.vars)
        d.cols = {k: v.copy() for k, v in self.cols.items()}
        d.types = dict(self.types)
        d.varlab = dict(self.varlab)
        d.vallab = dict(self.vallab)
        d.formats = dict(self.formats)
        d.labels = copy.deepcopy(self.labels)
        d.sortedby = list(self.sortedby)
        d.notes = list(self.notes)
        d.label = self.label
        d.ts = dict(self.ts) if self.ts else None
        d.n = self.n
        d.filename = self.filename
        return d

    @property
    def N(self):
        return self.n

    def set_obs(self, n):
        if n < self.n:
            raise StataError('observations cannot be reduced with set obs', 198)
        add = n - self.n
        for v in self.vars:
            if is_str_type(self.types[v]):
                self.cols[v] = np.concatenate([self.cols[v], np.array([''] * add, dtype=object)])
            else:
                self.cols[v] = np.concatenate([self.cols[v], np.full(add, np.nan)])
        self.n = n

    def has(self, name):
        return name in self.cols

    def isstr(self, name):
        return is_str_type(self.types[name])

    # ------------------------------------------------------------------
    def resolve(self, name, allow_abbrev=True, required=True):
        """Resolve a (possibly abbreviated or ~ wildcard) variable name."""
        if name in self.cols:
            return name
        if '~' in name:
            pat = name.replace('~', '*')
            m = [v for v in self.vars if fnmatch.fnmatchcase(v, pat)]
            if len(m) == 1:
                return m[0]
            if len(m) > 1:
                raise StataError(f'{name} ambiguous abbreviation', 111)
        elif allow_abbrev:
            m = [v for v in self.vars if v.startswith(name)]
            if len(m) == 1:
                return m[0]
            if len(m) > 1:
                raise StataError(f'{name} ambiguous abbreviation', 111)
        if required:
            raise StataError(f'variable {name} not found', 111)
        return None

    def expand_varlist(self, tokens, allow_new=False):
        out = []
        for t in tokens:
            if t in ('_all', '*') and not allow_new:
                out.extend(self.vars)
                continue
            if '-' in t and not t.startswith('-') and '*' not in t:
                a, b = t.split('-', 1)
                a, b = self.resolve(a), self.resolve(b)
                i, j = self.vars.index(a), self.vars.index(b)
                out.extend(self.vars[i:j + 1])
                continue
            if '*' in t or '?' in t:
                m = [v for v in self.vars if fnmatch.fnmatchcase(v, t)]
                if not m:
                    raise StataError(f'variable {t} not found', 111)
                out.extend(m)
                continue
            if allow_new and t not in self.cols:
                out.append(t)
            else:
                out.append(self.resolve(t))
        return out

    # ------------------------------------------------------------------
    def add_var(self, name, values, typ='float', fmt=None, label=None):
        check_name(name)
        if name in self.cols:
            raise StataError(f'variable {name} already defined', 110)
        if is_str_type(typ):
            vals = np.array(values, dtype=object)
        else:
            vals = store_cast(values, typ)
        self.vars.append(name)
        self.cols[name] = vals
        self.types[name] = typ
        self.formats[name] = fmt or default_format(typ)
        if label:
            self.varlab[name] = label

    def set_col(self, name, values):
        typ = self.types[name]
        if is_str_type(typ):
            self.cols[name] = np.array(values, dtype=object)
            mx = max((len(str(s)) for s in self.cols[name]), default=1)
            self.types[name] = 'str%d' % max(1, mx)
        else:
            self.cols[name] = store_cast(values, typ)

    def drop_vars(self, names):
        for n in names:
            self.vars.remove(n)
            for d in (self.cols, self.types, self.varlab, self.vallab, self.formats):
                d.pop(n, None)
        if self.ts and (self.ts.get('panel') in names or self.ts.get('time') in names):
            self.ts = None
        self.sortedby = [s for s in self.sortedby if s not in names]

    def keep_obs(self, mask):
        mask = np.asarray(mask, dtype=bool)
        for v in self.vars:
            self.cols[v] = self.cols[v][mask]
        self.n = int(mask.sum())

    def reorder(self, idx):
        for v in self.vars:
            self.cols[v] = self.cols[v][idx]

    def rename(self, old, new):
        check_name(new)
        if new in self.cols:
            raise StataError(f'{new} already defined', 110)
        i = self.vars.index(old)
        self.vars[i] = new
        for d in (self.cols, self.types, self.varlab, self.vallab, self.formats):
            if old in d:
                d[new] = d.pop(old)
        self.sortedby = [new if s == old else s for s in self.sortedby]
        if self.ts:
            for k in ('panel', 'time'):
                if self.ts.get(k) == old:
                    self.ts[k] = new

    def value_text(self, var, value):
        ln = self.vallab.get(var)
        if ln and ln in self.labels:
            lab = self.labels[ln]
            try:
                key = float(value)
            except (TypeError, ValueError):
                return None
            for k, t in lab.items():
                if float(k) == key:
                    return t
        return None

    def sort(self, keys, stable=True):
        if self.n == 0 or not keys:
            self.sortedby = list(keys)
            return
        arrays = []
        for k in reversed(keys):
            a = self.cols[k]
            if is_str_type(self.types[k]):
                arrays.append(np.array([str(s) for s in a], dtype=object))
            else:
                arrays.append(np.where(np.isnan(a), np.inf, a))
        idx = np.lexsort(arrays) if all(a.dtype != object for a in arrays) else _lexsort_obj(arrays)
        self.reorder(idx)
        self.sortedby = list(keys)


def _lexsort_obj(arrays):
    n = len(arrays[0])
    keys = list(zip(*reversed(arrays)))
    return np.array(sorted(range(n), key=lambda i: keys[i]), dtype=int)


_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]{0,31}$')
RESERVED = {'_all', '_b', '_coef', '_cons', '_n', '_N', '_pi', '_pred', '_rc', '_se', '_skip',
            'byte', 'double', 'float', 'if', 'in', 'int', 'long', 'using', 'with'}


def check_name(name):
    if not _NAME_RE.match(name) or name in RESERVED:
        raise StataError(f'{name} invalid name', 198)


def default_format(typ):
    if is_str_type(typ):
        try:
            w = int(typ[3:])
        except ValueError:
            w = 9
        return '%' + str(w) + 's'
    return {'byte': '%8.0g', 'int': '%8.0g', 'long': '%12.0g', 'float': '%9.0g',
            'double': '%10.0g'}[typ]


# ----------------------------------------------------------------------
_TYPEMAP = {'b': 'byte', 'h': 'int', 'l': 'long', 'f': 'float', 'd': 'double'}


def read_dta(path):
    import pandas as pd
    with pd.io.stata.StataReader(path) as r:
        df = r.read(convert_categoricals=False, convert_missing=False,
                    preserve_dtypes=True, convert_dates=False)
        try:
            varlabels = r.variable_labels()
        except Exception:
            varlabels = {}
        try:
            vlabs = r.value_labels()
        except Exception:
            vlabs = {}
        typlist = list(getattr(r, '_typlist', []))
        fmtlist = list(getattr(r, '_fmtlist', []))
        lbllist = list(getattr(r, '_lbllist', []))
        try:
            dlabel = r.data_label
        except Exception:
            dlabel = ''
    d = Data()
    d.n = len(df)
    for i, c in enumerate(df.columns):
        t = typlist[i] if i < len(typlist) else None
        col = df[c]
        if isinstance(t, int) or col.dtype == object:
            typ = 'str%d' % (t if isinstance(t, int) and t > 0 else max(1, int(col.astype(str).str.len().max() or 1)))
            vals = np.array(['' if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v) for v in col], dtype=object)
        else:
            typ = _TYPEMAP.get(t, 'double')
            vals = pd.to_numeric(col, errors='coerce').to_numpy(dtype=float)
        d.vars.append(c)
        d.cols[c] = vals
        d.types[c] = typ
        d.formats[c] = fmtlist[i] if i < len(fmtlist) and fmtlist[i] else default_format(typ)
        if varlabels.get(c):
            d.varlab[c] = varlabels[c]
        if i < len(lbllist) and lbllist[i]:
            d.vallab[c] = lbllist[i]
    for name, mp in (vlabs or {}).items():
        d.labels[name] = {float(k): str(v) for k, v in mp.items()}
    d.label = dlabel or ''
    return d


def write_dta(d, path):
    import pandas as pd
    cols = {}
    convert = {}
    for v in d.vars:
        if d.isstr(v):
            cols[v] = pd.Series(list(d.cols[v]), dtype=object)
        else:
            t = d.types[v]
            a = d.cols[v]
            if t in ('byte', 'int', 'long') and not np.isnan(a).any():
                dt = {'byte': 'int8', 'int': 'int16', 'long': 'int32'}[t]
                try:
                    cols[v] = pd.Series(a.astype(dt))
                except (OverflowError, ValueError):
                    cols[v] = pd.Series(a)
            elif t == 'float':
                cols[v] = pd.Series(a.astype(np.float32))
            else:
                cols[v] = pd.Series(a)
    df = pd.DataFrame(cols)
    vlabs = {}
    for v, ln in d.vallab.items():
        if ln in d.labels and not d.isstr(v):
            col = d.cols[v]
            ok = col[~np.isnan(col)]
            if ok.size and np.all(ok == np.trunc(ok)):
                vlabs[v] = {int(k): t for k, t in d.labels[ln].items()}
                df[v] = df[v].astype('float64')
    try:
        df.to_stata(path, write_index=False, version=118,
                    variable_labels={k: v[:80] for k, v in d.varlab.items() if k in d.cols},
                    value_labels=vlabs or None, data_label=(d.label or None))
    except TypeError:
        df.to_stata(path, write_index=False, version=118,
                    variable_labels={k: v[:80] for k, v in d.varlab.items() if k in d.cols})
