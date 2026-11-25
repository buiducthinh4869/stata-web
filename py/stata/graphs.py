"""Graphics: twoway, histogram, graph box/bar, coefplot, marginsplot, export (matplotlib)."""
import io
import math
import os
import re

import numpy as np
from scipy import stats as sps

from . import expr as E
from .errors import StataError
from .fmt import fmt_num, g_fmt, num_to_macro
from .interp import command, alias
from .syntax import parse_command, split_top, unquote, Options, quoted_strings, find_top, parse_numlist

_mpl = None


def mpl():
    global _mpl
    if _mpl is None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams.update({'axes.edgecolor': '#8a8f95', 'axes.linewidth': 0.8, 'xtick.color': '#3c4146',
                             'ytick.color': '#3c4146', 'axes.labelcolor': '#22272b', 'axes.titleweight': 'bold',
                             'figure.dpi': 110, 'legend.frameon': False, 'lines.solid_capstyle': 'round'})
        _mpl = plt
    return _mpl


# ---------------------------------------------------------------- colours
_COLORS = {
    'navy': (26, 71, 111), 'maroon': (144, 53, 59), 'forest_green': (85, 117, 47), 'dkorange': (227, 126, 0),
    'teal': (110, 142, 132), 'cranberry': (193, 5, 52), 'lavender': (147, 141, 210), 'khaki': (202, 194, 126),
    'sienna': (160, 82, 45), 'emidblue': (123, 146, 168), 'emerald': (45, 109, 102), 'brown': (156, 136, 71),
    'erose': (191, 161, 156), 'gold': (255, 210, 0), 'bluishgray': (217, 230, 235), 'black': (0, 0, 0),
    'white': (255, 255, 255), 'red': (255, 0, 0), 'blue': (0, 0, 255), 'green': (0, 128, 0),
    'yellow': (255, 255, 0), 'orange': (255, 127, 0), 'purple': (128, 0, 128), 'gray': (128, 128, 128),
    'grey': (128, 128, 128), 'dkgreen': (0, 100, 0), 'ltblue': (173, 216, 230), 'eltblue': (192, 218, 230),
    'edkblue': (62, 94, 126), 'dknavy': (0, 0, 64), 'olive': (128, 128, 0), 'olive_teal': (161, 188, 164),
    'pink': (255, 192, 203), 'magenta': (255, 0, 255), 'cyan': (0, 255, 255), 'lime': (0, 255, 0),
    'ltbluishgray': (214, 225, 230), 'ebblue': (0, 139, 188), 'mint': (152, 255, 152), 'sand': (194, 178, 128),
    'stone': (220, 210, 180), 'midblue': (0, 102, 204), 'midgreen': (0, 153, 0), 'mint_green': (150, 220, 160),
    'none': None, 'stc1': (26, 133, 255), 'stc2': (212, 17, 89), 'stc3': (0, 191, 127),
}
# Stata 18 'stcolor' scheme (cleaner than s2color)
PALETTE = ['#1a85ff', '#d41159', '#00a76f', '#ff9f1c', '#6a4c93', '#00a6c7', '#8c564b', '#e377c2',
           '#7f7f7f', '#bcbd22', '#17becf', '#1f3a93']


def stata_color(spec, default=None):
    """Return (r,g,b,a) tuple in 0..1 or None."""
    if spec is None:
        return default
    s = str(spec).strip().strip('"')
    if not s:
        return default
    alpha = 1.0
    m = re.match(r'^(.*?)%(\d+)$', s)
    if m:
        s, alpha = m.group(1), int(m.group(2)) / 100
    s = s.split()[0] if ' ' in s and not re.match(r'^\d', s) else s
    if s == 'none':
        return (0, 0, 0, 0)
    gm = re.match(r'^gs(\d+)$', s)
    if gm:
        v = int(gm.group(1)) / 16
        return (v, v, v, alpha)
    rm = re.match(r'^(\d+)\s+(\d+)\s+(\d+)$', s)
    if rm:
        return tuple(int(x) / 255 for x in rm.groups()) + (alpha,)
    if s.startswith('#'):
        h = s[1:]
        return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)) + (alpha,)
    c = _COLORS.get(s)
    if c is None:
        return default
    return tuple(x / 255 for x in c) + (alpha,)


_FONT = {'zero': 0.1, 'tiny': 6, 'vsmall': 7, 'small': 8.5, 'medsmall': 9.5, 'medium': 10.5, 'medlarge': 12,
         'large': 13.5, 'vlarge': 16, 'huge': 20}
_MSIZE = {'zero': 0.1, 'tiny': 4, 'vtiny': 2, 'vsmall': 9, 'small': 16, 'medsmall': 22, 'medium': 30,
          'medlarge': 40, 'large': 55, 'vlarge': 80, 'huge': 120}
_LW = {'none': 0, 'vvthin': 0.3, 'vthin': 0.5, 'thin': 0.8, 'medthin': 1.1, 'medium': 1.4, 'medthick': 2.0,
       'thick': 2.6, 'vthick': 3.4}
_LP = {'solid': '-', 'dash': (0, (5, 3)), 'dot': (0, (1, 2)), 'shortdash': (0, (3, 2)), 'longdash': (0, (8, 3)),
       'dash_dot': (0, (5, 2, 1, 2)), 'shortdash_dot': (0, (3, 2, 1, 2)), 'longdash_dot': (0, (8, 2, 1, 2)),
       'blank': (0, (0, 10))}
_MS = {'O': ('o', True), 'Oh': ('o', False), 'o': ('o', True), 'oh': ('o', False), 'D': ('D', True),
       'Dh': ('D', False), 'd': ('D', True), 'dh': ('D', False), 'S': ('s', True), 'Sh': ('s', False),
       's': ('s', True), 'sh': ('s', False), 'T': ('^', True), 'Th': ('^', False), 't': ('^', True),
       'th': ('^', False), 'X': ('x', True), 'x': ('x', True), '+': ('+', True), 'p': ('.', True),
       'point': ('.', True), 'i': ('', True), 'none': ('', True), 'circle': ('o', True),
       'circle_hollow': ('o', False), 'diamond': ('D', True), 'square': ('s', True), 'triangle': ('^', True)}


def _size(s, table, default):
    if s is None or s == '':
        return default
    s = str(s).strip().split(',')[0].split()[0] if str(s).strip() else ''
    if s in table:
        return table[s]
    try:
        return float(s) * (1.2 if table is _FONT else 1.0)
    except ValueError:
        m = re.match(r'^\*?([\d.]+)$', s)
        return default * float(m.group(1)) if m else default


# ---------------------------------------------------------------- figure wrapper
class Figure:
    def __init__(self, ip, nrows=1, ncols=1, figsize=None, share=False):
        plt = mpl()
        self.ip = ip
        self.fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (7.5 if ncols == 1 else 10, 5.4),
                                      sharex=share, sharey=share)
        self.bottom_extra = 0.0
        self.axes = list(np.atleast_1d(axes).ravel())
        self.ax = self.axes[0]
        self.legend_items = []
        self.legend_default = False
        self.series_labels = []
        self.title = None
        self.titles = {}
        self.region = None
        for ax in self.axes:
            _style_axes(ax)
        self.fig.patch.set_facecolor('white')
        self.legend_opts = None

    def set_default_titles(self, title=None, xtitle=None, ytitle=None):
        if title is not None:
            self.titles.setdefault('title', title)
        if xtitle is not None:
            self.titles.setdefault('xtitle', xtitle)
        if ytitle is not None:
            self.titles.setdefault('ytitle', ytitle)


def _style_axes(ax):
    ax.set_facecolor('white')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color='#e6e9ec', lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5, length=3)


def kdensity(x, w=None, n=200, bw=None, kernel='epan'):
    x = np.asarray(x, dtype=float)
    ok = ~np.isnan(x)
    x = x[ok]
    if w is not None:
        w = np.asarray(w, dtype=float)[ok]
        w = w / w.sum()
    else:
        w = np.full(len(x), 1 / len(x))
    if len(x) < 2:
        return np.array([]), np.array([])
    if bw is None:
        sd = math.sqrt(np.sum(w * (x - np.sum(w * x)) ** 2) * len(x) / (len(x) - 1))
        q75, q25 = np.quantile(x, [0.75, 0.25])
        m = min(sd, (q75 - q25) / 1.349) if q75 > q25 else sd
        bw = 0.9 * m / len(x) ** 0.2
        if bw <= 0:
            bw = sd or 1
    grid = np.linspace(x.min(), x.max(), n)
    u = (grid[:, None] - x[None, :]) / bw
    if kernel.startswith('gau'):
        K = np.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)
    else:
        K = np.where(np.abs(u) < math.sqrt(5), 0.75 * (1 - u * u / 5) / math.sqrt(5), 0)
    dens = (K * w[None, :]).sum(axis=1) / bw
    return grid, dens


# ---------------------------------------------------------------- option helpers
def _text_arg(raw):
    """Return (list_of_lines, Options) for a title-like option argument."""
    if raw is None:
        return None, Options('')
    k = find_top(raw, ',')
    main = raw if k < 0 else raw[:k]
    opts = Options(raw[k + 1:] if k >= 0 else '')
    lines = quoted_strings(main)
    if not lines and main.strip():
        lines = [main.strip()]
    return lines, opts


def apply_global_options(fig, opts):
    """Apply Stata global graph options to a Figure (titles, axes, lines, legend...)."""
    ax = fig.ax
    for key, mpl_name in (('title', 'title'), ('subtitle', 'subtitle'), ('xtitle', 'xtitle'),
                          ('ytitle', 'ytitle'), ('note', 'note'), ('caption', 'caption')):
        raw = opts.get(key)
        if raw is not None:
            lines, o = _text_arg(raw)
            fig.titles[key] = ('\n'.join(lines) if lines else '', o)
    for raw in opts.all('xline'):
        _ref_lines(ax, raw, 'x')
    for raw in opts.all('yline'):
        _ref_lines(ax, raw, 'y')
    for key in ('xlabel', 'ylabel'):
        raw = opts.get(key)
        if raw is not None:
            fig.titles[key] = raw
    for key in ('xscale', 'yscale'):
        raw = opts.get(key)
        if raw:
            if 'log' in raw:
                (ax.set_xscale if key == 'xscale' else ax.set_yscale)('log')
            m = re.search(r'range\(([^)]*)\)', raw)
            if m:
                vals = [float(v) for v in m.group(1).replace(',', ' ').split()]
                if len(vals) >= 2:
                    (ax.set_xlim if key == 'xscale' else ax.set_ylim)(vals[0], vals[-1])
    for raw in opts.all('text'):
        _text_annot(ax, raw)
    if opts.has('legend'):
        fig.legend_opts = opts.get('legend')
    gr = opts.get('graphregion')
    if gr:
        m = re.search(r'(?:f?color)\(([^)]*)\)', gr)
        if m:
            c = stata_color(m.group(1))
            if c:
                fig.fig.patch.set_facecolor(c)
    pr = opts.get('plotregion')
    if pr:
        m = re.search(r'(?:f?color)\(([^)]*)\)', pr)
        if m:
            c = stata_color(m.group(1))
            if c:
                for a in fig.axes:
                    a.set_facecolor(c)
    bgc = opts.get('bgcolor')
    if bgc:
        c = stata_color(bgc)
        if c:
            for a in fig.axes:
                a.set_facecolor(c)
    if opts.has('name'):
        fig.name = opts.get('name').split(',')[0].strip()
    if opts.has('scheme'):
        sc = opts.get('scheme')
        if 's1mono' in sc or 'white' in sc or 'plotplain' in sc:
            fig.fig.patch.set_facecolor('white')


def _ref_lines(ax, raw, axis):
    k = find_top(raw, ',')
    main = raw if k < 0 else raw[:k]
    o = Options(raw[k + 1:] if k >= 0 else '')
    vals = []
    for t in main.split():
        try:
            vals.extend(parse_numlist(t))
        except StataError:
            pass
    col = stata_color(o.get('lcolor') or o.get('lc'), stata_color('black' if axis == 'x' else 'gs8'))
    lw = _size(o.get('lwidth') or o.get('lw'), _LW, 0.9)
    ls = _LP.get((o.get('lpattern') or o.get('lp') or 'solid').strip(), '-')
    for v in vals:
        if axis == 'x':
            ax.axvline(v, color=col, lw=lw, ls=ls, zorder=1)
        else:
            ax.axhline(v, color=col, lw=lw, ls=ls, zorder=1)


def _text_annot(ax, raw):
    k = find_top(raw, ',')
    main = raw if k < 0 else raw[:k]
    o = Options(raw[k + 1:] if k >= 0 else '')
    strs = quoted_strings(main)
    nums = re.findall(r'-?[\d.]+', main.split('"')[0])
    if len(nums) >= 2 and strs:
        y, x = float(nums[0]), float(nums[1])
        place = (o.get('placement') or o.get('place') or 'c').strip()
        ha = {'e': 'left', 'w': 'right', 'c': 'center', 'n': 'center', 's': 'center', 'ne': 'left',
              'nw': 'right', 'se': 'left', 'sw': 'right'}.get(place, 'center')
        va = {'n': 'bottom', 's': 'top', 'ne': 'bottom', 'nw': 'bottom', 'se': 'top', 'sw': 'top'}.get(place, 'center')
        ax.text(x, y, '\n'.join(strs), ha=ha, va=va, fontsize=_size(o.get('size'), _FONT, 8.5))


def _apply_axis_labels(fig, d=None):
    ax = fig.ax
    for key in ('xlabel', 'ylabel'):
        raw = fig.titles.get(key)
        if raw is None:
            continue
        k = find_top(raw, ',')
        main = raw if k < 0 else raw[:k]
        o = Options(raw[k + 1:] if k >= 0 else '')
        toks = split_top(main)
        ticks, labels = [], []
        i = 0
        while i < len(toks):
            t = toks[i]
            if i + 1 < len(toks) and (toks[i + 1].startswith('"') or toks[i + 1].startswith('`"')):
                try:
                    ticks.append(float(t))
                    labels.append(unquote(toks[i + 1]))
                except ValueError:
                    pass
                i += 2
                continue
            try:
                for v in parse_numlist(t):
                    ticks.append(v)
                    labels.append(None)
            except StataError:
                pass
            i += 1
        fmt = o.get('format')
        setter = ax.set_xticks if key == 'xlabel' else ax.set_yticks
        lab_setter = ax.set_xticklabels if key == 'xlabel' else ax.set_yticklabels
        vl = getattr(fig, 'valuelabels_' + key[0], None)
        if ticks:
            setter(ticks)
            out = []
            for tv, lb in zip(ticks, labels):
                if lb is not None:
                    out.append(lb.replace('{newline}', '\n'))
                elif o.has('valuelabel', 5) and vl:
                    out.append(vl.get(float(tv), g_fmt(tv)))
                elif fmt:
                    out.append(fmt_num(tv, fmt).strip())
                else:
                    out.append(g_fmt(tv))
            lab_setter(out)
        elif fmt:
            from matplotlib.ticker import FuncFormatter
            f = FuncFormatter(lambda v, pos, fmt=fmt: fmt_num(v, fmt).strip())
            (ax.xaxis if key == 'xlabel' else ax.yaxis).set_major_formatter(f)
        elif o.has('valuelabel', 5) and vl:
            from matplotlib.ticker import FuncFormatter
            (ax.xaxis if key == 'xlabel' else ax.yaxis).set_major_formatter(
                FuncFormatter(lambda v, pos, vl=vl: vl.get(float(round(v)), '') if abs(v - round(v)) < 1e-9 else ''))
        ang = o.get('angle')
        if ang:
            try:
                a = float(ang.split()[0]) if ang.split()[0] not in ('horizontal', 'vertical') else (0 if ang.startswith('h') else 90)
            except ValueError:
                a = 0
            for lb in (ax.get_xticklabels() if key == 'xlabel' else ax.get_yticklabels()):
                lb.set_rotation(a)
                if key == 'xlabel' and a:
                    lb.set_ha('right')
        ls = o.get('labsize')
        if ls:
            ax.tick_params(axis=key[0], labelsize=_size(ls, _FONT, 8.5))
        if o.has('noticks'):
            ax.tick_params(axis=key[0], length=0)


def finish_graph(ip, fig, opts, name=None):
    """Titles, legend, layout; render to PNG and register."""
    plt = mpl()
    ax = fig.ax
    if opts is not None and opts is not False:
        pass
    t = fig.titles
    _apply_axis_labels(fig)

    def txt(key):
        v = t.get(key)
        if v is None:
            return None, Options('')
        if isinstance(v, tuple):
            return v
        return v, Options('')
    xt, xo = txt('xtitle')
    yt, yo = txt('ytitle')
    if xt is not None:
        ax.set_xlabel(xt, fontsize=_size(xo.get('size'), _FONT, 9.5))
    if yt is not None:
        ax.set_ylabel(yt, fontsize=_size(yo.get('size'), _FONT, 9.5))
    title, to = txt('title')
    sub, so = txt('subtitle')
    note, no_ = txt('note')
    cap, co = txt('caption')
    # legend
    _do_legend(fig)
    top = 0.975
    H = fig.fig.get_figheight()
    lh = 0.26 / H
    if title:
        fig.fig.suptitle(title, fontsize=_size(to.get('size'), _FONT, 13), y=0.985, fontweight='bold',
                         color=stata_color(to.get('color'), (0.1, 0.12, 0.14, 1)))
        top -= lh * (title.count('\n') + 1) + 0.01
    if sub:
        fig.fig.text(0.5, top, sub, ha='center', va='top', fontsize=_size(so.get('size'), _FONT, 10),
                     color=stata_color(so.get('color'), (0.3, 0.32, 0.35, 1)))
        top -= lh * 0.85 * (sub.count('\n') + 1)
    bottom = 0.01 + getattr(fig, 'bottom_extra', 0.0)
    if note:
        fig.fig.text(0.015, 0.012, note, ha='left', va='bottom', fontsize=_size(no_.get('size'), _FONT, 7.5),
                     color=stata_color(no_.get('color'), (0.35, 0.37, 0.4, 1)))
        bottom += (0.2 / H) * (note.count('\n') + 1) + 0.01
    if cap:
        fig.fig.text(0.02, bottom, cap, ha='left', va='bottom', fontsize=7.5)
        bottom += 0.035
    try:
        fig.fig.tight_layout(rect=(0, bottom, 1, top if (title or sub) else 0.995))
    except Exception:  # noqa: BLE001
        pass
    nm = name or getattr(fig, 'name', None) or 'Graph'
    register_graph(ip, fig.fig, nm)


def _do_legend(fig):
    ax = fig.ax
    items = fig.legend_items
    lo = fig.legend_opts
    if lo is not None:
        k = find_top(lo, ',')
        o = Options(lo)
        if lo.strip().startswith('off') or o.has('off'):
            return
        labels = {}
        for raw in o.all('label', 3):
            m = re.match(r'^\s*(\d+)\s+(.*)$', raw, re.S)
            if m:
                q = quoted_strings(m.group(2))
                labels[int(m.group(1))] = '\n'.join(q) if q else m.group(2)
        order = None
        raw = o.get('order')
        if raw:
            order = []
            toks = split_top(raw)
            i = 0
            while i < len(toks):
                try:
                    num = int(toks[i])
                except ValueError:
                    i += 1
                    continue
                lab = None
                j = i + 1
                q = []
                while j < len(toks) and (toks[j].startswith('"') or toks[j].startswith('`"')):
                    q.append(unquote(toks[j]))
                    j += 1
                if q:
                    lab = '\n'.join(q)
                order.append((num, lab))
                i = j
        handles = []
        names = []
        seq = order if order else [(i + 1, None) for i in range(len(items))]
        for num, lab in seq:
            if 1 <= num <= len(items):
                h, l0 = items[num - 1]
                l = lab or labels.get(num) or l0
                handles.append(h)
                names.append(l)
        if not handles:
            return
        pos = o.get('position') or o.get('pos') or '6'
        ring = o.get('ring')
        rows = o.get('rows') or o.get('row')
        cols = o.get('cols') or o.get('col')
        ncol = int(cols) if cols else (max(1, math.ceil(len(handles) / int(rows))) if rows else min(2, len(handles)))
        _place_legend(ax, handles, names, pos.strip(), ring, ncol, o.get('size'))
        return
    if items and (len(items) > 1 or fig.legend_default):
        handles = [h for h, _ in items]
        names = [l for _, l in items]
        _place_legend(ax, handles, names, '6', None, min(2, len(handles)), None)


def _place_legend(ax, handles, names, pos, ring, ncol, size):
    fs = _size(size, _FONT, 8.5)
    inside = ring is not None and ring.strip() == '0'
    locmap = {'12': 'upper center', '1': 'upper right', '2': 'upper right', '3': 'center right',
              '4': 'lower right', '5': 'lower right', '6': 'lower center', '7': 'lower left', '8': 'lower left',
              '9': 'center left', '10': 'upper left', '11': 'upper left', '0': 'center'}
    try:
        p = str(int(float(pos)))
    except ValueError:
        p = '6'
    if inside:
        ax.legend(handles, names, loc=locmap.get(p, 'best'), ncol=ncol, fontsize=fs, frameon=True)
        return
    if p == '6':
        ax.legend(handles, names, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=ncol, fontsize=fs,
                  frameon=True, edgecolor='#999999')
    elif p in ('3', '2', '4'):
        ax.legend(handles, names, loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=1, fontsize=fs)
    elif p in ('12', '1', '11'):
        ax.legend(handles, names, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=ncol, fontsize=fs)
    else:
        ax.legend(handles, names, loc=locmap.get(p, 'best'), ncol=ncol, fontsize=fs)


def register_graph(ip, mfig, name='Graph'):
    buf = io.BytesIO()
    mfig.savefig(buf, format='png', dpi=110, facecolor=mfig.get_facecolor())
    png = buf.getvalue()
    if not hasattr(ip, 'figs'):
        ip.figs = {}
    old = ip.figs.get(name)
    if old is not None and old is not mfig:
        mpl().close(old)
    ip.figs[name] = mfig
    ip.graphs[name] = png
    if name in ip.graph_order:
        ip.graph_order.remove(name)
    ip.graph_order.append(name)
    ip.current_graph = name
    for f in ip.graph_listeners:
        f(name, png)


# ---------------------------------------------------------------- twoway
TW_TYPES = {'scatter', 'line', 'connected', 'lfit', 'qfit', 'lfitci', 'qfitci', 'kdensity', 'histogram', 'bar',
            'rcap', 'rspike', 'rarea', 'area', 'function', 'lpoly', 'lowess', 'rbar', 'spike', 'dropline',
            'tsline', 'rcapsym', 'scatteri', 'dot'}


def _split_plots(body):
    """Return (list of plot strings, global option string)."""
    s = body.strip()
    plots = []
    if s.startswith('('):
        i = 0
        while i < len(s):
            while i < len(s) and s[i].isspace():
                i += 1
            if i < len(s) and s[i] == '(':
                depth = 0
                j = i
                inq = False
                while j < len(s):
                    ch = s[j]
                    if ch == '"':
                        inq = not inq
                    elif not inq:
                        if ch == '(':
                            depth += 1
                        elif ch == ')':
                            depth -= 1
                            if depth == 0:
                                break
                    j += 1
                plots.append(s[i + 1:j])
                i = j + 1
                continue
            if s[i:i + 2] == '||':
                i += 2
                continue
            break
        rest = s[i:].strip()
        if rest.startswith(','):
            rest = rest[1:]
        return plots, rest
    parts = [p_.strip() for p_ in re.split(r'\|\|', s)]
    last = parts[-1]
    k = find_top(last, ',')
    gl = ''
    if len(parts) == 1:
        return [last], ''
    if k >= 0:
        gl = last[k + 1:]
        parts[-1] = last[:k]
    # trailing global options after the last || segment
    return parts, gl


@command('twoway', 'tw', 'scatter', 'line', 'connected', 'histogram', 'hist', 'histo', 'kdensity')
def cmd_twoway(ip, cmd, rest, by):
    if cmd in ('histogram', 'hist', 'histo'):
        return cmd_histogram(ip, cmd, rest, by)
    if cmd == 'kdensity':
        return _standalone_kdensity(ip, rest)
    if cmd in ('scatter', 'line', 'connected'):
        rest = cmd + ' ' + rest
    plots, gl = _split_plots(rest)
    fig = Figure(ip)
    gopts = Options(gl)
    d = ip.data
    vl_x = vl_y = None
    if len(plots) == 1 and find_top(plots[0], ',') >= 0 and not rest.strip().startswith('('):
        # single plot: options after comma are both plot and global options
        k = find_top(plots[0], ',')
        gopts = Options(plots[0][k + 1:])
    byraw = gopts.get('by')
    if byraw:
        return _twoway_by(ip, plots, gopts, byraw)
    k_series = 0
    for ptxt in plots:
        k_series = _draw_plot(ip, fig, ptxt, k_series)
    xvar = getattr(fig, 'xvar', None)
    yvar = getattr(fig, 'yvar', None)
    if xvar and xvar in d.vallab:
        fig.valuelabels_x = d.labels.get(d.vallab[xvar], {})
    if yvar and yvar in d.vallab:
        fig.valuelabels_y = d.labels.get(d.vallab[yvar], {})
    if not hasattr(fig, 'valuelabels_y') and getattr(fig, 'yvar_is_cat', None):
        pass
    fig.set_default_titles(xtitle=getattr(fig, 'xtitle_default', None), ytitle=getattr(fig, 'ytitle_default', None))
    apply_global_options(fig, gopts)
    finish_graph(ip, fig, gopts)


def _by_levels(d, byraw):
    k = find_top(byraw, ',')
    var = d.resolve((byraw if k < 0 else byraw[:k]).split()[0])
    bo = Options(byraw[k + 1:] if k >= 0 else '')
    x = d.cols[var]
    if d.isstr(var):
        lv = sorted(set(x) - {''})
    else:
        lv = sorted(np.unique(x[~np.isnan(x)]))
    return var, lv, bo


def _twoway_by(ip, plots, gopts, byraw):
    d = ip.data
    var, lv, bo = _by_levels(d, byraw)
    n = len(lv)
    cols = int(bo.get('cols') or 0) or (min(n, 3) if n > 2 else n)
    rows = int(bo.get('rows') or 0) or math.ceil(n / cols)
    cols = math.ceil(n / rows)
    fig = Figure(ip, rows, cols, figsize=(min(4.2 * cols, 13), 3.4 * rows + 0.9), share=not bo.has('noixaxes'))
    for k, ax in enumerate(fig.axes):
        if k >= n:
            ax.set_visible(False)
            continue
        sub = Figure.__new__(Figure)
        sub.__dict__.update(fig.__dict__)
        sub.ax = ax
        sub.legend_items = []
        sub.extra_mask = (d.cols[var] == lv[k])
        ks = 0
        for ptxt in plots:
            ks = _draw_plot(ip, sub, ptxt, ks)
        lab = lv[k] if d.isstr(var) else (d.value_text(var, lv[k]) or g_fmt(lv[k]))
        ax.set_title(str(lab), fontsize=9.5, fontweight='bold', color='#22272b')
        if k == 0:
            fig.legend_items = sub.legend_items
            fig.xtitle_default = getattr(sub, 'xtitle_default', None)
            fig.ytitle_default = getattr(sub, 'ytitle_default', None)
    if not bo.has('noixtitle'):
        pass
    xt = fig.__dict__.get('xtitle_default')
    yt = fig.__dict__.get('ytitle_default')
    t = bo.get('title')
    if t:
        gopts.items.append(('title', t))
    apply_global_options(fig, gopts)
    for key, val in (('xtitle', xt), ('ytitle', yt)):
        v = fig.titles.pop(key, None)
        txt = v[0] if isinstance(v, tuple) else (v if v is not None else val)
        if txt:
            if key == 'xtitle':
                fig.fig.supxlabel(txt, fontsize=9.5, y=0.075 if len(fig.legend_items) > 1 else 0.02)
            else:
                fig.fig.supylabel(txt, fontsize=9.5)
    if len(fig.legend_items) > 1 and not (fig.legend_opts or '').startswith('off'):
        fig.fig.legend([h for h, _ in fig.legend_items], [l for _, l in fig.legend_items], loc='lower center',
                       bbox_to_anchor=(0.5, 0.0), ncol=min(4, len(fig.legend_items)), fontsize=8.5, frameon=False)
        fig.bottom_extra = 0.08
        fig.legend_items = []
    fig.legend_opts = 'off'
    note = fig.titles.get('note')
    if not note:
        fig.titles['note'] = (f'Graphs by {d.varlab.get(var, var)}', Options(''))
    finish_graph(ip, fig, gopts)


def _plot_opts(ptxt):
    k = find_top(ptxt, ',')
    main = ptxt if k < 0 else ptxt[:k]
    o = Options(ptxt[k + 1:] if k >= 0 else '')
    return main.strip(), o


def _marker(o, idx):
    col = stata_color(o.get('mcolor') or o.get('mc') or o.get('color'), stata_color(PALETTE[idx % len(PALETTE)]))
    ms = (o.get('msymbol') or o.get('ms') or 'O').strip()
    sym, filled = _MS.get(ms, ('o', True))
    size = _size(o.get('msize'), _MSIZE, 22)
    return col, sym, filled, size


def _line(o, idx, default_lw=1.4):
    col = stata_color(o.get('lcolor') or o.get('lc') or o.get('color'), stata_color(PALETTE[idx % len(PALETTE)]))
    lw = _size(o.get('lwidth') or o.get('lw'), _LW, default_lw)
    ls = _LP.get((o.get('lpattern') or o.get('lp') or 'solid').strip(), '-')
    return col, lw, ls


def _draw_plot(ip, fig, ptxt, idx):
    ax = fig.ax
    d = ip.data
    ptxt = ptxt.strip()
    m = re.match(r'^(twoway\s+)?(\w+)\s*(.*)$', ptxt, re.S)
    typ = m.group(2)
    body = m.group(3)
    if typ in ('tw', 'twoway'):
        return _draw_plot(ip, fig, body, idx)
    if typ == 'sc':
        typ = 'scatter'
    if typ not in TW_TYPES:
        # implied scatter: "y x"
        body = ptxt
        typ = 'scatter'
    p = parse_command(body)
    o = p.opts
    if typ == 'function':
        return _draw_function(ip, fig, p, idx)
    vars_ = d.expand_varlist(split_top(p.body))
    mask = ip.mark(p)
    if getattr(fig, 'extra_mask', None) is not None:
        mask = mask & fig.extra_mask
    label_from_plot = None
    for raw in o.all('legend'):
        mm = re.search(r'label\(\s*\d+\s+"([^"]*)"', raw)
        if mm:
            label_from_plot = mm.group(1)
    horizontal = o.has('horizontal', 3)

    def col(v):
        return d.cols[v].astype(float)
    if typ in ('scatter', 'line', 'connected', 'bar', 'area', 'spike', 'dropline', 'tsline', 'dot'):
        xv = vars_[-1]
        ys = vars_[:-1] if typ != 'tsline' else vars_
        if typ == 'tsline':
            xv = d.ts['time'] if d.ts else vars_[-1]
        x = col(xv)
        fig.xvar = xv
        fig.xtitle_default = d.varlab.get(xv, xv)
        if len(ys) == 1:
            fig.yvar = ys[0]
            fig.ytitle_default = d.varlab.get(ys[0], ys[0])
        for yv in ys:
            y = col(yv)
            sel = mask & ~np.isnan(x) & ~np.isnan(y)
            xs, ysel = x[sel], y[sel]
            lab = label_from_plot or d.varlab.get(yv, yv)
            if horizontal:
                xs, ysel = ysel, xs
            if typ == 'scatter':
                c, sym, filled, size = _marker(o, idx)
                jit = o.get('jitter')
                if jit:
                    rng = np.random.default_rng(0)
                    span = (np.ptp(ysel) or 1) * float(jit) / 100
                    ysel = ysel + rng.uniform(-span, span, len(ysel))
                    xs = xs + rng.uniform(-(np.ptp(xs) or 1) * float(jit) / 100, (np.ptp(xs) or 1) * float(jit) / 100, len(xs))
                h = ax.scatter(xs, ysel, s=size, marker=sym or 'o',
                               facecolors=c if filled else 'none', edgecolors=c, linewidths=0.8, zorder=3)
                if not sym:
                    h.set_visible(False)
                mlab = o.get('mlabel')
                if mlab:
                    lv = d.resolve(mlab.split()[0])
                    for xx, yy_, tt in zip(xs, ysel, d.cols[lv][sel]):
                        ax.annotate(str(tt), (xx, yy_), fontsize=7, xytext=(3, 3), textcoords='offset points')
            elif typ in ('line', 'connected', 'tsline'):
                c, lw, ls = _line(o, idx)
                if o.has('sort'):
                    oo = np.argsort(xs, kind='stable')
                    xs, ysel = xs[oo], ysel[oo]
                connect = o.get('connect') or o.get('c') or ''
                if 'asc' in connect or connect == 'L':
                    breaks = np.flatnonzero(np.diff(xs) < 0) + 1
                    segs = np.split(np.arange(len(xs)), breaks)
                else:
                    segs = [np.arange(len(xs))]
                h = None
                for sg in segs:
                    h, = ax.plot(xs[sg], ysel[sg], color=c, lw=lw, ls=ls,
                                 marker=('o' if typ == 'connected' else None), ms=4, zorder=2)
            elif typ in ('bar', 'spike', 'dropline'):
                c = stata_color(o.get('color') or o.get('fcolor') or o.get('bcolor'), stata_color(PALETTE[idx % len(PALETTE)]))
                if c is None and o.get('color'):
                    c = stata_color(o.get('color').split()[0])
                bw = float(o.get('barwidth') or 0.8)
                if typ == 'bar':
                    h = ax.bar(xs, ysel, width=bw, color=c,
                               edgecolor=stata_color(o.get('lcolor'), c), zorder=2)
                else:
                    h = ax.vlines(xs, 0, ysel, color=c)
            elif typ == 'area':
                c = stata_color(o.get('color'), stata_color(PALETTE[idx % len(PALETTE)]))
                h = ax.fill_between(xs, 0, ysel, color=c, lw=0)
            else:
                c, sym, filled, size = _marker(o, idx)
                h = ax.scatter(xs, ysel, color=c, s=size)
            fig.legend_items.append((h, lab))
            idx += 1
        return idx
    if typ in ('lfit', 'qfit', 'lfitci', 'qfitci', 'lpoly', 'lowess'):
        yv, xv = vars_[0], vars_[1]
        x, y = col(xv), col(yv)
        sel = mask & ~np.isnan(x) & ~np.isnan(y)
        xs, ys_ = x[sel], y[sel]
        grid = np.linspace(xs.min(), xs.max(), 100)
        deg = 2 if typ.startswith('q') else 1
        c, lw, ls = _line(o, idx, 1.6)
        if typ in ('lpoly', 'lowess'):
            gx, gy = _lpoly(xs, ys_, grid)
            h, = ax.plot(gx, gy, color=c, lw=lw, ls=ls)
        else:
            X = np.vander(xs, deg + 1)
            b, *_ = np.linalg.lstsq(X, ys_, rcond=None)
            G = np.vander(grid, deg + 1)
            yhat = G @ b
            if typ.endswith('ci'):
                e = ys_ - X @ b
                s2 = e @ e / (len(xs) - deg - 1)
                XtXi = np.linalg.inv(X.T @ X)
                se = np.sqrt(np.einsum('ij,jk,ik->i', G, XtXi, G) * s2)
                tcrit = sps.t.ppf(0.975, len(xs) - deg - 1)
                ax.fill_between(grid, yhat - tcrit * se, yhat + tcrit * se, color=(0.8, 0.8, 0.8, 0.6), lw=0)
            h, = ax.plot(grid, yhat, color=c, lw=lw, ls=ls)
        fig.legend_items.append((h, label_from_plot or 'Fitted values'))
        fig.xvar = xv
        fig.xtitle_default = fig.xtitle_default if getattr(fig, 'xtitle_default', None) else d.varlab.get(xv, xv)
        if not getattr(fig, 'ytitle_default', None):
            fig.ytitle_default = 'Fitted values'
        return idx + 1
    if typ == 'kdensity':
        v = vars_[0]
        x = col(v)[mask]
        bw = o.get('bwidth') or o.get('bw')
        gx, gy = kdensity(x, bw=float(bw) if bw else None, kernel=(o.get('kernel') or 'epan'))
        c, lw, ls = _line(o, idx)
        h, = ax.plot(gx, gy, color=c, lw=lw, ls=ls)
        fc = o.get('color') or o.get('fcolor')
        if o.get('recast') == 'area' or (fc and '%' in fc and o.has('lcolor')) or (fc and not o.has('lcolor')):
            if fc:
                ax.fill_between(gx, 0, gy, color=stata_color(fc), lw=0)
        fig.legend_items.append((h, label_from_plot or f'kdensity {v}'))
        fig.xtitle_default = fig.__dict__.get('xtitle_default') or d.varlab.get(v, v)
        fig.ytitle_default = fig.__dict__.get('ytitle_default') or 'kdensity ' + v
        fig.xvar = v
        return idx + 1
    if typ == 'histogram':
        v = vars_[0]
        x = col(v)[mask]
        x = x[~np.isnan(x)]
        edges = _hist_edges(x, o)
        cnt, _ = np.histogram(x, edges)
        dens = cnt / (len(x) * np.diff(edges))
        c = stata_color(o.get('color') or o.get('fcolor'), stata_color(PALETTE[idx % len(PALETTE)], None))
        lc = stata_color(o.get('lcolor'), c)
        yvals = cnt if o.has('frequency', 4) else (cnt / len(x) if o.has('fraction', 4) else
                                                   (100 * cnt / len(x) if o.has('percent') else dens))
        h = ax.bar(edges[:-1], yvals, width=np.diff(edges), align='edge', color=c, edgecolor=lc, lw=0.6)
        fig.legend_items.append((h, label_from_plot or d.varlab.get(v, v)))
        fig.xtitle_default = d.varlab.get(v, v)
        fig.ytitle_default = 'Frequency' if o.has('frequency', 4) else 'Density'
        return idx + 1
    if typ in ('rcap', 'rspike', 'rarea', 'rbar', 'rcapsym'):
        y1, y2, xv = vars_[0], vars_[1], vars_[2]
        a, b, x = col(y1), col(y2), col(xv)
        sel = mask & ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(x)
        c, lw, ls = _line(o, idx)
        if typ == 'rarea':
            oo = np.argsort(x[sel])
            fc = stata_color(o.get('color') or o.get('fcolor'), (c[0], c[1], c[2], 0.3))
            h = ax.fill_between(x[sel][oo], a[sel][oo], b[sel][oo], color=fc, lw=0)
        elif horizontal:
            h = ax.hlines(x[sel], a[sel], b[sel], color=c, lw=lw)
            if typ == 'rcap':
                ax.scatter(a[sel], x[sel], marker='|', color=c, s=80)
                ax.scatter(b[sel], x[sel], marker='|', color=c, s=80)
        else:
            if typ == 'rbar':
                h = ax.bar(x[sel], b[sel] - a[sel], bottom=a[sel], width=float(o.get('barwidth') or 0.6), color=c)
            else:
                h = ax.vlines(x[sel], a[sel], b[sel], color=c, lw=lw)
                if typ in ('rcap', 'rcapsym'):
                    ax.scatter(x[sel], a[sel], marker='_', color=c, s=80)
                    ax.scatter(x[sel], b[sel], marker='_', color=c, s=80)
        fig.legend_items.append((h, label_from_plot or f'{y1}/{y2}'))
        fig.xvar = xv if not horizontal else None
        if horizontal:
            fig.yvar = xv
        return idx + 1
    return idx


def _lpoly(x, y, grid, deg=0):
    sd = np.std(x)
    h = 1.06 * sd * len(x) ** -0.2 if sd > 0 else 1
    out = []
    for g in grid:
        w = np.exp(-0.5 * ((x - g) / h) ** 2)
        out.append(np.sum(w * y) / np.sum(w))
    return grid, np.array(out)


def _draw_function(ip, fig, p, idx):
    m = re.match(r'^\s*(?:(\w+)\s*=)?\s*(.*)$', p.body, re.S)
    expr = m.group(2)
    rng = p.opts.get('range')
    lo, hi = (0.0, 1.0)
    if rng:
        vals = rng.split()
        lo, hi = float(vals[0]), float(vals[1])
    x = np.linspace(lo, hi, 300)
    from .data import Data
    dd = Data()
    dd.n = len(x)
    dd.add_var('x', x, 'double')
    y = E.eval_expr(expr, E.Ctx(ip, dd))
    c, lw, ls = _line(p.opts, idx)
    h, = fig.ax.plot(x, np.broadcast_to(y, x.shape), color=c, lw=lw, ls=ls)
    fig.legend_items.append((h, expr))
    return idx + 1


def _hist_edges(x, o):
    if len(x) == 0:
        return np.array([0, 1])
    lo, hi = x.min(), x.max()
    if o.has('discrete', 4):
        u = np.unique(x)
        w = np.min(np.diff(u)) if len(u) > 1 else 1
        return np.concatenate([u - w / 2, [u[-1] + w / 2]])
    width = o.get('width', 1)
    start = o.get('start')
    if start:
        lo = float(start)
    if width:
        w = float(width)
        nb = int(math.ceil((hi - lo) / w)) or 1
        return lo + w * np.arange(nb + 1)
    nb = o.get('bin')
    if nb:
        nb = int(nb)
    else:
        n = len(x)
        nb = int(min(math.sqrt(n), 10 * math.log10(n))) if n > 1 else 1
        nb = max(nb, 1)
    return np.linspace(lo, hi, nb + 1)


def cmd_histogram(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    if p.opts.get('by'):
        v0 = d.resolve(split_top(p.body)[0])
        o2 = Options(p.opts.raw)
        o2.items = [(k, a) for k, a in o2.items if k != 'by']
        kind = 'frequency' if o2.has('frequency', 4) else ('percent' if o2.has('percent') else 'density')
        hopts = ' '.join(f'{k}({a})' if a is not None else k for k, a in o2.items
                         if k in ('width', 'bin', 'start', 'discrete', 'frequency', 'freq', 'percent', 'color', 'fcolor', 'lcolor'))
        glob = ' '.join(f'{k}({a})' if a is not None else k for k, a in o2.items
                        if k not in ('width', 'bin', 'start', 'discrete', 'frequency', 'freq', 'percent', 'color', 'fcolor', 'lcolor', 'normal', 'kdensity'))
        ifs = f' if {p.if_exp}' if p.if_exp else ''
        return cmd_twoway(ip, 'twoway', f'(histogram {v0}{ifs}, {hopts}), by({p.opts.get("by")}) {glob}', None)
    v = d.resolve(split_top(p.body)[0])
    mask = ip.mark(p)
    x = d.cols[v][mask].astype(float)
    x = x[~np.isnan(x)]
    o = p.opts
    edges = _hist_edges(x, o)
    cnt, _ = np.histogram(x, edges)
    fig = Figure(ip)
    ax = fig.ax
    if o.has('frequency', 4):
        yv, ylab = cnt, 'Frequency'
    elif o.has('fraction', 4):
        yv, ylab = cnt / len(x), 'Fraction'
    elif o.has('percent'):
        yv, ylab = 100 * cnt / len(x), 'Percent'
    else:
        yv, ylab = cnt / (len(x) * np.diff(edges)), 'Density'
    c = stata_color(o.get('color') or o.get('fcolor'), stata_color('khaki') if True else None)
    if not (o.get('color') or o.get('fcolor')):
        c = stata_color(PALETTE[0])[:3] + (0.75,)
    ax.bar(edges[:-1], yv, width=np.diff(edges), align='edge', color=c,
           edgecolor=stata_color(o.get('lcolor'), (1, 1, 1, 1)), lw=0.7)
    scale = 1.0
    if ylab == 'Frequency':
        scale = len(x) * np.mean(np.diff(edges))
    elif ylab == 'Fraction':
        scale = np.mean(np.diff(edges))
    elif ylab == 'Percent':
        scale = 100 * np.mean(np.diff(edges))
    if o.has('normal', 3):
        g = np.linspace(edges[0], edges[-1], 200)
        h, = ax.plot(g, sps.norm.pdf(g, x.mean(), x.std(ddof=1)) * scale, color=stata_color(PALETTE[1]), lw=1.8)
    if o.has('kdensity', 4):
        gx, gy = kdensity(x)
        ax.plot(gx, gy * scale, color='#22272b', lw=1.6, ls='--')
    fig.set_default_titles(xtitle=d.varlab.get(v, v), ytitle=ylab)
    apply_global_options(fig, o)
    finish_graph(ip, fig, o)
    ip.set_r(scalars={'N': len(x), 'bin': len(edges) - 1, 'start': edges[0], 'width': float(np.mean(np.diff(edges)))})


def _standalone_kdensity(ip, rest):
    p = parse_command(rest)
    d = ip.data
    v = d.resolve(split_top(p.body)[0])
    x = d.cols[v][ip.mark(p)]
    gx, gy = kdensity(x)
    fig = Figure(ip)
    fig.ax.plot(gx, gy, color=stata_color('navy'), lw=1.6)
    if p.opts.has('normal', 3):
        xx = x[~np.isnan(x)]
        fig.ax.plot(gx, sps.norm.pdf(gx, xx.mean(), xx.std(ddof=1)), color=stata_color('maroon'), lw=1.2)
    fig.set_default_titles(title='Kernel density estimate', xtitle=d.varlab.get(v, v), ytitle='Density')
    apply_global_options(fig, p.opts)
    finish_graph(ip, fig, p.opts)


# ---------------------------------------------------------------- graph command family
@command('graph', 'gr')
def cmd_graph(ip, cmd, rest, by):
    r = rest.strip()
    sub = r.split()[0] if r.split() else ''
    body = r[len(sub):].strip()
    if sub in ('twoway', 'tw'):
        return cmd_twoway(ip, 'twoway', body, by)
    if sub == 'export':
        return _graph_export(ip, body)
    if sub == 'save':
        return
    if sub in ('set', 'query', 'describe', 'dir'):
        return
    if sub == 'drop':
        for nm in body.split():
            if nm == '_all':
                ip.graphs.clear()
                ip.graph_order.clear()
            ip.graphs.pop(nm, None)
        return
    if sub == 'display':
        nm = body.split(',')[0].strip() or ip.current_graph
        if nm in ip.graphs:
            for f in ip.graph_listeners:
                f(nm, ip.graphs[nm])
        return
    if sub == 'combine':
        return _graph_combine(ip, body)
    if sub in ('box', 'hbox'):
        return _graph_box(ip, body, sub == 'hbox')
    if sub in ('bar', 'hbar', 'dot'):
        return _graph_bar(ip, body, sub)
    if sub == 'matrix':
        return _graph_matrix(ip, body)
    raise StataError(f'graph {sub} not supported', 198)


def _graph_export(ip, body):
    p = parse_command(body)
    fn = unquote(p.body.strip())
    path = ip.path(fn)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    nm = p.opts.get('name') or ip.current_graph
    figs = getattr(ip, 'figs', {})
    mfig = figs.get(nm)
    if mfig is None:
        raise StataError('no graph in memory', 693)
    if os.path.exists(path) and not p.opts.has('replace'):
        raise StataError(f'file {fn} already exists', 602)
    ext = os.path.splitext(path)[1].lower().lstrip('.') or 'png'
    w = p.opts.get('width')
    dpi = 110
    if w:
        dpi = max(50, float(w) / mfig.get_figwidth())
    if ext in ('png', 'jpg', 'jpeg', 'svg', 'pdf', 'eps', 'tif', 'tiff'):
        mfig.savefig(path, format={'jpg': 'jpeg', 'tif': 'tiff'}.get(ext, ext), dpi=dpi,
                     facecolor=mfig.get_facecolor())
    else:
        mfig.savefig(path + '.png', format='png', dpi=dpi)
    ip.txt(f'file {fn} saved as {ext.upper()} format\n')


def _graph_combine(ip, body):
    p = parse_command(body)
    names = p.body.split()
    figs = getattr(ip, 'figs', {})
    imgs = []
    for nm in names:
        nm = nm.replace('.gph', '')
        if nm not in ip.graphs:
            raise StataError(f'graph {nm} not found', 111)
        imgs.append(ip.graphs[nm])
    rows = int(p.opts.get('rows') or 1) if p.opts.has('rows') else None
    cols = int(p.opts.get('cols') or 1) if p.opts.has('cols') else None
    n = len(imgs)
    if rows:
        cols = math.ceil(n / rows)
    elif cols:
        rows = math.ceil(n / cols)
    else:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
    plt = mpl()
    fig = Figure(ip, rows, cols, figsize=(5 * cols, 4 * rows))
    import matplotlib.image as mpimg
    for k, ax in enumerate(fig.axes):
        ax.axis('off')
        if k < n:
            ax.imshow(mpimg.imread(io.BytesIO(imgs[k]), format='png'))
    apply_global_options(fig, p.opts)
    finish_graph(ip, fig, p.opts)


def _over_specs(o):
    specs = []
    for raw in o.all('over'):
        k = find_top(raw, ',')
        var = (raw if k < 0 else raw[:k]).strip()
        oo = Options(raw[k + 1:] if k >= 0 else '')
        relabel = {}
        rl = oo.get('relabel')
        if rl:
            toks = split_top(rl)
            for i in range(0, len(toks) - 1, 2):
                relabel[int(toks[i])] = unquote(toks[i + 1])
        specs.append((var, relabel, oo))
    return specs


def _graph_box(ip, body, horizontal):
    p = parse_command(body)
    d = ip.data
    vars_ = d.expand_varlist(split_top(p.body))
    mask = ip.mark(p)
    specs = _over_specs(p.opts)
    fig = Figure(ip)
    ax = fig.ax
    data, labels = [], []
    if specs:
        ov = d.resolve(specs[0][0])
        lv = sorted(np.unique(d.cols[ov][mask & ~np.isnan(d.cols[ov])]))
        for k, l in enumerate(lv, 1):
            for v in vars_:
                x = d.cols[v][mask & (d.cols[ov] == l)]
                data.append(x[~np.isnan(x)])
            labels.append(specs[0][1].get(k) or d.value_text(ov, l) or g_fmt(l))
    else:
        for v in vars_:
            x = d.cols[v][mask]
            data.append(x[~np.isnan(x)])
            labels.append(d.varlab.get(v, v))
    bp = ax.boxplot(data, vert=not horizontal, patch_artist=True, widths=0.5,
                    medianprops={'color': 'black'}, flierprops={'marker': 'o', 'ms': 3,
                                                                'markerfacecolor': stata_color('navy')})
    for k, b in enumerate(bp['boxes']):
        b.set_facecolor(stata_color(PALETTE[k % len(vars_)] if not specs else 'navy'))
    (ax.set_xticklabels if not horizontal else ax.set_yticklabels)(labels)
    if specs and specs[0][2].get('label'):
        m = re.search(r'angle\((\d+)\)', specs[0][2].get('label'))
        if m:
            for lb in ax.get_xticklabels():
                lb.set_rotation(int(m.group(1)))
                lb.set_ha('right')
    fig.set_default_titles(ytitle=d.varlab.get(vars_[0], vars_[0]) if len(vars_) == 1 else None)
    apply_global_options(fig, p.opts)
    finish_graph(ip, fig, p.opts)


def _graph_bar(ip, body, kind):
    p = parse_command(body)
    d = ip.data
    stat = 'mean'
    m = re.match(r'^\((\w+)\)\s*(.*)$', p.body.strip())
    b = p.body
    if m:
        stat, b = m.group(1), m.group(2)
    vars_ = d.expand_varlist(split_top(b))
    mask = ip.mark(p)
    specs = _over_specs(p.opts)
    fig = Figure(ip)
    ax = fig.ax

    def agg(x):
        x = x[~np.isnan(x)]
        if not x.size:
            return np.nan
        return {'mean': x.mean(), 'sum': x.sum(), 'count': x.size, 'median': np.median(x), 'max': x.max(),
                'min': x.min()}.get(stat, x.mean())
    groups = [()]
    if specs:
        levels = []
        for var, rl, oo in specs:
            v = d.resolve(var)
            if d.isstr(v):
                lv = sorted(set(d.cols[v][mask]))
            else:
                lv = sorted(np.unique(d.cols[v][mask & ~np.isnan(d.cols[v])]))
            levels.append((v, lv, rl))
        import itertools
        outer = levels[1] if len(levels) > 1 else None
        inner = levels[0]
        pos = 0
        ticks, ticklabs = [], []
        blabel = p.opts.get('blabel')
        bfmt = None
        if blabel:
            mm = re.search(r'format\(([^)]*)\)', blabel)
            bfmt = mm.group(1) if mm else '%9.0g'
        handles = []
        outer_lv = outer[1] if outer else [None]
        for oi, ol in enumerate(outer_lv):
            start = pos
            for ii, il in enumerate(inner[1]):
                sel = mask & (d.cols[inner[0]] == il)
                if outer:
                    sel &= (d.cols[outer[0]] == ol)
                val = agg(d.cols[vars_[0]][sel].astype(float))
                h = ax.bar(pos, val, width=0.8, color=stata_color(PALETTE[ii % len(PALETTE)]))
                if oi == 0:
                    handles.append(h)
                if blabel:
                    ax.text(pos, val, fmt_num(val, bfmt).strip(), ha='center', va='bottom', fontsize=8)
                lab = inner[2].get(ii + 1) or (d.value_text(inner[0], il) if not d.isstr(inner[0]) else il) or g_fmt(il)
                ticks.append(pos)
                ticklabs.append(lab)
                pos += 1
            if outer:
                olab = outer[2].get(oi + 1) or (ol if d.isstr(outer[0]) else (d.value_text(outer[0], ol) or g_fmt(ol)))
                ax.text((start + pos - 1) / 2, -0.12, olab, transform=ax.get_xaxis_transform(), ha='center',
                        va='top', fontsize=9)
            pos += 0.6
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabs, fontsize=8.5)
    else:
        for k, v in enumerate(vars_):
            ax.bar(k, agg(d.cols[v][mask].astype(float)), color=stata_color(PALETTE[k]))
        ax.set_xticks(range(len(vars_)))
        ax.set_xticklabels(vars_)
    fig.set_default_titles(ytitle=f'{stat} of {vars_[0]}')
    apply_global_options(fig, p.opts)
    finish_graph(ip, fig, p.opts)


# ---------------------------------------------------------------- diagnostics plots
@command('pnorm')
def cmd_pnorm(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    v = d.resolve(split_top(p.body)[0])
    x = d.cols[v][ip.mark(p)]
    x = np.sort(x[~np.isnan(x)])
    n = len(x)
    emp = np.arange(1, n + 1) / (n + 1)
    th = sps.norm.cdf((x - x.mean()) / x.std(ddof=1))
    fig = Figure(ip)
    fig.ax.scatter(emp, th, s=10, color=stata_color('navy'))
    fig.ax.plot([0, 1], [0, 1], color='black', lw=0.8)
    fig.set_default_titles(xtitle='Empirical P[i] = i/(N+1)', ytitle=f'Normal F[({v}-m)/s]')
    apply_global_options(fig, p.opts)
    finish_graph(ip, fig, p.opts)


@command('qnorm')
def cmd_qnorm(ip, cmd, rest, by):
    p = parse_command(rest)
    d = ip.data
    v = d.resolve(split_top(p.body)[0])
    x = np.sort(d.cols[v][~np.isnan(d.cols[v])])
    n = len(x)
    q = sps.norm.ppf(np.arange(1, n + 1) / (n + 1), x.mean(), x.std(ddof=1))
    fig = Figure(ip)
    fig.ax.scatter(q, x, s=10, color=stata_color('navy'))
    fig.ax.plot([q.min(), q.max()], [q.min(), q.max()], color='black', lw=0.8)
    fig.set_default_titles(xtitle='Inverse Normal', ytitle=v)
    apply_global_options(fig, p.opts)
    finish_graph(ip, fig, p.opts)


@command('rvfplot')
def cmd_rvfplot(ip, cmd, rest, by):
    p = parse_command(rest)
    est = ip.est
    f = est.get('fit') if est else None
    if f is None:
        raise StataError('rvfplot only after regress', 321)
    yhat = f.y - f.e
    fig = Figure(ip)
    fig.ax.scatter(yhat, f.e, s=10, color=stata_color('navy'))
    fig.set_default_titles(xtitle='Fitted values', ytitle='Residuals')
    apply_global_options(fig, p.opts)
    finish_graph(ip, fig, p.opts)


@command('marginsplot')
def cmd_marginsplot(ip, cmd, rest, by):
    p = parse_command(rest)
    m = getattr(ip, 'margins_last', None)
    if m is None:
        raise StataError('previous command was not margins', 301)
    rows = m['rows']
    fig = Figure(ip)
    ax = fig.ax
    if m['at']:
        xs = np.array([v for _, v in m['at']], dtype=float)
    else:
        xs = np.arange(1, len(rows) + 1, dtype=float)
    ys = np.array([r[1] for r in rows])
    lo = np.array([r[3] for r in rows])
    hi = np.array([r[4] for r in rows])
    ax.vlines(xs, lo, hi, color=stata_color('navy'), lw=1.4)
    ax.scatter(xs, lo, marker='_', color=stata_color('navy'), s=80)
    ax.scatter(xs, hi, marker='_', color=stata_color('navy'), s=80)
    ax.plot(xs, ys, '-o', color=stata_color('navy'), ms=5)
    if not m['at']:
        ax.set_xticks(xs)
        ax.set_xticklabels(m['labels'], fontsize=8.5)
    title = 'Average marginal effects of ' + m['xvar'] + ' with 95% CIs' if m['xvar'] else 'Predictive margins with 95% CIs'
    fig.set_default_titles(title=title, xtitle=m['over'] or (m['at'][0][0] if m['at'] else ''),
                           ytitle=f'Effects on Linear Prediction' if m['xvar'] else 'Linear Prediction')
    apply_global_options(fig, p.opts)
    finish_graph(ip, fig, p.opts)


# ---------------------------------------------------------------- coefplot
@command('coefplot')
def cmd_coefplot(ip, cmd, rest, by):
    from .interp import canon_coef
    k = find_top(rest, ',')
    main = rest if k < 0 else rest[:k]
    gopts = Options(rest[k + 1:] if k >= 0 else '')
    models = []
    s = main.strip()
    i = 0
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        if s[i] == '(':
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
            inner = s[i + 1:j]
            kk = find_top(inner, ',')
            nm = (inner if kk < 0 else inner[:kk]).strip()
            mo = Options(inner[kk + 1:] if kk >= 0 else '')
            models.append((nm, mo))
            i = j + 1
        elif s[i:i + 2] == '||':
            i += 2
        else:
            j = i
            while j < len(s) and not s[j].isspace():
                j += 1
            models.append((s[i:j], Options('')))
            i = j
    if not models:
        models = [('.', Options(''))]
    keep = gopts.get('keep')
    drop = gopts.get('drop')
    keep_p = split_top(keep) if keep else None
    drop_p = split_top(drop) if drop else []
    vertical = gopts.has('vertical', 4)
    fig = Figure(ip)
    ax = fig.ax
    import fnmatch

    def matches(nm, pats):
        cn = canon_coef(nm)
        for pt in pats:
            pc = canon_coef(pt.replace('*', 'STAR')).replace('STAR', '*')
            if fnmatch.fnmatchcase(cn, pc) or fnmatch.fnmatchcase(nm, pt):
                return True
        return False
    allcoefs = []
    data = []
    for mi, (nm, mo) in enumerate(models):
        est = ip.est if nm in ('.', '') else ip.stored.get(nm)
        if est is None:
            raise StataError(f'estimation result {nm} not found', 111)
        names = est['names']
        b = est['b']
        se = np.sqrt(np.maximum(np.diag(est['V']), 0))
        info = est.get('colinfo')
        stat = est.get('stat', 't')
        df = est.get('df_r')
        crit = sps.t.ppf(0.975, df) if stat == 't' and df else 1.96
        rows = []
        mkeep = mo.get('keep')
        mkeep_p = split_top(mkeep) if mkeep else keep_p
        for j, n_ in enumerate(names):
            if n_ == '_cons' and not (mkeep_p and matches(n_, mkeep_p)):
                continue
            if info and (info[j].base or info[j].omitted):
                continue
            if se[j] == 0 and b[j] == 0:
                continue
            if mkeep_p and not matches(n_, mkeep_p):
                continue
            if drop_p and matches(n_, drop_p):
                continue
            rows.append((n_, b[j], b[j] - crit * se[j], b[j] + crit * se[j]))
            if n_ not in allcoefs:
                allcoefs.append(n_)
        data.append((nm, mo, rows))
    swap = gopts.has('swapnames') or gopts.has('asequation')
    nmod = len(data)
    if swap:
        cats = [m[0] for m in data]
    else:
        cats = allcoefs
    pos = {c: k for k, c in enumerate(cats)}
    off = np.linspace(-0.15 * (nmod - 1), 0.15 * (nmod - 1), nmod) if nmod > 1 and not swap else np.zeros(nmod)
    for mi, (nm, mo, rows) in enumerate(data):
        col = stata_color(mo.get('mcolor') or gopts.get('mcolor'), stata_color(PALETTE[mi % len(PALETTE)]))
        ci_o = Options(mo.get('ciopts') or mo.get('ciopt') or gopts.get('ciopts') or '')
        ccol = stata_color(ci_o.get('lcolor') or ci_o.get('color'), col)
        clw = _size(ci_o.get('lwidth'), _LW, 1.4)
        sym, filled = _MS.get((mo.get('msymbol') or gopts.get('msymbol') or 'O').strip(), ('o', True))
        ms = _size(mo.get('msize') or gopts.get('msize'), _MSIZE, 30)
        lab = mo.get('label')
        lab = unquote(lab) if lab else (nm if nm != '.' else '')
        xs, ys, lo, hi = [], [], [], []
        for n_, bb, l_, h_ in rows:
            c_ = nm if swap else n_
            xs.append(pos[c_] + off[mi])
            ys.append(bb)
            lo.append(l_)
            hi.append(h_)
        xs = np.array(xs)
        if vertical:
            ax.vlines(xs, lo, hi, color=ccol, lw=clw)
            h = ax.scatter(xs, ys, color=col, marker=sym, s=ms, zorder=3, facecolors=col if filled else 'none')
        else:
            yy = -xs
            ax.hlines(yy, lo, hi, color=ccol, lw=clw)
            h = ax.scatter(ys, yy, color=col, marker=sym, s=ms, zorder=3, facecolors=col if filled else 'none')
        if nmod > 1 or mo.get('label'):
            fig.legend_items.append((h, lab))
    labels = [_coef_label(c, ip) for c in cats]
    bl = gopts.get('bylabel')
    if vertical:
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(labels, rotation=45 if len(cats) > 6 else 0, ha='right' if len(cats) > 6 else 'center',
                           fontsize=8)
    else:
        ax.set_yticks([-k for k in range(len(cats))])
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.grid(axis='x', color='#dfe6ea', lw=0.8)
        ax.grid(axis='y', visible=False)
    if gopts.has('xlabel') and vertical:
        fig.titles['xlabel'] = gopts.get('xlabel')
    apply_global_options(fig, gopts)
    if vertical and 'xlabel' in fig.titles:
        raw = fig.titles['xlabel']
        toks = split_top(raw.split(',')[0])
        if any(t.startswith('"') for t in toks):
            new_t, new_l = [], []
            for a, b_ in zip(toks[::2], toks[1::2]):
                try:
                    new_t.append(float(a) - 1)
                    new_l.append(unquote(b_))
                except ValueError:
                    pass
            ax.set_xticks(new_t)
            ax.set_xticklabels(new_l)
            fig.titles.pop('xlabel')
    finish_graph(ip, fig, gopts)


def _coef_label(name, ip):
    d = ip.data
    parts = name.split('#')
    out = []
    for p_ in parts:
        m = re.match(r'^(\d+(?:\.\d+)?)[bno]*\.(.+)$', p_)
        if m:
            v = m.group(2)
            lv = float(m.group(1))
            t = d.value_text(v, lv) if v in d.cols else None
            out.append(t if t else f'{num_to_macro(lv)}.{v}')
        else:
            v = re.sub(r'^c\.', '', p_)
            out.append(d.varlab.get(v, v) if (len(parts) == 1 and v in d.cols and False) else v)
    return ' # '.join(out)


def _graph_matrix(ip, body):
    p = parse_command(body)
    d = ip.data
    vars_ = d.expand_varlist(split_top(p.body))
    m = ip.mark(p)
    for v in vars_:
        m &= ~np.isnan(d.cols[v])
    k = len(vars_)
    half = p.opts.has('half')
    fig = Figure(ip, k, k, figsize=(min(2.3 * k, 12), min(2.3 * k, 12)))
    col = stata_color(p.opts.get('mcolor') or p.opts.get('color'), stata_color(PALETTE[0]))
    for i in range(k):
        for j in range(k):
            ax = fig.axes[i * k + j]
            if half and j > i:
                ax.set_visible(False)
                continue
            if i == j:
                ax.axis('off')
                ax.text(0.5, 0.5, d.varlab.get(vars_[i], vars_[i]) if len(d.varlab.get(vars_[i], '')) < 28 else vars_[i],
                        ha='center', va='center', fontsize=9, fontweight='bold', wrap=True, transform=ax.transAxes)
                continue
            ax.scatter(d.cols[vars_[j]][m], d.cols[vars_[i]][m], s=6, color=col[:3] + (0.55,), lw=0)
            ax.tick_params(labelsize=6.5)
            if j != 0 and not (half and j == 0):
                ax.set_yticklabels([])
            if i != k - 1:
                ax.set_xticklabels([])
    apply_global_options(fig, p.opts)
    finish_graph(ip, fig, p.opts)
