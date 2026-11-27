/* Web worker: runs the Python Stata emulator inside Pyodide. */
const PYODIDE_URL = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
importScripts(PYODIDE_URL + 'pyodide.js');

let py = null;
let mounted = null;
let sympyLoaded = false;

function post(type, payload) { self.postMessage(Object.assign({ type }, payload || {})); }

async function init() {
  post('status', { text: 'Đang tải Python (Pyodide)…' });
  py = await loadPyodide({ indexURL: PYODIDE_URL });
  post('status', { text: 'Đang tải numpy, scipy, pandas, matplotlib…' });
  await py.loadPackage(['numpy', 'scipy', 'pandas', 'matplotlib']);
  post('status', { text: 'Đang cài engine Stata…' });
  const z = await (await fetch('py.zip?v=' + Date.now())).arrayBuffer();
  py.unpackArchive(z, 'zip', { extractDir: '/home/pyodide/lib' });
  const ex = await (await fetch('examples/manifest.json?v=' + Date.now())).json();
  py.FS.mkdirTree('/work');
  py.FS.mkdirTree('/examples/data');
  py.FS.mkdirTree('/examples/do');
  for (const f of ex.data) {
    const b = new Uint8Array(await (await fetch('examples/data/' + encodeURIComponent(f))).arrayBuffer());
    py.FS.writeFile('/examples/data/' + f, b);
  }
  for (const f of ex.do) {
    const t = await (await fetch('examples/do/' + encodeURIComponent(f))).text();
    py.FS.writeFile('/examples/do/' + f, t);
  }
  self.exampleList = ex.do;
  py.globals.set('_post', (kind, a, b) => {
    if (kind === 'out') post('out', { style: a, text: b });
    else if (kind === 'graph') post('graph', { name: a, png: b.toJs() });
  });
  await py.runPythonAsync(`
import sys
sys.path.insert(0, '/home/pyodide/lib')
import matplotlib
matplotlib.use('Agg')
from stata import Interp
ip = Interp(workdir='/work')
ip.out.listeners.append(lambda st, tx: _post('out', st, tx))
ip.graph_listeners.append(lambda nm, png: _post('graph', nm, png))
`);
  post('ready', { examples: ex.do });
  await sendState();
}

async function sendState() {
  const s = py.runPython(`
import json, os
d = ip.data
vars_ = [{'name': v, 'label': d.varlab.get(v, ''), 'type': d.types[v], 'format': d.formats.get(v, ''),
          'vallab': d.vallab.get(v, '')} for v in d.vars]
files = []
for root, dirs, fs in os.walk('/work'):
    dirs[:] = [x for x in dirs if not x.startswith('.')]
    for f in fs:
        p = os.path.join(root, f)
        files.append({'path': os.path.relpath(p, '/work'), 'size': os.path.getsize(p)})
json.dumps({'vars': vars_, 'N': d.n, 'files': sorted(files, key=lambda x: x['path']),
            'cwd': os.getcwd(), 'est': (ip.est or {}).get('cmd', ''), 'label': d.label})
`);
  post('state', { state: JSON.parse(s) });
}

let xlsxLoaded = false;
async function ensurePackages(code) {
  if (!xlsxLoaded && /\b(import|export)\s+excel\b/.test(code)) {
    post('status', { text: 'Đang tải openpyxl cho Excel…' });
    // openpyxl is not in the Pyodide distribution: install the pure-Python wheel from PyPI
    await py.loadPackage(['micropip']);
    await py.runPythonAsync("import micropip\nawait micropip.install(['et_xmlfile', 'openpyxl'])");
    xlsxLoaded = true;
  }
  if (!sympyLoaded && /rddensity/.test(code)) {
    post('status', { text: 'Đang tải sympy cho rddensity…' });
    await py.loadPackage(['sympy']);
    sympyLoaded = true;
  }
}

async function run(code, echo) {
  await ensurePackages(code);
  py.globals.set('_code', code);
  py.globals.set('_echo', echo);
  const t0 = performance.now();
  const rc = await py.runPythonAsync('ip.run(_code, echo=_echo)');
  if (mounted) { await syncMounted(); }
  const browse = py.runPython(`
import json
b = getattr(ip, 'browse_request', None)
ip.browse_request = None
json.dumps(None if b is None else {'vars': b[0], 'n': int(len(b[1]))})
`);
  post('done', { rc, ms: performance.now() - t0, browse: JSON.parse(browse) });
  await sendState();
}

async function syncMounted() {
  await new Promise((res) => py.FS.syncfs(false, () => res()));
}

let initPromise = null;
self.onmessage = async (ev) => {
  const m = ev.data;
  try {
    if (m.type === 'init') { initPromise = initPromise || init(); await initPromise; return; }
    // wait until Pyodide and the engine are ready before handling uploads/runs
    if (initPromise) await initPromise;
    if (m.type === 'init') {}
    else if (m.type === 'run') await run(m.code, m.echo !== false);
    else if (m.type === 'upload') {
      const target = '/work/' + m.name;
      py.FS.mkdirTree(target.substring(0, target.lastIndexOf('/')) || '/work');
      py.FS.writeFile(target, new Uint8Array(m.data));
      if (mounted) await syncMounted();
      await sendState();
    } else if (m.type === 'download') {
      const data = py.FS.readFile('/work/' + m.path);
      post('file', { path: m.path, data });
    } else if (m.type === 'readtext') {
      const base = m.path.startsWith('/') ? '' : '/work/';
      const data = py.FS.readFile(base + m.path, { encoding: 'utf8' });
      post('text', { path: m.path, text: data, purpose: m.purpose });
    } else if (m.type === 'zip') {
      const out = py.runPython(`
import io, zipfile, os
buf = io.BytesIO()
with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, dirs, fs in os.walk('/work'):
        for f in fs:
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, '/work'))
buf.getvalue()`);
      post('file', { path: 'ket_qua_stata_web.zip', data: out.toJs() });
    } else if (m.type === 'delete') {
      py.FS.unlink('/work/' + m.path);
      if (mounted) await syncMounted();
      await sendState();
    } else if (m.type === 'data') {
      const js = py.runPython(`
import json, numpy as np
d = ip.data
start, cnt = ${m.start | 0}, ${m.count | 0}
cols = ${JSON.stringify(m.vars || [])} or d.vars
rows = []
for i in range(start, min(d.n, start + cnt)):
    r = []
    for v in cols:
        x = d.cols[v][i]
        if d.isstr(v):
            r.append(str(x))
        elif np.isnan(x):
            r.append('.')
        else:
            t = d.value_text(v, x)
            r.append(t if t is not None else ('%.7g' % x))
    rows.append(r)
json.dumps({'cols': cols, 'rows': rows, 'N': d.n, 'start': start})
`);
      post('data', { data: JSON.parse(js) });
    } else if (m.type === 'mount') {
      py.runPython("import os; os.chdir('/')");
      try { py.FS.unmount('/work'); } catch (e) { /* not mounted */ }
      const nativefs = await py.mountNativeFS('/work', m.handle);
      mounted = nativefs;
      py.runPython("import os; os.chdir('/work')");
      post('mounted', { name: m.handle.name });
      await sendState();
    }
  } catch (e) {
    post('out', { style: 'err', text: '\n[web error] ' + (e && e.message ? e.message : e) + '\n' });
    post('done', { rc: -1, ms: 0 });
  }
};
