/* UI for Stata Web. All computation happens in worker.js (Pyodide). */
const $ = (s) => document.querySelector(s);
let worker, busy = false, ready = false;
let history = [], hpos = 0;
let state = { vars: [], files: [], N: 0 };
let dataStart = 0;
const DCOUNT = 200;
let mountedName = null;
const resultsEl = $('#results');
const MAXCHARS = 2_000_000;
const esc = (s) => String(s ?? '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

function startWorker() {
  worker = new Worker('worker.js?v=' + (window.APP_V || Date.now()));
  worker.onmessage = onMsg;
  worker.postMessage({ type: 'init' });
}

let pendingScroll = false;
function append(style, text) {
  const span = document.createElement('span');
  span.className = style;
  span.textContent = text;
  resultsEl.appendChild(span);
  if (resultsEl.textContent.length > MAXCHARS) {
    while (resultsEl.firstChild && resultsEl.textContent.length > MAXCHARS * 0.7) resultsEl.removeChild(resultsEl.firstChild);
  }
  pendingScroll = true;
}
setInterval(() => { if (pendingScroll) { resultsEl.scrollTop = resultsEl.scrollHeight; pendingScroll = false; } }, 80);

const edChan = new BroadcastChannel('stata-web');
function setStatus(t) { $('#status').textContent = t; edChan.postMessage({ type: 'status', text: t }); }

// ---------------------------------------------------------------- editor (tabs + highlighting)
const ed = $('#editor');
const D = window.DoEditor({ textarea: ed, overlay: $('#hl'), lineno: $('#lineno'), tabbar: $('#dtabs'),
  onChange: () => { clearTimeout(window._sy); window._sy = setTimeout(() => edChan.postMessage({ type: 'load', text: D.text(), name: D.name() }), 500); } });
let popWin = null;

function openInEditor(name, text, show = true) {
  D.open(name, text);
  edChan.postMessage({ type: 'load', text, name });
  if (show && !(popWin && !popWin.closed)) showTab('editor');
}
function runAll() { showTab('results'); run(D.text(), true, `do ${D.name()}`); }
function runSel() {
  let s = ed.value.substring(ed.selectionStart, ed.selectionEnd);
  if (!s.trim()) {
    const v = ed.value, p = ed.selectionStart;
    const a = v.lastIndexOf('\n', p - 1) + 1, b = v.indexOf('\n', p);
    s = v.substring(a, b < 0 ? v.length : b);
  }
  showTab('results'); run(s, true, s.split('\n')[0].slice(0, 80));
}
function saveDo(name, text) {
  name = name || D.name(); text = text ?? D.text();
  const data = new TextEncoder().encode(text);
  worker.postMessage({ type: 'upload', name, data: data.buffer }, [data.buffer]);
  if (!mountedName) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], { type: 'text/plain' })); a.download = name; a.click();
  }
  setStatus(`Đã lưu ${name}`);
}
ed.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && (e.key === 'd' || e.key === 'D')) { e.preventDefault(); e.shiftKey ? runSel() : runAll(); }
  if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); saveDo(); }
});
$('#edRunAll').onclick = runAll;
$('#edRunSel').onclick = runSel;
$('#edSave').onclick = () => saveDo();
$('#doOpen').onchange = async (e) => { const f = e.target.files[0]; if (!f) return; openInEditor(f.name, await f.text()); e.target.value = ''; };
$('#edPop').onclick = () => {
  if (popWin && !popWin.closed) { popWin.focus(); return; }
  popWin = window.open('editor.html?v=' + (window.APP_V || ''), 'stataDoEditor', 'width=920,height=720');
  showTab('results');
};
edChan.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === 'hello') edChan.postMessage({ type: 'load', text: D.text(), name: D.name() });
  else if (m.type === 'sync') { D.open(m.name, m.text); }
  else if (m.type === 'run') { showTab('results'); run(m.code, true, m.label); }
  else if (m.type === 'save') { D.open(m.name, m.text); saveDo(m.name, m.text); }
};

// ---------------------------------------------------------------- worker messages
function onMsg(ev) {
  const m = ev.data;
  switch (m.type) {
    case 'status': setStatus(m.text); break;
    case 'ready':
      ready = true;
      setStatus('Sẵn sàng');
      $('#exSel').length = 1;
      for (const f of m.examples) { const o = document.createElement('option'); o.value = f; o.textContent = f; $('#exSel').appendChild(o); }
      append('txt', '  ___  ____  ____  ____  ____ (R)\n /__    /   ____/   /   ____/   Stata Web — mô phỏng Stata trong trình duyệt\n___/   /   /___/   /   /___/    cho giảng dạy suy diễn nhân quả\n\n');
      append('txt', 'Gõ lệnh vào ô Command, dùng menu phía trên, hoặc mở một do-file mẫu. Gõ  help  để xem danh sách lệnh.\n\n');
      break;
    case 'out': append(m.style, m.text); break;
    case 'graph': {
      addGraph(m.name, m.png);
      append('ginfo', `(đồ thị "${m.name}" hiển thị ở thẻ Graph)\n`);
      break;
    }
    case 'done':
      busy = false;
      setStatus(m.rc === 0 ? `Xong (${(m.ms / 1000).toFixed(1)}s)` : `Dừng với lỗi r(${m.rc})`);
      if (m.browse) { showTab('data'); dataStart = 0; requestData(); }
      break;
    case 'state': state = m.state; renderVars(); renderFiles(); if ($('#tab-data').classList.contains('active')) requestData(); break;
    case 'file': {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([m.data])); a.download = m.path.split('/').pop(); a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 4000);
      break;
    }
    case 'text': if (m.purpose === 'editor') openInEditor(m.path.split('/').pop(), m.text); break;
    case 'data': renderData(m.data); break;
    case 'mounted': mountedName = m.name; $('#cwd').textContent = '(' + m.name + ')'; append('txt', `\n(thư mục làm việc: ${m.name})\n`); break;
  }
}

function run(code, echo = true, record = null) {
  if (!ready) { setStatus('Engine chưa sẵn sàng…'); return; }
  if (busy) { setStatus('Đang chạy lệnh khác…'); return; }
  busy = true;
  setStatus('Đang chạy…');
  if (record) { history.push(record); hpos = history.length; addReview(record); }
  worker.postMessage({ type: 'run', code, echo });
}
function addReview(c) {
  const li = document.createElement('li');
  li.textContent = c; li.title = 'Bấm: đưa vào ô Command · Bấm đúp: chạy lại';
  li.onclick = () => { $('#cmd').value = c; $('#cmd').focus(); };
  li.ondblclick = () => run(c, true, c);
  $('#review').appendChild(li);
  li.scrollIntoView({ block: 'end' });
}

// ---------------------------------------------------------------- command line
$('#cmd').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    const c = e.target.value.trim();
    if (!c) return;
    e.target.value = '';
    showTab('results');
    run(c, true, c);
  } else if (e.key === 'ArrowUp') { if (hpos > 0) { hpos--; e.target.value = history[hpos]; } e.preventDefault(); }
  else if (e.key === 'ArrowDown') { if (hpos < history.length - 1) { hpos++; e.target.value = history[hpos]; } else { hpos = history.length; e.target.value = ''; } e.preventDefault(); }
  else if (e.key === 'PageUp') { resultsEl.scrollTop -= resultsEl.clientHeight * 0.9; }
  else if (e.key === 'PageDown') { resultsEl.scrollTop += resultsEl.clientHeight * 0.9; }
});

// ---------------------------------------------------------------- tabs
function showTab(t) {
  document.querySelectorAll('.tabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === t));
  document.querySelectorAll('.tab').forEach((d) => d.classList.toggle('active', d.id === 'tab-' + t));
  if (t === 'data') requestData();
  if (t === 'editor') D.paint();
}
document.querySelectorAll('.tabs button').forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
$('#exSel').onchange = (e) => {
  const f = e.target.value; if (!f) return;
  worker.postMessage({ type: 'readtext', path: '/examples/do/' + f, purpose: 'editor' });
  e.target.value = '';
};

// ---------------------------------------------------------------- variables
function renderVars() {
  const flt = $('#varFilter').value.toLowerCase();
  const tb = $('#vars tbody'); tb.innerHTML = '';
  for (const v of state.vars) {
    if (flt && !v.name.toLowerCase().includes(flt) && !v.label.toLowerCase().includes(flt)) continue;
    const tr = document.createElement('tr');
    tr.innerHTML = '<td></td><td></td>';
    tr.children[0].textContent = v.name; tr.children[1].textContent = v.label;
    tr.title = 'Bấm đúp để chèn tên biến vào ô Command';
    tr.onclick = () => {
      tb.querySelectorAll('tr').forEach((r) => r.classList.remove('sel')); tr.classList.add('sel');
      $('#props').innerHTML = `<b>${esc(v.name)}</b><br>Label: ${esc(v.label) || '—'}<br>Type: ${v.type}<br>Format: ${v.format}<br>Value label: ${esc(v.vallab) || '—'}<br><span class="muted">${state.N.toLocaleString()} obs</span>`;
    };
    tr.ondblclick = () => { const c = $('#cmd'); c.value += (c.value && !c.value.endsWith(' ') ? ' ' : '') + v.name; c.focus(); };
    tb.appendChild(tr);
  }
}
$('#varFilter').oninput = renderVars;

// ---------------------------------------------------------------- files
function renderFiles() {
  const ul = $('#files'); ul.innerHTML = '';
  for (const f of state.files) {
    const li = document.createElement('li');
    const fn = document.createElement('span'); fn.className = 'fn'; fn.textContent = f.path; fn.title = `${f.path} (${(f.size / 1024).toFixed(1)} KB) — bấm để mở`;
    fn.onclick = () => {
      const p = f.path.toLowerCase(), q = `"${f.path}"`;
      let c = null;
      if (p.endsWith('.do')) { worker.postMessage({ type: 'readtext', path: f.path, purpose: 'editor' }); return; }
      if (p.endsWith('.dta')) c = `use ${q}, clear`;
      else if (/\.(xlsx|xls)$/.test(p)) c = `import excel using ${q}, firstrow clear`;
      else if (/\.(csv|tsv)$/.test(p)) c = `import delimited using ${q}, clear`;
      else if (/\.(txt|log|tex)$/.test(p)) c = `type ${q}`;
      if (c) { showTab('results'); run(c, true, c); } else worker.postMessage({ type: 'download', path: f.path });
    };
    const dl = document.createElement('button'); dl.textContent = '⬇'; dl.title = 'Tải về máy';
    dl.onclick = () => worker.postMessage({ type: 'download', path: f.path });
    li.append(fn, dl);
    ul.appendChild(li);
  }
  if (!state.files.length) ul.innerHTML = '<li class="muted">(trống — tải file lên hoặc mở thư mục)</li>';
}
async function uploadFiles(list) {
  for (const f of list) {
    const buf = await f.arrayBuffer();
    worker.postMessage({ type: 'upload', name: f.name, data: buf }, [buf]);
    append('txt', `(đã tải lên ${f.name})\n`);
  }
}
$('#fileUp').onchange = (e) => { uploadFiles(e.target.files); e.target.value = ''; };
$('#btnZip').onclick = () => worker.postMessage({ type: 'zip' });
async function mountFolder() {
  if (!window.showDirectoryPicker) { alert('Trình duyệt này chưa hỗ trợ mở thư mục. Hãy dùng Chrome hoặc Edge, hoặc dùng "Tải file lên".'); return; }
  try { const h = await window.showDirectoryPicker({ mode: 'readwrite' }); worker.postMessage({ type: 'mount', handle: h }); } catch (e) { /* cancelled */ }
}
$('#btnMount').onclick = mountFolder;
let dragN = 0;
window.addEventListener('dragenter', (e) => { e.preventDefault(); dragN++; $('#drop').classList.add('on'); });
window.addEventListener('dragleave', () => { if (--dragN <= 0) $('#drop').classList.remove('on'); });
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => { e.preventDefault(); dragN = 0; $('#drop').classList.remove('on'); uploadFiles(e.dataTransfer.files); });

// ---------------------------------------------------------------- data browser / editor
function requestData() { if (ready && !busy) worker.postMessage({ type: 'data', start: dataStart, count: DCOUNT }); }
function renderData(d) {
  const t = $('#dtable');
  const strCols = new Set(state.vars.filter((v) => v.type.startsWith('str')).map((v) => v.name));
  const editable = $('#dEdit').checked;
  let h = '<thead><tr><th></th>' + d.cols.map((c) => `<th>${esc(c)}</th>`).join('') + '</tr></thead><tbody>';
  d.rows.forEach((r, i) => {
    h += `<tr><th>${d.start + i + 1}</th>` + r.map((x, k) =>
      `<td class="${strCols.has(d.cols[k]) ? 'str' : ''}" data-r="${d.start + i + 1}" data-c="${esc(d.cols[k])}"${editable ? ' contenteditable="true"' : ''}>${esc(x)}</td>`).join('') + '</tr>';
  });
  t.innerHTML = h + '</tbody>';
  $('#dInfo').textContent = d.N ? ` Quan sát ${d.start + 1}–${Math.min(d.N, d.start + DCOUNT)} / ${d.N.toLocaleString()} ` : ' (chưa có dữ liệu) ';
  if (editable) {
    t.querySelectorAll('td[contenteditable]').forEach((td) => {
      const orig = td.textContent;
      td.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); td.blur(); } if (e.key === 'Escape') { td.textContent = orig; td.blur(); } });
      td.addEventListener('blur', () => {
        const v = td.textContent.trim();
        if (v === orig) return;
        const isStr = td.classList.contains('str');
        let val = isStr ? `"${v.replace(/"/g, "'")}"` : (v === '' ? '.' : v);
        if (!isStr && v !== '.' && v !== '' && isNaN(Number(v))) {
          const vl = state.vars.find((x) => x.name === td.dataset.c);
          if (vl && vl.vallab) val = `"${v}":${vl.vallab}`; else { td.textContent = orig; setStatus('Giá trị phải là số'); return; }
        }
        const c = `replace ${td.dataset.c} = ${val} in ${td.dataset.r}`;
        run(c, true, c);
      });
    });
  }
}
$('#dPrev').onclick = () => { dataStart = Math.max(0, dataStart - DCOUNT); requestData(); };
$('#dNext').onclick = () => { if (dataStart + DCOUNT < state.N) { dataStart += DCOUNT; requestData(); } };
$('#dRefresh').onclick = requestData;
$('#dEdit').onchange = requestData;

// ---------------------------------------------------------------- graphs
// Each graph gets its own sub-tab. As in Stata, a graph drawn with name(x) replaces the tab named x;
// unnamed graphs (Stata name "Graph") are kept as separate tabs so students can compare them.
const GMAX = 40;
let gList = [], gSelId = null, gSeq = 0, gUnnamed = 0, gGridOn = false;
function addGraph(name, png) {
  const url = URL.createObjectURL(new Blob([png], { type: 'image/png' }));
  const old = name !== 'Graph' && gList.find((g) => g.name === name);
  if (old) { URL.revokeObjectURL(old.url); old.url = url; gSelId = old.id; }
  else {
    const g = { id: ++gSeq, name, url, label: name !== 'Graph' ? name : (++gUnnamed === 1 ? 'Graph' : `Graph #${gUnnamed}`) };
    gList.push(g); gSelId = g.id;
    while (gList.length > GMAX) URL.revokeObjectURL(gList.shift().url);
  }
  refreshGraphs();
}
const gLabel = (g) => g.label;
function closeGraph(id) {
  const i = gList.findIndex((g) => g.id === id); if (i < 0) return;
  URL.revokeObjectURL(gList[i].url); gList.splice(i, 1);
  if (gSelId === id) gSelId = gList.length ? gList[Math.min(i, gList.length - 1)].id : null;
  refreshGraphs();
}
function refreshGraphs() {
  const tabs = $('#gTabs'); tabs.innerHTML = '';
  for (const g of gList) {
    const t = document.createElement('div');
    t.className = 'gt' + (g.id === gSelId ? ' active' : ''); t.title = g.name;
    t.innerHTML = `<span>${esc(gLabel(g))}</span><button class="x" title="Đóng">×</button>`;
    t.onclick = (e) => { if (e.target.classList.contains('x')) return closeGraph(g.id); gSelId = g.id; gGridOn = false; refreshGraphs(); };
    tabs.appendChild(t);
  }
  const w = $('#gWrap'); const cur = gList.find((g) => g.id === gSelId);
  w.classList.toggle('grid', gGridOn); $('#gGrid').classList.toggle('on', gGridOn);
  if (gGridOn) {
    w.innerHTML = '';
    for (const g of gList) {
      const f = document.createElement('figure'); f.className = g.id === gSelId ? 'active' : '';
      f.innerHTML = `<img src="${g.url}" alt=""><figcaption>${esc(gLabel(g))}</figcaption>`;
      f.onclick = () => { gSelId = g.id; gGridOn = false; refreshGraphs(); };
      w.appendChild(f);
    }
  } else {
    w.innerHTML = cur ? `<img id="gImg" src="${cur.url}" alt="">` : '';
  }
  $('#gcount').textContent = gList.length ? `(${gList.length})` : '';
}
$('#gGrid').onclick = () => { gGridOn = !gGridOn; refreshGraphs(); };
$('#gClear').onclick = () => { for (const g of gList) URL.revokeObjectURL(g.url); gList = []; gSelId = null; gUnnamed = 0; refreshGraphs(); };
$('#gSave').onclick = () => { const g = gList.find((x) => x.id === gSelId); if (!g) return; const a = document.createElement('a'); a.href = g.url; a.download = gLabel(g).replace(/[^\w.-]+/g, '_') + '.png'; a.click(); };

// ---------------------------------------------------------------- menus + command dialogs
const V = '__vars__';
const DIALOGS = {
  describe: { t: 'Mô tả dữ liệu (describe)', f: [['vl', 'Biến (để trống = tất cả)', V]], b: (x) => `describe ${x.vl}` },
  summarize: { t: 'Thống kê tóm tắt (summarize)', f: [['vl', 'Biến', V], ['detail', 'Chi tiết (detail)', 'check'], ['by', 'Theo nhóm (bysort)', 'var']],
    b: (x) => `${x.by ? `bysort ${x.by}: ` : ''}summarize ${x.vl}${x.detail ? ', detail' : ''}` },
  tab1: { t: 'Bảng tần suất một chiều', f: [['v', 'Biến', 'var'], ['m', 'Hiện missing', 'check']], b: (x) => `tabulate ${x.v}${x.m ? ', missing' : ''}` },
  tab2: { t: 'Bảng chéo hai chiều', f: [['r', 'Biến hàng', 'var'], ['c', 'Biến cột', 'var'], ['row', 'Phần trăm hàng', 'check'], ['col', 'Phần trăm cột', 'check'], ['chi2', 'Kiểm định chi2', 'check']],
    b: (x) => `tabulate ${x.r} ${x.c}${(x.row || x.col || x.chi2) ? ',' : ''}${x.row ? ' row' : ''}${x.col ? ' column' : ''}${x.chi2 ? ' chi2' : ''}` },
  tabstat: { t: 'tabstat', f: [['vl', 'Biến', V], ['s', 'Thống kê', 'text:n mean sd min max'], ['by', 'Theo nhóm', 'var']],
    b: (x) => `tabstat ${x.vl}, statistics(${x.s})${x.by ? ` by(${x.by})` : ''}` },
  corr: { t: 'Tương quan (pwcorr)', f: [['vl', 'Biến', V], ['star', 'Đánh dấu p<0.05', 'check']], b: (x) => `pwcorr ${x.vl}${x.star ? ', star(0.05) sig' : ''}` },
  ttest: { t: 'Kiểm định t hai nhóm', f: [['v', 'Biến kết quả', 'var'], ['g', 'Biến nhóm (2 giá trị)', 'var']], b: (x) => `ttest ${x.v}, by(${x.g})` },
  reg: { t: 'Hồi quy tuyến tính (regress)', f: [['y', 'Biến phụ thuộc', 'var'], ['x', 'Biến giải thích', V], ['vce', 'Sai số chuẩn', 'sel:|robust|cluster'], ['cl', 'Biến cluster', 'var']],
    b: (x) => `regress ${x.y} ${x.x}${x.vce === 'robust' ? ', vce(robust)' : x.vce === 'cluster' ? `, vce(cluster ${x.cl})` : ''}` },
  logit: { t: 'Logit', f: [['y', 'Biến phụ thuộc (0/1)', 'var'], ['x', 'Biến giải thích', V]], b: (x) => `logit ${x.y} ${x.x}` },
  count: { t: 'Mô hình biến đếm (poisson / nbreg)', f: [['m', 'Mô hình', 'sel:poisson|nbreg'], ['y', 'Biến phụ thuộc (số đếm)', 'var'], ['x', 'Biến giải thích', V], ['irr', 'Hiện IRR', 'check'], ['r', 'Robust', 'check']],
    b: (x) => `${x.m} ${x.y} ${x.x}${x.irr || x.r ? ',' : ''}${x.irr ? ' irr' : ''}${x.r ? ' vce(robust)' : ''}` },
  ppml: { t: 'Gravity PPML (ppmlhdfe)', f: [['y', 'Kim ngạch thương mại', 'var'], ['x', 'Biến giải thích', V], ['fe', 'Hiệu ứng cố định (absorb)', V], ['cl', 'Biến cluster (tùy chọn)', 'var']],
    b: (x) => `ppmlhdfe ${x.y} ${x.x}${x.fe ? `, absorb(${x.fe})` : ''}${x.cl ? `${x.fe ? '' : ','} vce(cluster ${x.cl})` : ''}` },
  xtreg: { t: 'Hồi quy dữ liệu bảng (xtreg)', f: [['y', 'Biến phụ thuộc', 'var'], ['x', 'Biến giải thích', V], ['m', 'Mô hình', 'sel:fe|re|be'], ['r', 'Robust (cluster theo panel)', 'check']],
    b: (x) => `xtreg ${x.y} ${x.x}, ${x.m}${x.r ? ' vce(robust)' : ''}` },
  iv: { t: '2SLS (ivregress)', f: [['y', 'Biến phụ thuộc', 'var'], ['d', 'Biến nội sinh', 'var'], ['z', 'Biến công cụ', V], ['x', 'Biến ngoại sinh', V], ['r', 'Robust', 'check']],
    b: (x) => `ivregress 2sls ${x.y} ${x.x} (${x.d} = ${x.z})${x.r ? ', vce(robust)' : ''}` },
  rd: { t: 'RDD (rdrobust)', f: [['y', 'Biến kết quả', 'var'], ['x', 'Biến chạy', 'var'], ['c', 'Ngưỡng', 'text:0']], b: (x) => `rdrobust ${x.y} ${x.x}, c(${x.c})` },
  psm: { t: 'PSM (psmatch2)', f: [['t', 'Biến can thiệp', 'var'], ['x', 'Biến kiểm soát', V], ['y', 'Biến kết quả', 'var'], ['k', 'Số láng giềng', 'text:1']],
    b: (x) => `psmatch2 ${x.t} ${x.x}, outcome(${x.y}) neighbor(${x.k}) common logit` },
  did: { t: 'DID (didregress)', f: [['y', 'Biến kết quả', 'var'], ['t', 'Biến can thiệp', 'var'], ['g', 'Nhóm', 'var'], ['tm', 'Thời gian', 'var']],
    b: (x) => `didregress (${x.y}) (${x.t}), group(${x.g}) time(${x.tm})` },
  hist: { t: 'Histogram', f: [['v', 'Biến', 'var'], ['n', 'Đường phân phối chuẩn', 'check'], ['by', 'Theo nhóm', 'var']],
    b: (x) => `histogram ${x.v}${(x.n || x.by) ? ',' : ''}${x.n ? ' normal' : ''}${x.by ? ` by(${x.by})` : ''}` },
  scatter: { t: 'Biểu đồ phân tán', f: [['y', 'Trục Y', 'var'], ['x', 'Trục X', 'var'], ['fit', 'Thêm đường hồi quy', 'check']],
    b: (x) => x.fit ? `twoway (scatter ${x.y} ${x.x}) (lfit ${x.y} ${x.x})` : `scatter ${x.y} ${x.x}` },
  line: { t: 'Biểu đồ đường', f: [['y', 'Trục Y', V], ['x', 'Trục X', 'var']], b: (x) => `twoway line ${x.y} ${x.x}, sort` },
  box: { t: 'Box plot', f: [['v', 'Biến', 'var'], ['o', 'Theo nhóm', 'var']], b: (x) => `graph box ${x.v}${x.o ? `, over(${x.o})` : ''}` },
  bar: { t: 'Biểu đồ cột (trung bình)', f: [['v', 'Biến', 'var'], ['o', 'Theo nhóm', 'var']], b: (x) => `graph bar (mean) ${x.v}${x.o ? `, over(${x.o})` : ''}` },
  gen: { t: 'Tạo biến mới (generate)', f: [['n', 'Tên biến mới', 'text:'], ['e', 'Biểu thức', 'text:']], b: (x) => `generate ${x.n} = ${x.e}` },
  keepif: { t: 'Giữ quan sát theo điều kiện', f: [['e', 'Điều kiện (vd. year >= 2018)', 'text:']], b: (x) => `keep if ${x.e}` },
  xtset: { t: 'Khai báo dữ liệu bảng (xtset)', f: [['p', 'Biến panel', 'var'], ['t', 'Biến thời gian', 'var']], b: (x) => `xtset ${x.p} ${x.t}` },
};
const MENUS = [
  ['File', [['Mở dữ liệu (.dta/.csv/.xlsx) từ máy…', () => $('#fileUp').click()], ['Mở thư mục làm việc…', mountFolder], ['-'],
    ['Lưu dữ liệu hiện tại…', () => { const n = prompt('Tên file (.dta):', 'data.dta'); if (n) run(`save "${n}", replace`, true, `save "${n}", replace`); }],
    ['Tải tất cả kết quả (.zip)', () => worker.postMessage({ type: 'zip' })], ['-'],
    ['Do-file mới', () => { D.open('untitled' + (Date.now() % 1000) + '.do', ''); showTab('editor'); }], ['Mở do-file…', () => $('#doOpen').click()]]],
  ['Edit', [['Xóa cửa sổ Results', () => { resultsEl.innerHTML = ''; }], ['Sao chép Results', () => navigator.clipboard.writeText(resultsEl.textContent)]]],
  ['Data', [['Describe data', 'describe'], ['Data Editor (sửa)', () => { $('#dEdit').checked = true; showTab('data'); }], ['Data Browser (xem)', () => { $('#dEdit').checked = false; showTab('data'); }], ['-'],
    ['Tạo biến mới…', 'gen'], ['Giữ quan sát theo điều kiện…', 'keepif'], ['Khai báo dữ liệu bảng (xtset)…', 'xtset'], ['-'],
    ['Kiểm tra giá trị thiếu', () => run('misstable summarize', true, 'misstable summarize')], ['Codebook', () => run('codebook', true, 'codebook')]]],
  ['Graphics', [['Histogram…', 'hist'], ['Biểu đồ phân tán…', 'scatter'], ['Biểu đồ đường…', 'line'], ['Box plot…', 'box'], ['Biểu đồ cột…', 'bar']]],
  ['Statistics', [['Thống kê tóm tắt…', 'summarize'], ['Bảng tần suất…', 'tab1'], ['Bảng chéo…', 'tab2'], ['tabstat…', 'tabstat'], ['Tương quan…', 'corr'], ['Kiểm định t…', 'ttest'], ['-'],
    ['Hồi quy OLS…', 'reg'], ['Logit…', 'logit'], ['Poisson / NegBin…', 'count'], ['Gravity PPML (ppmlhdfe)…', 'ppml'], ['Dữ liệu bảng (xtreg)…', 'xtreg'], ['-'],
    ['IV / 2SLS…', 'iv'], ['RDD (rdrobust)…', 'rd'], ['PSM (psmatch2)…', 'psm'], ['DID (didregress)…', 'did']]],
  ['Help', [['Danh sách lệnh (help)', () => run('help', true, 'help')], ['Hướng dẫn sử dụng web', () => $('#helpDlg').showModal()]]],
];
function buildMenus() {
  const mb = $('#menubar');
  for (const [title, items] of MENUS) {
    const mi = document.createElement('div'); mi.className = 'mi'; mi.textContent = title;
    const dd = document.createElement('div'); dd.className = 'dd';
    for (const [lab, act] of items) {
      if (lab === '-') { dd.appendChild(document.createElement('hr')); continue; }
      const it = document.createElement('div'); it.textContent = lab;
      it.onclick = () => {
        if (typeof act === 'function') act();
        else if (DIALOGS[act]) openDialog(act);
        else { showTab('results'); run(act, true, act); }
      };
      dd.appendChild(it);
    }
    mi.appendChild(dd); mb.appendChild(mi);
  }
}
function openDialog(key) {
  const spec = DIALOGS[key];
  $('#cdTitle').textContent = spec.t;
  const body = $('#cdBody'); body.innerHTML = '';
  const get = {};
  for (const [id, lab, type] of spec.f) {
    const l = document.createElement('label'); l.textContent = lab; body.appendChild(l);
    let el;
    if (type === V || type === 'var') {
      el = document.createElement('select'); if (type === V) el.multiple = true;
      if (type === 'var') el.innerHTML = '<option value=""></option>';
      for (const v of state.vars) { const o = document.createElement('option'); o.value = v.name; o.textContent = v.label ? `${v.name} — ${v.label}` : v.name; el.appendChild(o); }
      get[id] = () => [...el.selectedOptions].map((o) => o.value).filter(Boolean).join(' ');
    } else if (type === 'check') {
      el = document.createElement('input'); el.type = 'checkbox'; l.prepend(el); get[id] = () => el.checked;
      el.onchange = () => preview(); continue;
    } else if (type.startsWith('sel:')) {
      el = document.createElement('select');
      for (const o of type.slice(4).split('|')) { const op = document.createElement('option'); op.value = o; op.textContent = o || '(mặc định)'; el.appendChild(op); }
      get[id] = () => el.value;
    } else {
      el = document.createElement('input'); el.type = 'text'; el.value = type.split(':')[1] || ''; get[id] = () => el.value.trim();
    }
    el.oninput = el.onchange = () => preview();
    body.appendChild(el);
  }
  function build() { const x = {}; for (const k in get) x[k] = get[k](); return spec.b(x).replace(/\s+/g, ' ').trim(); }
  function preview() { $('#cdPrev').textContent = '. ' + build(); }
  preview();
  $('#cdOk').onclick = () => { const c = build(); $('#cmdDlg').close(); showTab('results'); run(c, true, c); };
  $('#cdCancel').onclick = () => $('#cmdDlg').close();
  $('#cmdDlg').showModal();
}

// ---------------------------------------------------------------- misc
$('#btnStop').onclick = () => {
  if (!busy) return;
  worker.terminate(); busy = false; ready = false;
  append('err', '\n--Break--\n(Đã dừng. Engine được khởi động lại; dữ liệu trong bộ nhớ đã bị xóa.)\n');
  startWorker();
};
$('#btnHelp').onclick = (e) => { e.preventDefault(); $('#helpDlg').showModal(); };
buildMenus();
startWorker();
