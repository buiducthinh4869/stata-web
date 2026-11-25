/* UI for Stata Web. All computation happens in worker.js (Pyodide). */
const $ = (s) => document.querySelector(s);
let worker, busy = false, ready = false;
let history = [], hpos = 0;
let state = { vars: [], files: [], N: 0 };
let graphs = {};
let dataStart = 0;
const DCOUNT = 200;
let mountedName = null;
let edFile = 'untitled.do';
const resultsEl = $('#results');
const MAXCHARS = 2_000_000;

function startWorker() {
  worker = new Worker('worker.js?v=' + (window.APP_V || Date.now()));
  worker.onmessage = onMsg;
  worker.postMessage({ type: 'init' });
}

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
let pendingScroll = false;
setInterval(() => { if (pendingScroll) { resultsEl.scrollTop = resultsEl.scrollHeight; pendingScroll = false; } }, 80);

function setStatus(t) { $('#status').textContent = t; if (window.edChan) edChan.postMessage({ type: 'status', text: t }); }
const edChan = window.edChan = new BroadcastChannel('stata-web');
edChan.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === 'hello') edChan.postMessage({ type: 'load', text: $('#editor').value, name: edFile });
  else if (m.type === 'sync') { $('#editor').value = m.text; edFile = m.name; $('#edName').textContent = m.name; updateLines(); }
  else if (m.type === 'run') { showTab('results'); run(m.code, true, m.label); }
  else if (m.type === 'save') { $('#editor').value = m.text; edFile = m.name; saveDo(); }
};

function onMsg(ev) {
  const m = ev.data;
  switch (m.type) {
    case 'status': setStatus(m.text); break;
    case 'ready':
      ready = true;
      setStatus('Sẵn sàng');
      for (const f of m.examples) {
        const o = document.createElement('option'); o.value = f; o.textContent = f; $('#exSel').appendChild(o);
      }
      append('txt', '  ___  ____  ____  ____  ____ (R)\n /__    /   ____/   /   ____/   Stata Web — mô phỏng Stata trong trình duyệt\n___/   /   /___/   /   /___/    cho giảng dạy suy diễn nhân quả\n\n');
      append('txt', 'Gõ lệnh vào ô Command, hoặc chọn một do-file mẫu rồi bấm "Chạy tất cả".\n\n');
      break;
    case 'out': append(m.style, m.text); break;
    case 'graph': {
      const blob = new Blob([m.png], { type: 'image/png' });
      if (graphs[m.name]) URL.revokeObjectURL(graphs[m.name]);
      graphs[m.name] = URL.createObjectURL(blob);
      refreshGraphs(m.name);
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
      const blob = new Blob([m.data]);
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob); a.download = m.path.split('/').pop(); a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 4000);
      break;
    }
    case 'text':
      if (m.purpose === 'editor') {
        $('#editor').value = m.text; edFile = m.path.split('/').pop(); $('#edName').textContent = edFile; updateLines();
        edChan.postMessage({ type: 'load', text: m.text, name: edFile });
        if (!(popWin && !popWin.closed)) showTab('editor');
      }
      break;
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
  li.textContent = c; li.title = c;
  li.onclick = () => { $('#cmd').value = c; $('#cmd').focus(); };
  $('#review').appendChild(li);
  li.scrollIntoView({ block: 'end' });
}

// ---- command line
$('#cmd').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    const c = e.target.value.trim();
    if (!c) return;
    e.target.value = '';
    showTab('results');
    run(c, true, c);
  } else if (e.key === 'ArrowUp') { if (hpos > 0) { hpos--; e.target.value = history[hpos]; } e.preventDefault(); }
  else if (e.key === 'ArrowDown') { if (hpos < history.length - 1) { hpos++; e.target.value = history[hpos]; } else { hpos = history.length; e.target.value = ''; } e.preventDefault(); }
});

// ---- tabs
function showTab(t) {
  document.querySelectorAll('.tabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === t));
  document.querySelectorAll('.tab').forEach((d) => d.classList.toggle('active', d.id === 'tab-' + t));
  if (t === 'data') requestData();
}
document.querySelectorAll('.tabs button').forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));

// ---- editor
const ed = $('#editor');
function updateLines() {
  const n = ed.value.split('\n').length;
  $('#lineno').textContent = Array.from({ length: n }, (_, i) => i + 1).join('\n');
}
ed.addEventListener('input', updateLines);
ed.addEventListener('scroll', () => { $('#lineno').scrollTop = ed.scrollTop; });
ed.addEventListener('keydown', (e) => {
  if (e.key === 'Tab') { e.preventDefault(); const s = ed.selectionStart; ed.setRangeText('    ', s, ed.selectionEnd, 'end'); updateLines(); }
  if ((e.ctrlKey || e.metaKey) && (e.key === 'd' || e.key === 'D')) { e.preventDefault(); e.shiftKey ? runSel() : runAll(); }
  if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); saveDo(); }
});
function runAll() { showTab('results'); run(ed.value, true, `do ${edFile}`); }
function runSel() {
  let s = ed.value.substring(ed.selectionStart, ed.selectionEnd);
  if (!s.trim()) { // current line
    const v = ed.value, p = ed.selectionStart;
    const a = v.lastIndexOf('\n', p - 1) + 1, b = v.indexOf('\n', p);
    s = v.substring(a, b < 0 ? v.length : b);
  }
  showTab('results'); run(s, true, s.split('\n')[0].slice(0, 80));
}
$('#edRunAll').onclick = runAll;
$('#edRunSel').onclick = runSel;
function saveDo() {
  const data = new TextEncoder().encode(ed.value);
  worker.postMessage({ type: 'upload', name: edFile, data: data.buffer }, [data.buffer]);
  if (!mountedName) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([ed.value], { type: 'text/plain' })); a.download = edFile; a.click();
  }
  setStatus(`Đã lưu ${edFile}`);
}
$('#edSave').onclick = saveDo;
let popWin = null;
$('#edPop').onclick = () => {
  if (popWin && !popWin.closed) { popWin.focus(); return; }
  popWin = window.open('editor.html?v=' + (window.APP_V || ''), 'stataDoEditor', 'width=900,height=700');
  showTab('results');
};
$('#doOpen').onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  ed.value = await f.text(); edFile = f.name; $('#edName').textContent = f.name; updateLines(); e.target.value = '';
  edChan.postMessage({ type: 'load', text: ed.value, name: edFile });
};
$('#exSel').onchange = (e) => {
  const f = e.target.value; if (!f) return;
  worker.postMessage({ type: 'readtext', path: '/examples/do/' + f, purpose: 'editor' });
  e.target.value = '';
};

// ---- variables
function renderVars() {
  const flt = $('#varFilter').value.toLowerCase();
  const tb = $('#vars tbody'); tb.innerHTML = '';
  for (const v of state.vars) {
    if (flt && !v.name.toLowerCase().includes(flt) && !v.label.toLowerCase().includes(flt)) continue;
    const tr = document.createElement('tr');
    tr.innerHTML = `<td></td><td></td>`;
    tr.children[0].textContent = v.name; tr.children[1].textContent = v.label;
    tr.onclick = () => {
      tb.querySelectorAll('tr').forEach((r) => r.classList.remove('sel')); tr.classList.add('sel');
      $('#props').innerHTML = `<b>${esc(v.name)}</b><br>Label: ${esc(v.label) || '—'}<br>Type: ${v.type}<br>Format: ${v.format}<br>Value label: ${esc(v.vallab) || '—'}<br><span class="muted">${state.N.toLocaleString()} obs</span>`;
    };
    tr.ondblclick = () => { const c = $('#cmd'); c.value += (c.value && !c.value.endsWith(' ') ? ' ' : '') + v.name; c.focus(); };
    tb.appendChild(tr);
  }
}
$('#varFilter').oninput = renderVars;
const esc = (s) => String(s || '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

// ---- files
function renderFiles() {
  const ul = $('#files'); ul.innerHTML = '';
  for (const f of state.files) {
    const li = document.createElement('li');
    const fn = document.createElement('span'); fn.className = 'fn'; fn.textContent = f.path; fn.title = `${f.path} (${(f.size / 1024).toFixed(1)} KB)`;
    fn.onclick = () => {
      const p = f.path.toLowerCase();
      if (p.endsWith('.do')) worker.postMessage({ type: 'readtext', path: f.path, purpose: 'editor' });
      else if (p.endsWith('.dta')) { showTab('results'); run(`use "${f.path}", clear`, true, `use "${f.path}", clear`); }
      else if (/\.(txt|log|tex|csv)$/.test(p)) { showTab('results'); run(`type "${f.path}"`, true, `type "${f.path}"`); }
      else worker.postMessage({ type: 'download', path: f.path });
    };
    const dl = document.createElement('button'); dl.textContent = '⬇'; dl.title = 'Tải về';
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
$('#btnMount').onclick = async () => {
  if (!window.showDirectoryPicker) { alert('Trình duyệt này chưa hỗ trợ mở thư mục. Hãy dùng Chrome hoặc Edge, hoặc dùng nút "Tải file lên".'); return; }
  try {
    const h = await window.showDirectoryPicker({ mode: 'readwrite' });
    worker.postMessage({ type: 'mount', handle: h });
  } catch (e) { /* cancelled */ }
};
// drag and drop
let dragN = 0;
window.addEventListener('dragenter', (e) => { e.preventDefault(); dragN++; $('#drop').classList.add('on'); });
window.addEventListener('dragleave', () => { if (--dragN <= 0) $('#drop').classList.remove('on'); });
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => { e.preventDefault(); dragN = 0; $('#drop').classList.remove('on'); uploadFiles(e.dataTransfer.files); });

// ---- data browser
function requestData() { if (ready && !busy) worker.postMessage({ type: 'data', start: dataStart, count: DCOUNT }); }
function renderData(d) {
  const t = $('#dtable');
  const strCols = new Set(state.vars.filter((v) => v.type.startsWith('str')).map((v) => v.name));
  let h = '<thead><tr><th></th>' + d.cols.map((c) => `<th>${esc(c)}</th>`).join('') + '</tr></thead><tbody>';
  d.rows.forEach((r, i) => {
    h += `<tr><th>${d.start + i + 1}</th>` + r.map((x, k) => `<td class="${strCols.has(d.cols[k]) ? 'str' : ''}">${esc(x)}</td>`).join('') + '</tr>';
  });
  t.innerHTML = h + '</tbody>';
  $('#dInfo').textContent = d.N ? ` Quan sát ${d.start + 1}–${Math.min(d.N, d.start + DCOUNT)} / ${d.N.toLocaleString()} ` : ' (chưa có dữ liệu) ';
}
$('#dPrev').onclick = () => { dataStart = Math.max(0, dataStart - DCOUNT); requestData(); };
$('#dNext').onclick = () => { if (dataStart + DCOUNT < state.N) { dataStart += DCOUNT; requestData(); } };
$('#dRefresh').onclick = requestData;

// ---- graphs
function refreshGraphs(sel) {
  const s = $('#gSel'); s.innerHTML = '';
  for (const n of Object.keys(graphs)) { const o = document.createElement('option'); o.value = n; o.textContent = n; s.appendChild(o); }
  if (sel) s.value = sel;
  $('#gImg').src = graphs[s.value] || '';
  $('#gcount').textContent = Object.keys(graphs).length ? `(${Object.keys(graphs).length})` : '';
}
$('#gSel').onchange = () => refreshGraphs();
$('#gSave').onclick = () => { const n = $('#gSel').value; if (!graphs[n]) return; const a = document.createElement('a'); a.href = graphs[n]; a.download = n + '.png'; a.click(); };

// ---- misc
$('#btnStop').onclick = () => {
  if (!busy) return;
  worker.terminate(); busy = false; ready = false;
  append('err', '\n--Break--\n(Đã dừng. Engine được khởi động lại; dữ liệu trong bộ nhớ đã bị xóa.)\n');
  startWorker();
};
$('#btnHelp').onclick = (e) => { e.preventDefault(); $('#helpDlg').showModal(); };
updateLines();
startWorker();
