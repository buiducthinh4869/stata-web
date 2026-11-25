/* Shared do-file editor: syntax highlighting overlay + multiple tabs. */
(function () {
  const CMDS = ('use save clear set gen generate replace egen drop keep rename label sort gsort by bysort merge append ' +
    'collapse reshape expand describe des codebook list summarize sum su tabulate tab tabstat ttest mean correlate pwcorr ' +
    'regress reg xtreg reghdfe areg ivregress ivreg2 logit probit predict test testparm lincom margins marginsplot ' +
    'estimates est eststo esttab estout outreg2 estpost hausman estat xtset tsset twoway tw scatter line histogram graph ' +
    'kdensity coefplot psmatch2 pstest psgraph teffects kmatch rdrobust rdplot rddensity rdbwselect synth synth2 csdid ' +
    'csdid_plot sdid didregress xtdidregress ritest weakivtest display di local global scalar matrix mat foreach ' +
    'forvalues forval while if else quietly qui capture cap noisily preserve restore import export log count recode ' +
    'encode decode destring tostring format order duplicates misstable xtsum xtdescribe tempfile tempvar postfile post ' +
    'postclose file cd mkdir do run exit version ssc net help which assert confirm vif').split(' ');
  const KW = new Set(CMDS);
  const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  function highlight(src) {
    let out = '';
    let i = 0;
    const n = src.length;
    let lineStart = true;
    let inBlock = false;
    while (i < n) {
      if (inBlock) {
        const j = src.indexOf('*/', i);
        const end = j < 0 ? n : j + 2;
        out += '<span class="hc">' + esc(src.slice(i, end)) + '</span>';
        i = end; inBlock = false; continue;
      }
      const c = src[i];
      if (src.startsWith('/*', i)) { inBlock = true; continue; }
      if (lineStart) {
        const m = /^[ \t]*/.exec(src.slice(i));
        const ws = m[0];
        if (src[i + ws.length] === '*') {
          const j = src.indexOf('\n', i);
          const end = j < 0 ? n : j;
          out += '<span class="hc">' + esc(src.slice(i, end)) + '</span>';
          i = end; continue;
        }
        out += ws; i += ws.length;
        const w = /^[A-Za-z_][A-Za-z0-9_]*/.exec(src.slice(i));
        if (w && KW.has(w[0])) { out += '<span class="hk">' + w[0] + '</span>'; i += w[0].length; }
        lineStart = false; continue;
      }
      if (c === '\n') { out += '\n'; i++; lineStart = true; continue; }
      if (src.startsWith('//', i) && (i === 0 || /\s/.test(src[i - 1]))) {
        const j = src.indexOf('\n', i);
        const end = j < 0 ? n : j;
        out += '<span class="hc">' + esc(src.slice(i, end)) + '</span>';
        i = end; continue;
      }
      if (c === '"') {
        let j = src.indexOf('"', i + 1);
        const nl = src.indexOf('\n', i + 1);
        if (j < 0 || (nl >= 0 && nl < j)) j = (nl < 0 ? n : nl) - 1;
        out += '<span class="hs">' + esc(src.slice(i, j + 1)) + '</span>';
        i = j + 1; continue;
      }
      if (c === '`') {
        const j = src.indexOf("'", i + 1);
        const nl = src.indexOf('\n', i + 1);
        if (j > 0 && (nl < 0 || j < nl)) { out += '<span class="hm">' + esc(src.slice(i, j + 1)) + '</span>'; i = j + 1; continue; }
      }
      if (c === '$') {
        const w = /^\$\{?[A-Za-z_][A-Za-z0-9_]*\}?/.exec(src.slice(i));
        if (w) { out += '<span class="hm">' + esc(w[0]) + '</span>'; i += w[0].length; continue; }
      }
      const w = /^[A-Za-z_][A-Za-z0-9_]*/.exec(src.slice(i));
      if (w) {
        const prev = src.slice(Math.max(0, i - 2), i);
        const word = w[0];
        out += (KW.has(word) && /(:\s|^\s|\{ )$/.test(prev)) ? '<span class="hk">' + word + '</span>' : word;
        i += word.length; continue;
      }
      out += esc(c); i++;
    }
    return out + '\n';
  }

  /** Attach highlighting + tabs to an editor area.
   * opts: {textarea, overlay, lineno, tabbar, nameEl, onChange} */
  window.DoEditor = function (opts) {
    const ta = opts.textarea, ov = opts.overlay, ln = opts.lineno;
    const self = { docs: [{ name: 'untitled.do', text: '' }], cur: 0 };
    function paint() {
      ov.innerHTML = highlight(ta.value);
      const nlines = ta.value.split('\n').length;
      ln.textContent = Array.from({ length: nlines }, (_, k) => k + 1).join('\n');
      ov.scrollTop = ta.scrollTop; ov.scrollLeft = ta.scrollLeft; ln.scrollTop = ta.scrollTop;
    }
    function renderTabs() {
      if (!opts.tabbar) return;
      opts.tabbar.innerHTML = '';
      self.docs.forEach((d, k) => {
        const b = document.createElement('span');
        b.className = 'dtab' + (k === self.cur ? ' on' : '');
        b.textContent = d.name;
        b.onclick = () => self.select(k);
        const x = document.createElement('i'); x.textContent = '×'; x.title = 'Đóng';
        x.onclick = (e) => { e.stopPropagation(); self.close(k); };
        b.appendChild(x);
        opts.tabbar.appendChild(b);
      });
      const plus = document.createElement('span'); plus.className = 'dtab plus'; plus.textContent = '+'; plus.title = 'Do-file mới';
      plus.onclick = () => self.open('untitled' + (self.docs.length + 1) + '.do', '');
      opts.tabbar.appendChild(plus);
      if (opts.nameEl) opts.nameEl.textContent = self.docs[self.cur].name;
    }
    self.save = () => { self.docs[self.cur].text = ta.value; };
    self.select = (k) => { self.save(); self.cur = k; ta.value = self.docs[k].text; paint(); renderTabs(); };
    self.open = (name, text) => {
      self.save();
      const ex = self.docs.findIndex((d) => d.name === name);
      if (ex >= 0) { self.docs[ex].text = text; self.cur = ex; }
      else if (self.docs.length === 1 && !self.docs[0].text && self.docs[0].name.startsWith('untitled')) { self.docs[0] = { name, text }; self.cur = 0; }
      else { self.docs.push({ name, text }); self.cur = self.docs.length - 1; }
      ta.value = self.docs[self.cur].text; paint(); renderTabs();
    };
    self.close = (k) => {
      self.save();
      self.docs.splice(k, 1);
      if (!self.docs.length) self.docs.push({ name: 'untitled.do', text: '' });
      self.cur = Math.min(self.cur, self.docs.length - 1);
      ta.value = self.docs[self.cur].text; paint(); renderTabs();
    };
    self.name = () => self.docs[self.cur].name;
    self.rename = (n) => { self.docs[self.cur].name = n; renderTabs(); };
    self.text = () => ta.value;
    self.setText = (t) => { ta.value = t; self.save(); paint(); };
    self.paint = paint;
    ta.addEventListener('input', () => { self.save(); paint(); if (opts.onChange) opts.onChange(); });
    ta.addEventListener('scroll', () => { ov.scrollTop = ta.scrollTop; ov.scrollLeft = ta.scrollLeft; ln.scrollTop = ta.scrollTop; });
    ta.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') { e.preventDefault(); ta.setRangeText('    ', ta.selectionStart, ta.selectionEnd, 'end'); self.save(); paint(); }
    });
    paint(); renderTabs();
    return self;
  };
})();
