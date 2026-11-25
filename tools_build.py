"""Build py.zip (engine) and examples/manifest.json. Run before committing."""
import json, os, zipfile
root = os.path.dirname(os.path.abspath(__file__))
with zipfile.ZipFile(os.path.join(root, 'py.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
    for pkg in ('stata', 'rdpkg'):
        for f in sorted(os.listdir(os.path.join(root, 'py', pkg))):
            if f.endswith('.py'):
                z.write(os.path.join(root, 'py', pkg, f), f'{pkg}/{f}')
ex = {'data': sorted(os.listdir(os.path.join(root, 'examples', 'data'))),
      'do': sorted(os.listdir(os.path.join(root, 'examples', 'do')))}
json.dump(ex, open(os.path.join(root, 'examples', 'manifest.json'), 'w'), ensure_ascii=False, indent=1)
print('built', os.path.getsize(os.path.join(root, 'py.zip')), 'bytes;', len(ex['do']), 'do-files')

import re, time
v = time.strftime('%Y%m%d%H%M%S')
for fn in ('index.html', 'editor.html'):
    path = os.path.join(root, fn)
    txt = open(path, encoding='utf-8').read()
    txt = re.sub(r'(\?v=)(__V__|\d{14})', lambda m: m.group(1) + v, txt)
    txt = re.sub(r'(APP_V=")(__V__|\d{14})', lambda m: m.group(1) + v, txt)
    open(path, 'w', encoding='utf-8').write(txt)
print('version', v)
