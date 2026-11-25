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
