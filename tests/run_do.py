import sys, os, time
import matplotlib; matplotlib.use("Agg")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'py'))
import matplotlib
matplotlib.use('Agg')
from stata import Interp
wd = sys.argv[2] if len(sys.argv) > 2 else '/tmp/stwork'
os.makedirs(wd, exist_ok=True)
src = open(os.path.abspath(sys.argv[1]), encoding="utf-8", errors="replace").read()
ip = Interp(workdir=wd)

t = time.time()
rc = ip.run(src)
sys.stdout.write(ip.out.text())
print(f'\n#### rc={rc} time={time.time()-t:.1f}s')
