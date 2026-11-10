# motion_check.py — continuidade temporal: diferenca media entre quadros consecutivos de cada plano
import sys, os, time
os.environ.setdefault('FILM_H', '720')
sys.path.insert(0, '/home/claude/filme/src')
import numpy as np, cv2
import timeline
from core import *
names = sys.argv[1].split(',') if len(sys.argv) > 1 else timeline.ORDER
nfr = int(sys.argv[2]) if len(sys.argv) > 2 else 20
for name in names:
    sh = timeline.make(name); sh.setup()
    i0 = int(sh.n * 0.45)
    prev = None; ds = []; t0 = time.time(); means = []
    strip = []
    for i in range(i0, i0 + nfr):
        fr = sh.frame(i).astype(np.float32)
        means.append(fr.mean())
        if prev is not None:
            ds.append(np.abs(fr - prev).mean())
        prev = fr
        if (i - i0) % 4 == 0 and len(strip) < 5:
            strip.append(cv2.resize(fr.astype(np.uint8), (512, 288), interpolation=cv2.INTER_AREA))
    el = (time.time() - t0) / nfr
    ds = np.array(ds)
    print('%-11s dif media %.2f  max %.2f  min %.2f  (razao max/mediana %.2f)  lum %.1f..%.1f  %.2fs/q' % (
        name, ds.mean(), ds.max(), ds.min(), ds.max() / max(np.median(ds), 1e-3), min(means), max(means), el), flush=True)
    cv2.imwrite('/home/claude/filme/work/mot_%s.jpg' % name, np.ascontiguousarray(np.hstack(strip)[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 85])
