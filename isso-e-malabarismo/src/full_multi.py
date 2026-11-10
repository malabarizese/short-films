import sys, os, time
os.environ['FILM_H'] = '1440'
sys.path.insert(0, '/home/claude/filme/src')
import numpy as np, cv2
import timeline
from core import *
for spec in sys.argv[1:]:
    name, t = spec.split('@'); t = float(t)
    sh = timeline.make(name); t0 = time.time(); sh.setup()
    fr = sh.frame(int(round(t * FPS))); t1 = time.time()
    fr = sh.frame(int(round(t * FPS)) + 1); t2 = time.time()
    print('%s: preparo+1o %.1fs, seguinte %.2fs' % (name, t1 - t0, t2 - t1), flush=True)
    cv2.imwrite('/home/claude/filme/work/full_%s.png' % name, np.ascontiguousarray(fr[..., ::-1]))
    cv2.imwrite('/home/claude/filme/work/fullh_%s.jpg' % name, cv2.resize(np.ascontiguousarray(fr[..., ::-1]), (1280, 720), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])
